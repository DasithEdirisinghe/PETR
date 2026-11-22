#!/usr/bin/env python3
"""Trace V2 query roles from PETR's official checkpoint into pedestrian AP.

Use V6 solely to configure the V2 rules; load PETR-R50's epoch-24 weights.
V2 roles and original/rule HM assignments come from prepare_rule_v2_layer.
The top-300 mapping reproduces NMSFreeCoder. AP uses
nuScenes accumulate/calc_ap on official filtered train GT and newly inferred
global detections for the selected frames. It is a subset diagnostic, not
official full-train AP. The audit runs only final-layer rule preparation;
ordinary PETR classification/regression losses are not evaluated.
"""

import argparse
import csv
import importlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from reranking_oracle import greedy_match_records, official_ap, oracle_order


def stable_move(order, selected, front):
    selected = set(selected)
    a = [i for i in order if i in selected]
    b = [i for i in order if i not in selected]
    return a + b if front else b + a


def deterministic_train_config(config):
    """Retain train GT while using the unaugmented inference image view."""
    data = config.data.train.copy()
    pipeline = []
    for step in data['pipeline']:
        step = dict(step)
        if step['type'] == 'GlobalRotScaleTransImage':
            continue
        if step['type'] == 'ResizeCropFlipImage':
            step['training'] = False
        pipeline.append(step)
    data['pipeline'] = pipeline
    data['test_mode'] = False
    # NuScenesDataset normally replaces an empty-GT training item with a
    # randomly chosen sample. Auditing needs the requested token itself.
    data['filter_empty_gt'] = False
    return data


def retained_entries(logits, encoded, coder, pedestrian_id):
    """Reproduce NMSFreeCoder topk and post-center filtering, with query IDs."""
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

    n_classes = coder.num_classes
    if logits.size(1) != n_classes:
        raise ValueError('Class dimension differs from NMSFreeCoder')
    scores, flat = logits.sigmoid().reshape(-1).topk(coder.max_num)
    labels = flat % n_classes
    queries = flat // n_classes
    decoded = denormalize_bbox(encoded[queries], coder.pc_range)
    bounds = torch.as_tensor(coder.post_center_range, device=decoded.device,
                             dtype=decoded.dtype)
    keep = ((decoded[:, :3] >= bounds[:3]).all(1) &
            (decoded[:, :3] <= bounds[3:]).all(1))
    if coder.score_threshold:
        keep &= scores > coder.score_threshold
    keep &= labels == pedestrian_id
    for rank in torch.nonzero(keep, as_tuple=False).flatten().tolist():
        yield (int(queries[rank]), float(scores[rank]),
               decoded[rank].detach().cpu().tolist(), rank + 1)


def detection_box(token, center, score, name='pedestrian'):
    from nuscenes.eval.detection.data_classes import DetectionBox
    return DetectionBox(
        sample_token=token, translation=tuple(float(x) for x in center[:3]),
        size=(1.0, 1.0, 1.0), rotation=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0), ego_translation=(0.0, 0.0, 0.0), num_pts=1,
        detection_name=name, detection_score=float(score), attribute_name='')


def roles_for_frame(query_count, class_id, prepared, geometry, logits,
                    encoded, gt, labels, rule_cfg):
    rule, original, positives, duplicates, unsupported = prepared[:5]
    positive = {int(q) for b, q, c in positives if int(b) == 0 and int(c) == class_id}
    duplicate = {int(q) for b, q, c in duplicates if int(b) == 0 and int(c) == class_id}
    negative = {int(q) for b, q, c in unsupported if int(b) == 0 and int(c) == class_id}
    wrong_class = set()
    if geometry is not None and gt.size(0):
        from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
        overlap = geometry[0]
        pred = denormalize_bbox(encoded[0], None)
        distance = torch.linalg.vector_norm(
            pred[:, None, :2] - gt[None, :, :2], dim=-1)
        best, owner = overlap.max(dim=1)
        second = (overlap.topk(2, dim=1).values[:, 1]
                  if gt.size(0) > 1 else torch.zeros_like(best))
        clear = ((best >= float(rule_cfg.get('min_projected_iou', 0.15))) &
                 (best-second >= float(rule_cfg.get('owner_iou_margin', 0.05))))
        same = labels == class_id
        if same.any():
            supported = ((overlap[:, same] > float(rule_cfg.get('unsupported_max_iou', 0.05))) |
                         (distance[:, same] < float(rule_cfg.get('unsupported_min_center', 4.0)))).any(1)
        else:
            supported = torch.zeros(query_count, dtype=torch.bool,
                                    device=logits.device)
        mask = (clear & ~supported & (labels[owner] != class_id) &
                (rule[0].labels != class_id) &
                (logits[0, :, class_id].sigmoid() >=
                 float(rule_cfg.get('unsupported_min_score', 0.05))))
        wrong_class = set(torch.nonzero(mask, as_tuple=False).flatten().tolist())
    result = []
    for query in range(query_count):
        if query in positive:
            role = 'certified_winner'
        elif query in duplicate:
            role = 'inferior_duplicate'
        elif query in negative:
            role = ('wrong_class' if query in wrong_class
                    else 'unsupported_background')
        elif (int(rule[0].gt_inds[query]) > 0 and
              int(rule[0].labels[query]) == class_id):
            role = 'hm_fallback'
        else:
            role = 'undecided'
        result.append(role)
    return result


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_thresholds(gt_by_sample, entries, cfg, thresholds):
    predictions = sorted((entry['box'] for entry in entries),
                         key=lambda box: box.detection_score, reverse=True)
    # Scores are unique in practice; stable sort preserves prediction/role map
    # when they are tied. Use object identity to map the corresponding role.
    roles_by_box = {id(entry['box']): entry['role'] for entry in entries}
    roles = [roles_by_box[id(box)] for box in predictions]
    baseline_order = list(range(len(predictions)))
    result = []
    for threshold in thresholds:
        baseline_ap = official_ap(gt_by_sample, predictions, None, cfg,
                                  threshold, 'pedestrian')
        oracle_rank, selected = oracle_order(gt_by_sample, predictions,
                                             threshold)
        oracle_ap = official_ap(gt_by_sample, predictions, oracle_rank, cfg,
                                threshold, 'pedestrian')
        records = greedy_match_records(gt_by_sample, predictions,
                                       baseline_order, threshold)
        baseline_tp = {i for i, _ in records.values()}
        last_selected = max(selected) if selected else -1
        baseline_fp = [i for i in baseline_order if i not in baseline_tp]
        actions = {
            'promote_certified': stable_move(
                baseline_order,
                [i for i, role in enumerate(roles) if role == 'certified_winner'],
                front=True),
            'demote_duplicates': stable_move(
                baseline_order,
                [i for i in baseline_fp if roles[i] == 'inferior_duplicate'],
                front=False),
            'demote_unsupported': stable_move(
                baseline_order,
                [i for i in baseline_fp
                 if roles[i] == 'unsupported_background'], front=False),
            'demote_wrong_class': stable_move(
                baseline_order,
                [i for i in baseline_fp if roles[i] == 'wrong_class'],
                front=False),
        }
        negatives = [i for i in baseline_fp if roles[i] in
                     ('inferior_duplicate', 'unsupported_background',
                      'wrong_class')]
        positives = [i for i, role in enumerate(roles)
                     if role == 'certified_winner']
        actions['combined_v2_roles'] = stable_move(
            stable_move(baseline_order, negatives, front=False),
            positives, front=True)
        action_ap = {name: official_ap(gt_by_sample, predictions, order, cfg,
                                       threshold, 'pedestrian')
                     for name, order in actions.items()}
        for role in ('certified_winner', 'inferior_duplicate',
                     'unsupported_background', 'wrong_class', 'hm_fallback',
                     'undecided'):
            indices = [i for i, value in enumerate(roles) if value == role]
            role_fp = [i for i in indices if i not in baseline_tp]
            result.append({
                'threshold_m': threshold, 'role': role,
                'gt_count': sum(map(len, gt_by_sample.values())),
                'retained_pedestrian_count': len(predictions),
                'role_retained_count': len(indices),
                'role_baseline_tp_count': len(indices)-len(role_fp),
                'role_baseline_fp_count': len(role_fp),
                'role_fp_above_oracle_representative_count': sum(
                    i < last_selected for i in role_fp),
                'role_oracle_selected_count': sum(i in selected for i in indices),
                'role_mean_score': sum(predictions[i].detection_score
                                       for i in indices)/len(indices)
                if indices else '',
                'baseline_ap_percent': 100*baseline_ap,
                'oracle_ap_percent': 100*oracle_ap,
                'oracle_gap_pp': 100*(oracle_ap-baseline_ap),
                **{name+'_ap_percent': 100*ap for name, ap in action_ap.items()},
                **{name+'_gain_pp': 100*(ap-baseline_ap)
                   for name, ap in action_ap.items()},
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--reference-results-json', required=True, type=Path,
                        help='Use the same train sample tokens as the saved PETR oracle')
    parser.add_argument('--data-root', required=True, type=Path,
                        help='nuScenes root for official filtered train GT')
    parser.add_argument('--max-frames', type=int, default=500,
                        help='0 processes all train frames; default is a pilot')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=(0.5, 1.0, 2.0, 4.0))
    args = parser.parse_args()
    if not args.config.is_file() or not args.checkpoint.is_file():
        parser.error('Config and checkpoint must exist')
    if args.reference_results_json and (args.data_root is None or
            not (args.data_root/'v1.0-trainval').is_dir()):
        parser.error('Reference audit requires --data-root with nuScenes metadata')
    if args.max_frames < 0 or any(t not in (0.5, 1.0, 2.0, 4.0)
                                   for t in args.thresholds):
        parser.error('Invalid max-frames or threshold')
    config = Config.fromfile(str(args.config))
    if config.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**config.custom_imports)
    if config.get('plugin', False):
        importlib.import_module('projects.mmdet3d_plugin')
    if config.model.pts_bbox_head.get('rule_v2_cfg') is None:
        parser.error('Config must enable rule_v2_cfg')
    class_names = list(config.class_names)
    class_id = class_names.index('pedestrian')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = build_dataset(deterministic_train_config(config))
    reference_predictions = None
    if args.reference_results_json:
        if not args.reference_results_json.is_file():
            parser.error('Missing reference results JSON')
        reference = json.loads(args.reference_results_json.read_text())
        tokens = set(reference['results'])
        indices = [i for i, info in enumerate(dataset.data_infos)
                   if info['token'] in tokens]
        if len(indices) != len(tokens):
            parser.error('Reference results include tokens outside the train dataset')
    else:
        indices = list(range(len(dataset)))
    if args.max_frames and len(indices) > args.max_frames:
        positions = np.linspace(0, len(indices)-1, args.max_frames, dtype=int)
        indices = [indices[int(i)] for i in positions]
    info_by_token = {dataset.data_infos[i]['token']: dataset.data_infos[i]
                     for i in indices}
    if len(info_by_token) != len(indices):
        raise ValueError('Requested train indices contain duplicate sample tokens')
    from torch.utils.data import Subset
    loader = build_dataloader(Subset(dataset, indices),
                              samples_per_gpu=1, workers_per_gpu=0,
                              num_gpus=1, dist=False, shuffle=False, seed=0)
    model = build_model(config.model, train_cfg=config.get('train_cfg'),
                        test_cfg=config.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, str(args.checkpoint), map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    head = model.module.pts_bbox_head
    original_loss = head.loss
    import projects.mmdet3d_plugin.models.dense_heads.ray_rule_v2 as v2
    original_prepare = v2.prepare_rule_v2_layer
    original_geometry = v2._camera_geometry
    captured = []
    geometry_trace = []

    def capture_geometry(*call_args, **call_kwargs):
        result = original_geometry(*call_args, **call_kwargs)
        geometry_trace.append(result)
        return result

    def capture(*call_args, **call_kwargs):
        geometry_trace.clear()
        prepared = original_prepare(*call_args, **call_kwargs)
        captured.append((call_args, prepared,
                         geometry_trace[-1] if geometry_trace else None))
        return prepared

    v2._camera_geometry = capture_geometry
    v2.prepare_rule_v2_layer = capture

    def audit_loss(gt_bboxes_list, gt_labels_list, preds_dicts,
                   gt_bboxes_ignore=None, img_metas=None):
        """Run only final-layer V2 analysis; avoid ordinary loss on empty GT."""
        if gt_bboxes_ignore is not None or img_metas is None:
            raise ValueError('Audit expects ordinary PETR GT and image metadata')
        final_logits = preds_dicts['all_cls_scores'][-1].float()
        final_encoded = preds_dicts['all_bbox_preds'][-1].float()
        device = final_logits.device
        gt_tensors = [torch.cat((boxes.gravity_center, boxes.tensor[:, 3:]),
                                dim=1).to(device)
                      for boxes in gt_bboxes_list]
        v2.prepare_rule_v2_layer(
            final_logits, final_encoded, gt_tensors, gt_labels_list,
            img_metas, head.assigner, head.rule_v2_cfg)
        return {'loss_audit': final_logits.sum() * 0.0}

    head.loss = audit_loss
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_by_sample = {}
    retained = []
    trace_path = args.output_dir/'retained_pedestrian_query_roles.csv'
    trace_fields = ['sample_token', 'query_id', 'top300_rank', 'score', 'role',
                    'original_hm_gt', 'rule_gt', 'gt_count', 'in_eval_range']
    raw_counts = Counter()
    frame_count = 0
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    eval_cfg = config_factory('detection_cvpr_2019')
    pedestrian_range = float(eval_cfg.class_range['pedestrian'])
    from pyquaternion import Quaternion
    try:
        with trace_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=trace_fields)
            writer.writeheader()
            for data in loader:
                if args.max_frames and frame_count >= args.max_frames:
                    break
                captured.clear()
                with torch.no_grad():
                    model(return_loss=True, **data)
                if not captured:
                    raise RuntimeError('V2 preparation was not called')
                call_args, prepared, geometry = captured[-1]
                logits, encoded, gt_list, label_list, metas = call_args[:5]
                if logits.size(0) != 1:
                    raise AssertionError('Expected one frame per batch')
                token = str(metas[0]['sample_idx'])
                if token in gt_by_sample:
                    raise AssertionError('Duplicate train sample token')
                if token not in info_by_token:
                    raise RuntimeError('Loader returned an unrequested token: '
                                       '{}. Check dataset resampling.'.format(token))
                info = info_by_token[token]
                rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                translation = np.asarray(info['lidar2ego_translation'])

                def in_eval_range(center):
                    ego = rotation.dot(np.asarray(center[:3])) + translation
                    return math.hypot(float(ego[0]), float(ego[1])) <= pedestrian_range

                def global_center(center):
                    ego = rotation.dot(np.asarray(center[:3])) + translation
                    world_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
                    world_translation = np.asarray(info['ego2global_translation'])
                    return world_rotation.dot(ego) + world_translation

                gt = gt_list[0]
                labels = label_list[0]
                gt_by_sample[token] = [
                    detection_box(token, gt[g].tolist(), 1.0)
                    for g in range(gt.size(0))
                    if int(labels[g]) == class_id and in_eval_range(gt[g].tolist())]
                rule, original = prepared[:2]
                roles = roles_for_frame(
                    logits.size(1), class_id, prepared, geometry, logits,
                    encoded, gt, labels, head.rule_v2_cfg)
                raw_counts.update(roles)
                for q, score, center, rank in retained_entries(
                        logits[0], encoded[0], head.bbox_coder, class_id):
                    role = roles[q]
                    box = detection_box(token, global_center(center), score)
                    in_range = in_eval_range(center)
                    if in_range:
                        retained.append({'box': box, 'role': role})
                    writer.writerow({
                        'sample_token': token, 'query_id': q,
                        'top300_rank': rank, 'score': score, 'role': role,
                        'original_hm_gt': int(original[0].gt_inds[q])-1,
                        'rule_gt': int(rule[0].gt_inds[q])-1,
                        'gt_count': int(gt.size(0)),
                        'in_eval_range': int(in_range),
                    })
                frame_count += 1
                if frame_count % 50 == 0:
                    print('Traced {} train frames'.format(frame_count), flush=True)
                captured.clear()
    finally:
        head.loss = original_loss
        v2.prepare_rule_v2_layer = original_prepare
        v2._camera_geometry = original_geometry
    if not retained or not any(gt_by_sample.values()):
        raise RuntimeError('No pedestrian detections or GTs in traced frames')
    if args.reference_results_json:
        from nuscenes import NuScenes
        from nuscenes.eval.common.loaders import load_gt
        from nuscenes.eval.detection.data_classes import DetectionBox
        from nuscenes.eval.detection.evaluate import NuScenesEval
        nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                        verbose=False)
        expected = set(load_gt(nusc, 'train', DetectionBox,
                               verbose=False).sample_tokens)
        if not set(gt_by_sample).issubset(expected):
            raise ValueError('Traced sample outside official train split')
        completed = dict(reference)
        completed['results'] = dict(reference['results'])
        completed['results'].update({token: [] for token in expected-set(reference['results'])})
        source_dir = args.output_dir/'source_eval'
        source_dir.mkdir(exist_ok=True)
        completed_path = source_dir/'completed_for_loader_results_nusc.json'
        completed_path.write_text(json.dumps(completed))
        evaluator = NuScenesEval(
            nusc, config=eval_cfg, result_path=str(completed_path),
            eval_set='train', output_dir=str(source_dir), verbose=False)
        gt_by_sample = {
            token: [box for box in evaluator.gt_boxes[token]
                    if box.detection_name == 'pedestrian']
            for token in gt_by_sample}
        reference_predictions = [
            box for token in gt_by_sample for box in evaluator.pred_boxes[token]
            if box.detection_name == 'pedestrian']
    rows = analyze_thresholds(gt_by_sample, retained, eval_cfg, args.thresholds)
    reference_ap = {
        threshold: 100*official_ap(gt_by_sample, reference_predictions, None,
                                   eval_cfg, threshold, 'pedestrian')
        for threshold in args.thresholds}
    for row in rows:
        row['reference_baseline_ap_percent'] = reference_ap[row['threshold_m']]
        row['reinference_minus_reference_pp'] = (
            row['baseline_ap_percent'] - reference_ap[row['threshold_m']])
    write_csv(args.output_dir/'pedestrian_role_ap_decomposition.csv', rows,
              list(rows[0]))
    summary = {
        'config': str(args.config), 'checkpoint': str(args.checkpoint),
        'reference_results_json': str(args.reference_results_json)
        if args.reference_results_json else None,
        'frames': frame_count, 'retained_pedestrian_count': len(retained),
        'reference_pedestrian_count': len(reference_predictions),
        'raw_query_role_counts': dict(raw_counts),
        'mean_baseline_ap_percent': sum(r['baseline_ap_percent'] for r in rows)/len(rows),
        'mean_oracle_ap_percent': sum(r['oracle_ap_percent'] for r in rows)/len(rows),
        'mean_combined_v2_roles_ap_percent': sum(
            r['combined_v2_roles_ap_percent'] for r in rows)/len(rows),
        'mean_reference_baseline_ap_percent': sum(reference_ap.values())/len(reference_ap),
        'mean_reinference_minus_reference_pp': sum(
            r['reinference_minus_reference_pp'] for r in rows)/len(rows),
        'interpretation': ('GT-informed counterfactual on sampled train frames '
                           'using exact final-layer V2 roles and retained detections. '
                           'With reference results, GT is official filtered train GT '
                           'and predictions are transformed to global coordinates. '
                           'This is subset AP, not full-split official AP. FP-only '
                           'demotion uses GT labels and is not an inference-time '
                           'scoring policy.'),
    }
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
