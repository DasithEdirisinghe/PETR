#!/usr/bin/env python3
"""Trace official car-TP queries and their last-layer GT-ray attention on R1-f val.

The saved AP rank ledger defines the cohort. This script reruns unmodified PETR
inference only to recover each final prediction's query ID and attention.
"""

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/cross_rig_attention'))

from analyze_pair import feature_rays, rays_intersect_box  # noqa: E402
from compare_r1_r1f_cars import gt_frames  # noqa: E402
from match_saved_final_queries import match_saved_queries  # noqa: E402
from score_r1f_car_queries import (  # noqa: E402
    CONFIG, CHECKPOINTS, CaptureCrossAttention, attention_selected,
    build_petr_checkpoint)
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, gt_targets, prepare_one, relocate_dataset_paths)
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox  # noqa: E402

MODELS = {'R1': 'r1', 'R1-f': 'r1f'}
FIELDS = ('sample_token', 'gt_index', 'model', 'prediction_index', 'query_index',
          'car_score', 'query_score_delta', 'query_center_delta_m',
          'gt_attention_mass', 'gt_cell_fraction', 'gt_cell_count',
          'valid_cell_count', 'reference_error_m',
          'layer_1_error_m', 'layer_2_error_m', 'layer_3_error_m',
          'layer_4_error_m', 'layer_5_error_m', 'layer_6_error_m',
          'recovery_margin_m')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--box-expansion', type=float, default=0.5,
                        help='Metres added to GT box for the ray mask; default 0.5.')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Smoke test; uses the first N cohort frames.')
    parser.add_argument('--output-dir', type=Path, default=ROOT /
                        'experiments/cross_rig_attention/output/r1f_val/common_tp_attention')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def cohort_from_rank_ledger(path):
    """Official AP@4m TPs common to the two checkpoints, keyed by GT identity."""
    matches = {model: {} for model in MODELS}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['model'] not in matches or row['is_tp'] != '1':
                continue
            key = (row['sample_token'], int(row['matched_gt_index']))
            if key in matches[row['model']]:
                raise ValueError('GT matched twice: {} {}'.format(row['model'], key))
            matches[row['model']][key] = {
                'prediction_index': int(row['prediction_index']),
                'score': float(row['car_score'])}
    common = set(matches['R1']) & set(matches['R1-f'])
    by_model = {model: defaultdict(list) for model in MODELS}
    for token, gt_index in sorted(common):
        for model in MODELS:
            by_model[model][token].append((gt_index, matches[model][token, gt_index]))
    return by_model, common


def verify_dnd_cohort(path, cohort, common):
    """If paired-GT DnD results exist, require identical final TP box pairs."""
    if not path.exists():
        return
    indexed = {model: {(token, gt_index): match['prediction_index']
                       for token, targets in cohort[model].items()
                       for gt_index, match in targets}
               for model in MODELS}
    expected = {(token, indexed['R1-f'][token, gi], indexed['R1'][token, gi])
                for token, gi in common}
    with path.open(newline='') as handle:
        actual = {(row['sample_token'], int(row['native_prediction_index']),
                   int(row['cross_prediction_index']))
                  for row in csv.DictReader(handle)
                  if row['gt_class'] == 'car' and row['transition'] == 'both_tp'}
    if expected != actual:
        raise ValueError('Common-TP rank ledger differs from DnD paired-GT cohort')


def decode_query_ids(head, classes, output, saved_boxes, info, metas, dataset):
    """Attach query IDs to the standard coder and PCCR formatting path."""
    coder = head.bbox_coder
    scores = output['all_cls_scores'][-1, 0].float().sigmoid().reshape(-1)
    top_scores, flat = scores.topk(coder.max_num)
    labels = flat % coder.num_classes
    query_ids = flat // coder.num_classes
    boxes = denormalize_bbox(
        output['all_bbox_preds'][-1, 0][query_ids].float(), coder.pc_range)
    bounds = torch.as_tensor(coder.post_center_range, device=boxes.device)
    keep = (boxes[:, :3] >= bounds[:3]).all(1)
    keep &= (boxes[:, :3] <= bounds[3:]).all(1)
    if coder.score_threshold:
        keep &= top_scores > coder.score_threshold
    top_scores, labels, query_ids = top_scores[keep], labels[keep], query_ids[keep]

    # This is the actual head decode used by PETR. The saved PCCR JSON is
    # shorter than the coder output because the dataset subsequently applies
    # class-specific radial ranges in ego coordinates.
    decoded_boxes, decoded_scores, decoded_labels = head.get_bboxes(output, metas)[0]
    if len(query_ids) != len(decoded_scores):
        raise ValueError('Query/coder count mismatch: {} vs {}'.format(
            len(query_ids), len(decoded_scores)))
    if not torch.allclose(top_scores, decoded_scores, atol=1e-5, rtol=0):
        raise ValueError('Query/coder score order mismatch')
    if not torch.equal(labels, decoded_labels):
        raise ValueError('Query/coder label order mismatch')

    if tuple(classes) != tuple(dataset.CLASSES):
        raise ValueError('Checkpoint and PCCR dataset class orders differ')
    names = [dataset.CLASSES[int(label)] for label in decoded_labels]
    eval_boxes = dataset._lidar_boxes_to_eval_boxes(
        decoded_boxes, names, decoded_scores, info['token'])
    supported_queries = [int(query_ids[index]) for index, name in enumerate(names)
                         if name in dataset.class_range]
    if len(eval_boxes) != len(supported_queries):
        raise ValueError('PCCR class filtering lost query alignment')
    paired = sorted(zip(eval_boxes, supported_queries),
                    key=lambda item: item[0].detection_score,
                    reverse=True)[:dataset.MAX_BOXES_PER_SAMPLE]
    final = [(global_box, query) for local_box, query in paired
             for global_box in [dataset._to_global(info, local_box)]
             if global_box is not None]
    candidates = [{'class_name': box.detection_name,
                   'score': float(box.detection_score),
                   'center_xy': box.translation[:2],
                   'query_index': query} for box, query in final]
    saved = [{'class_name': box['detection_name'],
              'score': float(box['detection_score']),
              'center_xy': box['translation'][:2]} for box in saved_boxes]
    return match_saved_queries(candidates, saved, return_diagnostics=True)


def reference_xy(head, query):
    if head.oracle_adapter is not None:
        raise ValueError('Baseline-only trace: oracle adapter is active')
    ref = head.reference_points.weight[query, :2].detach().float()
    span = torch.as_tensor((head.pc_range[3]-head.pc_range[0],
                            head.pc_range[4]-head.pc_range[1]), device=ref.device)
    origin = torch.as_tensor(head.pc_range[:2], device=ref.device)
    return ref*span+origin


def trace_frame(model, dataset, index, frame, targets, saved_boxes,
                device, expansion):
    data = prepare_one(dataset, index,
                       device.index if device.type == 'cuda' else device)
    images, metas = first_augmentation(data)
    head = model.pts_bbox_head
    if head.bbox_coder.__class__.__name__ != 'NMSFreeCoder':
        raise ValueError('Expected standard PETR NMSFreeCoder')
    with torch.no_grad():
        features = model.extract_feat(img=images, img_metas=metas)
        gt_boxes, gt_labels, _ = gt_targets(dataset, index, device, head.pc_range)
        with CaptureCrossAttention(head) as capture:
            output = head(features, metas)
        query_ids, query_audit = decode_query_ids(
            head, model.CLASSES, output, saved_boxes,
            dataset.data_infos[index], metas, dataset)
        pairs = []
        for gt_index, match in targets:
            locations = np.flatnonzero(frame['gt_indices'] == gt_index)
            if len(locations) != 1:
                raise ValueError('GT {} missing or duplicated in frame'.format(gt_index))
            local = int(locations[0])
            tracer_index = frame['tracer_indices'][local]
            if tracer_index is None or int(tracer_index) >= len(gt_boxes):
                raise ValueError('Evaluator TP GT absent from model GT targets')
            tracer_index = int(tracer_index)
            if int(gt_labels[tracer_index]) != list(model.CLASSES).index('car'):
                raise ValueError('Evaluator TP is not a car in model targets')
            pred_index = match['prediction_index']
            if saved_boxes[pred_index]['detection_name'] != 'car':
                raise ValueError('Evaluator TP prediction is not a car')
            if abs(saved_boxes[pred_index]['detection_score']-match['score']) > 1e-4:
                raise ValueError('Evaluator TP score differs from saved prediction')
            pairs.append((gt_index, tracer_index, pred_index,
                          int(query_ids[pred_index]), match['score']))
        layout = features[0].shape
        source = {'key_layout': {'num_cameras': layout[1],
                                 'height': layout[-2], 'width': layout[-1]},
                  'lidar2img': metas[0]['lidar2img'],
                  'pad_shape': metas[0]['pad_shape']}
        origins, directions, _ = feature_rays(source)
        module = head.transformer.decoder.layers[-1].attentions[1]
        attention, valid = attention_selected(
            module, capture.calls[6], [pair[3] for pair in pairs])
        if not valid.any():
            raise ValueError('Last-layer attention has no valid feature cells')
        rows = []
        for position, (gt_index, tracer_index, pred_index, query, score) in enumerate(pairs):
            gt_box = gt_boxes[tracer_index].detach().float().cpu().numpy()
            mask = rays_intersect_box(origins, directions, gt_box, expansion)
            inside = mask & valid
            weights = attention[position]
            if not np.isfinite(weights[valid]).all():
                raise ValueError('Non-finite last-layer TP-query attention')
            total = float(weights[valid].sum())
            mass = float(weights[inside].sum()/total) if inside.any() and total > 0 else None
            if mass is not None and not (-1e-6 <= mass <= 1+1e-6):
                raise ValueError('GT-region attention mass is outside [0, 1]')
            gt_center = gt_boxes[tracer_index, :2].float()
            ref_error = float(torch.norm(reference_xy(head, query)-gt_center))
            row = {'sample_token': dataset.data_infos[index]['token'],
                   'gt_index': gt_index, 'prediction_index': pred_index,
                   'query_index': query, 'car_score': score,
                   'query_score_delta': query_audit[pred_index]['score_delta'],
                   'query_center_delta_m': query_audit[pred_index]['center_delta_m'],
                   'gt_attention_mass': mass,
                   'gt_cell_fraction': float(inside.sum()/valid.sum()),
                   'gt_cell_count': int(inside.sum()),
                   'valid_cell_count': int(valid.sum()),
                   'reference_error_m': ref_error}
            for layer in range(6):
                center = output['all_bbox_preds'][layer, 0, query, :2].float()
                row['layer_{}_error_m'.format(layer+1)] = float(
                    torch.norm(center-gt_center))
            if row['layer_6_error_m'] > 4.05:
                raise ValueError('Retraced final center is >5 cm beyond the saved TP threshold')
            row['recovery_margin_m'] = ref_error-row['layer_6_error_m']
            rows.append(row)
        return rows


def median(values):
    return statistics.median(values) if values else None


def quantiles(values):
    return ([float(np.percentile(values, 25)),
             float(np.percentile(values, 75))] if values else None)


def summarize(rows, common_count, processed_frames, expansion):
    by_model = {name: {} for name in MODELS}
    for row in rows:
        key = (row['sample_token'], row['gt_index'])
        by_model[row['model']][key] = row
    paired = set(by_model['R1']) & set(by_model['R1-f'])
    summary = {'cohort': 'official car TP@4m in both models on R1-f val',
               'common_tp_total': common_count, 'paired_traced': len(paired),
               'processed_frames': processed_frames,
               'box_expansion_m': expansion, 'models': {},
               'attention_definition': 'Last decoder layer, mean over heads; '
               'mass on GT-box-intersecting valid feature-cell rays across all cameras.'}
    for model in MODELS:
        selected = [by_model[model][key] for key in paired]
        valid_mass = [row['gt_attention_mass'] for row in selected
                      if row['gt_attention_mass'] is not None]
        margins = [row['recovery_margin_m'] for row in selected]
        summary['models'][model] = {
            'n': len(selected), 'n_attention_valid': len(valid_mass),
            'median_gt_attention_mass': median(valid_mass),
            'median_query_score_delta': median([row['query_score_delta']
                                                for row in selected]),
            'max_query_score_delta': max(row['query_score_delta']
                                         for row in selected),
            'median_query_center_delta_m': median([row['query_center_delta_m']
                                                   for row in selected]),
            'max_query_center_delta_m': max(row['query_center_delta_m']
                                            for row in selected),
            'median_gt_cell_fraction': median([row['gt_cell_fraction'] for row in selected]),
            'median_reference_error_m': median([row['reference_error_m'] for row in selected]),
            'median_final_error_m': median([row['layer_6_error_m'] for row in selected]),
            'median_recovery_margin_m': median(margins),
            'recovery_margin_iqr_m': quantiles(margins),
            'fraction_positive_recovery': sum(value > 0 for value in margins)/len(margins),
            'median_layer_error_m': [median([row['layer_{}_error_m'.format(layer)]
                                             for row in selected]) for layer in range(1, 7)]}
    differences = [by_model['R1-f'][key]['gt_attention_mass']-
                   by_model['R1'][key]['gt_attention_mass'] for key in paired
                   if by_model['R1'][key]['gt_attention_mass'] is not None and
                   by_model['R1-f'][key]['gt_attention_mass'] is not None]
    summary['paired_median_r1f_minus_r1_attention_mass'] = median(differences)
    summary['paired_attention_comparisons'] = len(differences)
    summary['paired_median_r1f_minus_r1_reference_error_m'] = median([
        by_model['R1-f'][key]['reference_error_m']-
        by_model['R1'][key]['reference_error_m'] for key in paired])
    summary['paired_median_r1f_minus_r1_final_error_m'] = median([
        by_model['R1-f'][key]['layer_6_error_m']-
        by_model['R1'][key]['layer_6_error_m'] for key in paired])
    summary['paired_median_r1f_minus_r1_recovery_margin_m'] = median([
        by_model['R1-f'][key]['recovery_margin_m']-
        by_model['R1'][key]['recovery_margin_m'] for key in paired])
    return summary


def main():
    args = arguments()
    if args.box_expansion < 0 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError('Invalid box expansion or frame limit')
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/R1-f/'
    definition.ann_file = 'data/pccr/R1-f/R1-f_infos_val.pkl'
    definition.test_mode = True
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    frames = gt_frames(ROOT / definition.ann_file)
    token_to_index = {item['token']: i for i, item in enumerate(dataset.data_infos)}
    rank_path = ROOT / ('experiments/cross_rig_attention/output/r1f_val/'
                        'car_ap_rank_audit/ranked_car_predictions.csv')
    with rank_path.with_name('summary.json').open() as handle:
        rank_summary = json.load(handle)
    if rank_summary['rig'] != 'R1-f' or rank_summary['threshold_m'] != 4.0:
        raise ValueError('Rank ledger must be official R1-f car AP@4m replay')
    for model_name in MODELS:
        model_summary = rank_summary['models'][model_name]
        if abs(model_summary['replayed_car_ap']-
               model_summary['official_car_ap']) > 1e-8:
            raise ValueError('Rank ledger AP does not match official evaluator')
    cohort, common = cohort_from_rank_ledger(rank_path)
    if not common:
        raise ValueError('No common car TPs in the AP rank ledger')
    verify_dnd_cohort(ROOT / 'experiments/cross_rig_attention/output/r1f_val/'
                      'dnd3d_car_4m/paired_gt.csv', cohort, common)
    tokens = sorted(cohort['R1'])
    if set(tokens) != set(cohort['R1-f']) or not set(tokens) <= set(token_to_index):
        raise ValueError('Cohort tokens do not match model/dataset frames')
    tokens = tokens[:args.max_frames]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    for model_name, short in MODELS.items():
        checkpoint = CHECKPOINTS[short]
        stat = checkpoint.stat()
        ledger_stat = rank_path.stat()
        signature = {'checkpoint': str(checkpoint), 'size': stat.st_size,
                     'mtime_ns': stat.st_mtime_ns, 'box_expansion_m': args.box_expansion,
                     'rank_ledger': str(rank_path),
                     'rank_ledger_size': ledger_stat.st_size,
                     'rank_ledger_mtime_ns': ledger_stat.st_mtime_ns,
                     'schema': 2}
        model = build_petr_checkpoint(cfg, dataset, short, device)
        with (ROOT / ('experiments/cross_rig_attention/output/r1f_val/{}/'
                      'formatted/results_pccr.json'.format(model_name))).open() as handle:
            saved = json.load(handle)['results']
        for count, token in enumerate(tokens, 1):
            target = output_dir / 'cases' / short / (token+'.json')
            if target.exists() and not args.force:
                with target.open() as handle:
                    case = json.load(handle)
                if case['signature'] != signature or len(case['rows']) != len(cohort[model_name][token]):
                    raise ValueError('Cached case has different settings/cohort: {}'.format(target))
                continue
            rows = trace_frame(model, dataset, token_to_index[token], frames[token],
                               cohort[model_name][token], saved[token], device,
                               args.box_expansion)
            for row in rows:
                row['model'] = model_name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.partial.json')
            with temporary.open('w') as handle:
                json.dump({'signature': signature, 'rows': rows}, handle)
            os.replace(str(temporary), str(target))
            if count % 50 == 0:
                print('{}: {}/{} frames'.format(model_name, count, len(tokens)), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    rows = []
    for model_name, short in MODELS.items():
        for token in tokens:
            with (output_dir / 'cases' / short / (token+'.json')).open() as handle:
                for row in json.load(handle)['rows']:
                    row.setdefault('recovery_margin_m',
                                   row['reference_error_m']-row['layer_6_error_m'])
                    rows.append(row)
    with (output_dir / 'per_common_tp.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, len(common), len(tokens), args.box_expansion)
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print('Saved', output_dir / 'per_common_tp.csv')
    print('Saved', output_dir / 'summary.json')
    if args.max_frames is None:
        from plot_common_tp_geometry_attention import plot as plot_focused  # noqa: E402
        plot_focused(output_dir / 'summary.json',
                     output_dir / 'common_tp_geometry_attention.svg')


if __name__ == '__main__':
    main()
