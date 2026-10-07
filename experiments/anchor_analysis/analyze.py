#!/usr/bin/env python
"""Analyze how PETR's learned 3D reference points relate to detections.

This experiment is deliberately read-only: it does not patch PETR or alter its
forward computation.  It retains the raw query index, matches queries to GT
with PETR's training-time Hungarian assigner, and tracks the final assignment
back through every decoder layer.
"""

from __future__ import print_function

import argparse
import csv
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.anchor_analysis.plots import make_all_plots  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze PETR learned anchors and decoder corrections.')
    parser.add_argument('--config', default=str(
        REPO_ROOT / 'projects/configs/petr/'
        'petr_r50dcn_gridmask_p4_nuscenes_local.py'))
    parser.add_argument('--checkpoint', default=str(
        REPO_ROOT / 'ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth'))
    parser.add_argument('--output-dir', default=str(
        REPO_ROOT / 'experiments/anchor_analysis/outputs'))
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of validation samples; 0 means all.')
    parser.add_argument('--sample-index', type=int, default=None,
                        help='Analyze only one explicit dataset index.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sanity-only', action='store_true',
                        help='Run one sample and strict consistency checks.')
    return parser.parse_args()


def import_plugin(cfg):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', '')
    module_path = os.path.dirname(plugin_dir).replace('/', '.')
    importlib.import_module(module_path)


def relocate_dataset_paths(dataset, data_root):
    """Repair paths embedded by a nuScenes pickle made in another checkout."""
    root = Path(data_root).resolve()

    def relocate(value):
        if not isinstance(value, str) or os.path.exists(value):
            return value
        normalized = value.replace('\\', '/')
        for directory in ('samples', 'sweeps', 'maps'):
            marker = '/' + directory + '/'
            if marker in normalized:
                candidate = root / directory / normalized.split(marker, 1)[1]
                if candidate.exists():
                    return str(candidate)
        return value

    for info in dataset.data_infos:
        if 'lidar_path' in info:
            info['lidar_path'] = relocate(info['lidar_path'])
        for camera in info.get('cams', {}).values():
            if 'data_path' in camera:
                camera['data_path'] = relocate(camera['data_path'])
        for sweep in info.get('sweeps', []):
            if 'data_path' in sweep:
                sweep['data_path'] = relocate(sweep['data_path'])


def select_indices(length, num_samples, sample_index, seed):
    if sample_index is not None:
        if sample_index < 0 or sample_index >= length:
            raise IndexError('--sample-index is outside the dataset')
        return [sample_index]
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    if num_samples > 0:
        indices = indices[:num_samples]
    return indices


def prepare_one(dataset, index, scatter_device):
    sample = dataset.prepare_test_data(index)
    if sample is None:
        raise RuntimeError('Dataset returned no test sample at index {}'.format(index))
    data = collate([sample], samples_per_gpu=1)
    return scatter(data, [scatter_device])[0]


def first_nonempty_index(dataset, indices, pc_range):
    """Choose a deterministic sanity sample containing at least one GT."""
    bounds = torch.tensor(pc_range)
    for index in indices:
        annotation = dataset.get_ann_info(index)
        labels = torch.as_tensor(annotation['gt_labels_3d'])
        centers = annotation['gt_bboxes_3d'].gravity_center
        in_range = ((centers >= bounds[:3]) &
                    (centers <= bounds[3:6])).all(dim=1)
        if bool(((labels >= 0) & in_range).any()):
            return index
    raise RuntimeError('No non-empty ground-truth sample exists in the dataset')


def first_augmentation(data):
    """Unwrap the one augmentation produced by MultiScaleFlipAug3D."""
    img = data['img']
    img_metas = data['img_metas']
    if isinstance(img, (list, tuple)):
        img = img[0]
    if (isinstance(img_metas, (list, tuple)) and img_metas and
            isinstance(img_metas[0], (list, tuple))):
        img_metas = img_metas[0]
    return img, img_metas


def gt_tensor_and_labels(dataset, index, device, pc_range):
    """Build the same valid, in-range GT target set used for PETR training."""
    annotation = dataset.get_ann_info(index)
    boxes = annotation['gt_bboxes_3d']
    labels = annotation['gt_labels_3d']
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    box_tensor = torch.cat(
        (boxes.gravity_center, boxes.tensor[:, 3:]), dim=1).to(device)
    bounds = box_tensor.new_tensor(pc_range)
    centers = box_tensor[:, :3]
    in_range = ((centers >= bounds[:3]) &
                (centers <= bounds[3:6])).all(dim=1)
    valid_class = labels >= 0
    keep = in_range & valid_class
    return box_tensor[keep], labels[keep]


def metric_anchors(head):
    normalized = head.reference_points.weight.detach()
    pc_range = normalized.new_tensor(head.pc_range)
    return normalized * (pc_range[3:6] - pc_range[:3]) + pc_range[:3]


def positive_assignment(head, bbox_pred, cls_score, gt_boxes, gt_labels):
    result = head.assigner.assign(
        bbox_pred, cls_score, gt_boxes, gt_labels, None)
    positive = result.gt_inds > 0
    query_indices = torch.nonzero(positive, as_tuple=False).squeeze(1)
    gt_indices = result.gt_inds[query_indices] - 1
    return query_indices, gt_indices


def center_xyz(bbox_predictions):
    # PETR layout: cx, cy, w, l, cz, h, sin(yaw), cos(yaw), vx, vy.
    return bbox_predictions[..., [0, 1, 4]]


def as_float(value):
    return float(value.detach().cpu().item())


def vector_fields(prefix, vector):
    return {
        prefix + '_x': as_float(vector[0]),
        prefix + '_y': as_float(vector[1]),
        prefix + '_z': as_float(vector[2]),
    }


def analyze_sample(head, outputs, gt_boxes, gt_labels, anchors, dataset_index,
                   sample_token, class_names, score_threshold):
    all_cls = outputs['all_cls_scores'][:, 0]
    all_bbox = outputs['all_bbox_preds'][:, 0]
    final_queries, final_gts = positive_assignment(
        head, all_bbox[-1], all_cls[-1], gt_boxes, gt_labels)
    gt_centers = gt_boxes[:, :3]
    records = []

    for layer in range(all_cls.shape[0]):
        layer_centers = center_xyz(all_bbox[layer])
        probs = all_cls[layer].sigmoid()
        predicted_labels = probs.argmax(dim=-1)
        for query_tensor, gt_tensor in zip(final_queries, final_gts):
            query_index = int(query_tensor.item())
            gt_index = int(gt_tensor.item())
            anchor = anchors[query_index]
            prediction = layer_centers[query_index]
            target = gt_centers[gt_index]
            gt_label = int(gt_labels[gt_index].item())
            predicted_label = int(predicted_labels[query_index].item())
            matched_score = probs[query_index, gt_label]
            anchor_bev = torch.norm(anchor[:2] - target[:2])
            correction_bev = torch.norm(prediction[:2] - anchor[:2])
            final_error_bev = torch.norm(prediction[:2] - target[:2])
            record = {
                'dataset_index': int(dataset_index),
                'sample_token': sample_token,
                'decoder_layer': layer + 1,
                'query_index': query_index,
                'gt_index': gt_index,
                'gt_label': gt_label,
                'gt_class': class_names[gt_label],
                'predicted_label': predicted_label,
                'predicted_class': class_names[predicted_label],
                'class_correct': int(predicted_label == gt_label),
                'matched_class_score': as_float(matched_score),
                'above_score_threshold': int(as_float(matched_score) >= score_threshold),
                'anchor_distance_bev': as_float(anchor_bev),
                'correction_distance_bev': as_float(correction_bev),
                'final_error_bev': as_float(final_error_bev),
                'anchor_distance_3d': as_float(torch.norm(anchor - target)),
                'correction_distance_3d': as_float(torch.norm(prediction - anchor)),
                'final_error_3d': as_float(torch.norm(prediction - target)),
            }
            record.update(vector_fields('anchor', anchor))
            record.update(vector_fields('prediction', prediction))
            record.update(vector_fields('gt', target))
            records.append(record)

    # Independent matching is recorded compactly to quantify assignment changes.
    final_map = {int(g.item()): int(q.item())
                 for q, g in zip(final_queries, final_gts)}
    stability = []
    for layer in range(all_cls.shape[0]):
        layer_queries, layer_gts = positive_assignment(
            head, all_bbox[layer], all_cls[layer], gt_boxes, gt_labels)
        layer_map = {int(g.item()): int(q.item())
                     for q, g in zip(layer_queries, layer_gts)}
        common = sorted(set(final_map).intersection(layer_map))
        stable = sum(layer_map[g] == final_map[g] for g in common)
        stability.append({
            'dataset_index': int(dataset_index),
            'sample_token': sample_token,
            'decoder_layer': layer + 1,
            'num_gt': int(gt_boxes.shape[0]),
            'num_common_gt': len(common),
            'num_same_query_as_final': stable,
            'fraction_same_query_as_final': (
                float(stable) / len(common) if common else 0.0),
        })

    coverage = []
    if gt_centers.shape[0] > 0:
        distances = torch.cdist(anchors[:, :2], gt_centers[:, :2])
        nearest_distance, nearest_query = distances.min(dim=0)
        for gt_index in range(gt_centers.shape[0]):
            label = int(gt_labels[gt_index].item())
            coverage.append({
                'dataset_index': int(dataset_index),
                'sample_token': sample_token,
                'gt_index': gt_index,
                'gt_label': label,
                'gt_class': class_names[label],
                'nearest_anchor_query': int(nearest_query[gt_index].item()),
                'nearest_anchor_distance_bev': as_float(nearest_distance[gt_index]),
                'selected_final_query': final_map.get(gt_index, -1),
            })
    return records, coverage, stability


def write_csv(path, rows):
    if not rows:
        return
    with open(str(path), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(records, coverage, anchors, score_threshold):
    final_layer = max(row['decoder_layer'] for row in records)
    final = [row for row in records if row['decoder_layer'] == final_layer]
    thresholds = (0.5, 1.0, 2.0, 4.0)
    summary = {
        'num_queries': int(anchors.shape[0]),
        'num_unique_final_matched_queries': len(set(
            row['query_index'] for row in final)),
        'fraction_queries_used_by_final_matching': float(len(set(
            row['query_index'] for row in final))) / int(anchors.shape[0]),
        'num_decoder_layers': int(final_layer),
        'num_analyzed_gt_assignments': len(final),
        'score_threshold': score_threshold,
        'mean_nearest_anchor_coverage_bev_m': float(np.mean([
            row['nearest_anchor_distance_bev'] for row in coverage])),
        'median_selected_anchor_distance_bev_m': percentile(
            [row['anchor_distance_bev'] for row in final], 50),
        'median_correction_distance_bev_m': percentile(
            [row['correction_distance_bev'] for row in final], 50),
        'median_final_error_bev_m': percentile(
            [row['final_error_bev'] for row in final], 50),
    }
    for threshold in thresholds:
        successes = [
            row['class_correct'] and row['above_score_threshold'] and
            row['final_error_bev'] <= threshold for row in final]
        summary['query_success_at_{:.1f}m'.format(threshold)] = (
            float(np.mean(successes)))
    return summary


def run_sanity_checks(head, outputs, anchors, records):
    assert anchors.ndim == 2 and anchors.shape[1] == 3
    assert anchors.shape[0] == head.num_query
    assert outputs['all_cls_scores'].shape[2] == head.num_query
    assert outputs['all_bbox_preds'].shape[2] == head.num_query
    assert torch.isfinite(anchors).all()
    assert torch.isfinite(outputs['all_bbox_preds']).all()
    if not records:
        raise AssertionError('No matched GT/query records were produced')
    final_layer = int(outputs['all_cls_scores'].shape[0])
    final_records = [r for r in records if r['decoder_layer'] == final_layer]
    if len(final_records) != len(set(r['gt_index'] for r in final_records)):
        raise AssertionError('Final Hungarian assignment is not one-to-one')


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    os.chdir(str(REPO_ROOT))
    if not Path(args.config).is_file():
        raise FileNotFoundError('Config does not exist: {}'.format(args.config))
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(
            'Checkpoint does not exist: {}'.format(args.checkpoint))
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    relocate_dataset_paths(dataset, cfg.data.test.data_root)

    # Keep train_cfg: it constructs the exact Hungarian assigner used by PETR.
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    device = torch.device(args.device)
    model = model.to(device).eval()
    scatter_device = device.index if device.type == 'cuda' else device
    head = model.pts_bbox_head
    if not hasattr(head, 'assigner'):
        raise RuntimeError('PETR head has no training Hungarian assigner')
    anchors = metric_anchors(head)

    if args.sanity_only and args.sample_index is None:
        candidates = select_indices(len(dataset), 0, None, args.seed)
        indices = [first_nonempty_index(dataset, candidates, head.pc_range)]
    else:
        sample_count = 1 if args.sanity_only else args.num_samples
        indices = select_indices(
            len(dataset), sample_count, args.sample_index, args.seed)
    all_records, all_coverage, all_stability = [], [], []
    with torch.no_grad():
        for position, index in enumerate(indices):
            data = prepare_one(dataset, index, scatter_device)
            img, img_metas = first_augmentation(data)
            features = model.extract_feat(img=img, img_metas=img_metas)
            outputs = head(features, img_metas)
            gt_boxes, gt_labels = gt_tensor_and_labels(
                dataset, index, device, head.pc_range)
            token = dataset.data_infos[index].get('token', str(index))
            records, coverage, stability = analyze_sample(
                head, outputs, gt_boxes, gt_labels, anchors, index, token,
                list(model.CLASSES), args.score_threshold)
            all_records.extend(records)
            all_coverage.extend(coverage)
            all_stability.extend(stability)
            if args.sanity_only:
                run_sanity_checks(head, outputs, anchors, records)
            print('[{}/{}] index={} token={} gt={}'.format(
                position + 1, len(indices), index, token, len(gt_labels)),
                flush=True)

    if not all_records:
        raise RuntimeError('Analysis produced no matched records')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'query_layer_records.csv', all_records)
    write_csv(output_dir / 'anchor_coverage.csv', all_coverage)
    write_csv(output_dir / 'assignment_stability.csv', all_stability)
    np.save(str(output_dir / 'learned_anchors_xyz.npy'),
            anchors.detach().cpu().numpy())
    summary = summarize(
        all_records, all_coverage, anchors, args.score_threshold)
    summary.update({
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'num_samples': len(indices),
        'seed': args.seed,
        'note': ('Query success is an analysis diagnostic based on final-layer '
                 'Hungarian pairs; it is not official nuScenes recall or AP.'),
    })
    with open(str(output_dir / 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    make_all_plots(all_records, all_coverage, all_stability,
                   anchors.detach().cpu().numpy(), output_dir,
                   args.score_threshold)
    print('Analysis complete: {}'.format(output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
