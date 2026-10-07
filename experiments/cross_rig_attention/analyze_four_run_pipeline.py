#!/usr/bin/env python3
"""Object-aligned comparison of four PETR rig/checkpoint runs."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/cross_rig_attention'))
from analyze_pair import (feature_rays, load_trace, rays_intersect_box,
                          target_metrics)  # noqa: E402
from paired_selection_indices import ensure_source_tracer_indices  # noqa: E402

LABELS = {('R1', 'r1'): 'R1 image · R1 model',
          ('R1', 'r1f'): 'R1 image · R1-f model',
          ('R1-f', 'r1'): 'R1-f image · R1 model',
          ('R1-f', 'r1f'): 'R1-f image · R1-f model'}
COLORS = {('R1', 'r1'): '#2463a6', ('R1', 'r1f'): '#8aafcf',
          ('R1-f', 'r1'): '#d14d40', ('R1-f', 'r1f'): '#209b83'}
FEATURES = ('backbone_0', 'backbone_1', 'fpn_0', 'fpn_1', 'input_proj')


def flatten_map(tensor, cameras):
    """Return [camera*y*x, channel] from BNCHW/BCHW maps."""
    if tensor.ndim == 5:
        tensor = tensor[0]
    if tensor.ndim != 4 or tensor.shape[0] != cameras:
        raise ValueError('Unexpected camera feature map shape: {}'.format(tuple(tensor.shape)))
    return tensor.float().permute(0, 2, 3, 1).reshape(-1, tensor.shape[1]).numpy(), tensor.shape[-2:]


def car_feature_auc(tensor, capture, gt_box, expansion):
    cameras = capture['num_cameras']
    features, (height, width) = flatten_map(tensor, cameras)
    ray_source = {
        'key_layout': {'num_cameras': cameras, 'height': height, 'width': width},
        'lidar2img': capture['lidar2img'], 'pad_shape': capture['pad_shape']}
    origins, directions, camera_ids = feature_rays(ray_source)
    positives = rays_intersect_box(origins, directions, gt_box, expansion)
    visible_cameras = np.unique(camera_ids[positives])
    negatives = (~positives) & np.isin(camera_ids, visible_cameras)
    pos_indices = np.flatnonzero(positives)
    neg_indices = np.flatnonzero(negatives)
    if len(pos_indices) < 4 or len(neg_indices) < 4:
        return None, len(pos_indices)
    # Within-run held-out prototype. Raw channels are never compared across models.
    rng = np.random.RandomState(17)
    rng.shuffle(pos_indices)
    split = len(pos_indices) // 2
    train, held_out = pos_indices[:split], pos_indices[split:]
    neg_indices = rng.choice(neg_indices,
                             min(len(neg_indices), max(200, 10*len(held_out))),
                             replace=False)
    norm = np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    unit = features / norm
    prototype = unit[train].mean(axis=0)
    prototype /= max(np.linalg.norm(prototype), 1e-8)
    positive_scores = unit[held_out] @ prototype
    negative_scores = unit[neg_indices] @ prototype
    auc = ((positive_scores[:, None] > negative_scores).mean() +
           .5*(positive_scores[:, None] == negative_scores).mean())
    return float(auc), len(pos_indices)


def one_target(trace):
    targets = trace['summary']['targets']
    if len(targets) != 1:
        raise ValueError('Expected one GT car per trace')
    return next(iter(targets.items()))


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--box-expansion', type=float, default=.5)
    args = parser.parse_args()
    with args.selection.open() as handle:
        pairs = json.load(handle)[:args.max_examples]
    if not pairs:
        raise ValueError('Empty paired selection')
    pairs = ensure_source_tracer_indices(pairs, args.split)
    output = args.output_dir
    feature_rows, decoder_rows, cases = [], [], []
    for pair_id, pair in enumerate(pairs):
        case = {'pair_index': pair_id, 'scene': pair['scene_name'],
                'frame': pair['scene_frame_index'],
                'gt_pair_distance_m': pair['cross_rig_gt_distance_m']}
        for rig in ('R1', 'R1-f'):
            token = pair['sample_token'] if rig == 'R1' else pair['source_sample_token']
            gt_index = int(pair['tracer_gt_index'] if rig == 'R1' else
                           pair['source_tracer_gt_index'])
            rig_dir = ROOT / 'experiments/cross_rig_attention/output' / (
                'r1_{}'.format(args.split) if rig == 'R1' else
                'r1f_{}'.format(args.split))
            for model in ('r1', 'r1f'):
                trace_dir = rig_dir / 'traces' / model / '{}_gt{:03d}'.format(
                    token, gt_index)
                trace = load_trace(trace_dir)
                name, target = one_target(trace)
                if target['gt_index'] != gt_index or target['gt_class'] != 'car':
                    raise ValueError('Trace/selection GT mismatch: {}'.format(trace_dir))
                capture_path = output / 'tensors' / rig.replace('-', '') / model / (
                    token + '.pt')
                capture = torch.load(str(capture_path), map_location='cpu')
                if capture['sample_token'] != token or capture['model'] != model:
                    raise ValueError('Stage capture mismatch: {}'.format(capture_path))
                # This checks the upstream capture is the same deterministic model run.
                trace_logits = trace['tensors']['all_cls_scores']
                difference = float((trace_logits.float() -
                                    capture['all_cls_scores'].float()).abs().max())
                if difference > .02:  # capture is fp16; trace logits are fp32
                    raise AssertionError('Trace/capture logits differ: {} {}'.format(
                        difference, capture_path))
                case['{}_{}_query'.format(rig, model)] = target['query_index']
                maps = {'backbone_{}'.format(i): value for i, value in
                        enumerate(capture['backbone'])}
                maps.update({'fpn_{}'.format(i): value for i, value in
                             enumerate(capture['fpn'])})
                maps['input_proj'] = capture['input_proj']
                for stage in FEATURES:
                    auc, count = car_feature_auc(maps[stage], capture,
                                                 target['gt_box'], args.box_expansion)
                    feature_rows.append({'pair_index': pair_id, 'rig': rig,
                                         'model': model, 'stage': stage,
                                         'car_auc': auc, 'gt_ray_count': count})
                # PE is not scored by feature AUROC: geometry is not an appearance classifier.
                for row in target_metrics('{}_{}'.format(rig, model), trace,
                                          name, target, args.box_expansion, 200):
                    decoder_rows.append({
                        'pair_index': pair_id, 'rig': rig, 'model': model,
                        'layer': row['layer'], 'query_index': target['query_index'],
                        'reference_error_m': target['reference_to_gt_bev_m'],
                        'center_error_m': row['bev_error_m'],
                        'car_confidence': row['confidence'],
                        'gt_ray_enrichment': (
                            row['gt_attention_mass'] /
                            (row['gt_ray_count']/row['valid_token_count'])
                            if row['gt_ray_count'] and row['valid_token_count'] else None),
                        'gt_attention_mass': row['gt_attention_mass'],
                        'H_G3D_utility': row.get('H_G3D_directional_utility'),
                        'E_G3D_utility': row.get('E_G3D_directional_utility'),
                        'H_X_utility': row.get('H_X_directional_utility'),
                        'E_X_utility': row.get('E_X_directional_utility'),
                    })
        cases.append(case)
        print('Analyzed four-run car', pair_id+1, '/', len(pairs), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('feature_evidence.csv', feature_rows),
                           ('decoder_trajectory.csv', decoder_rows)]:
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'paired_cases.json').open('w') as handle:
        json.dump(cases, handle, indent=2)

    figure, grid = plt.subplots(2, 3, figsize=(18, 10))
    axes = grid.flat
    summary = {'num_paired_cars': len(pairs), 'conditions': {}}
    for condition in LABELS:
        rig, model = condition
        label, color = LABELS[condition], COLORS[condition]
        stage_auc = [median([r['car_auc'] for r in feature_rows
                             if (r['rig'], r['model'], r['stage']) ==
                             (rig, model, stage)]) for stage in FEATURES]
        axes[0].plot(range(len(FEATURES)), stage_auc, '-o', color=color,
                     label=label)
        subset = [r for r in decoder_rows if (r['rig'], r['model']) == condition]
        centers = [median([r['center_error_m'] for r in subset if r['layer'] == layer])
                   for layer in range(1, 7)]
        enrichments = [median([r['gt_ray_enrichment'] for r in subset
                               if r['layer'] == layer]) for layer in range(1, 7)]
        geometry_utility = [median([
            (r['H_G3D_utility'] or 0.0) + (r['E_G3D_utility'] or 0.0)
            for r in subset if r['layer'] == layer]) for layer in range(1, 7)]
        appearance_utility = [median([
            (r['H_X_utility'] or 0.0) + (r['E_X_utility'] or 0.0)
            for r in subset if r['layer'] == layer]) for layer in range(1, 7)]
        axes[1].plot(range(1, 7), centers, '-o', color=color, label=label)
        axes[2].plot(range(1, 7), enrichments, '-o', color=color, label=label)
        axes[3].scatter([list(LABELS).index(condition)],
                        [median([r['reference_error_m'] for r in subset])],
                        color=color, s=75)
        axes[4].plot(range(1, 7), geometry_utility, '-o', color=color, label=label)
        axes[5].plot(range(1, 7), appearance_utility, '-o', color=color, label=label)
        summary['conditions']['{}_{}'.format(rig, model)] = {
            'feature_auc': dict(zip(FEATURES, stage_auc)),
            'reference_error_m': median([r['reference_error_m'] for r in subset]),
            'decoder_center_error_m': centers,
            'gt_ray_enrichment': enrichments,
            'geometry_utility': geometry_utility,
            'appearance_utility': appearance_utility,
            'final_car_confidence': median([r['car_confidence'] for r in subset
                                           if r['layer'] == 6]),
        }
    axes[0].set_xticks(range(len(FEATURES)))
    axes[0].set_xticklabels(FEATURES, rotation=25, ha='right')
    axes[0].set_title('Car evidence at upstream stages')
    axes[0].set_ylabel('Held-out GT-ray vs background AUROC')
    axes[1].set_title('Object-aligned decoder localization')
    axes[1].set_xlabel('Decoder layer')
    axes[1].set_ylabel('Median BEV center error (m)')
    axes[2].set_title('Query attention to car-supporting rays')
    axes[2].set_xlabel('Decoder layer')
    axes[2].set_ylabel('Median enrichment vs uniform attention')
    axes[3].set_title('Selected query reference location')
    axes[3].set_ylabel('Median reference-to-GT BEV error (m)')
    axes[3].set_xticks(range(4))
    axes[3].set_xticklabels(
        ['R1/R1', 'R1/R1-f', 'R1-f/R1', 'R1-f/R1-f'],
        rotation=20, ha='right')
    axes[4].set_title('G3D contribution to GT-ray attention')
    axes[4].set_xlabel('Decoder layer')
    axes[4].set_ylabel('Leave-term-out attention-mass change')
    axes[5].set_title('Image X contribution to GT-ray attention')
    axes[5].set_xlabel('Decoder layer')
    axes[5].set_ylabel('Leave-term-out attention-mass change')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle('Same physical cars · both rigs × both checkpoints',
                    x=.03, ha='left', fontsize=15, fontweight='bold')
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(output / 'four_run_pipeline.png', dpi=220)
    figure.savefig(output / 'four_run_pipeline.svg')
    plt.close(figure)
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print('Saved four-run comparison:', output)


if __name__ == '__main__':
    main()
