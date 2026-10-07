#!/usr/bin/env python3
"""Measure car-query reference-to-GT recovery on the full R1-f validation set."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, gt_targets, prepare_one, relocate_dataset_paths)

CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
CHECKPOINTS = {
    'r1': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth',
    'r1f': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth',
}


def build(cfg, dataset, model_key, device):
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(CHECKPOINTS[model_key]),
                                 map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.to(device).eval()
    return model


def checkpoint_signature(model_key):
    path = CHECKPOINTS[model_key]
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size,
            'mtime_ns': getattr(stat, 'st_mtime_ns', int(stat.st_mtime*1e9))}


def frame_rows(model, dataset, index, device, score_threshold):
    head = model.pts_bbox_head
    data = prepare_one(dataset, index, device.index if device.type == 'cuda'
                       else device)
    images, metas = first_augmentation(data)
    with torch.no_grad():
        features = model.extract_feat(img=images, img_metas=metas)
        outputs = head(features, metas)
        gt_boxes, gt_labels, _ = gt_targets(dataset, index, device,
                                            head.pc_range)
        car_label = list(model.CLASSES).index('car')
        car_gt = torch.nonzero(gt_labels == car_label).flatten().tolist()
        if not car_gt:
            return []
        final_boxes = outputs['all_bbox_preds'][-1, 0]
        final_cls = outputs['all_cls_scores'][-1, 0]
        assignment = head.assigner.assign(final_boxes, final_cls,
                                          gt_boxes, gt_labels, None)
        assigned = {}
        for query in torch.nonzero(assignment.gt_inds > 0).flatten().tolist():
            gt = int(assignment.gt_inds[query].item()) - 1
            if gt in car_gt:
                if gt in assigned:
                    raise AssertionError('Hungarian assignment reused GT car')
                assigned[gt] = query
        pc_range = torch.as_tensor(head.pc_range, device=device,
                                   dtype=head.reference_points.weight.dtype)
        refs = head.reference_points.weight.detach()
        metric_refs = refs*(pc_range[3:6]-pc_range[:3])+pc_range[:3]
        probs = final_cls.sigmoid()
        rows = []
        for gt in car_gt:
            gt_center = gt_boxes[gt, :3].detach().float().cpu().numpy()
            row = {'gt_index': gt, 'gt_x': float(gt_center[0]),
                   'gt_y': float(gt_center[1]), 'gt_z': float(gt_center[2]),
                   'assigned': gt in assigned}
            if gt not in assigned:
                rows.append(row)
                continue
            query = assigned[gt]
            reference = metric_refs[query].detach().float().cpu().numpy()
            centers = outputs['all_bbox_preds'][:, 0, query][:, [0, 1, 4]]
            centers = centers.detach().float().cpu().numpy()
            car_score = float(probs[query, car_label])
            top_class = int(probs[query].argmax())
            initial_xy = float(np.linalg.norm(reference[:2]-gt_center[:2]))
            initial_xyz = float(np.linalg.norm(reference-gt_center))
            errors = np.linalg.norm(centers[:, :2]-gt_center[None, :2], axis=1)
            errors_xyz = np.linalg.norm(centers-gt_center[None], axis=1)
            direction = gt_center[:2]-reference[:2]
            distance = max(float(np.linalg.norm(direction)), 1e-8)
            progress = ((centers[:, :2]-reference[None, :2]) @ direction /
                        distance)
            row.update({
                'query_index': query,
                'reference_xyz': reference.tolist(),
                'predicted_centers_xyz': centers.tolist(),
                'initial_error_xy_m': initial_xy,
                'initial_error_xyz_m': initial_xyz,
                'layer_error_xy_m': errors.tolist(),
                'layer_error_xyz_m': errors_xyz.tolist(),
                'layer_gain_xy_m': (initial_xy-errors).tolist(),
                'layer_progress_toward_gt_m': progress.tolist(),
                'final_car_score': car_score,
                'final_class_correct': top_class == car_label,
                'above_score_threshold': car_score >= score_threshold,
                'final_error_le_05m': bool(errors[-1] <= .5),
                'final_error_le_1m': bool(errors[-1] <= 1.0),
                'final_error_le_2m': bool(errors[-1] <= 2.0),
            })
            rows.append(row)
    return rows


def summarize(root, models, score_threshold, tokens):
    summaries = {}
    per_car = {}
    for key in models:
        files = [root / 'frames' / key / (token+'.json') for token in tokens]
        frames = [json.load(path.open()) for path in files]
        rows = [dict(row, token=frame['token']) for frame in frames
                for row in frame['cars']]
        assigned = [row for row in rows if row['assigned']]
        eligible = [row for row in assigned if row['final_class_correct'] and
                    row['final_car_score'] >= score_threshold]
        if not rows:
            raise ValueError('No processed GT cars for {}'.format(key))
        summary = {
            'processed_frames': len(frames), 'gt_cars': len(rows),
            'assigned_cars': len(assigned),
            'eligible_cars': len(eligible),
            'score_threshold': score_threshold,
            'gt_cars_without_assignment': len(rows)-len(assigned),
            'assigned': {}, 'eligible': {},
        }
        for label, cohort in [('assigned', assigned), ('eligible', eligible)]:
            if not cohort:
                continue
            initial = np.asarray([r['initial_error_xy_m'] for r in cohort])
            layer_errors = np.asarray([r['layer_error_xy_m'] for r in cohort])
            gains = initial[:, None]-layer_errors
            summary[label].update({
                'n': len(cohort),
                'mean_initial_xy_m': float(initial.mean()),
                'median_initial_xy_m': float(np.median(initial)),
                'mean_layer_error_xy_m': layer_errors.mean(axis=0).tolist(),
                'median_layer_error_xy_m': np.median(layer_errors, axis=0).tolist(),
                'mean_layer_gain_xy_m': gains.mean(axis=0).tolist(),
                'median_layer_gain_xy_m': np.median(gains, axis=0).tolist(),
                'fraction_final_improved_over_reference': float(
                    (gains[:, -1] > 0).mean()),
                'fraction_final_le_05m': float((layer_errors[:, -1] <= .5).mean()),
                'fraction_final_le_1m': float((layer_errors[:, -1] <= 1).mean()),
                'fraction_final_le_2m': float((layer_errors[:, -1] <= 2).mean()),
                'fraction_layer6_worse_than_layer1': float(
                    (layer_errors[:, -1] > layer_errors[:, 0]).mean()),
                'median_fraction_initial_error_recovered_when_ref_gt_1m': (
                    float(np.median(1-layer_errors[initial >= 1, -1] /
                                    initial[initial >= 1]))
                    if (initial >= 1).any() else None),
            })
        summaries[key] = summary
        per_car[key] = rows
        with (root / ('per_car_{}.json'.format(key))).open('w') as handle:
            json.dump(rows, handle)
    if 'r1' in per_car and 'r1f' in per_car:
        paired = {}
        for key in ('r1', 'r1f'):
            paired[key] = {(row['token'], row['gt_index']): row
                           for row in per_car[key] if row['assigned']}
        keys = sorted(set(paired['r1']) & set(paired['r1f']))
        if keys:
            source = [paired['r1'][key] for key in keys]
            target = [paired['r1f'][key] for key in keys]
            summaries['paired_both_assigned'] = {
                'n': len(keys),
                'fraction_r1f_reference_farther': float(np.mean([
                    b['initial_error_xy_m'] > a['initial_error_xy_m']
                    for a, b in zip(source, target)])),
                'fraction_r1f_final_closer': float(np.mean([
                    b['layer_error_xy_m'][-1] < a['layer_error_xy_m'][-1]
                    for a, b in zip(source, target)])),
                'mean_r1f_minus_r1_recovery_m': float(np.mean([
                    (b['initial_error_xy_m']-b['layer_error_xy_m'][-1]) -
                    (a['initial_error_xy_m']-a['layer_error_xy_m'][-1])
                    for a, b in zip(source, target)])),
            }
    with (root / 'summary.json').open('w') as handle:
        json.dump(summaries, handle, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'r1': '#d14d40', 'r1f': '#209b83'}
    for key in models:
        entry = summaries[key]['assigned']
        if not entry:
            continue
        x = np.arange(7)
        errors = [entry['mean_initial_xy_m']] + entry['mean_layer_error_xy_m']
        axes[0].plot(x, errors, '-o', color=colors[key], label=key.upper())
        gain = [0.0] + entry['mean_layer_gain_xy_m']
        axes[1].plot(x, gain, '-o', color=colors[key], label=key.upper())
    for axis in axes:
        axis.set_xticks(range(7))
        axis.set_xticklabels(['Ref', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'])
        axis.grid(alpha=.2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].set_title('Mean BEV center error')
    axes[0].set_ylabel('Distance to GT (m)')
    axes[1].set_title('Mean recovery from initial reference')
    axes[1].set_ylabel('Initial error − layer error (m)')
    axes[1].axhline(0, color='#64748b', linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle('R1-f validation cars · checkpoint-specific Hungarian query',
                 x=.04, ha='left', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(root / 'car_reference_recovery.png', dpi=220)
    fig.savefig(root / 'car_reference_recovery.svg')
    plt.close(fig)
    print('Saved summary and plot:', root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--models', nargs='+', choices=['r1', 'r1f'],
                        default=['r1', 'r1f'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Use for a quick smoke test; omit for all frames.')
    parser.add_argument('--score-threshold', type=float, default=.35)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError('--max-frames must be positive')
    if not 0 <= args.score_threshold <= 1:
        raise ValueError('--score-threshold must be in [0,1]')
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/R1-f/'
    definition.ann_file = 'data/pccr/R1-f/R1-f_infos_{}.pkl'.format(args.split)
    definition.test_mode = True
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    output = args.output_dir or ROOT / 'experiments/cross_rig_attention/output' / (
        'r1f_{}'.format(args.split)) / 'reference_recovery'
    output.mkdir(parents=True, exist_ok=True)
    limit = min(len(dataset), args.max_frames) if args.max_frames else len(dataset)
    tokens = [dataset.data_infos[index]['token'] for index in range(limit)]
    device = torch.device(args.device)
    for key in args.models:
        frame_dir = output / 'frames' / key
        frame_dir.mkdir(parents=True, exist_ok=True)
        signature = checkpoint_signature(key)
        missing = []
        for index in range(limit):
            token = dataset.data_infos[index]['token']
            path = frame_dir / (token+'.json')
            if not path.is_file():
                missing.append((index, token))
                continue
            with path.open() as handle:
                cached = json.load(handle)
            if cached.get('checkpoint_signature') != signature:
                raise ValueError('Cached frame uses a different checkpoint: {}. '
                                 'Choose another --output-dir before rerunning.'.format(path))
        print(key, 'frames remaining:', len(missing), 'of', limit, flush=True)
        if not missing:
            continue
        model = build(cfg, dataset, key, device)
        for offset, (index, token) in enumerate(missing):
            rows = frame_rows(model, dataset, index, device,
                              args.score_threshold)
            path = frame_dir / (token+'.json')
            temporary = frame_dir / (token+'.partial.json')
            with temporary.open('w') as handle:
                json.dump({'token': token, 'index': index, 'cars': rows,
                           'checkpoint_signature': signature}, handle)
            os.replace(str(temporary), str(path))
            if (offset+1) % 25 == 0 or offset+1 == len(missing):
                print(key, offset+1, '/', len(missing), 'new frames', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    summarize(output, args.models, args.score_threshold, tokens)


if __name__ == '__main__':
    main()
