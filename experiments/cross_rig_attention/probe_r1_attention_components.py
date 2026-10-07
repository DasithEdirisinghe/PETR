#!/usr/bin/env python3
"""Frozen-R1 sensitivity test for X-key, G3D-key, and whole-key contributions.

Uses unchanged paired images/calibration. GT selects the same physical car and
scores the output; no GT is passed into PETR's forward computation.
"""

import argparse
import csv
import importlib
import json
import os
import pickle
import sys
from contextlib import contextmanager
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, prepare_one, relocate_dataset_paths)
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox  # noqa: E402
from plot_car_bev_localization import quaternion_matrix  # noqa: E402


CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
CHECKPOINT = ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth'
BRANCHES = ('x_key', 'g3d_key', 'all_key')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.75, 1.0, 1.25])
    parser.add_argument('--branches', nargs='+', choices=BRANCHES,
                        default=list(BRANCHES))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--paired-selection', default=None,
                        help='Use the same paired cohort as the trace stage.')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def dataset_for(cfg, rig, split):
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/{}/'.format(rig)
    definition.ann_file = 'data/pccr/{0}/{0}_infos_{1}.pkl'.format(rig, split)
    definition.test_mode = True
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    return dataset


def radial_error(center, gt, info):
    rotation = quaternion_matrix(info['lidar2ego_rotation'])
    translation = np.asarray(info['lidar2ego_translation'], dtype=np.float64)
    ego_lidar = -rotation.T @ translation
    ray = np.asarray(gt[:2]) - ego_lidar[:2]
    ray /= max(np.linalg.norm(ray), 1e-8)
    return float(np.dot(np.asarray(center[:2])-np.asarray(gt[:2]), ray))


def score_query(outputs, query_index, car_id, gt_box, info):
    encoded = outputs['all_bbox_preds'][:, 0, query_index]
    decoded = denormalize_bbox(encoded, None).detach().float().cpu().numpy()
    confidence = outputs['all_cls_scores'][:, 0, query_index, car_id].sigmoid()
    confidence = confidence.detach().float().cpu().numpy()
    result = []
    for layer, (box, score) in enumerate(zip(decoded, confidence), 1):
        result.append({
            'layer': layer,
            'center_error_m': float(np.linalg.norm(box[:2]-gt_box[:2])),
            'radial_ego_m': radial_error(box, gt_box, info),
            'car_confidence': float(score),
        })
    return result


@contextmanager
def component_scale(head, branch, alpha):
    """Scale one key branch; keep V unchanged and restore methods afterwards."""
    originals = []
    if branch == 'g3d_key':
        original = head.position_embeding

        def scaled_geometry(*args, **kwargs):
            embedding, mask = original(*args, **kwargs)
            return embedding * alpha, mask

        head.position_embeding = scaled_geometry
        originals.append((head, 'position_embeding', original))
    else:
        for layer in head.transformer.decoder.layers:
            module = layer.attentions[1]
            original = module.forward

            def scaled_cross_attention(*args, _original=original, **kwargs):
                values = list(args)
                query = kwargs.get('query', values[0] if values else None)
                key = kwargs.get('key', values[1] if len(values) > 1 else None)
                value = kwargs.get('value', values[2] if len(values) > 2 else None)
                base_key = query if key is None else key
                base_value = base_key if value is None else value
                scaled_key = base_key * alpha
                if len(values) > 1:
                    values[1] = scaled_key
                else:
                    kwargs['key'] = scaled_key
                # Explicitly retain the original V if PETR omitted value.
                if len(values) > 2:
                    values[2] = base_value
                else:
                    kwargs['value'] = base_value
                if branch == 'all_key':
                    key_pos = kwargs.get('key_pos',
                                         values[5] if len(values) > 5 else None)
                    if key_pos is not None:
                        if len(values) > 5:
                            values[5] = key_pos * alpha
                        else:
                            kwargs['key_pos'] = key_pos * alpha
                return _original(*values, **kwargs)

            module.forward = scaled_cross_attention
            originals.append((module, 'forward', original))
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def max_output_difference(a, b):
    return float(max(
        (a['all_bbox_preds']-b['all_bbox_preds']).abs().max(),
        (a['all_cls_scores']-b['all_cls_scores']).abs().max()))


def main():
    args = arguments()
    if args.max_examples < 1 or not args.alphas or any(a <= 0 for a in args.alphas):
        raise ValueError('Need positive examples and positive alpha values')
    if 1.0 not in args.alphas:
        raise ValueError('--alphas must include 1.0 for the exact-replay check')
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    datasets = {rig: dataset_for(cfg, rig, args.split) for rig in ('R1', 'R1-f')}
    token_indices = {rig: {row['token']: index for index, row in
                           enumerate(dataset.data_infos)}
                     for rig, dataset in datasets.items()}
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(CHECKPOINT), map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', datasets['R1'].CLASSES)
    car_id = list(model.CLASSES).index('car')
    device = torch.device(args.device)
    model.to(device).eval()
    scatter_device = device.index if device.type == 'cuda' else device
    base = ROOT / 'experiments/cross_rig_attention/output'
    r1 = base / 'r1_{}'.format(args.split)
    r1f = base / 'r1f_{}'.format(args.split)
    selection_path = (Path(args.paired_selection) if args.paired_selection else
                      r1 / 'traces/paired_selection.json')
    with selection_path.open('r') as handle:
        pairs = json.load(handle)[:args.max_examples]
    if not pairs:
        raise ValueError('Paired selection is empty: {}'.format(selection_path))
    output = Path(args.output_dir or (
        selection_path.parent / 'sensitivity' if args.paired_selection else
        r1f / 'traces/r1_component_sensitivity'))
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'per_car_per_layer.csv').exists() and not args.force:
        print('Reusing existing intervention result:', output)
        return
    rows = []
    for pair_index, pair in enumerate(pairs):
        source_token = pair['sample_token']
        source_gt_index = int(pair['tracer_gt_index'])
        trace_path = (r1 / 'traces/r1' /
                      '{}_gt{:03d}/trace_summary.json'.format(
                          source_token, source_gt_index))
        with trace_path.open('r') as handle:
            target = next(iter(json.load(handle)['targets'].values()))
        query_index = target['query_index']
        gt_box = np.asarray(target['gt_box'])
        for rig, token in [('R1', source_token),
                           ('R1-f', pair['source_sample_token'])]:
            dataset = datasets[rig]
            index = token_indices[rig][token]
            info = dataset.data_infos[index]
            prepared = prepare_one(dataset, index, scatter_device)
            images, metas = first_augmentation(prepared)
            metas[0]['camera_names'] = list(info['cams'])
            with torch.no_grad():
                features = model.extract_feat(img=images.clone(), img_metas=metas)
                baseline = model.pts_bbox_head(features, metas)
                if rig == 'R1':
                    baseline_center = denormalize_bbox(
                        baseline['all_bbox_preds'][-1, 0, query_index],
                        None)[:2].detach().float().cpu().numpy()
                    saved_center = np.asarray(target['layer_box_decoded'][-1][:2])
                    if np.linalg.norm(baseline_center - saved_center) > .01:
                        raise AssertionError('R1 baseline does not reproduce '
                                             'the cached trace for car {}'.format(pair_index))
                for branch in args.branches:
                    for alpha in args.alphas:
                        with component_scale(model.pts_bbox_head, branch, alpha):
                            altered = model.pts_bbox_head(features, metas)
                        if alpha == 1.0:
                            difference = max_output_difference(baseline, altered)
                            if difference > 1e-5:
                                raise AssertionError('{} alpha=1 replay differs by {}'.format(
                                    branch, difference))
                        for layer_row in score_query(
                                altered, query_index, car_id, gt_box, info):
                            layer_row.update({
                                'pair_index': pair_index, 'rig': rig,
                                'sample_token': token, 'query_index': query_index,
                                'branch': branch, 'alpha': alpha,
                                'gt_ego_distance_m': pair.get('gt_ego_distance_m'),
                                'selection_group': pair.get('group'),
                            })
                            rows.append(layer_row)
            print('Interventions:', pair_index+1, '/', len(pairs), rig, flush=True)
    with (output / 'per_car_per_layer.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'num_paired_cars': len(pairs), 'alphas': args.alphas,
               'branches': args.branches, 'conditions': {}}
    for rig in ('R1', 'R1-f'):
        summary['conditions'][rig] = {}
        for branch in args.branches:
            entries = {}
            baseline_by_pair = {r['pair_index']: r for r in rows
                                if r['rig'] == rig and r['branch'] == branch and
                                r['alpha'] == 1.0 and r['layer'] == 6}
            for alpha in args.alphas:
                subset = [r for r in rows if r['rig'] == rig and
                          r['branch'] == branch and r['alpha'] == alpha and
                          r['layer'] == 6]
                entries[str(alpha)] = {
                    'mean_center_error_m': float(np.mean(
                        [r['center_error_m'] for r in subset])),
                    'mean_radial_ego_m': float(np.mean(
                        [r['radial_ego_m'] for r in subset])),
                    'mean_car_confidence': float(np.mean(
                        [r['car_confidence'] for r in subset])),
                    'mean_delta_center_error_m': float(np.mean([
                        r['center_error_m'] - baseline_by_pair[
                            r['pair_index']]['center_error_m'] for r in subset])),
                    'fraction_cars_improved': float(np.mean([
                        r['center_error_m'] < baseline_by_pair[
                            r['pair_index']]['center_error_m'] for r in subset])),
                }
            summary['conditions'][rig][branch] = entries
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    colors = {'x_key': '#2563eb', 'g3d_key': '#ef7d32',
              'all_key': '#64748b'}
    for axis, rig in zip(axes, ('R1', 'R1-f')):
        for branch in args.branches:
            values = [summary['conditions'][rig][branch][str(alpha)][
                'mean_center_error_m'] for alpha in args.alphas]
            axis.plot(args.alphas, values, '-o', color=colors[branch],
                      label=branch.replace('_', ' '))
        axis.axvline(1, color='#253047', linewidth=1, linestyle=':')
        axis.set_title('{} frames'.format(rig))
        axis.set_xlabel('Key-component scale α  (1 = original model)')
        axis.set_ylabel('Mean fixed-query BEV center error (m)')
        axis.grid(alpha=.2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('Frozen R1 PETR: component sensitivity on the same cars',
                 x=.04, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output / 'component_sensitivity.png', dpi=220)
    fig.savefig(output / 'component_sensitivity.svg')
    plt.close(fig)
    print('Saved component sensitivity:', output)


if __name__ == '__main__':
    main()
