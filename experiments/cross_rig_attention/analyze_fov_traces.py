#!/usr/bin/env python3
"""Analyze the same eight cars' decoder trajectories and GT-ray attention."""

import argparse
import csv
import json
import pickle
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_pair import load_trace, target_metrics
from plot_car_bev_localization import quaternion_matrix


ROOT = Path(__file__).resolve().parents[2]
COLORS = {'r1': '#2563eb', 'r1f': '#ef7d32'}
RIG_STYLES = {'R1': '-', 'R1-f': '--'}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--box-expansion', type=float, default=.5)
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def info_by_token(rig, split):
    path = ROOT / 'data/pccr' / rig / '{}_infos_{}.pkl'.format(rig, split)
    with path.open('rb') as handle:
        return {row['token']: row for row in pickle.load(handle)['infos']}


def radial_error(decoded, gt, info):
    rotation = quaternion_matrix(info['lidar2ego_rotation'])
    translation = np.asarray(info['lidar2ego_translation'], dtype=np.float64)
    ego_lidar = -rotation.T @ translation
    ray = np.asarray(gt[:2]) - ego_lidar[:2]
    ray /= max(np.linalg.norm(ray), 1e-8)
    return float(np.dot(np.asarray(decoded[:2]) - np.asarray(gt[:2]), ray))


def main():
    args = arguments()
    r1_dir = ROOT / 'experiments/cross_rig_attention/output' / 'r1_{}'.format(args.split)
    r1f_dir = ROOT / 'experiments/cross_rig_attention/output' / 'r1f_{}'.format(args.split)
    with (r1_dir / 'traces/paired_selection.json').open('r') as handle:
        pairs = json.load(handle)[:args.max_examples]
    with (r1f_dir / 'analysis/trace_candidates.json').open('r') as handle:
        source_candidates = {
            (row['sample_token'], row['gt_index']): row
            for row in json.load(handle)}
    if not pairs:
        raise ValueError('No paired cars to analyze')
    infos = {'R1': info_by_token('R1', args.split),
             'R1-f': info_by_token('R1-f', args.split)}
    records = []
    for pair_number, pair in enumerate(pairs):
        for rig, root, token, gt_index in [
                ('R1', r1_dir, pair['sample_token'], pair['tracer_gt_index']),
                ('R1-f', r1f_dir, pair['source_sample_token'],
                 pair.get('source_tracer_gt_index', source_candidates[
                     (pair['source_sample_token'], pair['source_gt_index'])][
                         'tracer_gt_index']))]:
            for model in ('r1', 'r1f'):
                directory = (root / 'traces' / model /
                             '{}_gt{:03d}'.format(token, gt_index))
                trace = load_trace(directory)
                targets = trace['summary']['targets']
                if len(targets) != 1:
                    raise ValueError('Expected one target in ' + str(directory))
                target_name, target = next(iter(targets.items()))
                if target['gt_index'] != gt_index or target['gt_class'] != 'car':
                    raise ValueError('GT mismatch in ' + str(directory))
                metrics = target_metrics(
                    '{} / {}'.format(rig, model), trace, target_name, target,
                    args.box_expansion, top_k=200)
                for layer, row in enumerate(metrics):
                    predicted = target['layer_box_decoded'][layer]
                    row.update({
                        'pair_number': pair_number,
                        'scene_name': pair['scene_name'],
                        'scene_frame_index': pair['scene_frame_index'],
                        'rig': rig, 'model': model,
                        'sample_token': token,
                        'radial_ego_m': radial_error(
                            predicted, target['gt_box'], infos[rig][token]),
                    })
                    base_rate = (row['gt_ray_count']/row['valid_token_count']
                                 if row['valid_token_count'] else 0)
                    row['gt_attention_enrichment'] = (
                        row['gt_attention_mass']/base_rate if base_rate else None)
                    row.pop('camera_attention_mass', None)
                    records.append(row)
    output = Path(args.output_dir or r1_dir / 'traces/fov_diagnostic')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'paired_layer_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(
            *(row.keys() for row in records))))
        writer.writeheader()
        writer.writerows(records)

    fields = [('bev_error_m', 'BEV center error (m)'),
              ('radial_ego_m', 'Signed ego-radial error (m)'),
              ('gt_attention_enrichment', 'GT-ray attention enrichment'),
              ('confidence', 'Car confidence')]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    summary = {}
    for rig in ('R1', 'R1-f'):
        for model in ('r1', 'r1f'):
            subset = [r for r in records if r['rig'] == rig and r['model'] == model]
            key = '{}_{}'.format(rig, model)
            summary[key] = {}
            for axis, (field, title) in zip(axes.flat, fields):
                layers = sorted(set(r['layer'] for r in subset))
                values = []
                for layer in layers:
                    observed = [float(r[field]) for r in subset
                                if r['layer'] == layer and r[field] is not None and
                                np.isfinite(float(r[field]))]
                    values.append(statistics.median(observed) if observed else np.nan)
                axis.plot(layers, values, marker='o', linewidth=2,
                          linestyle=RIG_STYLES[rig], color=COLORS[model],
                          label='{} images / {} model'.format(rig, model))
                axis.set_title(title, loc='left')
                axis.set_xlabel('Decoder layer')
                axis.set_xticks(layers)
                axis.grid(alpha=.2)
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                summary[key][field] = values
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Same physical cars across FoVs: decoder and attention diagnostics',
                 x=.04, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / 'paired_layer_metrics.png', dpi=220)
    fig.savefig(output / 'paired_layer_metrics.svg')
    plt.close(fig)
    with (output / 'paired_layer_summary.json').open('w') as handle:
        json.dump({'num_paired_cars': len(pairs), 'median_by_layer': summary,
                   'attention_metric': 'Attention mass on feature rays intersecting '
                   'GT box expanded by {} m, divided by their token fraction.'.format(
                       args.box_expansion)}, handle, indent=2)
    print('Saved paired trace analysis:', output)


if __name__ == '__main__':
    main()
