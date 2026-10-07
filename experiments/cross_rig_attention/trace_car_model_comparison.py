#!/usr/bin/env python3
"""Trace selected target-rig cars through both vanilla PETR checkpoints."""

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_r1_r1f_cars import gt_frames


ROOT = Path(__file__).resolve().parents[2]
TRACER = ROOT / 'experiments/query_trace/trace_query.py'
CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
CKPTS = {
    'r1': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth',
    'r1f': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth',
}


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--target-rig', choices=['R1', 'R1-f'], default='R1-f')
    parser.add_argument('--selection-rig', choices=['R1', 'R1-f'], default=None,
                        help='Rig whose trace_candidates.json selects the physical cars. '
                             'Defaults to --target-rig.')
    parser.add_argument('--max-pair-distance', type=float, default=0.25,
                        help='Maximum world-XY GT center gap across paired rigs (m).')
    parser.add_argument('--analysis-dir', default=None)
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def trace_one(row, model_key, split, target_rig, output_dir, device, force):
    token = row['sample_token']
    gt_index = row['tracer_gt_index']
    run_dir = output_dir / model_key / '{}_gt{:03d}'.format(token, gt_index)
    summary_path = run_dir / 'trace_summary.json'
    if force or not summary_path.is_file():
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(TRACER), '--config', str(CONFIG),
            '--checkpoint', str(CKPTS[model_key]),
            '--data-root', 'data/pccr/{}/'.format(target_rig),
            '--ann-file', 'data/pccr/{0}/{0}_infos_{1}.pkl'.format(target_rig, split),
            '--sample-token', token, '--gt-indices', str(gt_index),
            '--classes', 'car', '--device', device,
            '--run-label', '.', '--output-dir', str(run_dir),
        ]
        print('Tracing {} car {} on {}'.format(model_key, gt_index, token), flush=True)
        with (run_dir / 'trace.log').open('w') as handle:
            subprocess.run(command, cwd=str(ROOT), stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
    with summary_path.open('r') as handle:
        summary = json.load(handle)
    if summary['sample_token'] != token:
        raise ValueError('Trace sample token mismatch: ' + str(summary_path))
    targets = list(summary['targets'].values())
    if len(targets) != 1:
        raise ValueError('Expected one traced car in ' + str(summary_path))
    target = targets[0]
    if target['gt_class'] != 'car' or target['gt_index'] != gt_index:
        raise ValueError('Traced GT mismatch: ' + str(summary_path))
    if np.linalg.norm(np.asarray(target['gt_box'][:2]) -
                      np.asarray([row['gt_lidar_x'], row['gt_lidar_y']])) > .01:
        raise ValueError('GT center mismatch: ' + str(summary_path))
    return target, summary_path


def scene_frame_lookup(rig, split):
    data_root = ROOT / 'data/pccr' / rig
    table_root = data_root / 'v1.0-trainval'
    with (table_root / 'sample.json').open('r') as handle:
        samples = {row['token']: row for row in json.load(handle)}
    with (table_root / 'scene.json').open('r') as handle:
        scenes = {row['token']: row['name'] for row in json.load(handle)}
    with (data_root / '{}_infos_{}.pkl'.format(rig, split)).open('rb') as handle:
        infos = pickle.load(handle)['infos']
    frames_by_scene = {}
    for info in infos:
        sample = samples.get(info['token'])
        if sample is None:
            raise ValueError('Annotation token absent from {} sample.json: {}'.format(
                rig, info['token']))
        name = scenes[sample['scene_token']]
        frames_by_scene.setdefault(name, []).append(info)
    by_token = {}
    for name, frames in frames_by_scene.items():
        frames.sort(key=lambda info: info['timestamp'])
        for index, info in enumerate(frames):
            by_token[info['token']] = (name, index)
    return by_token, {name: [info['token'] for info in frames]
                      for name, frames in frames_by_scene.items()}


def pair_selected_cars(candidates, selection_rig, target_rig, split,
                       max_distance):
    source_lookup, _ = scene_frame_lookup(selection_rig, split)
    _, target_scenes = scene_frame_lookup(target_rig, split)
    target_frames = gt_frames(
        ROOT / 'data/pccr' / target_rig /
        '{}_infos_{}.pkl'.format(target_rig, split))
    paired = []
    for source in candidates:
        scene, frame_index = source_lookup[source['sample_token']]
        if scene not in target_scenes or frame_index >= len(target_scenes[scene]):
            raise ValueError('No synchronized {} frame for {} frame {}'.format(
                target_rig, scene, frame_index))
        token = target_scenes[scene][frame_index]
        frame = target_frames[token]
        distances = np.linalg.norm(
            frame['centers'] - np.asarray([source['gt_x'], source['gt_y']]),
            axis=1)
        if not len(distances):
            raise ValueError('No eligible GT cars in paired frame {}'.format(token))
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance > max_distance:
            raise ValueError('GT car pair {} → {} is {:.3f} m apart '
                             '(limit {:.3f} m); refusing a different object'.format(
                                 source['sample_token'], token, distance,
                                 max_distance))
        tracer_index = frame['tracer_indices'][index]
        if tracer_index is None:
            raise ValueError('Paired car is not in tracer GT set: {}'.format(token))
        target = dict(source)
        target.update({
            'sample_token': token,
            'gt_index': int(frame['gt_indices'][index]),
            'tracer_gt_index': int(tracer_index),
            'gt_x': float(frame['centers'][index, 0]),
            'gt_y': float(frame['centers'][index, 1]),
            'gt_lidar_x': float(frame['lidar_centers'][index, 0]),
            'gt_lidar_y': float(frame['lidar_centers'][index, 1]),
            'source_rig': selection_rig,
            'source_sample_token': source['sample_token'],
            'source_gt_index': source['gt_index'],
            'source_tracer_gt_index': source['tracer_gt_index'],
            'scene_name': scene,
            'scene_frame_index': frame_index,
            'cross_rig_gt_distance_m': distance,
        })
        paired.append(target)
    return paired


def main():
    options = args()
    if options.max_examples < 1:
        raise ValueError('--max-examples must be positive')
    if options.max_pair_distance <= 0:
        raise ValueError('--max-pair-distance must be positive')
    selection_rig = options.selection_rig or options.target_rig
    analysis = (Path(options.analysis_dir) if options.analysis_dir else
                ROOT / 'experiments/cross_rig_attention/output' /
                '{}_{}'.format('r1' if selection_rig == 'R1' else 'r1f',
                               options.split) / 'analysis')
    with (analysis / 'car_summary.json').open('r') as handle:
        recorded_rig = json.load(handle).get('target_rig', 'R1-f')
    if recorded_rig != selection_rig:
        raise ValueError('Analysis targets {}, but --selection-rig is {}'.format(
            recorded_rig, selection_rig))
    with (analysis / 'trace_candidates.json').open('r') as handle:
        candidates = json.load(handle)[:options.max_examples]
    if not candidates:
        raise RuntimeError('No selected cars; run car comparison first')
    output = (ROOT / 'experiments/cross_rig_attention/output' /
              '{}_{}'.format('r1' if options.target_rig == 'R1' else 'r1f',
                             options.split) / 'traces')
    output.mkdir(parents=True, exist_ok=True)
    if selection_rig != options.target_rig:
        candidates = pair_selected_cars(
            candidates, selection_rig, options.target_rig, options.split,
            options.max_pair_distance)
        with (output / 'paired_selection.json').open('w') as handle:
            json.dump(candidates, handle, indent=2)
    records = []
    for car in candidates:
        for model_key in CKPTS:
            target, path = trace_one(car, model_key, options.split,
                                     options.target_rig,
                                     output, options.device, options.force)
            for layer, error in enumerate(target['layer_prediction_to_gt_bev_m'], 1):
                record = {
                    'sample_token': car['sample_token'],
                    'gt_index': car['gt_index'],
                    'group': car['group'], 'model': model_key,
                    'query_index': target['query_index'],
                    'layer': layer,
                    'center_error_m': float(error),
                    'car_confidence': float(target['layer_confidence'][layer-1]),
                    'initial_reference_error_m': float(
                        target['reference_to_gt_bev_m']),
                    'trace_summary': str(path),
                }
                impacts = target.get('layer_component_impact_percent', [])
                if layer <= len(impacts):
                    for term, value in impacts[layer-1].items():
                        record['impact_' + term.replace('-', '_') + '_pct'] = float(value)
                records.append(record)
    csv_path = output / 'layer_car_comparison.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(
            *(record.keys() for record in records))))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for model_key, color, label in [('r1', '#2563eb', 'R1-trained'),
                                    ('r1f', '#ef7d32', 'R1-f-trained')]:
        rows = [r for r in records if r['model'] == model_key]
        for axis, field, ylabel in [(axes[0], 'center_error_m', 'BEV center error (m)'),
                                    (axes[1], 'car_confidence', 'Car confidence')]:
            layers = sorted(set(r['layer'] for r in rows))
            medians = [np.median([r[field] for r in rows if r['layer'] == layer])
                       for layer in layers]
            axis.plot(layers, medians, marker='o', markerfacecolor='white',
                      markeredgewidth=2, linewidth=2.4, color=color, label=label)
            axis.set_xticks(layers)
            axis.set_xlabel('Decoder layer')
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.2)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('Selected GT cars on the same {} frames (median)'.format(
        options.target_rig),
                 x=.05, ha='left', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output / 'layer_car_comparison.png', dpi=220)
    fig.savefig(output / 'layer_car_comparison.svg')
    plt.close(fig)
    print('Saved layer analysis:', csv_path)


if __name__ == '__main__':
    main()
