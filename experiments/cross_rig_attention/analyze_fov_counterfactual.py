#!/usr/bin/env python3
"""Compare frozen-PETR FoV counterfactuals using official AP and car-center error."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_r1_r1f_cars import assign, finite_stats, gt_frames, read_predictions


COLORS = {'raw': '#64748b', 'image_warp': '#2563eb',
          'feature_warp': '#ef7d32', 'source_view': '#16a34a'}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--score-threshold', type=float, default=.35)
    parser.add_argument('--association-distance', type=float, default=4.0)
    return parser.parse_args()


def main():
    args = arguments()
    root = Path(args.experiment_dir)
    frames = gt_frames(args.ann_file)
    modes = [name for name in COLORS if
             (root / name / 'formatted/results_pccr.json').is_file()]
    if 'raw' not in modes:
        raise FileNotFoundError('Full-split raw results are required for comparison')
    detections = {}
    ap = {}
    for mode in modes:
        predictions = read_predictions(root / mode / 'formatted/results_pccr.json')
        if set(predictions) != set(frames):
            raise ValueError('{} prediction token mismatch'.format(mode))
        detections[mode] = predictions
        with (root / mode / 'formatted/metrics_summary.json').open() as handle:
            ap[mode] = json.load(handle)['label_aps']['car']

    rows = []
    for token, frame in frames.items():
        matches = {mode: assign(frame['centers'], detections[mode][token],
                                args.score_threshold, args.association_distance)
                   for mode in modes}
        for index, gt in enumerate(frame['centers']):
            ray = gt - frame['ego_origin']
            ray /= max(np.linalg.norm(ray), 1e-8)
            row = {'sample_token': token,
                   'gt_index': int(frame['gt_indices'][index]),
                   'ego_distance_m': float(np.linalg.norm(
                       gt - frame['ego_origin']))}
            for mode in modes:
                pred_index = matches[mode].get(index)
                row[mode] = None
                if pred_index is not None:
                    pred = detections[mode][token][pred_index]
                    delta = pred['center'] - gt
                    row[mode] = {
                        'score': pred['score'],
                        'center_error_m': float(np.linalg.norm(delta)),
                        'radial_ego_m': float(np.dot(delta, ray)),
                        'lateral_ego_m': float(np.cross(ray, delta)),
                    }
            rows.append(row)

    report = {'gt_cars': len(rows), 'score_threshold': args.score_threshold,
              'association_distance_m': args.association_distance,
              'modes': {}, 'paired_with_raw': {}, 'distance_bins': []}
    for mode in modes:
        matched = [r[mode] for r in rows if r[mode] is not None]
        report['modes'][mode] = {
            'official_car_ap': ap[mode], 'matched_count': len(matched),
            'coverage': len(matched)/len(rows) if rows else 0,
            'center_error_m': finite_stats(x['center_error_m'] for x in matched),
            'radial_ego_m': finite_stats(x['radial_ego_m'] for x in matched),
        }
        both = [r for r in rows if r['raw'] is not None and r[mode] is not None]
        report['paired_with_raw'][mode] = {
            'count': len(both),
            'raw_center_error_m': finite_stats(r['raw']['center_error_m'] for r in both),
            'mode_center_error_m': finite_stats(r[mode]['center_error_m'] for r in both),
            'raw_radial_ego_m': finite_stats(r['raw']['radial_ego_m'] for r in both),
            'mode_radial_ego_m': finite_stats(r[mode]['radial_ego_m'] for r in both),
        }
    for low in range(0, 50, 10):
        subset = [r for r in rows if low <= r['ego_distance_m'] < low+10]
        report['distance_bins'].append({
            'range_m': [low, low+10], 'gt_count': len(subset),
            'matched': {mode: sum(r[mode] is not None for r in subset)
                        for mode in modes}})
    with (root / 'comparison.json').open('w') as handle:
        json.dump(report, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    x = np.arange(4)
    width = .8/len(modes)
    for index, mode in enumerate(modes):
        values = [100 * float(ap[mode][str(threshold)])
                  for threshold in (.5, 1.0, 2.0, 4.0)]
        bars = axes[0].bar(x-.4+width*(index+.5), values, width=width,
                           color=COLORS[mode], label=mode.replace('_', ' '))
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x()+bar.get_width()/2, value+.7,
                         '{:.1f}'.format(value), ha='center', fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['0.5 m', '1 m', '2 m', '4 m'])
    axes[0].set_ylabel('Official car AP (%)')
    axes[0].set_title('Actual detection quality')
    axes[0].legend(frameon=False)
    bins = report['distance_bins']
    x = np.arange(len(bins))
    for mode in modes:
        values = [100*b['matched'][mode]/b['gt_count'] if b['gt_count'] else 0
                  for b in bins]
        axes[1].plot(x, values, '-o', color=COLORS[mode], linewidth=2.3,
                     label=mode.replace('_', ' '))
        for xx, yy in zip(x, values):
            axes[1].text(xx, yy+2, '{:.0f}'.format(yy), ha='center', fontsize=8,
                         color=COLORS[mode])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['{}–{}\n(n={})'.format(*b['range_m'], b['gt_count'])
                             for b in bins])
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel('GT cars matched (%)')
    axes[1].set_title('Where detections recover')
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Frozen PETR: calibration-consistent FoV normalization',
                 x=.04, ha='left', fontsize=17, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(root / 'comparison.png', dpi=220)
    fig.savefig(root / 'comparison.svg')
    plt.close(fig)
    print('Saved:', root / 'comparison.json')


if __name__ == '__main__':
    main()
