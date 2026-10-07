#!/usr/bin/env python3
"""Compare two PETR checkpoints' normal car detections on one target rig."""

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_car_bev_localization import lidar_to_global, quaternion_matrix


THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'adult',
           'child', 'traffic_light', 'traffic_sign'}
COLORS = {'r1': '#2563eb', 'r1f': '#ef7d32'}
LABELS = {'r1': 'R1-trained PETR', 'r1f': 'R1-f-trained PETR'}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--r1-results', required=True)
    parser.add_argument('--r1f-results', required=True)
    parser.add_argument('--r1-metrics', required=True)
    parser.add_argument('--r1f-metrics', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-rig', choices=['R1', 'R1-f'], default='R1-f')
    parser.add_argument('--score-threshold', type=float, default=0.35)
    parser.add_argument('--association-distance', type=float, default=4.0,
                        help='Maximum BEV center distance for diagnostic one-to-one matching.')
    parser.add_argument('--nearest-distance', type=float, default=10.0,
                        help='Maximum distance for a low-score nearest-candidate probe.')
    parser.add_argument('--examples-per-group', type=int, default=12)
    return parser.parse_args()


def gt_frames(path):
    with open(path, 'rb') as handle:
        infos = pickle.load(handle)['infos']
    frames = {}
    for info in infos:
        token = info['token']
        names = np.asarray(info['gt_names'])
        boxes = np.asarray(info['gt_boxes'])
        keep = names == 'car'
        if 'num_lidar_pts' in info:
            count = np.asarray(info['num_lidar_pts'])
            if 'num_radar_pts' in info:
                count = count + np.asarray(info['num_radar_pts'])
            keep &= count > 0
        lidar_indices = np.flatnonzero(keep)
        centers = boxes[keep, :3]
        gravity_centers = boxes[:, :3].copy()
        gravity_centers[:, 2] += boxes[:, 5] / 2.0
        in_tracer = np.isin(names, list(CLASSES)) & np.all(
            (gravity_centers >= [-51.2, -51.2, -6.0]) &
            (gravity_centers <= [51.2, 51.2, 6.0]), axis=1)
        rotation = quaternion_matrix(info['lidar2ego_rotation'])
        ego = centers @ rotation.T + np.asarray(info['lidar2ego_translation'])
        in_range = np.linalg.norm(ego[:, :2], axis=1) < 50.0
        centers = centers[in_range]
        lidar_indices = lidar_indices[in_range]
        global_centers = lidar_to_global(centers, info)
        frames[token] = {
            'centers': global_centers[:, :2],
            'lidar_centers': centers[:, :2],
            'gt_indices': lidar_indices,
            'tracer_indices': [int(in_tracer[:raw].sum())
                               if in_tracer[raw] else None
                               for raw in lidar_indices],
            'ego_origin': np.asarray(info['ego2global_translation'])[:2],
        }
    return frames


def read_predictions(path):
    with open(path, 'r') as handle:
        payload = json.load(handle)
    output = {}
    for token, boxes in payload['results'].items():
        output[token] = [
            {'center': np.asarray(box['translation'][:2], dtype=float),
             'score': float(box['detection_score'])}
            for box in boxes if box['detection_name'] == 'car']
    return output


def assign(centers, predictions, score_threshold, max_distance):
    """Confidence-ordered, one-to-one diagnostic assignment at one distance."""
    assigned = {}
    used = set()
    ordered = sorted(enumerate(predictions),
                     key=lambda item: item[1]['score'], reverse=True)
    for pred_index, pred in ordered:
        if pred['score'] < score_threshold:
            continue
        remaining = [i for i in range(len(centers)) if i not in used]
        if not remaining:
            break
        distances = np.linalg.norm(centers[remaining] - pred['center'], axis=1)
        closest = int(np.argmin(distances))
        if distances[closest] < max_distance:
            gt_index = remaining[closest]
            assigned[gt_index] = pred_index
            used.add(gt_index)
    return assigned


def nearest(ground_truth, predictions, max_distance):
    if not predictions:
        return None, None
    distances = np.asarray([np.linalg.norm(p['center'] - ground_truth)
                            for p in predictions])
    index = int(np.argmin(distances))
    return (index, float(distances[index])) if distances[index] < max_distance else (None, None)


def finite_stats(values):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    if not len(values):
        return {'count': 0, 'mean': None, 'median': None, 'p90': None}
    return {'count': int(len(values)), 'mean': float(values.mean()),
            'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90))}


def model_values(row, prefix, center, origin, prediction):
    if prediction is None:
        return
    delta = prediction['center'] - center
    ray = center - origin
    norm = np.linalg.norm(ray)
    if norm > 1e-6:
        radial = float(np.dot(delta, ray / norm))
        lateral = float(np.cross(ray / norm, delta))
    else:
        radial, lateral = None, None
    row.update({
        prefix + '_score': prediction['score'],
        prefix + '_pred_x': float(prediction['center'][0]),
        prefix + '_pred_y': float(prediction['center'][1]),
        prefix + '_error_m': float(np.linalg.norm(delta)),
        prefix + '_radial_ego_m': radial,
        prefix + '_lateral_ego_m': lateral,
    })


def build_rows(frames, predictions, args):
    rows = []
    for token, frame in frames.items():
        centers = frame['centers']
        lists = {key: predictions[key].get(token, []) for key in LABELS}
        assigned = {key: assign(centers, lists[key], args.score_threshold,
                                args.association_distance) for key in LABELS}
        for local_index, center in enumerate(centers):
            row = {
                'sample_token': token,
                'gt_index': int(frame['gt_indices'][local_index]),
                'tracer_gt_index': frame['tracer_indices'][local_index],
                'gt_x': float(center[0]), 'gt_y': float(center[1]),
                'gt_lidar_x': float(frame['lidar_centers'][local_index][0]),
                'gt_lidar_y': float(frame['lidar_centers'][local_index][1]),
                'gt_ego_distance_m': float(np.linalg.norm(
                    center - frame['ego_origin'])),
            }
            for key in LABELS:
                match_index = assigned[key].get(local_index)
                prediction = lists[key][match_index] if match_index is not None else None
                row[key + '_matched'] = int(prediction is not None)
                model_values(row, key, center, frame['ego_origin'], prediction)
                nearest_index, nearest_distance = nearest(
                    center, lists[key], args.nearest_distance)
                row[key + '_nearest_distance_m'] = nearest_distance
                row[key + '_nearest_score'] = (
                    lists[key][nearest_index]['score']
                    if nearest_index is not None else None)
            if row['r1_matched'] and row['r1f_matched']:
                row['group'] = 'both'
                row['model_displacement_m'] = float(np.linalg.norm(
                    np.asarray([row['r1_pred_x'] - row['r1f_pred_x'],
                                row['r1_pred_y'] - row['r1f_pred_y']])))
            elif row['r1f_matched']:
                row['group'] = 'r1f_only'
            elif row['r1_matched']:
                row['group'] = 'r1_only'
            else:
                row['group'] = 'neither'
            rows.append(row)
    return rows


def read_ap(path):
    with open(path, 'r') as handle:
        values = json.load(handle)['label_aps']['car']
    return {str(k): float(v) for k, v in values.items()}


def summary(rows, args):
    counts = Counter(row['group'] for row in rows)
    result = {'target_rig': args.target_rig,
              'num_gt_cars': len(rows), 'score_threshold': args.score_threshold,
              'association_distance_m': args.association_distance,
              'groups': dict(counts), 'models': {}}
    for key, metric_path in [('r1', args.r1_metrics), ('r1f', args.r1f_metrics)]:
        errors = [r.get(key + '_error_m') for r in rows]
        matched = [r for r in rows if r[key + '_matched']]
        result['models'][key] = {
            'label': LABELS[key], 'official_car_ap': read_ap(metric_path),
            'matched_count': len(matched),
            'diagnostic_coverage': len(matched) / len(rows) if rows else 0,
            'matched_center_error_m': finite_stats(errors),
            'radial_ego_m': finite_stats(
                r.get(key + '_radial_ego_m') for r in matched),
            'lateral_ego_m': finite_stats(
                r.get(key + '_lateral_ego_m') for r in matched),
            'nearest_candidate_distance_m': finite_stats(
                r.get(key + '_nearest_distance_m') for r in rows),
        }
    both = [r for r in rows if r['group'] == 'both']
    result['paired_models'] = {
        key: {
            'center_error_m': finite_stats(r[key + '_error_m'] for r in both),
            'radial_ego_m': finite_stats(r.get(key + '_radial_ego_m') for r in both),
            'lateral_ego_m': finite_stats(r.get(key + '_lateral_ego_m') for r in both),
        } for key in LABELS}
    result['distance_bins'] = []
    for low in range(0, 50, 10):
        subset = [r for r in rows if low <= r['gt_ego_distance_m'] < low + 10]
        result['distance_bins'].append({
            'range_m': [low, low + 10], 'gt_count': len(subset),
            'groups': dict(Counter(r['group'] for r in subset)),
            'models': {key: {'matched_count': sum(r[key + '_matched'] for r in subset),
                             'coverage': sum(r[key + '_matched'] for r in subset)
                             / len(subset) if subset else 0}
                       for key in LABELS},
        })
    result['paired_center_displacement_m'] = finite_stats(
        r['model_displacement_m'] for r in both)
    result['paired_r1_minus_r1f_error_m'] = finite_stats(
        r['r1_error_m'] - r['r1f_error_m'] for r in both)
    return result


def style(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.grid(alpha=.22)
    axis.set_axisbelow(True)


def plots(rows, report, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelcolor': '#253047', 'text.color': '#253047',
                         'savefig.facecolor': 'white'})
    bins = report['distance_bins']
    x = np.arange(len(bins))
    fig, axis = plt.subplots(figsize=(12.5, 6.7))
    fig.suptitle('Car detection coverage on {}'.format(report['target_rig']), x=.08, y=.98,
                 ha='left', fontsize=19, fontweight='bold')
    fig.text(.08, .915, 'Same validation frames  |  diagnostic match: score ≥ {:.2f}, center < {:.1f} m  |  {} GT cars'.format(
        report['score_threshold'], report['association_distance_m'], report['num_gt_cars']),
        fontsize=10, color='#57647a')
    for index, key in enumerate(LABELS):
        values = [100 * b['models'][key]['coverage'] for b in bins]
        bars = axis.bar(x + (index-.5)*.38, values, width=.36,
                        color=COLORS[key], label=LABELS[key])
        for bar, value, b in zip(bars, values, bins):
            axis.text(bar.get_x()+bar.get_width()/2, value+1.3,
                      '{:.0f}%'.format(value), ha='center', va='bottom',
                      fontsize=10, fontweight='bold', color=COLORS[key])
    axis.axvspan(1.5, 4.5, color='#fff3e9', zorder=-1)
    axis.set_xticks(x)
    axis.set_xticklabels(['{}–{} m\n(n={})'.format(*b['range_m'], b['gt_count'])
                          for b in bins])
    axis.set_ylim(0, 110)
    axis.set_ylabel('GT cars with a matched detection (%)')
    axis.set_xlabel('GT distance from ego vehicle')
    axis.legend(frameon=False, loc='upper right', ncol=2)
    style(axis)
    fig.tight_layout(rect=(.04, .04, .99, .87))
    for suffix in ('png', 'svg'):
        fig.savefig(output / ('car_coverage_by_distance.' + suffix), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('What changes on identical {} frames?'.format(report['target_rig']), x=.06,
                 y=.98, ha='left', fontsize=19, fontweight='bold')
    fig.text(.06, .91, 'Diagnostic detection outcomes and official car AP are different metrics',
             color='#57647a', fontsize=10)
    groups = [('both', 'Both models', '#64748b'),
              ('r1f_only', 'R1-f only', COLORS['r1f']),
              ('r1_only', 'R1 only', COLORS['r1']),
              ('neither', 'Neither', '#cbd5e1')]
    for i, (key, label, color) in enumerate(groups):
        count = report['groups'].get(key, 0)
        axes[0].barh(i, count, color=color, height=.66)
        axes[0].text(count + 22, i, '{}  ({:.1f}%)'.format(
            count, 100*count/report['num_gt_cars']), va='center', fontweight='bold')
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels([g[1] for g in groups])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(report['groups'].values()) * 1.38)
    axes[0].set_xlabel('Number of GT cars')
    axes[0].set_title('Which cars get a final detection?', loc='left', pad=14)
    style(axes[0])
    thresholds = ['0.5', '1.0', '2.0', '4.0']
    for i, key in enumerate(LABELS):
        ap = report['models'][key]['official_car_ap']
        values = [100 * ap[t] for t in thresholds]
        bars = axes[1].bar(x=np.arange(4)+(i-.5)*.36, height=values,
                           width=.34, color=COLORS[key], label=LABELS[key])
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x()+bar.get_width()/2, value+1,
                         '{:.1f}'.format(value), ha='center', fontsize=9,
                         color=COLORS[key], fontweight='bold')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([t+' m' for t in thresholds])
    axes[1].set_ylim(0, max(65, max(100*v for key in LABELS for v in
                                    report['models'][key]['official_car_ap'].values())*1.18))
    axes[1].set_ylabel('Official car AP (%)')
    axes[1].set_title('How accurate are the detections?', loc='left', pad=14)
    axes[1].legend(frameon=False, loc='upper left')
    style(axes[1])
    fig.tight_layout(rect=(.03, .03, .99, .87), w_pad=4)
    for suffix in ('png', 'svg'):
        fig.savefig(output / ('car_coverage_and_error.' + suffix), dpi=220)
    plt.close(fig)

    both = [r for r in rows if r['group'] == 'both']
    if both:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle('Same cars, different centers', x=.06, y=.98,
                     ha='left', fontsize=19, fontweight='bold')
        fig.text(.06, .91, 'Paired comparison on {} cars detected by both models; positive radial = farther from ego'.format(
            len(both)), color='#57647a', fontsize=10)
        fields = [('center_error_m', 'Center error  |  lower is better'),
                  ('radial_ego_m', 'Signed ego-radial error  |  zero is best')]
        for axis, (field, title) in zip(axes, fields):
            vals = [report['paired_models'][key][field]['mean'] for key in LABELS]
            for i, (key, val) in enumerate(zip(LABELS, vals)):
                axis.scatter(val, i, s=190, color=COLORS[key], zorder=3)
                axis.text(val, i-.17, '{:+.2f} m'.format(val) if field == 'radial_ego_m'
                          else '{:.2f} m'.format(val), ha='center', va='bottom',
                          color=COLORS[key], fontweight='bold')
            axis.plot(vals, [0, 1], color='#a8b2c1', linewidth=2, zorder=1)
            axis.axvline(0, color='#253047', linewidth=1)
            axis.set_yticks([0, 1])
            axis.set_yticklabels([LABELS[key] for key in LABELS])
            axis.invert_yaxis()
            axis.set_ylim(1.5, -.5)
            axis.set_xlabel('Mean error (m)')
            axis.set_title(title, loc='left', pad=14)
            axis.set_xlim(min(-.25, min(vals)-.3), max(1.7, max(vals)+.3))
            style(axis)
        fig.tight_layout(rect=(.03, .03, .99, .87), w_pad=4)
        for suffix in ('png', 'svg'):
            fig.savefig(output / ('car_signed_errors.' + suffix), dpi=220)
        plt.close(fig)


def write_insights(report, output):
    b = report['distance_bins'][2]
    p = report['paired_models']
    lines = [
        '# {} validation: car diagnostic'.format(report['target_rig']), '',
        'Normal PETR inference from two independently trained checkpoints on the same {} frames.'.format(
            report['target_rig']),
        '',
        '- {} GT cars. At score ≥ {:.2f} and BEV center < {:.1f} m, R1 matches {} ({:.1f}%), '
        'while R1-f matches {} ({:.1f}%).'.format(
            report['num_gt_cars'], report['score_threshold'],
            report['association_distance_m'], report['models']['r1']['matched_count'],
            100*report['models']['r1']['diagnostic_coverage'],
            report['models']['r1f']['matched_count'],
            100*report['models']['r1f']['diagnostic_coverage']),
        '- At 20–30 m: R1 matches {}/{} ({:.1f}%), R1-f matches {}/{} ({:.1f}%).'.format(
            b['models']['r1']['matched_count'], b['gt_count'],
            100*b['models']['r1']['coverage'], b['models']['r1f']['matched_count'],
            b['gt_count'], 100*b['models']['r1f']['coverage']),
        '- On the {} cars matched by both: mean center error R1 {:.2f} m vs R1-f {:.2f} m; '
        'mean signed ego-radial error R1 {:+.2f} m vs R1-f {:+.2f} m.'.format(
            p['r1']['center_error_m']['count'], p['r1']['center_error_m']['mean'],
            p['r1f']['center_error_m']['mean'], p['r1']['radial_ego_m']['mean'],
            p['r1f']['radial_ego_m']['mean']),
        '', 'Interpretation: compare missed cars by distance and signed BEV center bias '
        'on cars found by both models. This is observational: it does not isolate '
        'camera-ray depth, the attention mechanism, or a causal model component.',
        '', 'Diagnostic matching is not official AP. Official AP uses the evaluator’s ranking '
        'and matching rules; see car_coverage_and_error.png for AP at each threshold.',
    ]
    (output / 'car_insights.md').write_text('\n'.join(lines) + '\n')


def main():
    args = arguments()
    if args.score_threshold < 0 or args.association_distance <= 0:
        raise ValueError('Score threshold must be non-negative and distance positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = gt_frames(args.ann_file)
    predictions = {'r1': read_predictions(args.r1_results),
                   'r1f': read_predictions(args.r1f_results)}
    expected = set(frames)
    for key in LABELS:
        actual = set(predictions[key])
        if actual != expected:
            raise ValueError('{} result tokens do not match {}: expected {}, '
                             'actual {}, common {}. Check the annotation split.'.format(
                                 key, args.ann_file, len(expected), len(actual),
                                 len(expected & actual)))
    rows = build_rows(frames, predictions, args)
    report = summary(rows, args)
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with (output / 'car_per_gt.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'car_summary.json').open('w') as handle:
        json.dump(report, handle, indent=2)
    target_key = 'r1' if args.target_rig == 'R1' else 'r1f'
    source_key = 'r1f' if target_key == 'r1' else 'r1'
    examples = sorted((r for r in rows if r['group'] == target_key + '_only'
                       and r['tracer_gt_index'] is not None),
                      key=lambda r: (-r[target_key + '_score'], r['sample_token']))[
                          :args.examples_per_group]
    examples += sorted((r for r in rows if r['group'] == 'both'
                        and r['tracer_gt_index'] is not None),
                       key=lambda r: r[source_key + '_error_m'] -
                       r[target_key + '_error_m'],
                       reverse=True)[:args.examples_per_group]
    seen = set()
    selected = []
    for row in examples:
        key = (row['sample_token'], row['gt_index'])
        if key not in seen:
            selected.append(row)
            seen.add(key)
    with (output / 'trace_candidates.json').open('w') as handle:
        json.dump(selected, handle, indent=2)
    plots(rows, report, output)
    write_insights(report, output)
    print('GT cars:', len(rows), 'Groups:', dict(report['groups']))
    print('Saved analysis to:', output)


if __name__ == '__main__':
    main()
