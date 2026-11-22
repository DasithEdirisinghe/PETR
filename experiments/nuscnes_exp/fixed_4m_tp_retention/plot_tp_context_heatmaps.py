#!/usr/bin/env python3
"""Plot mean GT visibility and neighboring-GT preference of PETR 4 m TPs.

Uses the saved nuScenes result JSON and evaluator-filtered validation GT.
No PETR inference is run. The assigned GT is explicitly excluded when
measuring the nearest *other* same-class GT to each prediction center.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'petr_localization_visualization'))
import pipeline
from plot_tp_visibility_bands import render_visibility_bands

DISTANCE_LABELS = tuple(entry[1] for entry in pipeline.DIST)
ERROR_LABELS = tuple(entry[1] for entry in pipeline.ERROR)
# nuScenes stores visibility intervals, not an exact visible fraction.
VISIBILITY_MIDPOINT_PERCENT = {'1': 20, '2': 50, '3': 70, '4': 90}
TP_FIELDS = (
    'class', 'sample_token', 'prediction_identifier', 'score',
    'assigned_gt_annotation_token', 'assigned_gt_instance_token',
    'gt_distance_m', 'distance_bin', 'assigned_gt_error_m', 'error_bin',
    'visibility_token', 'visibility_midpoint_percent', 'nearest_other_gt_m',
    'nearest_other_gt_center_global', 'nearest_other_gt_is_matched',
    'nearest_other_gt_owner_score', 'nearest_other_gt_owner_higher_score',
    'other_gt_closer_than_assigned', 'other_gt_within_4m',
    'pred_center_inside_any_other_gt_bev',
)
SUMMARY_FIELDS = (
    'class', 'distance_bin', 'error_bin', 'tp_count',
    'visibility_0_40_count', 'visibility_40_60_count',
    'visibility_60_80_count', 'visibility_80_100_count',
    'mean_visibility_percent', 'low_visibility_count', 'low_visibility_percent',
    'nearest_other_gt_available_count', 'mean_nearest_other_gt_m',
    'median_nearest_other_gt_m', 'other_gt_within_4m_count',
    'other_gt_within_4m_percent', 'other_gt_closer_count',
    'other_gt_closer_percent', 'pred_center_inside_other_gt_count',
    'pred_center_inside_other_gt_percent',
)


def point_inside_bev_box(x, y, box):
    """Test the point against the box's rotated BEV footprint."""
    dx, dy = x - box.center[0], y - box.center[1]
    rotation = box.rotation_matrix
    local_x = rotation[0, 0] * dx + rotation[1, 0] * dy
    local_y = rotation[0, 1] * dx + rotation[1, 1] * dy
    return (abs(local_x) <= box.wlh[1] / 2 + 1e-8 and
            abs(local_y) <= box.wlh[0] / 2 + 1e-8)


def other_gt_context(evaluator, sample_token, name, pred, assigned_gt,
                     owners):
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    px, py = pred.translation[:2]
    nearest, nearest_distance, inside_other = None, float('inf'), False
    for other in evaluator.gt_boxes[sample_token]:
        if other.detection_name != name or other is assigned_gt:
            continue
        distance = math.hypot(px - other.translation[0],
                              py - other.translation[1])
        if distance < nearest_distance:
            nearest, nearest_distance = other, distance
        # The half-diagonal bounds the oriented footprint; skip distant boxes.
        half_diagonal = math.hypot(other.size[0], other.size[1]) / 2
        if distance <= half_diagonal + 1e-8:
            box = Box(other.translation, other.size, Quaternion(other.rotation))
            if point_inside_bev_box(px, py, box):
                inside_other = True
    if nearest is None:
        return None, None, False, None, False, False
    owner_score = owners.get(id(nearest))
    return (nearest, nearest_distance, inside_other, owner_score,
            nearest_distance < 4.0,
            nearest_distance < evaluator.cfg.dist_fcn_callable(assigned_gt, pred))


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize(rows, names):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['class'], row['distance_bin'], row['error_bin']].append(row)
        grouped['all classes', row['distance_bin'], row['error_bin']].append(row)
    output = []
    for name in names:
        for distance_label in DISTANCE_LABELS:
            for error_label in ERROR_LABELS:
                cell = grouped[name, distance_label, error_label]
                count = len(cell)
                visibility = {token: sum(row['visibility_token'] == token
                                         for row in cell) for token in ('1', '2', '3', '4')}
                low = visibility['1'] + visibility['2']
                other_distances = [row['nearest_other_gt_m'] for row in cell
                                   if row['nearest_other_gt_m'] is not None]
                within = sum(row['other_gt_within_4m'] for row in cell)
                closer = sum(row['other_gt_closer_than_assigned'] for row in cell)
                inside = sum(row['pred_center_inside_any_other_gt_bev']
                             for row in cell)
                output.append({
                    'class': name, 'distance_bin': distance_label,
                    'error_bin': error_label, 'tp_count': count,
                    'visibility_0_40_count': visibility['1'],
                    'visibility_40_60_count': visibility['2'],
                    'visibility_60_80_count': visibility['3'],
                    'visibility_80_100_count': visibility['4'],
                    'mean_visibility_percent':
                        sum(VISIBILITY_MIDPOINT_PERCENT[row['visibility_token']]
                            for row in cell) / count if count else None,
                    'low_visibility_count': low,
                    'low_visibility_percent': 100 * low / count if count else None,
                    'nearest_other_gt_available_count': len(other_distances),
                    'mean_nearest_other_gt_m':
                        sum(other_distances) / len(other_distances)
                        if other_distances else None,
                    'median_nearest_other_gt_m':
                        median(other_distances) if other_distances else None,
                    'other_gt_within_4m_count': within,
                    'other_gt_within_4m_percent':
                        100 * within / count if count else None,
                    'other_gt_closer_count': closer,
                    'other_gt_closer_percent':
                        100 * closer / count if count else None,
                    'pred_center_inside_other_gt_count': inside,
                    'pred_center_inside_other_gt_percent':
                        100 * inside / count if count else None,
                })
    return output


def plot_heatmap(rows, names, metric, title, note, path, upper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    lookup = {(row['class'], row['distance_bin'], row['error_bin']): row
              for row in rows}
    panel_rows = math.ceil(len(names) / 2)
    fig, axes = plt.subplots(panel_rows, 2,
                             figsize=(15.5, 3.5 * panel_rows), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    color = plt.cm.Blues
    image = None
    for panel, name in enumerate(names):
        ax = axes[panel]
        values = np.full((len(DISTANCE_LABELS), len(ERROR_LABELS)), np.nan)
        for i, distance_label in enumerate(DISTANCE_LABELS):
            for j, error_label in enumerate(ERROR_LABELS):
                row = lookup[name, distance_label, error_label]
                value = row[metric]
                if value is not None:
                    values[i, j] = value
        ax.set_facecolor('#edf1f5')
        image = ax.imshow(np.ma.masked_invalid(values), cmap=color,
                          vmin=0, vmax=upper, aspect='auto')
        ax.set_title(name.replace('_', ' '), loc='left', fontsize=13,
                     fontweight='bold', pad=12)
        ax.set_xticks(range(len(ERROR_LABELS)))
        ax.set_xticklabels([label + ' m' for label in ERROR_LABELS], fontsize=9)
        ax.set_yticks(range(len(DISTANCE_LABELS)))
        ax.set_yticklabels([label + ' m' for label in DISTANCE_LABELS], fontsize=9)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                       length=0)
        ax.set_xticks(np.arange(-.5, len(ERROR_LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(DISTANCE_LABELS), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='minor', length=0)
        for i, distance_label in enumerate(DISTANCE_LABELS):
            for j, error_label in enumerate(ERROR_LABELS):
                row = lookup[name, distance_label, error_label]
                value = row[metric]
                if not row['tp_count']:
                    label = 'No 4 m TPs'
                elif value is None:
                    label = 'No other GT\n0/{:,} TPs'.format(row['tp_count'])
                elif metric == 'mean_visibility_percent':
                    label = '{:.1f}%\n{:,} TPs'.format(value, row['tp_count'])
                elif metric == 'other_gt_closer_percent':
                    label = '{:.1f}%\n{:,}/{:,} TPs'.format(
                        value, row['other_gt_closer_count'], row['tp_count'])
                elif metric.endswith('_percent'):
                    label = '{:.1f}%\n{:,} TPs'.format(value, row['tp_count'])
                else:
                    label = '{:.2f} m\nother GT: {:,}/{:,}'.format(
                        value, row['nearest_other_gt_available_count'],
                        row['tp_count'])
                text_color = 'white' if value is not None and value > upper * .65 else '#203040'
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        color=text_color, linespacing=1.25)
    for ax in axes[len(names):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=19, fontweight='bold', x=.05, ha='left', y=.995)
    fig.text(.05, .968, note, fontsize=10, color='#465669', va='top')
    fig.subplots_adjust(left=.08, right=.93, top=.92, bottom=.035,
                        hspace=.35, wspace=.18)
    fig.colorbar(image, ax=list(axes[:len(names)]), fraction=.015, pad=.012,
                 label='Percent' if metric.endswith('_percent') else 'Metres')
    fig.savefig(path.with_suffix('.png'), dpi=160)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent /
                        'outputs/baseline')
    parser.add_argument('--classes', nargs='+', choices=pipeline.CLASSES,
                        default=list(pipeline.CLASSES))
    args = parser.parse_args()
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved PETR predictions or nuScenes data root missing')
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    evaluator = NuScenesEval(nusc, config_factory('detection_cvpr_2019'),
                            str(args.results_json), 'val',
                            str(args.output_dir / 'devkit_work'), verbose=False)
    all_tokens = set(evaluator.gt_boxes.sample_tokens)
    tp_rows = []
    for name in args.classes:
        matches = list(pipeline.official_pairs(evaluator, all_tokens, [name]))
        owners = {id(gt): pred.detection_score
                  for _, _, _, pred, gt, _ in matches}
        for _, sample_token, pred_index, pred, gt, error in matches:
            ann = pipeline.annotation(nusc, gt)
            visibility = ann['visibility_token']
            if visibility not in ('1', '2', '3', '4'):
                raise ValueError('Unexpected nuScenes visibility token: ' + visibility)
            distance = math.hypot(*gt.ego_translation[:2])
            distance_label, _ = pipeline.band(distance, pipeline.DIST)
            error_label, _ = pipeline.band(error, pipeline.ERROR)
            nearest, other_distance, inside, owner_score, within4, closer = (
                other_gt_context(evaluator, sample_token, name, pred, gt, owners))
            tp_rows.append({
                'class': name, 'sample_token': sample_token,
                'prediction_identifier': '{}:{}'.format(sample_token, pred_index),
                'score': pred.detection_score,
                'assigned_gt_annotation_token': ann['token'],
                'assigned_gt_instance_token': ann['instance_token'],
                'gt_distance_m': distance, 'distance_bin': distance_label,
                'assigned_gt_error_m': error, 'error_bin': error_label,
                'visibility_token': visibility,
                'visibility_midpoint_percent':
                    VISIBILITY_MIDPOINT_PERCENT[visibility],
                'nearest_other_gt_m': other_distance,
                'nearest_other_gt_center_global':
                    (','.join(str(value) for value in nearest.translation)
                     if nearest is not None else ''),
                'nearest_other_gt_is_matched': owner_score is not None,
                'nearest_other_gt_owner_score': owner_score,
                'nearest_other_gt_owner_higher_score':
                    owner_score is not None and owner_score > pred.detection_score,
                'other_gt_closer_than_assigned': closer,
                'other_gt_within_4m': within4,
                'pred_center_inside_any_other_gt_bev': inside,
            })
        print('{}: {:,} matched 4 m TPs'.format(name, len(matches)), flush=True)
    names = ['all classes'] + args.classes
    summary = summarize(tp_rows, names)
    write_csv(args.output_dir / 'tp_context.csv', TP_FIELDS, tp_rows)
    write_csv(args.output_dir / 'tp_context_by_class_range_error.csv',
              SUMMARY_FIELDS, summary)
    render_visibility_bands(summary, args.output_dir, names)
    plot_heatmap(summary, names, 'other_gt_closer_percent',
                 'PETR 4 m TPs · prediction closer to another GT',
                 'Each TP scores 100% if another same-class GT center is closer to the prediction than its assigned GT, else 0%.',
                 args.output_dir / 'tp_other_gt_closer_heatmap', 100)
    print('Wrote context CSVs and heatmaps to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
