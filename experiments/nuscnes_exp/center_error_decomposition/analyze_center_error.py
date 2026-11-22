#!/usr/bin/env python3
"""Decompose baseline PETR's official 4 m TP center errors in ego BEV axes."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0))
FIELDS = (
    'sample_token', 'class', 'score', 'center_error_m', 'gt_range_m',
    'radial_signed_m', 'tangential_signed_m', 'radial_abs_m',
    'tangential_abs_m', 'error_bin_m',
)


def error_bin(distance):
    for lower, upper in BINS:
        if lower <= distance < upper:
            return '{}-{}'.format(lower, upper)
    raise ValueError('A 4 m match must have center error < 4 m')


def match_at_four_m(evaluator):
    """Replay the devkit's confidence order and closest-unmatched-GT rule."""
    matches = []
    for class_name in evaluator.cfg.class_names:
        predictions = [box for box in evaluator.pred_boxes.all
                       if box.detection_name == class_name]
        ranked = [index for _, index in sorted(
            (box.detection_score, index)
            for index, box in enumerate(predictions))][::-1]
        taken = set()
        for index in ranked:
            pred = predictions[index]
            token = pred.sample_token
            nearest_index, nearest_distance = None, float('inf')
            for gt_index, gt in enumerate(evaluator.gt_boxes[token]):
                if gt.detection_name != class_name or (token, gt_index) in taken:
                    continue
                distance = evaluator.cfg.dist_fcn_callable(gt, pred)
                if distance < nearest_distance:
                    nearest_index, nearest_distance = gt_index, distance
            if nearest_index is not None and nearest_distance < 4.0:
                taken.add((token, nearest_index))
                matches.append((pred, evaluator.gt_boxes[token][nearest_index],
                                nearest_distance))
    return matches


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['error_bin_m']].append(row)
    summary = []
    for lower, upper in BINS:
        label = '{}-{}'.format(lower, upper)
        group = groups[label]
        summary.append({
            'center_error_bin_m': label,
            'count': len(group),
            'percent_of_4m_tps': 100 * len(group) / len(rows) if rows else None,
            'mean_center_error_m': mean([r['center_error_m'] for r in group]),
            'mean_signed_radial_m': mean([r['radial_signed_m'] for r in group]),
            'mean_abs_radial_m': mean([r['radial_abs_m'] for r in group]),
            'mean_signed_tangential_m': mean([r['tangential_signed_m'] for r in group]),
            'mean_abs_tangential_m': mean([r['tangential_abs_m'] for r in group]),
            'mean_score': mean([r['score'] for r in group]),
            'mean_gt_range_m': mean([r['gt_range_m'] for r in group]),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='nuScenes root containing v1.0-trainval metadata')
    parser.add_argument('--results-json', type=Path, required=True,
                        help='baseline PETR formatted results_nusc.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.data_root.exists():
        parser.error('nuScenes data root does not exist: {}'.format(args.data_root))
    if not args.results_json.is_file():
        parser.error('formatted predictions do not exist: {}'.format(args.results_json))

    # Keep devkit imports inside main so --help and syntax checks work without
    # the cluster's PETR environment.
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                   verbose=False)
    evaluator = NuScenesEval(
        nusc, config=config_factory('detection_cvpr_2019'),
        result_path=str(args.results_json), eval_set='val',
        output_dir=str(args.output_dir), verbose=False)
    matches = match_at_four_m(evaluator)

    ego_origins = {}
    rows = []
    for pred, gt, distance in matches:
        token = pred.sample_token
        if token not in ego_origins:
            sample = nusc.get('sample', token)
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_origins[token] = nusc.get(
                'ego_pose', sample_data['ego_pose_token'])['translation'][:2]
        ego_x, ego_y = ego_origins[token]
        ray_x, ray_y = gt.translation[0] - ego_x, gt.translation[1] - ego_y
        gt_range = math.hypot(ray_x, ray_y)
        if gt_range <= 1e-8:
            raise ValueError('GT at ego origin in sample {}'.format(token))
        ux, uy = ray_x / gt_range, ray_y / gt_range
        dx = pred.translation[0] - gt.translation[0]
        dy = pred.translation[1] - gt.translation[1]
        radial = dx * ux + dy * uy
        tangential = -dx * uy + dy * ux
        if abs(math.hypot(radial, tangential) - distance) > 1e-5:
            raise ValueError('Radial/tangential decomposition mismatch')
        rows.append({
            'sample_token': token, 'class': pred.detection_name,
            'score': float(pred.detection_score),
            'center_error_m': float(distance), 'gt_range_m': gt_range,
            'radial_signed_m': radial, 'tangential_signed_m': tangential,
            'radial_abs_m': abs(radial), 'tangential_abs_m': abs(tangential),
            'error_bin_m': error_bin(distance),
        })

    with (args.output_dir / 'center_error_matches.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    class_rows = []
    for class_name in evaluator.cfg.class_names:
        for item in summarize([row for row in rows if row['class'] == class_name]):
            class_rows.append({'class': class_name, **item})
    with (args.output_dir / 'center_error_by_class.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    with (args.output_dir / 'center_error_summary.json').open('w') as handle:
        json.dump({'match_threshold_m': 4.0, 'matches': len(rows),
                   'coordinate_frame': 'ego-centered BEV, radial toward GT',
                   'bin_rule': 'lower inclusive, upper exclusive, as in nuScenes matching',
                   'source': str(args.results_json), 'bins': summary}, handle, indent=2)
    print('4 m true positives:', len(rows))
    print('{:<9} {:>8} {:>11} {:>11} {:>11} {:>11} {:>11}'.format(
        'Error m', 'Count', '% of TPs', 'Mean e', '|Radial|', '|Tang.|',
        'Radial bias'))
    for row in summary:
        print('{:<9} {:>8} {:>10.1f}% {:>11.3f} {:>11.3f} {:>11.3f} {:>11.3f}'.format(
            row['center_error_bin_m'], row['count'],
            row['percent_of_4m_tps'] or 0,
            row['mean_center_error_m'] or 0,
            row['mean_abs_radial_m'] or 0,
            row['mean_abs_tangential_m'] or 0,
            row['mean_signed_radial_m'] or 0))


if __name__ == '__main__':
    main()
