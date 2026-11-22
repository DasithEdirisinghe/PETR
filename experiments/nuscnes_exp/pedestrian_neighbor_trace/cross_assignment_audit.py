#!/usr/bin/env python3
"""Audit whether official TPs are duplicates of previously claimed nearby GTs.

The input population is every officially matched GT whose best-3D candidate is
a different prediction.  For each official TP assigned to Ga, the audit checks
previously claimed same-class GTs Gb in the same sample.  A cross-assignment is
recorded when the TP is both closer to Gb and has higher projected 2D IoU with
Gb than with Ga.  The script also records whether the unused best-3D candidate
fits Ga better, exposing the full duplicate -> reassignment -> unused-box chain.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from localization_capacity import load_filtered_class
from pedestrian_image_alignment import (camera_records, center_distance,
                                        measurements, reference_camera)

EPS = 1e-6
CLASSES = ('car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')


def as_float(row, key):
    return float(row[key]) if row.get(key, '') != '' else None


def as_int(row, key):
    return int(row[key]) if row.get(key, '') != '' else None


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(count, total):
    return 100*count/total if total else 0.0


def main():
    base = Path(__file__).resolve().parent
    petr = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-name', required=True, choices=CLASSES)
    parser.add_argument('--model-name', default='PETR')
    parser.add_argument('--all-gt-csv', type=Path, required=True)
    parser.add_argument('--results-json', type=Path, default=petr /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=petr.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=4.0)
    args = parser.parse_args()

    with args.all_gt_csv.open(newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    target_rows = [row for row in all_rows
                   if row['official_match'] == '1' and
                   row['different_3d_candidate'] == '1']
    if not target_rows:
        parser.error('No matched different-candidate rows found')

    # Prediction IDs in all_gt.csv index the same stable confidence-sorted list.
    owner_by_gt = {}
    for row in all_rows:
        pi = as_int(row, 'selected_prediction_id')
        if pi is not None:
            owner_by_gt[(row['sample_token'], int(row['gt_index']))] = pi

    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                    verbose=False)
    gt_by_sample, predictions = load_filtered_class(
        args.results_json, args.data_root, args.class_name)
    top10_count = max(1, math.ceil(.1*len(predictions)))
    view_cache = {}
    camera_record_cache = {}

    def view(token, gi):
        key = (token, gi)
        if key not in view_cache:
            if token not in camera_record_cache:
                camera_record_cache[token] = camera_records(nusc, token)
            view_cache[key] = reference_camera(
                camera_record_cache[token], gt_by_sample[token][gi])
        return view_cache[key]

    output = []
    counts = Counter()
    for position, row in enumerate(target_rows, 1):
        token, ga = row['sample_token'], int(row['gt_index'])
        ps_index = int(row['selected_prediction_id'])
        ps = predictions[ps_index]
        selected_center = float(row['selected_center_m'])
        selected_iou2d = as_float(row, 'selected_projected_iou2d')
        selected_iou3d = float(row['selected_iou3d'])

        prior = []
        for gb, gt in enumerate(gt_by_sample[token]):
            if gb == ga:
                continue
            owner = owner_by_gt.get((token, gb))
            if owner is None or owner >= ps_index:
                continue
            distance = center_distance(ps, gt)
            if distance >= args.threshold:
                continue
            _, camera, gt_rect = view(token, gb)
            metric = measurements(ps, gt, camera, gt_rect)
            prior.append((gb, owner, metric))

        counts['has_prior_claimed_gt_within_threshold'] += bool(prior)
        qualifying = []
        for gb, owner, metric in prior:
            center_better = metric['center_m'] + EPS < selected_center
            iou2d_better = selected_iou2d is not None and \
                metric['projected_iou2d'] is not None and \
                metric['projected_iou2d'] > selected_iou2d + EPS
            if center_better:
                counts['any_prior_gt_closer'] += 1
                # Count once per current GT, not once per neighbouring GT.
                break
        for gb, owner, metric in prior:
            center_better = metric['center_m'] + EPS < selected_center
            iou2d_better = selected_iou2d is not None and \
                metric['projected_iou2d'] is not None and \
                metric['projected_iou2d'] > selected_iou2d + EPS
            if center_better and iou2d_better:
                qualifying.append((gb, owner, metric))

        if qualifying:
            gb, owner, other = max(
                qualifying,
                key=lambda item: (item[2]['projected_iou2d']-selected_iou2d,
                                  selected_center-item[2]['center_m'],
                                  item[2]['iou3d']-selected_iou3d))
        elif prior:
            gb, owner, other = min(prior, key=lambda item: item[2]['center_m'])
        else:
            gb, owner, other = None, None, None

        cross = bool(qualifying)
        cross_3d = cross and other['iou3d'] > selected_iou3d + EPS
        alt_center = as_float(row, 'best_3d_center_m')
        alt_iou2d = as_float(row, 'best_3d_projected_iou2d')
        alt_iou3d = as_float(row, 'best_3d_iou3d')
        alt_better_center_2d = alt_center is not None and alt_iou2d is not None and \
            selected_iou2d is not None and alt_center + EPS < selected_center and \
            alt_iou2d > selected_iou2d + EPS
        alt_better_3d = alt_iou3d is not None and \
            alt_iou3d > selected_iou3d + EPS
        cascade = cross and alt_better_center_2d
        strict_cascade = cascade and alt_better_3d

        counts['cross_assigned_center_and_2d'] += cross
        counts['cross_assigned_center_2d_and_3d'] += cross_3d
        counts['alternative_better_current_center_and_2d'] += alt_better_center_2d
        counts['strictly_higher_alternative_3d_iou'] += alt_better_3d
        counts['full_assignment_cascade'] += cascade
        counts['strict_full_assignment_cascade'] += strict_cascade
        counts['cross_assigned_top10_percent'] += cross and ps_index < top10_count

        output.append({
            'sample_token': token, 'gt_index': ga,
            'visibility_token': row['visibility_token'],
            'selected_prediction_id': ps_index,
            'selected_score': row['selected_score'],
            'selected_top10_percent': int(ps_index < top10_count),
            'selected_assigned_center_m': selected_center,
            'selected_assigned_projected_iou2d': selected_iou2d,
            'selected_assigned_iou3d': selected_iou3d,
            'prior_claimed_gt_count_within_threshold': len(prior),
            'cross_assigned_duplicate': int(cross),
            'cross_assigned_duplicate_strict_3d': int(cross_3d),
            'other_gt_index': '' if gb is None else gb,
            'other_gt_owner_prediction_id': '' if owner is None else owner,
            'other_gt_owner_score': '' if owner is None else predictions[owner].detection_score,
            'selected_other_center_m': '' if other is None else other['center_m'],
            'selected_other_projected_iou2d': '' if other is None else other['projected_iou2d'],
            'selected_other_iou3d': '' if other is None else other['iou3d'],
            'alternative_prediction_id': row['best_3d_prediction_id'],
            'alternative_score': row['best_3d_score'],
            'alternative_center_m': alt_center,
            'alternative_projected_iou2d': alt_iou2d,
            'alternative_iou3d': alt_iou3d,
            'alternative_better_current_center_and_2d': int(alt_better_center_2d),
            'alternative_strictly_better_3d_iou': int(alt_better_3d),
            'full_assignment_cascade': int(cascade),
            'strict_full_assignment_cascade': int(strict_cascade),
        })
        if position % 1000 == 0:
            print(f'Processed {position:,}/{len(target_rows):,}', flush=True)

    total = len(output)
    summary = {
        'model_name': args.model_name, 'class_name': args.class_name,
        'threshold_m': args.threshold,
        'different_candidate_case_count': total,
        'prediction_count': len(predictions),
        'top10_prediction_count': top10_count,
        'counts': dict(counts),
        'percent_of_different_candidate_cases': {
            key: percent(value, total) for key, value in counts.items()
        },
        'definitions': {
            'cross_assigned_duplicate':
                'Official TP is closer to, and has higher projected 2D IoU with, another same-class GT already claimed by an earlier prediction.',
            'full_assignment_cascade':
                'Cross-assigned duplicate TP and the different best-3D candidate has lower center error and higher projected 2D IoU for the official assigned GT.',
            'strict_full_assignment_cascade':
                'Full assignment cascade plus strictly higher alternative 3D IoU for the assigned GT.'
        }
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/'cases.csv', output)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = [f'# {args.model_name} {args.class_name} cross-assignment audit', '',
             f'Different-candidate cases: {total:,}. Threshold: {args.threshold:g} m.', '',
             '| Measure | Count | Percent |', '|---|---:|---:|']
    for key, value in counts.items():
        lines.append(f'| {key.replace("_", " ")} | {value:,} | {percent(value,total):.2f}% |')
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
