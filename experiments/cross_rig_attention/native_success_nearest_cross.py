#!/usr/bin/env python3
"""For native-detected GT cars, inspect the closest cross-rig final output."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / 'experiments/cross_rig_attention/output'
FIELDS = (
    'rig', 'sample_token', 'gt_index', 'gt_x', 'gt_y',
    'native_model', 'native_prediction_index', 'native_score',
    'native_center_distance_m', 'cross_model', 'cross_prediction_index',
    'cross_prediction_count_in_frame', 'cross_box_x', 'cross_box_y',
    'cross_class', 'cross_score', 'cross_center_distance_m',
    'cross_car_class', 'cross_within_1m', 'cross_within_4m',
    'cross_nearest_shared_by_gt_count',
    'cross_car_matched_within_4m', 'cross_matched_car_prediction_index',
    'cross_matched_car_score', 'cross_matched_car_distance_m',
)


def predictions(path):
    with path.open() as handle:
        payload = json.load(handle)['results']
    return {token: [(tuple(float(value) for value in box['translation'][:2]),
                     box['detection_name'], float(box['detection_score']))
                    for box in boxes] for token, boxes in payload.items()}


def gt_frames_from_analysis(path):
    """Read GT centers already exported by the existing car comparison."""
    frames = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            frame = frames.setdefault(row['sample_token'],
                                      {'centers': [], 'gt_indices': []})
            frame['centers'].append((float(row['gt_x']), float(row['gt_y'])))
            frame['gt_indices'].append(int(row['gt_index']))
    return frames


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def native_car_matches(centers, outputs):
    """Evaluator-style car matching: descending score, nearest available GT."""
    taken = set()
    matches = {}
    ordered = sorted((i for i, box in enumerate(outputs) if box[1] == 'car'),
                     key=lambda i: -outputs[i][2])
    for index in ordered:
        point = outputs[index][0]
        available = [i for i in range(len(centers)) if i not in taken]
        if not available:
            break
        distances = [math.dist(centers[i], point) for i in available]
        choice = min(range(len(distances)), key=distances.__getitem__)
        if distances[choice] < 4.0:
            gt_index = available[choice]
            matches[gt_index] = (index, float(distances[choice]))
            taken.add(gt_index)
    return matches


def closest_box(gt_center, outputs):
    """Use any final class; resolve identical-distance boxes by higher score."""
    if not outputs:
        return None
    distances = [math.dist(gt_center, box[0]) for box in outputs]
    index = min(range(len(outputs)),
                key=lambda i: (distances[i], -outputs[i][2], i))
    return index, distances[index]


def run_rig(rig):
    native = rig
    cross = 'R1-f' if rig == 'R1' else 'R1'
    directory = OUTPUT_ROOT / ('r1_val' if rig == 'R1' else 'r1f_val')
    frames = gt_frames_from_analysis(directory / 'analysis/car_per_gt.csv')
    outputs = {model: predictions(directory / model / 'formatted/results_pccr.json')
               for model in (native, cross)}
    for model, results in outputs.items():
        if not set(frames).issubset(set(results)):
            raise ValueError('{} {} results omit GT-car validation tokens'
                             .format(rig, model))

    rows = []
    for token, frame in frames.items():
        centers = frame['centers']
        native_outputs = outputs[native][token]
        cross_outputs = outputs[cross][token]
        native_matches = native_car_matches(centers, native_outputs)
        cross_matches = native_car_matches(centers, cross_outputs)
        selections = {gt: closest_box(centers[gt], cross_outputs)
                      for gt in native_matches}
        shared = Counter(item[0] for item in selections.values() if item is not None)
        for gt, (native_index, native_distance) in native_matches.items():
            center = centers[gt]
            chosen = selections[gt]
            row = dict.fromkeys(FIELDS, '')
            row.update({
                'rig': rig, 'sample_token': token,
                'gt_index': int(frame['gt_indices'][gt]),
                'gt_x': float(center[0]), 'gt_y': float(center[1]),
                'native_model': native,
                'native_prediction_index': native_index,
                'native_score': native_outputs[native_index][2],
                'native_center_distance_m': native_distance,
                'cross_model': cross,
                'cross_prediction_count_in_frame': len(cross_outputs),
                'cross_car_matched_within_4m': int(gt in cross_matches),
            })
            if gt in cross_matches:
                cross_index, cross_distance = cross_matches[gt]
                row.update({
                    'cross_matched_car_prediction_index': cross_index,
                    'cross_matched_car_score': cross_outputs[cross_index][2],
                    'cross_matched_car_distance_m': cross_distance,
                })
            if chosen is not None:
                index, distance = chosen
                box = cross_outputs[index]
                row.update({
                    'cross_prediction_index': index,
                    'cross_box_x': box[0][0],
                    'cross_box_y': box[0][1],
                    'cross_class': box[1], 'cross_score': box[2],
                    'cross_center_distance_m': distance,
                    'cross_car_class': int(box[1] == 'car'),
                    'cross_within_1m': int(distance < 1.0),
                    'cross_within_4m': int(distance < 4.0),
                    'cross_nearest_shared_by_gt_count': shared[index],
                })
            rows.append(row)
    return rows, {'native_model': native, 'cross_model': cross,
                  'validation_frames': len(outputs[native]),
                  'frames_with_gt_cars': len(frames)}


def summarize(rows, metadata):
    distances = [float(row['cross_center_distance_m']) for row in rows
                 if row['cross_center_distance_m'] != '']
    classes = Counter(row['cross_class'] or 'no_output' for row in rows)
    paired = [row for row in rows if row['cross_car_matched_within_4m']]
    outcomes = Counter()
    for row in rows:
        if not row['cross_class']:
            outcome = 'no_output_in_frame'
        elif row['cross_class'] == 'car':
            outcome = 'car_within_4m' if row['cross_within_4m'] else 'car_beyond_4m'
        else:
            outcome = ('other_class_within_4m' if row['cross_within_4m']
                       else 'other_class_beyond_4m')
        outcomes[outcome] += 1
    return dict(metadata, native_success_gt_cars=len(rows),
                cross_car_matched_within_4m=len(paired),
                native_mean_distance_on_paired_m=(statistics.mean(
                    float(row['native_center_distance_m']) for row in paired)
                    if paired else None),
                cross_mean_distance_on_paired_m=(statistics.mean(
                    float(row['cross_matched_car_distance_m']) for row in paired)
                    if paired else None),
                nearest_cross_class_counts=dict(classes),
                nearest_cross_outcomes=dict(outcomes),
                nearest_cross_distance_median=(statistics.median(distances)
                                                 if distances else None),
                nearest_cross_distance_p90=(percentile(distances, 90)
                                            if distances else None),
                nearest_cross_within_1m=sum(row['cross_within_1m'] == 1 for row in rows),
                nearest_cross_within_4m=sum(row['cross_within_4m'] == 1 for row in rows),
                gt_cars_sharing_nearest_box=sum(
                    isinstance(row['cross_nearest_shared_by_gt_count'], int) and
                    row['cross_nearest_shared_by_gt_count'] > 1 for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=OUTPUT_ROOT / 'native_success_nearest_cross')
    args = parser.parse_args()
    all_rows = []
    summary = {'selection': 'native car true positive at strict 4 m',
               'cross_lookup': 'closest final output of any class, no distance cutoff',
               'rigs': {}}
    for rig in ('R1', 'R1-f'):
        rows, metadata = run_rig(rig)
        all_rows.extend(rows)
        summary['rigs'][rig] = summarize(rows, metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'per_gt_nearest_cross.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print('Saved', args.output_dir / 'per_gt_nearest_cross.csv')
    print('Saved', args.output_dir / 'summary.json')


if __name__ == '__main__':
    main()
