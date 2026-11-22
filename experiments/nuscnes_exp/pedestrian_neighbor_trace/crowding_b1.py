#!/usr/bin/env python3
"""Compare B1 frequency by fixed same-class GT crowding across eval thresholds."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
CROWDING_M = {
    'pedestrian': 2.0, 'bicycle': 2.0, 'motorcycle': 2.0,
    'traffic_cone': 2.0, 'car': 4.0, 'truck': 4.0, 'bus': 4.0,
    'trailer': 4.0, 'construction_vehicle': 4.0, 'barrier': 4.0,
}


def crowded_ids(frame_positions, threshold):
    """GT is crowded iff another same-class GT has XY center distance < Tc."""
    result = set()
    for boxes in frame_positions.values():
        for i, (gt_id, x, y) in enumerate(boxes):
            for other_id, ox, oy in boxes[i + 1:]:
                if math.hypot(x - ox, y - oy) < threshold:
                    result.add(gt_id)
                    result.add(other_id)
    return result


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else math.nan


def analyze(transitions_path, summary_path):
    with summary_path.open(newline='') as handle:
        source = {(row['class'], float(row['threshold_m'])): row
                  for row in csv.DictReader(handle)}
    expected = {(name, threshold) for name in CROWDING_M for threshold in THRESHOLDS}
    if set(source) != expected:
        raise ValueError('Transition summary must have exactly 10 classes x 4 thresholds')

    # GT positions are identical at each evaluation threshold. Read them once
    # from 0.5 m records; retain only the small fields needed at other thresholds.
    frames = defaultdict(list)
    records = defaultdict(list)
    seen = set()
    with transitions_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            name = row['class']
            threshold = float(row['threshold_m'])
            if (name, threshold) not in expected:
                raise ValueError('Unexpected class or evaluation threshold')
            key = (name, row['gt_id'])
            unique = (name, threshold, row['gt_id'])
            if unique in seen:
                raise ValueError('Duplicate GT/threshold record: ' + str(unique))
            seen.add(unique)
            if threshold == 0.5:
                x, y = json.loads(row['gt_box_json'])['translation'][:2]
                frames[name, row['sample_token']].append((row['gt_id'], x, y))
            both = row['baseline_prediction_id'] != '' and row['oracle_prediction_id'] != ''
            b1 = row['subtype'] == 'B1'
            if b1 and not both:
                raise ValueError('B1 case is not matched by both methods')
            records[name, threshold].append((key, both, b1))

    crowded = {}
    for name, distance in CROWDING_M.items():
        crowded[name] = crowded_ids(
            {token: boxes for (cls, token), boxes in frames.items() if cls == name},
            distance)
    result = []
    for name in CROWDING_M:
        population = {
            gt_id for (cls, token), boxes in frames.items() if cls == name
            for gt_id, _, _ in boxes}
        for threshold in THRESHOLDS:
            subset = records[name, threshold]
            if len(subset) != len(population) or {key[1] for key, _, _ in subset} != population:
                raise ValueError('GT population changes across evaluation thresholds: ' + name)
            counts = Counter()
            for (_, gt_id), both, b1 in subset:
                group = 'crowded' if gt_id in crowded[name] else 'non_crowded'
                counts[group + '_total_gt'] += 1
                counts[group + '_both_matched'] += both
                counts[group + '_b1'] += b1
            saved = source[name, threshold]
            if sum(counts[g + '_total_gt'] for g in ('crowded', 'non_crowded')) != int(saved['total_gt']):
                raise ValueError('GT population disagrees with transition summary')
            if sum(counts[g + '_both_matched'] for g in ('crowded', 'non_crowded')) != int(saved['both_matched']):
                raise ValueError('Both-matched count disagrees with transition summary')
            if sum(counts[g + '_b1'] for g in ('crowded', 'non_crowded')) != int(saved['subtype_B1']):
                raise ValueError('B1 count disagrees with transition summary')
            crowded_rate = percentage(counts['crowded_b1'], counts['crowded_both_matched'])
            non_crowded_rate = percentage(counts['non_crowded_b1'], counts['non_crowded_both_matched'])
            result.append({
                'class': name, 'eval_threshold_m': threshold,
                'crowding_threshold_m': CROWDING_M[name],
                'total_gt': len(population),
                'crowded_total_gt': counts['crowded_total_gt'],
                'non_crowded_total_gt': counts['non_crowded_total_gt'],
                'crowded_both_matched': counts['crowded_both_matched'],
                'non_crowded_both_matched': counts['non_crowded_both_matched'],
                'crowded_b1': counts['crowded_b1'],
                'non_crowded_b1': counts['non_crowded_b1'],
                'crowded_b1_percent': crowded_rate,
                'non_crowded_b1_percent': non_crowded_rate,
                'difference_percentage_points': crowded_rate - non_crowded_rate,
            })
    if len(result) != 40:
        raise ValueError('Expected exactly 40 class-threshold rows')
    return result


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transitions-csv', type=Path, default=base / 'outputs/assignment_transition/gt_transitions.csv')
    parser.add_argument('--summary-csv', type=Path, default=base / 'outputs/assignment_transition/all_threshold_summary.csv')
    parser.add_argument('--output-csv', type=Path, default=base / 'outputs/assignment_transition/crowding_b1_summary.csv')
    args = parser.parse_args()
    for path in (args.transitions_csv, args.summary_csv):
        if not path.is_file():
            parser.error('Missing ' + str(path))
    rows = analyze(args.transitions_csv, args.summary_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote {} validated rows to {}'.format(len(rows), args.output_csv))
    for row in rows:
        if row['eval_threshold_m'] == 2.0:
            print('{}: crowded {:.2f}% ({}/{}), non-crowded {:.2f}% ({}/{})'.format(
                row['class'], row['crowded_b1_percent'], row['crowded_b1'],
                row['crowded_both_matched'], row['non_crowded_b1_percent'],
                row['non_crowded_b1'], row['non_crowded_both_matched']))


if __name__ == '__main__':
    main()
