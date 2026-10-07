#!/usr/bin/env python3
"""Build a balanced paired-car cohort across distance and detection outcomes."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from trace_car_model_comparison import pair_selected_cars


ROOT = Path(__file__).resolve().parents[2]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--per-stratum', type=int, default=5,
                        help='Maximum cars per 10-m × outcome stratum.')
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--max-pair-distance', type=float, default=.25)
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def main():
    args = arguments()
    if args.per_stratum < 1:
        raise ValueError('--per-stratum must be positive')
    root = ROOT / 'experiments/cross_rig_attention/output'
    path = root / 'r1f_{}'.format(args.split) / 'analysis/car_per_gt.csv'
    strata = defaultdict(list)
    with path.open(newline='') as handle:
        for source in csv.DictReader(handle):
            if source['tracer_gt_index'] in ('', 'None'):
                continue
            distance = float(source['gt_ego_distance_m'])
            band = min(int(distance//10), 4)
            source['gt_index'] = int(source['gt_index'])
            source['tracer_gt_index'] = int(source['tracer_gt_index'])
            source['gt_x'] = float(source['gt_x'])
            source['gt_y'] = float(source['gt_y'])
            strata[(band, source['group'])].append(source)
    rng = random.Random(args.seed)
    selected = []
    for key in sorted(strata):
        population = strata[key]
        selected.extend(rng.sample(population,
                                   min(args.per_stratum, len(population))))
    selected.sort(key=lambda row: (row['sample_token'], row['gt_index']))
    paired = pair_selected_cars(selected, 'R1-f', 'R1', args.split,
                                args.max_pair_distance)
    output = Path(args.output_dir or root / 'r1f_{}'.format(args.split) /
                  'traces/cohorts/balanced_{}'.format(args.per_stratum))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'paired_selection.json').open('w') as handle:
        json.dump(paired, handle, indent=2)
    counts = Counter((int(float(row['gt_ego_distance_m'])//10), row['group'])
                     for row in selected)
    with (output / 'cohort_summary.json').open('w') as handle:
        json.dump({'num_paired_cars': len(paired), 'seed': args.seed,
                   'per_stratum': args.per_stratum,
                   'strata_counts': {'{}-{}m_{}'.format(
                       10*band, 10*(band+1), group): count
                       for (band, group), count in sorted(counts.items())},
                   'max_gt_pair_distance_m': max(
                       (row['cross_rig_gt_distance_m'] for row in paired),
                       default=None)}, handle, indent=2)
    print('Saved balanced paired cohort:', output, 'cars:', len(paired))


if __name__ == '__main__':
    main()
