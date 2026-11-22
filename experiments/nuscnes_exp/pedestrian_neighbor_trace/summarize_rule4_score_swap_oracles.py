#!/usr/bin/env python3
"""Aggregate completed score-swap oracle evaluations across models."""

import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    threshold_rows, class_rows, overall_rows = [], [], []
    for model_dir in sorted(path for path in args.root.iterdir()
                            if path.is_dir()):
        required = [model_dir/name for name in
                    ('threshold_ap.csv', 'class_mean_ap.csv', 'overall.csv')]
        if not all(path.is_file() for path in required):
            continue
        threshold_rows.extend(read_csv(required[0]))
        class_rows.extend(read_csv(required[1]))
        overall_rows.extend(read_csv(required[2]))
    if not overall_rows:
        parser.error('No completed score-swap evaluations under {}'.format(args.root))
    write_csv(args.root/'threshold_ap_all_models.csv', threshold_rows)
    write_csv(args.root/'class_mean_ap_all_models.csv', class_rows)
    write_csv(args.root/'overall_all_models.csv', overall_rows)
    print(args.root/'overall_all_models.csv')
    for row in overall_rows:
        print('{model} | {variant}: swaps={swaps}, '
              '{analyzed_class_count}-class mAP={analyzed_classes_map} '
              '(delta {delta_analyzed_classes_map}), official mAP={official_mean_ap} '
              '(delta {delta_official_mean_ap})'.format(**row))


if __name__ == '__main__':
    main()
