#!/usr/bin/env python3
"""Combine completed pedestrian GT-pointing coverage runs."""

import argparse
import csv
import statistics
from pathlib import Path


VISIBILITY_ORDER = ('0-40%', '40-60%', '60-80%', '80-100%', 'all')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model_dir in sorted(path for path in args.root.iterdir()
                            if path.is_dir()):
        path = model_dir/'pedestrian/per_gt.csv'
        if not path.is_file():
            continue
        with path.open(newline='') as handle:
            gt_rows = list(csv.DictReader(handle))
        model_name = model_dir.name
        groups = [(label, [row for row in gt_rows
                           if row['visibility_label'] == label])
                  for label in VISIBILITY_ORDER[:-1]]
        groups.append(('all', gt_rows))
        for label, group in groups:
            scores = [float(row['max_pointing_score']) for row in group
                      if row['max_pointing_score'] != '']
            rows.append({
                'model_slug': model_name,
                'visibility': label,
                'gt_count': len(group),
                'covered_gt': len(scores),
                'uncovered_gt': len(group)-len(scores),
                'coverage_percent': 100*len(scores)/len(group) if group else 0,
                'minimum_max_pointing_score': min(scores) if scores else '',
                'mean_max_pointing_score':
                    sum(scores)/len(scores) if scores else '',
                'median_max_pointing_score':
                    statistics.median(scores) if scores else '',
                'maximum_max_pointing_score': max(scores) if scores else '',
            })
    if not rows:
        parser.error('No completed per_gt.csv files under {}'.format(args.root))
    output = args.root/'comparison.csv'
    write_csv(output, rows)
    print(output)
    for row in rows:
        print('{model_slug} | {visibility}: {covered_gt}/{gt_count} '
              '({coverage_percent:.1f}%), mean max score '
              '{mean_max_pointing_score}'.format(**row))


if __name__ == '__main__':
    main()
