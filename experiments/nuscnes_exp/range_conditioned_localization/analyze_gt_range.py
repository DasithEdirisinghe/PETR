#!/usr/bin/env python3
"""Summarize 4 m matched PETR center errors by GT ego range."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


RANGE_LABELS = ('0–10 m', '10–20 m', '20–30 m', '30–40 m', '40–50 m', '>50 m')
ERROR_BANDS = (
    (0.0, 0.5, 'lt_0.5m'),
    (0.5, 1.0, '0.5_to_1m'),
    (1.0, 2.0, '1_to_2m'),
    (2.0, 4.0, '2_to_4m'),
)


def range_bin(metres):
    if metres < 0:
        raise ValueError('GT ego range cannot be negative')
    if metres < 10:
        return RANGE_LABELS[0]
    if metres < 20:
        return RANGE_LABELS[1]
    if metres < 30:
        return RANGE_LABELS[2]
    if metres < 40:
        return RANGE_LABELS[3]
    if metres <= 50:
        return RANGE_LABELS[4]
    return RANGE_LABELS[5]


def summarize(rows):
    by_range = defaultdict(list)
    for row in rows:
        by_range[range_bin(row['gt_range_m'])].append(row)
    result = []
    total = len(rows)
    for label in RANGE_LABELS:
        group = by_range[label]
        n = len(group)
        item = {
            'gt_range': label,
            'matched_predictions': n,
            'range_share_percent': 100 * n / total if total else None,
        }
        for lower, upper, key in ERROR_BANDS:
            item['percent_error_{}'.format(key)] = (
                100 * sum(lower <= row['center_error_m'] < upper
                          for row in group) / n if n else None)
        item['mean_abs_radial_m'] = (
            sum(row['radial_abs_m'] for row in group) / n if n else None)
        item['mean_abs_tangential_m'] = (
            sum(row['tangential_abs_m'] for row in group) / n if n else None)
        result.append(item)
    return result


def display(value, digits=1):
    return '—' if value is None else '{:.{}f}'.format(value, digits)


def markdown_table(rows):
    lines = [
        '| GT ego range | Matched predictions | Share of 4 m matches | '
        'Error <0.5 m | Error 0.5–1 m | '
        'Error 1–2 m | Error 2–4 m | Mean abs radial | Mean abs tangential |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        values = [row['gt_range'], '{:,}'.format(row['matched_predictions']),
                  display(row['range_share_percent']) + '%']
        if row['matched_predictions']:
            shares = [row['percent_error_{}'.format(key)]
                      for _, _, key in ERROR_BANDS]
            rounded = [round(share, 1) for share in shares[:3]]
            # Keep the displayed disjoint shares at exactly 100.0% despite
            # rounding; the JSON and CSV retain the unrounded values.
            rounded.append(round(100.0 - sum(rounded), 1))
            values.extend('{:.1f}%'.format(value) for value in rounded)
        else:
            values.extend('—' for _ in ERROR_BANDS)
        values.extend(display(row[key], 2) + ' m' if row[key] is not None else '—'
                      for key in ('mean_abs_radial_m', 'mean_abs_tangential_m'))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', type=Path, default=experiment_dir.parent /
                        'center_error_decomposition/outputs/baseline/center_error_matches.csv')
    parser.add_argument('--output-dir', type=Path, default=experiment_dir /
                        'outputs/baseline')
    args = parser.parse_args()

    matches = []
    with args.matches.open(newline='') as handle:
        for row in csv.DictReader(handle):
            matches.append({
                'class': row['class'],
                'center_error_m': float(row['center_error_m']),
                'gt_range_m': float(row['gt_range_m']),
                'radial_abs_m': float(row['radial_abs_m']),
                'tangential_abs_m': float(row['tangential_abs_m']),
            })
    if not matches:
        raise ValueError('No matches in {}'.format(args.matches))
    if any(row['center_error_m'] >= 4 for row in matches):
        raise ValueError('Input must contain only official <4 m matches')

    overall = summarize(matches)
    by_class = []
    for class_name in sorted({row['class'] for row in matches}):
        for item in summarize([row for row in matches if row['class'] == class_name]):
            by_class.append({'class': class_name, **item})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'gt_range_summary.json').open('w') as handle:
        json.dump({'source': str(args.matches), 'match_threshold_m': 4.0,
                   'percentage_denominator': '4 m matched predictions in each GT range bin',
                   'error_bands': 'disjoint, lower inclusive and upper exclusive',
                   'range_bin_rule': 'lower inclusive, upper exclusive; 40–50 includes 50 m',
                   'rows': overall}, handle, indent=2)
    for filename, rows in (('gt_range_summary.csv', overall),
                           ('gt_range_by_class.csv', by_class)):
        with (args.output_dir / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    table = markdown_table(overall)
    (args.output_dir / 'gt_range_summary.md').write_text(
        '# Baseline PETR localization by GT ego range\n\n' + table + '\n\n'
        'The range-share percentages sum to 100% across rows and use all '
        '4 m matched predictions as the denominator. '
        'The four error-band percentages are disjoint and sum to 100% within '
        'each populated range bin. The displayed final percentage absorbs '
        'rounding to one decimal; JSON and CSV retain unrounded values. '
        'Radial and tangential errors are in ego-centered BEV.\n')
    print(table)
    print('\nSaved to:', args.output_dir)


if __name__ == '__main__':
    main()
