#!/usr/bin/env python3
"""Summarize AP/oracle and Rule-4 results for the inferred train subset."""

import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    oracle = args.root/'oracle_ap'
    summary_path = oracle/'summary.json'
    class_path = oracle/'class_mean_ap_comparison.csv'
    if not summary_path.is_file() or not class_path.is_file():
        parser.error('Oracle outputs are incomplete under {}'.format(oracle))
    summary = json.loads(summary_path.read_text())
    class_rows = rows(class_path)
    rule4 = {}
    for path in sorted((args.root/'rule4_4m_margin_0p05').glob('*/summary.json')):
        item = json.loads(path.read_text())
        rule4[item['class_name']] = item
    lines = [
        '# PETR-R50 nuScenes inferred training-subset diagnostic', '',
        'This evaluates only the {:,} sample tokens present in the saved prediction JSON; '
        'it is not official full-train AP.'.format(summary['analyzed_sample_count']), '',
        '- Baseline subset mAP: {:.3f}%'.format(summary['baseline_mean_ap_percent']),
        '- Fixed-box oracle subset mAP: {:.3f}%'.format(
            summary['oracle_rank_mean_ap_percent']),
        '- Oracle headroom: {:.3f} percentage points'.format(
            summary['gain_percentage_points']), '',
        '| Class | Baseline AP | Oracle AP | Gain | Rule-4 alternatives | Rule-4 % matched |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in class_rows:
        rule = rule4.get(row['class'])
        lines.append('| {} | {:.3f}% | {:.3f}% | {:.3f} pp | {} | {} |'.format(
            row['class'], float(row['baseline_mean_ap_percent']),
            float(row['oracle_rank_mean_ap_percent']),
            float(row['gain_percentage_points']),
            '{:,}'.format(rule['rule4_alternative_count']) if rule else '—',
            '{:.2f}%'.format(rule['rule4_alternative_percent_of_matched'])
            if rule else '—'))
    (args.root/'README.md').write_text('\n'.join(lines)+'\n')
    print(args.root/'README.md')


if __name__ == '__main__':
    main()

