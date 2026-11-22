#!/usr/bin/env python3
"""Create compact CSV and Markdown summaries for PETR-R50 train diagnostics."""

import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    metrics_path = args.root/'official_eval/metrics_summary.json'
    oracle_dir = args.root/'oracle_ap'
    if not metrics_path.is_file() or not (oracle_dir/'class_threshold_comparison.csv').is_file():
        parser.error('Official evaluation or oracle outputs are incomplete under {}'.format(args.root))

    metrics = json.loads(metrics_path.read_text())
    threshold_rows = read_csv(oracle_dir/'class_threshold_comparison.csv')
    class_rows = read_csv(oracle_dir/'class_mean_ap_comparison.csv')
    rule4 = {}
    rule4_root = args.root/'rule4_4m_margin_0p05'
    for path in sorted(rule4_root.glob('*/summary.json')):
        data = json.loads(path.read_text())
        rule4[data['class_name']] = data

    details = []
    for row in threshold_rows:
        details.append({
            'class': row['class'],
            'threshold_m': row['threshold_m'],
            'baseline_ap_percent': row['baseline_ap_percent'],
            'maximum_recall_percent': row['maximum_recall_percent'],
            'oracle_ap_percent': row['oracle_rank_ap_percent'],
            'oracle_gain_pp': row['oracle_ap_gain_percentage_points'],
        })
    write_csv(args.root/'train_ap_oracle_by_class_threshold.csv', details)

    combined = []
    for row in class_rows:
        name = row['class']
        rule = rule4.get(name)
        combined.append({
            'class': name,
            'baseline_mean_ap_percent': row['baseline_mean_ap_percent'],
            'oracle_mean_ap_percent': row['oracle_rank_mean_ap_percent'],
            'oracle_gain_pp': row['gain_percentage_points'],
            'matched_gt_at_4m': rule['matched_gt_count'] if rule else '',
            'rule4_alternatives': rule['rule4_alternative_count'] if rule else '',
            'rule4_percent_of_matched': rule['rule4_alternative_percent_of_matched']
            if rule else '',
        })
    write_csv(args.root/'train_class_summary.csv', combined)

    oracle_summary = json.loads((oracle_dir/'summary.json').read_text())
    lines = [
        '# PETR-R50 nuScenes training-split diagnostic', '',
        'These are in-sample diagnostic values, not validation performance.', '',
        '- Official train mAP: {:.3f}%'.format(100*metrics['mean_ap']),
        '- Official train NDS: {:.3f}%'.format(100*metrics['nd_score']),
        '- Fixed-box oracle mean AP: {:.3f}%'.format(
            oracle_summary['oracle_rank_mean_ap_percent']),
        '- Fixed-box oracle headroom: {:.3f} percentage points'.format(
            oracle_summary['gain_percentage_points']), '',
        '| Class | Baseline mean AP | Oracle mean AP | Gain | Rule-4 cases | Rule-4 % matched |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in combined:
        lines.append('| {cls} | {b:.3f}% | {o:.3f}% | {g:.3f} pp | {r} | {p} |'.format(
            cls=row['class'], b=float(row['baseline_mean_ap_percent']),
            o=float(row['oracle_mean_ap_percent']), g=float(row['oracle_gain_pp']),
            r=row['rule4_alternatives'] or '—',
            p=('{:.2f}%'.format(float(row['rule4_percent_of_matched']))
               if row['rule4_percent_of_matched'] != '' else '—')))
    (args.root/'README.md').write_text('\n'.join(lines)+'\n')
    print(args.root/'README.md')


if __name__ == '__main__':
    main()

