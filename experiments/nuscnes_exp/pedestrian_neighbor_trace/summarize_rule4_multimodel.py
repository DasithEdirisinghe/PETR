#!/usr/bin/env python3
"""Create one comparison CSV from completed Rule-4 analyses."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob('*/*/summary.json')):
        summary = json.loads(path.read_text())
        metrics = summary['mean_metrics_rule4_cases']
        rows.append({
            'model_slug': path.parents[1].name,
            'model': summary['model_name'],
            'class': summary['class_name'],
            'matched_gt': summary['matched_gt_count'],
            'rule4_alternative': summary['rule4_alternative_count'],
            'rule4_percent': summary['rule4_alternative_percent_of_matched'],
            'no_rule4_alternative': summary['no_rule4_alternative_count'],
            'lower_confidence_count': summary['alternative_lower_confidence_count'],
            'lower_confidence_percent': summary['alternative_lower_confidence_percent'],
            'official_tp_mean_score': metrics['official_tp']['score'],
            'alternative_mean_score': metrics['alternative']['score'],
            'official_tp_mean_iou2d': metrics['official_tp']['projected_iou2d'],
            'alternative_mean_iou2d': metrics['alternative']['projected_iou2d'],
            'official_tp_mean_iou3d': metrics['official_tp']['iou3d'],
            'alternative_mean_iou3d': metrics['alternative']['iou3d'],
            'official_tp_mean_center_m': metrics['official_tp']['center_m'],
            'alternative_mean_center_m': metrics['alternative']['center_m'],
        })
    if not rows:
        parser.error('No Rule-4 summaries found under {}'.format(args.root))
    output = args.root/'comparison.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print('{model} | {class}: {rule4_alternative:,}/{matched_gt:,} '
              '({rule4_percent:.1f}%), lower confidence '
              '{lower_confidence_percent:.1f}%'.format(**row))


if __name__ == '__main__':
    main()
