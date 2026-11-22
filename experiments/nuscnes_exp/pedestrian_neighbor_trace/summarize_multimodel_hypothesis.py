#!/usr/bin/env python3
"""Combine multi-model hypothesis and cross-assignment summaries."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for class_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            hypothesis_path = class_dir/'hypothesis'/'summary.json'
            cross_path = class_dir/'cross_assignment'/'summary.json'
            if not hypothesis_path.is_file():
                continue
            hypothesis = json.loads(hypothesis_path.read_text())
            cross = json.loads(cross_path.read_text()) if cross_path.is_file() else {}
            hc = hypothesis['counts']
            different = hypothesis['matched_gt_with_different_best_3d']
            assigned_other = hc.get('best_3d_already_tp_other_gt', 0)
            cc = cross.get('counts', {})
            rows.append({
                'model_slug': model_dir.name,
                'model': hypothesis.get('model_name', model_dir.name),
                'class': hypothesis.get('class_name', class_dir.name),
                'threshold_m': hypothesis['threshold_m'],
                'gt_count': hypothesis['gt_count'],
                'official_matched_gt': hypothesis['official_matched_gt'],
                'different_best_3d': different,
                'best_3d_alternative_unmatched': different-assigned_other,
                'best_3d_alternative_assigned_other_gt': assigned_other,
                'strictly_higher_alternative_3d_iou': hc.get('better_3d_iou', 0),
                'preferred_any_better': hc.get('preferred_any_better', ''),
                'preferred_unmatched_better': hc.get('preferred_unmatched_better', ''),
                'preferred_assigned_other_better': hc.get('preferred_assigned_other_better', ''),
                'cross_assigned_center_and_2d': cc.get('cross_assigned_center_and_2d', ''),
                'cross_assigned_center_2d_and_3d': cc.get('cross_assigned_center_2d_and_3d', ''),
                'full_assignment_cascade': cc.get('full_assignment_cascade', ''),
                'strict_full_assignment_cascade': cc.get('strict_full_assignment_cascade', ''),
            })
    if not rows:
        parser.error(f'No completed summaries under {args.root}')
    output = args.root/'comparison.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(output)
    for row in rows:
        print('{model} | {class}: different={different_best_3d:,}, '
              'unmatched={best_3d_alternative_unmatched:,}, '
              'cross={cross_assigned_center_and_2d}, '
              'strict_cascade={strict_full_assignment_cascade}'.format(**row))


if __name__ == '__main__':
    main()
