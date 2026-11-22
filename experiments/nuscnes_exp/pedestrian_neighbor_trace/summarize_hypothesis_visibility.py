#!/usr/bin/env python3
"""Summarize nuScenes visibility for PETR pedestrian hypothesis groups."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_hypothesis_pair_gallery import load_metadata

VISIBILITY = {1: '0-40%', 2: '40-60%', 3: '60-80%', 4: '80-100%'}


def truth(value):
    return str(value).strip() == '1'


def main():
    base = Path(__file__).resolve().parent
    workspace = base.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-gt-csv', type=Path,
                        default=base/'outputs/pedestrian_image_alignment_4m/all_gt.csv')
    parser.add_argument('--data-root', type=Path,
                        default=workspace/'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path,
                        default=base/'outputs/pedestrian_image_alignment_4m')
    args = parser.parse_args()
    with args.all_gt_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    gt_by_token, _ = load_metadata(args.data_root, rows)
    for row in rows:
        boxes = gt_by_token[row['sample_token']]
        gi = int(row['gt_index'])
        if gi >= len(boxes):
            raise ValueError('GT filtering order mismatch for '+row['gt_id'])
        row['visibility'] = boxes[gi]['visibility']
    groups = {
        'all_gt': rows,
        'official_matched_gt': [r for r in rows if truth(r['official_match'])],
        'official_unmatched_gt': [r for r in rows if not truth(r['official_match'])],
        'different_best_3d_candidate': [r for r in rows
                                        if truth(r['different_3d_candidate'])],
        'joint_better_lower_score': [r for r in rows
                                     if truth(r['joint_better_lower_score'])],
        'joint_better_unmatched_alternative': [r for r in rows
                                                if truth(r['joint_better_unmatched_alt'])],
    }
    output = []
    for name, selected in groups.items():
        counts = Counter(int(r['visibility']) for r in selected)
        for visibility in range(1, 5):
            output.append({
                'group': name, 'visibility_token': visibility,
                'visibility_percent': VISIBILITY[visibility],
                'count': counts[visibility], 'group_total': len(selected),
                'percent_within_group': 100*counts[visibility]/len(selected)
                if selected else 0,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'visibility_distribution.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    summary = {name: {'total': len(selected),
                      'visibility': {str(v): {
                          'range': VISIBILITY[v],
                          'count': sum(int(r['visibility']) == v for r in selected),
                          'percent': 100*sum(int(r['visibility']) == v for r in selected)/len(selected)
                          if selected else 0} for v in range(1, 5)}}
               for name, selected in groups.items()}
    (args.output_dir/'visibility_distribution.json').write_text(
        json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
