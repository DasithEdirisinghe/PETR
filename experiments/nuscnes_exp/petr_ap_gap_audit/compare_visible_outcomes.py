#!/usr/bin/env python3
"""Compare retained and lost original 4 m TPs with 80–100% visible GTs."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['visibility_token'] == '4' and row['pred4_id']:
                yield row


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    n = len(rows)
    value = lambda field: [float(row[field]) for row in rows]
    available_other = [float(row['nearest_other_gt_to_pred_m']) for row in rows
                       if row['nearest_other_gt_to_pred_m']]
    radial = value('pred4_radial_error_m')
    return {
        'count': n,
        'median_gt_range_m': statistics.median(value('gt_range_m')),
        'median_prediction_score': statistics.median(value('pred4_score')),
        'mean_prediction_score': statistics.mean(value('pred4_score')),
        'median_class_score_rank_percent': statistics.median(
            value('pred4_rank_percent')),
        'other_gt_closer_percent': 100 * sum(
            row['other_gt_closer'] == '1' for row in rows) / n,
        'other_gt_within_4m_percent': 100 * sum(
            distance < 4 for distance in available_other) / n,
        'median_signed_radial_error_m': statistics.median(radial),
        'prediction_farther_than_gt_percent': 100 * sum(v > 0 for v in radial) / n,
        'mean_abs_radial_error_m': statistics.mean(abs(v) for v in radial),
        'mean_abs_tangential_error_m': statistics.mean(
            abs(v) for v in value('pred4_tangential_error_m')),
    }


def report(path, summary):
    lookup = {(row['class'], row['distance_bin'], row['outcome']): row
              for row in summary}
    classes = sorted({row['class'] for row in summary})
    lines = ['# PETR: 80–100% visible GTs, retained versus lost 4 m TPs', '',
             'Only original 4 m TP predictions assigned to visibility-token-4 '
             'GTs are included. Retained means the same prediction is still a '
             'TP after independent 0.5 m matching; it may match a different '
             'GT. Lost means that prediction is an FP at 0.5 m.', '',
             '## All GT distances', '',
             '| Class | Retained / lost | Median GT range (retained / lost) | '
             'Median score (retained / lost) | Another GT closer '
             '(retained / lost) |',
             '|---|---:|---:|---:|---:|']
    for name in classes:
        retained = lookup[name, 'all', 'retained']
        lost = lookup[name, 'all', 'lost']
        lines.append('| {} | {:,} / {:,} | {:.1f} / {:.1f} m | '
                     '{:.3f} / {:.3f} | {:.1f}% / {:.1f}% |'.format(
                         name, retained['count'], lost['count'],
                         retained['median_gt_range_m'], lost['median_gt_range_m'],
                         retained['median_prediction_score'],
                         lost['median_prediction_score'],
                         retained['other_gt_closer_percent'],
                         lost['other_gt_closer_percent']))
    lines += ['', '## How to interpret this comparison', '',
              '- Compare rows within the **same class and GT distance bin** '
              'in `high_visibility_outcomes_by_class_range.csv` before '
              'attributing a difference to score or scene context. GT distance '
              'can affect both score and localization.',
              '- Radial and tangential errors are provided for diagnosis, but '
              'the lost group necessarily has larger center error by definition. '
              'That difference alone does not explain the failure.',
              '- Another-GT-closer and nearby-GT flags are geometric context, '
              'not evidence that PETR confused object identities.',
              '- An 80–100% nuScenes visibility token describes the GT across '
              'six cameras; it does not mean each camera image shows the '
              'entire object.', '',
              'For visual inspection, compare retained and lost examples from '
              'the **same class, 10 m range bin, and similar score**. Show all '
              'six images with projected GT/prediction centers, plus a BEV crop '
              'containing nearby same-class GTs. This avoids mistaking a '
              'distance or score difference for a visual feature difference.', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).resolve().parent /
                        'outputs/gt_transitions.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    groups = defaultdict(list)
    for row in read_rows(args.input):
        outcome = ('retained' if row['pred4_status_at_0.5m'].startswith('matched_')
                   else 'lost')
        groups[row['class'], 'all', outcome].append(row)
        groups[row['class'], row['distance_bin'], outcome].append(row)
    if not groups:
        raise ValueError('No original 4 m TPs assigned to 80–100% visible GTs')
    summary = []
    for (name, distance_bin, outcome), rows in sorted(groups.items()):
        summary.append({'class': name, 'distance_bin': distance_bin,
                        'outcome': outcome, **summarize(rows)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'high_visibility_outcomes_by_class_range.csv',
              summary)
    report(args.output_dir / 'high_visibility_comparison.md', summary)
    print('Wrote high-visibility comparison in {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
