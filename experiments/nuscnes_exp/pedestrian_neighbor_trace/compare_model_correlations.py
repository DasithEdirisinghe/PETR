#!/usr/bin/env python3
"""Compute StreamPETR confidence–3D-IoU correlations and compare with PETR.

The underlying correlation calculation is confidence_quality_correlation.py
for both models: official nuScenes-filtered predictions, best same-class 3D
IoU per prediction, and no TP assignment or distance threshold.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


CORRELATIONS = ('pearson_all', 'spearman_all', 'pearson_positive_iou',
                'spearman_positive_iou')
LABELS = ('Pearson all', 'Spearman all', 'Pearson IoU>0', 'Spearman IoU>0')


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row['class'] for row in rows}) != len(rows):
        raise ValueError('Expected unique nonempty class rows in {}'.format(path))
    missing = set(CORRELATIONS) - set(rows[0])
    if missing:
        raise ValueError('Missing {} in {}'.format(sorted(missing), path))
    return rows


def fmt(value):
    return '—' if value is None else '{:.3f}'.format(value)


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    default_stream = (root.parent / 'StreamPETR/test/'
                      'stream_petr_r50_noflash_428q_nui_60e/'
                      'Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--petr-csv', type=Path, default=base / 'outputs/'
                        'confidence_quality_correlation/class_correlation.csv')
    parser.add_argument('--stream-results-json', type=Path, default=default_stream)
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs/'
                        'confidence_quality_correlation/model_comparison')
    parser.add_argument('--recompute', action='store_true',
                        help='Recalculate StreamPETR even if the matching saved output exists')
    args = parser.parse_args()
    petr = read_rows(args.petr_csv)
    if not args.stream_results_json.is_file():
        parser.error('StreamPETR result JSON missing: {}'.format(
            args.stream_results_json))
    if not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('nuScenes metadata missing: {}'.format(args.data_root))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / 'streampetr'
    stream_csv = stream_dir / 'class_correlation.csv'
    source_path = stream_dir / 'source.json'
    source_matches = False
    if stream_csv.is_file() and source_path.is_file():
        source = json.loads(source_path.read_text())
        source_matches = (Path(source['results_json']).resolve() ==
                          args.stream_results_json.resolve() and
                          Path(source['data_root']).resolve() ==
                          args.data_root.resolve() and
                          source.get('classes') == [row['class'] for row in petr])
    if args.recompute or not source_matches:
        command = [sys.executable, str(base / 'confidence_quality_correlation.py'),
                   '--results-json', str(args.stream_results_json),
                   '--data-root', str(args.data_root),
                   '--output-dir', str(stream_dir),
                   '--classes'] + [row['class'] for row in petr]
        print('Computing StreamPETR correlations with the existing PETR pipeline...',
              flush=True)
        subprocess.run(command, check=True)
    stream = read_rows(stream_csv)
    by_class = {row['class']: row for row in stream}
    if set(by_class) != {row['class'] for row in petr}:
        raise ValueError('PETR and StreamPETR class sets differ')
    comparison = []
    for base_row in petr:
        other = by_class[base_row['class']]
        row = {'class': base_row['class'],
               'petr_prediction_count': int(base_row['prediction_count']),
               'streampetr_prediction_count': int(other['prediction_count']),
               'petr_positive_iou_fraction': float(base_row['positive_iou_fraction']),
               'streampetr_positive_iou_fraction': float(other['positive_iou_fraction'])}
        for field in CORRELATIONS:
            left = float(base_row[field]) if base_row[field] else None
            right = float(other[field]) if other[field] else None
            row['petr_' + field] = left
            row['streampetr_' + field] = right
            row['delta_' + field] = (right - left
                                    if left is not None and right is not None else None)
        comparison.append(row)
    means = {'class': 'Mean across classes'}
    for field in CORRELATIONS:
        for prefix in ('petr_', 'streampetr_', 'delta_'):
            values = [row[prefix + field] for row in comparison
                      if row[prefix + field] is not None]
            means[prefix + field] = statistics.mean(values) if values else None
    csv_path = args.output_dir / 'class_correlation_comparison.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    markdown = ['# PETR versus StreamPETR confidence–3D-IoU correlation', '',
                'Each prediction is paired with its best same-class GT 3D IoU '
                'in the same validation frame. Both models use the same '
                'nuScenes maximum-box and class-range filters. No TP matching '
                'or center-distance threshold is used. The positive subset '
                'contains only predictions with best 3D IoU > 0.', '',
                '| Class | PETR n | StreamPETR n | PETR positive % | StreamPETR positive % '
                '| PETR Pearson all | StreamPETR Pearson all | PETR Spearman all '
                '| StreamPETR Spearman all | PETR Pearson IoU>0 | StreamPETR Pearson IoU>0 '
                '| PETR Spearman IoU>0 | StreamPETR Spearman IoU>0 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        markdown.append('| {} | {:,} | {:,} | {:.1f} | {:.1f} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            row['class'].replace('_', ' ').title(), row['petr_prediction_count'],
            row['streampetr_prediction_count'],
            100 * row['petr_positive_iou_fraction'],
            100 * row['streampetr_positive_iou_fraction'],
            *(fmt(row[prefix + field]) for field in CORRELATIONS
              for prefix in ('petr_', 'streampetr_'))))
    markdown.append('| **Mean across classes** | — | — | — | — | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
        *(fmt(means[prefix + field]) for field in CORRELATIONS
          for prefix in ('petr_', 'streampetr_'))))
    markdown += ['', 'Macro means weight each class equally. Correlations '
                 'describe association, not causal improvement in AP.', '']
    report_path = args.output_dir / 'class_correlation_comparison.md'
    report_path.write_text('\n'.join(markdown))
    print('Wrote {} and {}'.format(csv_path, report_path), flush=True)
    for field, label in zip(CORRELATIONS, LABELS):
        print('{}: PETR {} → StreamPETR {} (Δ {})'.format(
            label, fmt(means['petr_' + field]),
            fmt(means['streampetr_' + field]),
            fmt(means['delta_' + field])), flush=True)


if __name__ == '__main__':
    main()
