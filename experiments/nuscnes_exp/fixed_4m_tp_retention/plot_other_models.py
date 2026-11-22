#!/usr/bin/env python3
"""Reuse PETR's fixed-4m heatmaps for StreamPETR and OPEN match records."""

import argparse
import csv
from pathlib import Path

from plot_tp_retention import BANDS, RANGES, THRESHOLDS, disjoint_rows, draw, summarize


SOURCES = {
    'StreamPETR': Path('../StreamPETR/experiments/center_error_by_range_outputs/matched_centers.csv'),
    'OPEN': Path('../OPEN/experiments/open_center_error_by_range_outputs/matched_centers.csv'),
}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(SOURCES), required=True)
    parser.add_argument('--matches', type=Path,
                        help='Override the saved matched_centers.csv path')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    petr_root = here.parents[2]
    source = args.matches or (petr_root / SOURCES[args.model]).resolve()
    output = args.output_dir or here / 'outputs' / args.model.lower()
    if not source.is_file():
        parser.error('Match records not found: {}'.format(source))

    matches = []
    with source.open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'match_threshold_m', 'class', 'gt_range_m', 'center_error_m'}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError('Missing columns: {}'.format(required - set(reader.fieldnames or ())))
        for row in reader:
            if float(row['match_threshold_m']) == 4.0:
                matches.append(row)
    if not matches:
        raise ValueError('No 4 m matches in {}'.format(source))
    rows, classes = summarize(matches)
    bands = disjoint_rows(rows)
    total = next(row['tp_at_4m'] for row in rows
                 if row['class'] == 'all classes' and row['gt_range'] == 'All ranges'
                 and row['threshold_m'] == 4.0)
    if total != len(matches):
        raise AssertionError('The 4 m TP total does not match the source rows')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'fixed_4m_tp_error_bands.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bands[0]))
        writer.writeheader()
        writer.writerows(bands)
    with (output / 'fixed_4m_tp_retention.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    band_plot = output / 'fixed_4m_tp_error_bands.svg'
    retention_plot = output / 'fixed_4m_tp_retention.svg'
    draw(band_plot, rows, classes, disjoint=True, model_name=args.model)
    draw(retention_plot, rows, classes, model_name=args.model)
    with (output / 'README.md').open('w') as handle:
        handle.write('# {} original 4 m TP localization\n\n'.format(args.model))
        handle.write('Source: `{}`. Only the original 4 m matches were used. '
                     'Rows group matches by GT ego range; columns are the '
                     'disjoint `[0,0.5)`, `[0.5,1)`, `[1,2)`, `[2,4)` m '
                     'center-error bands. Cell percentages divide the original '
                     '4 m TP count within each class/range row; they are not AP. '
                     'The cumulative retention figure is also provided.\n\n'.format(source))
        handle.write('Original 4 m TPs: {:,}.\n'.format(total))
    print('{} original 4 m TPs: {:,}'.format(args.model, total))
    print(band_plot)
    print(retention_plot)


if __name__ == '__main__':
    main()
