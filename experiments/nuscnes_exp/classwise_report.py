#!/usr/bin/env python3
"""Join baseline PETR diagnostics into classwise tables and one SVG dashboard."""

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
RANGES = ('0–10 m', '10–20 m', '20–30 m', '30–40 m', '40–50 m', '>50 m')
ERRORS = ('0.0-0.5', '0.5-1.0', '1.0-2.0', '2.0-4.0')
COLORS = ('#2374ab', '#40a59b', '#f2bd52', '#e88455', '#d5dce3')


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    return '{:,}'.format(int(value))


def pct(numerator, denominator):
    return '{:.1f}%'.format(100 * numerator / denominator) if denominator else '—'


def mean(value):
    return '{:.2f}'.format(float(value)) if value not in ('', None) else '—'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] + ['---:'] * (len(headers) - 1)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def dashboard(path, classes, cov, loc, err):
    """A single class-by-range view of GT coverage and localization."""
    left, cell_w, right_w, top, row_h = 150, 215, 230, 196, 112
    width = left + len(RANGES) * cell_w + right_w + 30
    height = top + len(classes) * row_h + 120
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" '
             'width="{}" height="{}">'.format(width, height, width, height),
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#233247}'
             '.title{font-size:25px;font-weight:bold}.sub{font-size:13px}'
             '.head{font-size:15px;font-weight:bold}.class{font-size:16px;font-weight:bold}'
             '.small{font-size:11px}.count{font-size:12px;font-weight:bold}</style>']

    def txt(x, y, value, style='small', fill=None):
        extra = ' fill="{}"'.format(fill) if fill else ''
        parts.append('<text x="{:.1f}" y="{:.1f}" class="{}"{}>{}</text>'.format(
            x, y, style, extra, escape(str(value))))

    def rect(x, y, w, h, color, radius=0):
        parts.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                     'rx="{}" fill="{}"/>'.format(x, y, max(w, 0), h, radius, color))

    txt(28, 39, 'PETR baseline · class and GT-range localization', 'title')
    txt(28, 64, 'Each range cell shows GT count, predictions matched within 4 m, '
        'GT coverage by center-distance band, and mean absolute radial/tangential error.', 'sub')
    legend = (('<0.5 m', COLORS[0]), ('0.5–1 m', COLORS[1]),
              ('1–2 m', COLORS[2]), ('2–4 m', COLORS[3]), ('No <4 m match', COLORS[4]))
    for i, (label, color) in enumerate(legend):
        x = 30 + i * 165
        rect(x, 86, 14, 14, color, 2)
        txt(x + 21, 98, label, 'sub')
    txt(30, 126, 'Stacked bars divide all eligible GT; percentages in the final column divide '
        'matched predictions. R/T = mean |radial| / |tangential| error in metres.', 'sub')
    txt(30, 147, 'The threshold bands represent GT coverage from independent matching at each '
        'threshold; they do not imply a fixed prediction-to-GT assignment.', 'sub')
    txt(30, top - 14, 'Class', 'head')
    for j, label in enumerate(RANGES):
        txt(left + j * cell_w + 9, top - 14, label, 'head')
    txt(left + len(RANGES) * cell_w + 9, top - 14, 'All ranges · matched', 'head')

    keys = ('error_lt_0.5m', 'error_0.5_to_1m', 'error_1_to_2m',
            'error_2_to_4m', 'no_match_lt_4m')
    for i, name in enumerate(classes):
        y = top + i * row_h
        if i % 2 == 0:
            rect(20, y, width - 40, row_h, '#eef3f8')
        txt(30, y + 27, name, 'class')
        total = cov[name, 'Total']
        txt(30, y + 47, '{} GT'.format(number(total['total_gt'])), 'small')
        for j, label in enumerate(RANGES):
            x = left + j * cell_w + 8
            row = cov[name, label]
            gt = int(row['total_gt'])
            matched = int(row['matched_lt_4m'])
            if not gt:
                txt(x, y + 43, 'No eligible GT', 'small')
                continue
            txt(x, y + 22, '{} GT · {} matched ({})'.format(
                number(gt), number(matched), pct(matched, gt)), 'count')
            cursor = x
            for k, key in enumerate(keys):
                segment = 190 * int(row[key]) / gt
                if segment:
                    rect(cursor, y + 34, segment, 17, COLORS[k])
                cursor += segment
            item = loc[name, label]
            if matched:
                txt(x, y + 72, 'R {:.2f} m    T {:.2f} m'.format(
                    float(item['mean_abs_radial_m']),
                    float(item['mean_abs_tangential_m'])), 'small')
            else:
                txt(x, y + 72, 'R/T —', 'small')
        x = left + len(RANGES) * cell_w + 8
        matched = int(total['matched_lt_4m'])
        txt(x, y + 22, '{} matched ({})'.format(
            number(matched), pct(matched, int(total['total_gt']))), 'count')
        cursor = x
        for k, band in enumerate(ERRORS):
            count = int(err[name, band]['count'])
            segment = 195 * count / matched if matched else 0
            if segment:
                rect(cursor, y + 34, segment, 17, COLORS[k])
            cursor += segment
        txt(x, y + 72, 'Error bands among <4 m matches', 'small')
    txt(30, height - 45, 'Center distance = Euclidean BEV XY distance between prediction and GT. '
        'R/T errors use the ego-to-GT radial axis.', 'sub')
    txt(30, height - 25, 'This figure shows counts and localization error, not AP; AP also depends '
        'on confidence ranking and false positives.', 'sub')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='experiments/nuscnes_exp directory')
    args = parser.parse_args()
    root = args.root
    coverage = read_csv(root / 'gt_range_detection_coverage/outputs/baseline/gt_coverage_by_class.csv')
    centers = read_csv(root / 'center_error_decomposition/outputs/baseline/center_error_by_class.csv')
    ranges = read_csv(root / 'range_conditioned_localization/outputs/baseline/gt_range_by_class.csv')
    cov = {(row['class'], row['gt_range']): row for row in coverage}
    loc = {(row['class'], row['gt_range']): row for row in ranges}
    err = {(row['class'], row['center_error_bin_m']): row for row in centers}
    classes = sorted({row['class'] for row in coverage})

    center_rows = []
    coverage_rows = []
    range_rows = []
    for name in classes:
        total = cov[name, 'Total']
        gt_count = int(total['total_gt'])
        matched_count = int(total['matched_lt_4m'])
        error_count = sum(int(err[name, band]['count']) for band in ERRORS)
        if error_count != matched_count:
            raise ValueError('{}: center-error matches {} != coverage matches {}'.format(
                name, error_count, matched_count))
        for band in ERRORS:
            row = err[name, band]
            count = int(row['count'])
            center_rows.append((name, band, number(gt_count), number(matched_count),
                                number(count), pct(count, matched_count),
                                mean(row['mean_center_error_m']),
                                mean(row['mean_abs_radial_m']),
                                mean(row['mean_abs_tangential_m']),
                                mean(row['mean_score']), mean(row['mean_gt_range_m'])))
        for label in RANGES + ('Total',):
            row = cov[name, label]
            gt = int(row['total_gt'])
            matched = int(row['matched_lt_4m'])
            coverage_rows.append((name, label, number(gt), number(matched),
                                  pct(matched, gt), number(int(row['no_match_lt_4m'])),
                                  number(int(row['error_lt_0.5m'])),
                                  number(int(row['error_0.5_to_1m'])),
                                  number(int(row['error_1_to_2m'])),
                                  number(int(row['error_2_to_4m']))))
            if label == 'Total':
                continue
            item = loc[name, label]
            if int(item['matched_predictions']) != matched:
                raise ValueError('{} {}: range-localization and coverage counts differ'.format(
                    name, label))
            values = [float(item['percent_error_' + key]) if matched else None for key in
                      ('lt_0.5m', '0.5_to_1m', '1_to_2m', '2_to_4m')]
            range_rows.append((name, label, number(gt), number(matched), pct(matched, gt),
                               *('{:.1f}%'.format(value) if value is not None else '—'
                                 for value in values),
                               mean(item['mean_abs_radial_m']),
                               mean(item['mean_abs_tangential_m'])))

    outputs = [
        (root / 'center_error_decomposition/outputs/baseline/center_error_by_class.md',
         '# Center error decomposition by class\n\n'
         'GT count is all evaluator-eligible GT in the class. Matched predictions '
         'are original same-class matches at <4 m. Error-bin percentages use '
         'those matched predictions as the denominator within each class.\n\n' +
         table(('Class', 'Error bin (m)', 'GT count', 'Matched predictions <4 m',
                'Predictions in bin', '% of matched', 'Mean center error (m)',
                'Mean |radial| (m)', 'Mean |tangential| (m)', 'Mean score',
                'Mean GT range (m)'), center_rows)),
        (root / 'gt_range_detection_coverage/outputs/baseline/gt_coverage_by_class.md',
         '# GT detection coverage by class and range\n\n'
         'GT count is all evaluator-eligible GT. Matched predictions <4 m '
         'equals matched GT <4 m: each match claims one GT. Counts in the four '
         'error columns are disjoint GT coverage bands derived from matching '
         'independently at each threshold; they do not identify a fixed '
         'prediction across thresholds.\n\n' +
         table(('Class', 'GT range', 'GT count', 'Matched predictions <4 m',
                'GT coverage <4 m', 'Unmatched GT', '<0.5 m', '0.5–1 m',
                '1–2 m', '2–4 m'), coverage_rows)),
        (root / 'range_conditioned_localization/outputs/baseline/gt_range_by_class.md',
         '# Range-conditioned localization by class\n\n'
         'GT count is all evaluator-eligible GT in each class and range. '
         'Matched predictions are the original <4 m matches. The four '
         'disjoint error percentages use matched predictions in the row as '
         'their denominator; GT coverage uses all GT in the row.\n\n' +
         table(('Class', 'GT range', 'GT count', 'Matched predictions <4 m',
                'GT coverage <4 m', '<0.5 m', '0.5–1 m', '1–2 m', '2–4 m',
                'Mean |radial| (m)', 'Mean |tangential| (m)'), range_rows)),
    ]
    for path, content in outputs:
        path.write_text(content + '\n')
        print(path)
    plot_path = root / 'classwise_localization_dashboard.svg'
    dashboard(plot_path, classes, cov, loc, err)
    print(plot_path)


if __name__ == '__main__':
    main()
