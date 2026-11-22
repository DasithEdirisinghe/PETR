#!/usr/bin/env python3
"""Render class × GT-range × threshold AP heatmaps with GT/prediction counts."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


RANGES = ('0–10 m', '10–20 m', '20–30 m', '30–40 m', '40–50 m', '>50 m',
          'All ranges')
THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')


def color(value):
    if value is None:
        return '#ebeff3'
    t = max(0.0, min(1.0, value / 100))
    low, high = (239, 246, 252), (20, 93, 143)
    rgb = tuple(round(a + t * (b - a)) for a, b in zip(low, high))
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def read_rows(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['ap_percent'] = float(row['ap_percent']) if row['ap_percent'] else None
        for key in ('gt_count', 'prediction_count', 'matched_gt_count'):
            row[key] = int(row[key])
    return rows


def overall_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['gt_range'], row['threshold_m'])].append(row)
    overall = []
    for label in RANGES:
        for threshold in THRESHOLDS:
            subset = grouped[label, threshold]
            aps = [row['ap_percent'] for row in subset if row['ap_percent'] is not None]
            overall.append({
                'class': 'all classes', 'gt_range': label, 'threshold_m': threshold,
                'ap_percent': sum(aps) / len(aps) if aps else None,
                'gt_count': sum(row['gt_count'] for row in subset),
                'prediction_count': sum(row['prediction_count'] for row in subset),
                'matched_gt_count': sum(row['matched_gt_count'] for row in subset),
            })
    return overall


def render(path, classes, lookup):
    panel_w, panel_h = 745, 475
    margin_x, top = 30, 147
    width = 2 * panel_w + 2 * margin_x
    height = top + ((len(classes) + 1) // 2) * panel_h + 78
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" '
              'width="{}" height="{}">'.format(width, height, width, height),
              '<rect width="100%" height="100%" fill="#f8fafc"/>',
              '<style>text{font-family:Arial,sans-serif;fill:#253447}'
              '.title{font-size:25px;font-weight:bold}.note{font-size:13px}'
              '.head{font-size:16px;font-weight:bold}.row{font-size:12px}'
              '.ap{font-size:15px;font-weight:bold}.count{font-size:10px}'
              '.light{fill:#ffffff}</style>']

    def text(x, y, value, style='row', anchor=None, light=False):
        attrs = ' text-anchor="{}"'.format(anchor) if anchor else ''
        if light:
            style += ' light'
        pieces.append('<text x="{:.1f}" y="{:.1f}" class="{}"{}>{}</text>'.format(
            x, y, style, attrs, escape(str(value))))

    def rect(x, y, w, h, fill, stroke=None):
        outline = ' stroke="{}"'.format(stroke) if stroke else ''
        pieces.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                      'height="{:.1f}" fill="{}"{}/>'.format(
                          x, y, w, h, fill, outline))

    text(30, 40, 'PETR baseline · range-sliced AP', 'title')
    text(30, 65, 'Color and large number = AP. Each cell also shows matched GT / '
         'eligible GT and number of filtered predictions.', 'note')
    text(30, 87, 'GT use GT ego range; predictions use predicted ego range. '
         'Each slice is ranked and matched separately using nuScenes AP rules.', 'note')
    text(30, 109, '“All ranges” is the unsliced official class AP; “all classes” '
         'is the class-mean AP for each slice.', 'note')
    for i, (label, value) in enumerate((('0%', 0), ('25%', 25), ('50%', 50),
                                         ('75%', 75), ('100%', 100))):
        x = 1000 + i * 94
        rect(x, 94, 18, 16, color(value))
        text(x + 23, 107, label, 'note')

    for i, name in enumerate(classes):
        x0 = margin_x + (i % 2) * panel_w
        y0 = top + (i // 2) * panel_h
        rect(x0, y0, panel_w - 12, panel_h - 12, '#ffffff', '#dce4ec')
        text(x0 + 18, y0 + 28, name.replace('_', ' '), 'head')
        grid_x, grid_y, cell_w, cell_h = x0 + 160, y0 + 57, 139, 53
        for j, threshold in enumerate(THRESHOLDS):
            text(grid_x + (j + 0.5) * cell_w, grid_y - 12,
                 '<{} m'.format(threshold), 'row', 'middle')
        for row_index, label in enumerate(RANGES):
            y = grid_y + row_index * cell_h
            text(x0 + 18, y + 28, label, 'row')
            if label == 'All ranges':
                rect(x0 + 15, y - 3, 710, 1.5, '#8397a9')
            for j, threshold in enumerate(THRESHOLDS):
                row = lookup[name, label, threshold]
                ap = row['ap_percent']
                x = grid_x + j * cell_w
                rect(x + 2, y + 1, cell_w - 4, cell_h - 3, color(ap))
                light = ap is not None and ap >= 55
                if ap is None:
                    text(x + cell_w / 2, y + 23, 'No GT', 'ap', 'middle')
                    text(x + cell_w / 2, y + 42,
                         'P {:,}'.format(row['prediction_count']), 'count', 'middle')
                else:
                    text(x + cell_w / 2, y + 18, '{:.1f}%'.format(ap),
                         'ap', 'middle', light)
                    text(x + cell_w / 2, y + 33,
                         'TP {:,} / GT {:,}'.format(
                             row['matched_gt_count'], row['gt_count']),
                         'count', 'middle', light)
                    text(x + cell_w / 2, y + 46,
                         'Pred {:,}'.format(row['prediction_count']),
                         'count', 'middle', light)
    text(30, height - 35, 'Range-sliced AP is a diagnostic extension of the '
         'official metric, not an official nuScenes leaderboard score. '
         'TP counts are per-threshold greedy matches.', 'note')
    pieces.append('</svg>')
    path.write_text('\n'.join(pieces))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, default=here /
                        'outputs/baseline/range_sliced_ap.csv')
    parser.add_argument('--output-dir', type=Path, default=here / 'outputs/baseline')
    args = parser.parse_args()
    rows = read_rows(args.input_csv)
    classes = ['all classes'] + list(dict.fromkeys(r['class'] for r in rows))
    complete = rows + overall_rows(rows)
    lookup = {(r['class'], r['gt_range'], r['threshold_m']): r for r in complete}
    for name in classes:
        for label in RANGES:
            for threshold in THRESHOLDS:
                if (name, label, threshold) not in lookup:
                    raise ValueError('Missing cell: {} {} {}'.format(name, label, threshold))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = args.output_dir / 'range_sliced_ap_heatmap.svg'
    render(svg, classes, lookup)
    lines = ['# PETR baseline: AP by class, ego range, and threshold', '',
             '![Range-sliced AP heatmap](range_sliced_ap_heatmap.svg)', '',
             'Each cell shows AP, matched GT / eligible GT, and the number of '
             'filtered predictions. GT and predictions are separately assigned '
             'to ego-range bins using their own centers before matching. '
             'This range-sliced AP is a diagnostic extension; the `All ranges` '
             'row reproduces official unsliced AP. Empty GT bins have no AP. '
             'The `all classes` panel is the unweighted mean of classes with '
             'eligible GT in each slice. Predictions crossing range-bin '
             'boundaries can affect the sliced values.', '']
    for name in classes:
        lines += ['## ' + name.replace('_', ' '), '',
                  '| GT range | GT | Predictions | AP@0.5 m (TP) | AP@1 m (TP) | '
                  'AP@2 m (TP) | AP@4 m (TP) |',
                  '|---|---:|---:|---:|---:|---:|']
        for label in RANGES:
            first = lookup[name, label, THRESHOLDS[0]]
            cells = []
            for threshold in THRESHOLDS:
                row = lookup[name, label, threshold]
                cells.append('{:.2f}% ({:,})'.format(row['ap_percent'],
                    row['matched_gt_count']) if row['ap_percent'] is not None else '—')
            lines.append('| {} | {:,} | {:,} | {} |'.format(
                label, first['gt_count'], first['prediction_count'],
                ' | '.join(cells)))
        lines.append('')
    report = args.output_dir / 'range_sliced_ap.md'
    report.write_text('\n'.join(lines))
    print(svg)
    print(report)


if __name__ == '__main__':
    main()
