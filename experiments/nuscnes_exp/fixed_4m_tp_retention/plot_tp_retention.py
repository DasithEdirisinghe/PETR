#!/usr/bin/env python3
"""Plot where the original PETR 4 m true positives survive stricter thresholds."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


RANGES = ('0–10 m', '10–20 m', '20–30 m', '30–40 m', '40–50 m', '>50 m',
          'All ranges')
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
BANDS = ('0–0.5 m', '0.5–1 m', '1–2 m', '2–4 m')
FIELDS = ('class', 'gt_range', 'threshold_m', 'tp_count', 'tp_at_4m',
          'retained_percent')


def range_bin(distance):
    if distance < 0:
        raise ValueError('GT ego range cannot be negative')
    if distance < 10:
        return RANGES[0]
    if distance < 20:
        return RANGES[1]
    if distance < 30:
        return RANGES[2]
    if distance < 40:
        return RANGES[3]
    if distance <= 50:
        return RANGES[4]
    return RANGES[5]


def summarize(matches):
    errors = defaultdict(list)
    classes = list(dict.fromkeys(row['class'] for row in matches))
    for row in matches:
        label = range_bin(float(row['gt_range_m']))
        error = float(row['center_error_m'])
        if not 0 <= error < 4:
            raise ValueError('Input must contain only fixed original <4 m matches')
        errors[row['class'], label].append(error)
        errors[row['class'], 'All ranges'].append(error)
        errors['all classes', label].append(error)
        errors['all classes', 'All ranges'].append(error)
    rows = []
    for name in ['all classes'] + classes:
        for label in RANGES:
            values = errors[name, label]
            denominator = len(values)
            for threshold in THRESHOLDS:
                tp = sum(error < threshold for error in values)
                rows.append({'class': name, 'gt_range': label,
                             'threshold_m': threshold, 'tp_count': tp,
                             'tp_at_4m': denominator,
                             'retained_percent': 100 * tp / denominator
                             if denominator else None})
    return rows, ['all classes'] + classes


def summarize_range_csv(source_rows):
    """Recover exact band counts from the saved unrounded classwise shares."""
    classes = list(dict.fromkeys(row['class'] for row in source_rows))
    band_keys = ('lt_0.5m', '0.5_to_1m', '1_to_2m', '2_to_4m')
    bands = defaultdict(lambda: [0, 0, 0, 0])
    for source in source_rows:
        n = int(source['matched_predictions'])
        if not n:
            continue
        counts = [round(float(source['percent_error_' + key]) * n / 100)
                  for key in band_keys]
        if sum(counts) != n:
            raise ValueError('Band percentages cannot recover exact counts for {} {}'.format(
                source['class'], source['gt_range']))
        name, label = source['class'], source['gt_range']
        for scope in ((name, label), (name, 'All ranges'),
                      ('all classes', label), ('all classes', 'All ranges')):
            for i, count in enumerate(counts):
                bands[scope][i] += count
    rows = []
    for name in ['all classes'] + classes:
        for label in RANGES:
            values = bands[name, label]
            denominator = sum(values)
            for i, threshold in enumerate(THRESHOLDS):
                tp = sum(values[:i + 1])
                rows.append({'class': name, 'gt_range': label,
                             'threshold_m': threshold, 'tp_count': tp,
                             'tp_at_4m': denominator,
                             'retained_percent': 100 * tp / denominator
                             if denominator else None})
    return rows, ['all classes'] + classes


def fill(percent):
    if percent is None:
        return '#e8edf2'
    t = max(0.0, min(1.0, percent / 100))
    low, high = (234, 245, 251), (17, 96, 143)
    values = [round(a + t * (b - a)) for a, b in zip(low, high)]
    return '#{:02x}{:02x}{:02x}'.format(*values)


def disjoint_rows(rows):
    lookup = {(r['class'], r['gt_range'], r['threshold_m']): r for r in rows}
    output = []
    for name in dict.fromkeys(r['class'] for r in rows):
        for label in RANGES:
            previous = 0
            for threshold, band in zip(THRESHOLDS, BANDS):
                current = lookup[name, label, threshold]
                count = current['tp_count'] - previous
                total = current['tp_at_4m']
                output.append({'class': name, 'gt_range': label,
                               'error_band_m': band, 'tp_count': count,
                               'tp_at_4m': total,
                               'percent_of_4m_tps': 100 * count / total
                               if total else None})
                previous = current['tp_count']
            if previous != lookup[name, label, 4.0]['tp_at_4m']:
                raise AssertionError('Error bands do not partition 4 m TPs')
    return output


def draw(path, rows, classes, disjoint=False, model_name='PETR baseline'):
    lookup = {(r['class'], r['gt_range'], r['threshold_m']): r for r in rows}
    panel_w, panel_h = 735, 438
    left, top = 30, 140
    width = left * 2 + 2 * panel_w
    height = top + ((len(classes) + 1) // 2) * panel_h + 75
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" '
           'width="{}" height="{}">'.format(width, height, width, height),
           '<rect width="100%" height="100%" fill="#f8fafc"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#263748}'
           '.title{font-size:25px;font-weight:bold}.note{font-size:13px}'
           '.head{font-size:17px;font-weight:bold}.label{font-size:12px}'
           '.pct{font-size:16px;font-weight:bold}.count{font-size:11px}'
           '.light{fill:#ffffff}</style>']

    def text(x, y, value, style='label', anchor=None, light=False):
        align = ' text-anchor="{}"'.format(anchor) if anchor else ''
        css = style + (' light' if light else '')
        svg.append('<text x="{:.1f}" y="{:.1f}" class="{}"{}>{}</text>'.format(
            x, y, css, align, escape(str(value))))

    def rect(x, y, w, h, color, border=None):
        stroke = ' stroke="{}"'.format(border) if border else ''
        svg.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                   'height="{:.1f}" fill="{}"{}/>'.format(x, y, w, h, color, stroke))

    if disjoint:
        text(30, 38, '{} · disjoint center-error bands of 4 m TPs'.format(
            model_name), 'title')
        text(30, 64, 'Color = share of original 4 m TPs in that error band. '
             'Each cell prints the exact TP count / 4 m TP count.', 'note')
        text(30, 88, 'The four bands partition each GT-range row, so their '
             'percentages sum to 100%. These are fixed-pair counts, not AP.', 'note')
        all_tps = lookup['all classes', 'All ranges', 4.0]['tp_at_4m']
        text(30, 111, '“All ranges” pools a class; “all classes” pools all '
             '{:,} original 4 m TPs. No rematching occurs.'.format(all_tps), 'note')
    else:
        text(30, 38, '{} · retention of original 4 m true positives'.format(
            model_name), 'title')
        text(30, 64, 'Color = fraction of the same 4 m matched predictions within '
             'each stricter threshold. Every cell prints that percentage and TP count / 4 m TP count.', 'note')
        text(30, 88, 'Rows are GT ego-range bins. “All ranges” pools each class; '
             '“all classes” pools all 4 m matches. These are fixed-pair counts, not AP.', 'note')
        text(30, 111, 'Each 4 m match stays assigned to its original GT at every '
             'threshold; there is no rematching.', 'note')

    for index, name in enumerate(classes):
        x0 = left + (index % 2) * panel_w
        y0 = top + (index // 2) * panel_h
        rect(x0, y0, panel_w - 12, panel_h - 12, '#ffffff', '#dce5ec')
        text(x0 + 18, y0 + 29, name.replace('_', ' '), 'head')
        grid_x, grid_y, cw, ch = x0 + 156, y0 + 58, 138, 48
        for j, threshold in enumerate(THRESHOLDS):
            text(grid_x + (j + 0.5) * cw, grid_y - 12,
                 BANDS[j] if disjoint else '<{} m'.format(threshold),
                 'label', 'middle')
        for i, label in enumerate(RANGES):
            y = grid_y + i * ch
            if label == 'All ranges':
                rect(x0 + 15, y - 3, 706, 1.5, '#8ca0b1')
            text(x0 + 18, y + 28, label, 'label')
            previous = 0
            for j, threshold in enumerate(THRESHOLDS):
                row = lookup[name, label, threshold]
                x = grid_x + j * cw
                count = row['tp_count'] - previous if disjoint else row['tp_count']
                previous = row['tp_count']
                total = row['tp_at_4m']
                percent = 100 * count / total if total else None
                rect(x + 2, y + 1, cw - 4, ch - 3, fill(percent))
                if percent is None:
                    text(x + cw / 2, y + 29, 'No 4 m TPs', 'label', 'middle')
                else:
                    light = percent >= 58
                    text(x + cw / 2, y + 20, '{:.1f}%'.format(percent),
                         'pct', 'middle', light)
                    text(x + cw / 2, y + 36,
                         '{:,} / {:,} TPs'.format(count, total),
                         'count', 'middle', light)
    text(30, height - 35, 'Center error is Euclidean distance in BEV XY. '
         'Thresholds use strict < comparisons, matching the nuScenes evaluator.', 'note')
    svg.append('</svg>')
    path.write_text('\n'.join(svg))


def markdown(path, rows, classes):
    lookup = {(r['class'], r['gt_range'], r['threshold_m']): r for r in rows}
    lines = ['# PETR baseline: retention of fixed 4 m true positives', '',
             '![Fixed 4 m TP retention by class, GT range, and threshold]('
             'fixed_4m_tp_retention.svg)', '',
             'Every row starts with the **same predictions originally matched '
             'to GT within 4 m**. The percentages say what fraction of those '
             'fixed matches also have center error below 0.5, 1, or 2 m. '
             'No prediction–GT assignments are changed. Each cell gives '
             '`TP count (percentage of 4 m TPs)`. These values are **not AP** '
             'or recall over all GT.', '']
    for name in classes:
        lines += ['## ' + name.replace('_', ' '), '',
                  '| GT ego range | 4 m TPs | <0.5 m | <1 m | <2 m | <4 m |',
                  '|---|---:|---:|---:|---:|---:|']
        for label in RANGES:
            base = lookup[name, label, 4.0]
            cells = []
            for threshold in THRESHOLDS:
                row = lookup[name, label, threshold]
                cells.append('{:,} ({:.1f}%)'.format(
                    row['tp_count'], row['retained_percent'])
                    if row['retained_percent'] is not None else '—')
            lines.append('| {} | {:,} | {} |'.format(
                label, base['tp_at_4m'], ' | '.join(cells)))
        lines.append('')
    path.write_text('\n'.join(lines))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--range-summary', type=Path, default=here.parent /
                        'range_conditioned_localization/outputs/baseline/gt_range_by_class.csv')
    parser.add_argument('--output-dir', type=Path, default=here / 'outputs/baseline')
    args = parser.parse_args()
    with args.range_summary.open(newline='') as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError('No range summaries in {}'.format(args.range_summary))
    rows, classes = summarize_range_csv(source_rows)
    total = next(r for r in rows if r['class'] == 'all classes'
                 and r['gt_range'] == 'All ranges' and r['threshold_m'] == 4.0)
    if total['tp_count'] != 116212:
        raise AssertionError('Expected 116,212 original 4 m matches; got {}'.format(
            total['tp_count']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'fixed_4m_tp_retention.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plot = args.output_dir / 'fixed_4m_tp_retention.svg'
    draw(plot, rows, classes)
    bands = disjoint_rows(rows)
    with (args.output_dir / 'fixed_4m_tp_error_bands.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bands[0]))
        writer.writeheader()
        writer.writerows(bands)
    band_plot = args.output_dir / 'fixed_4m_tp_error_bands.svg'
    draw(band_plot, rows, classes, disjoint=True)
    report = args.output_dir / 'fixed_4m_tp_retention.md'
    markdown(report, rows, classes)
    with (args.output_dir / 'fixed_4m_tp_error_bands.md').open('w') as handle:
        handle.write('# PETR baseline: disjoint center-error bands\n\n'
                     '![Disjoint error-band heatmap](fixed_4m_tp_error_bands.svg)\n\n'
                     'Each class/range row partitions its original 4 m true '
                     'positives into `[0,0.5)`, `[0.5,1)`, `[1,2)`, and `[2,4)` m. '
                     'Color is the percentage of 4 m TPs in one band; cell text '
                     'gives the exact count. The four percentages sum to 100% '
                     'within every populated row. These are not AP values.\n')
    print('Original 4 m TPs: {:,}'.format(total['tp_count']))
    print(plot)
    print(band_plot)
    print(report)


if __name__ == '__main__':
    main()
