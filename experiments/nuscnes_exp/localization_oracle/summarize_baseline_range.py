#!/usr/bin/env python3
"""Plot baseline PETR GT coverage by class, GT range, and AP distance threshold."""

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


RANGES = ('0–10 m', '10–20 m', '20–30 m', '30–40 m', '40–50 m', '>50 m')
THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')
COLORS = ('#2474aa', '#20a29a', '#ecae39', '#d66b4b')


def percentage(row, threshold):
    value = row['recall_lt_{}m_percent'.format(threshold)]
    return float(value) if value else None


def draw_svg(path, classes, records):
    panel_w, panel_h, columns = 735, 310, 2
    margin_x, top = 35, 160
    width = margin_x * 2 + columns * panel_w
    height = top + ((len(classes) + 1) // columns) * panel_h + 85
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" '
             'width="{}" height="{}">'.format(width, height, width, height),
             '<rect width="100%" height="100%" fill="#fafbfd"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#243448}'
             '.title{font-size:25px;font-weight:bold}.note{font-size:13px}'
             '.head{font-size:17px;font-weight:bold}.axis{font-size:11px}'
             '.value{font-size:10px;font-weight:bold}</style>']

    def text(x, y, value, style='axis', anchor=None, color=None):
        attrs = (' text-anchor="{}"'.format(anchor) if anchor else '') + (
            ' fill="{}"'.format(color) if color else '')
        parts.append('<text x="{:.1f}" y="{:.1f}" class="{}"{}>{}</text>'.format(
            x, y, style, attrs, escape(str(value))))

    def line(x1, y1, x2, y2, color, width=1, dash=None):
        attrs = ' stroke-dasharray="{}"'.format(dash) if dash else ''
        parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                     'stroke="{}" stroke-width="{}"{}/>'.format(
                         x1, y1, x2, y2, color, width, attrs))

    text(35, 40, 'Baseline PETR · GT coverage by class and ego range', 'title')
    text(35, 66, 'Each point is the percentage of all evaluator-eligible GT matched '
         'within that center-distance threshold. The final “All” point summarizes '
         'the class across ranges.', 'note')
    text(35, 89, 'These are GT coverage rates, not AP: AP also depends on ranked '
         'confidence scores and false positives.', 'note')
    for i, threshold in enumerate(THRESHOLDS):
        x = 42 + i * 170
        line(x, 117, x + 24, 117, COLORS[i], 4)
        text(x + 32, 121, '<{} m'.format(threshold), 'note')

    for index, name in enumerate(classes):
        col, row_number = index % columns, index // columns
        x0 = margin_x + col * panel_w
        y0 = top + row_number * panel_h
        plot_x, plot_y = x0 + 53, y0 + 42
        plot_w, plot_h = 642, 185
        text(x0 + 6, y0 + 24, name.replace('_', ' '), 'head')
        for level in (0, 25, 50, 75, 100):
            yy = plot_y + plot_h * (1 - level / 100)
            line(plot_x, yy, plot_x + plot_w, yy, '#d9e2eb')
            text(plot_x - 7, yy + 4, str(level), 'axis', 'end')
        rows = [records[name, label] for label in RANGES]
        rows = [r for r in rows if int(r['total_gt']) > 0]
        rows.append(records[name, 'Total'])
        step = plot_w / len(rows)
        xs = [plot_x + step * (i + 0.5) for i in range(len(rows))]
        if len(rows) > 1:
            separator = (xs[-2] + xs[-1]) / 2
            line(separator, plot_y, separator, plot_y + plot_h, '#8899a9', 1, '4 4')
        for x, r in zip(xs, rows):
            text(x, plot_y + plot_h + 18, 'All' if r['gt_range'] == 'Total'
                 else r['gt_range'].replace(' m', ''), 'axis', 'middle')
            text(x, plot_y + plot_h + 34, 'n={:,}'.format(int(r['total_gt'])),
                 'axis', 'middle')
        for i, threshold in enumerate(THRESHOLDS):
            points = []
            for x, r in zip(xs, rows):
                value = percentage(r, threshold)
                y = plot_y + plot_h * (1 - value / 100)
                points.append((x, y, value))
            for a, b in zip(points, points[1:]):
                if b is points[-1]:
                    continue  # Keep the overall summary visually separate.
                line(a[0], a[1], b[0], b[1], COLORS[i], 2.5)
            for x, y, value in points:
                parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="3.8" fill="{}"/>'.format(
                    x, y, COLORS[i]))
                if x == xs[-1]:
                    text(x + (i - 1.5) * 26, y - 8, '{:.0f}%'.format(value),
                         'value', 'middle', COLORS[i])
        line(x0 + 4, y0 + panel_h - 4, x0 + panel_w - 22,
             y0 + panel_h - 4, '#e3e9ef')
    text(35, height - 38, 'Center distance is Euclidean distance in BEV XY. '
         'Matching is replayed independently at each threshold.', 'note')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage-csv', type=Path, default=here.parent /
                        'gt_range_detection_coverage/outputs/baseline/gt_coverage_by_class.csv')
    parser.add_argument('--overall-csv', type=Path, default=here.parent /
                        'gt_range_detection_coverage/outputs/baseline/gt_coverage_by_range.csv')
    parser.add_argument('--output-dir', type=Path, default=here / 'outputs/baseline')
    args = parser.parse_args()
    with args.coverage_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with args.overall_csv.open(newline='') as handle:
        overall_rows = list(csv.DictReader(handle))
    records = {(r['class'], r['gt_range']): r for r in rows}
    records.update({('all classes', r['gt_range']): r for r in overall_rows})
    classes = ['all classes'] + list(dict.fromkeys(r['class'] for r in rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = args.output_dir / 'baseline_gt_coverage_by_class.svg'
    draw_svg(svg, classes, records)
    lines = ['# Baseline PETR: classwise GT coverage by ego range', '',
             '![GT coverage by class, range, and center-distance threshold]('
             'baseline_gt_coverage_by_class.svg)', '',
             'The percentages below use **all evaluator-eligible GT in the '
             'class and range bin** as the denominator. Each threshold uses '
             'independent, same-class confidence-ordered matching. The `All` '
             'row pools all GT for that class across ranges. These are GT '
             'coverage rates, **not AP**; AP also uses confidence ranking and '
             'false positives.', '']
    ap_path = args.output_dir / 'ap_comparison.csv'
    if ap_path.is_file():
        with ap_path.open(newline='') as handle:
            baseline = next(r for r in csv.DictReader(handle)
                            if r['oracle'] == 'baseline')
        lines += ['The same-run baseline official AP is **{:.2f}% @0.5 m**, '
                  '**{:.2f}% @1 m**, **{:.2f}% @2 m**, and **{:.2f}% @4 m**; '
                  'overall mAP is **{:.2f}%**. These AP figures should not be '
                  'read as GT coverage percentages.'.format(
                      *(float(baseline[k]) for k in
                        ('AP@0.5m', 'AP@1.0m', 'AP@2.0m', 'AP@4.0m', 'mAP'))), '']
    for name in classes:
        lines += ['## ' + name.replace('_', ' '), '',
                  '| GT ego range | GT count | Matched GT <4 m | GT % <0.5 m | '
                  'GT % <1 m | GT % <2 m | GT % <4 m |',
                  '|---|---:|---:|---:|---:|---:|']
        for label in RANGES + ('Total',):
            row = records[name, label]
            gt = int(row['total_gt'])
            vals = ['{:.1f}%'.format(percentage(row, t)) if gt else '—'
                    for t in THRESHOLDS]
            lines.append('| {} | {:,} | {:,} | {} |'.format(
                'All ranges' if label == 'Total' else label, gt,
                int(row['matched_lt_4m']), ' | '.join(vals)))
        lines.append('')
    output = args.output_dir / 'classwise_summary.md'
    output.write_text('\n'.join(lines))
    print(output)
    print(svg)


if __name__ == '__main__':
    main()
