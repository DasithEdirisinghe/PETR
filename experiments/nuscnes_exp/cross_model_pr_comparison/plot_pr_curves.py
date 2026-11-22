#!/usr/bin/env python3
"""Plot official nuScenes PR curves for PETR and StreamPETR from saved metrics."""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent
DEV_WS = HERE.parents[3]
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
COLORS = {'PETR': '#d44b43', 'StreamPETR': '#2563a5'}


def svg_start(width, height):
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" '
            'width="{}" height="{}">'.format(width, height, width, height),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263444}'
            '.title{font-size:27px;font-weight:bold}'
            '.subtitle{font-size:14px;fill:#596879}'
            '.panel{font-size:18px;font-weight:bold}'
            '.tick{font-size:12px;fill:#526477}'
            '.legend{font-size:14px;font-weight:bold}'
            '.cell{font-size:15px;font-weight:bold}</style>']


def text(parts, x, y, value, css='tick', anchor=None, color=None):
    attrs = ' text-anchor="{}"'.format(anchor) if anchor else ''
    if color:
        attrs += ' style="fill:{}"'.format(color)
    parts.append('<text x="{:.1f}" y="{:.1f}" class="{}"{}>{}</text>'.format(
        x, y, css, attrs, escape(str(value))))


def line(parts, x1, y1, x2, y2, color, width=1, dash=None):
    extra = ' stroke-dasharray="{}"'.format(dash) if dash else ''
    parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                 'stroke="{}" stroke-width="{}"{}/>'.format(
                     x1, y1, x2, y2, color, width, extra))


def rect(parts, x, y, width, height, color, stroke=None):
    extra = ' stroke="{}"'.format(stroke) if stroke else ''
    parts.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                 'height="{:.1f}" fill="{}"{}/>'.format(
                     x, y, width, height, color, extra))


def polyline(parts, points, color, width=2.5):
    coords = ' '.join('{:.2f},{:.2f}'.format(x, y) for x, y in points)
    parts.append('<polyline fill="none" stroke="{}" stroke-width="{}" '
                 'stroke-linecap="round" stroke-linejoin="round" '
                 'points="{}"/>'.format(color, width, coords))


def calculated_ap(detail, min_recall, min_precision):
    first = round(100 * min_recall) + 1
    clipped = [max(0.0, value - min_precision)
               for value in detail['precision'][first:]]
    return sum(clipped) / len(clipped) / (1.0 - min_precision)


def read_model(label, summary_path, details_path):
    with summary_path.open() as handle:
        summary = json.load(handle)
    with details_path.open() as handle:
        details = json.load(handle)
    cfg = summary['cfg']
    if sorted(float(value) for value in cfg['dist_ths']) != list(THRESHOLDS):
        raise ValueError('{} has unexpected AP thresholds'.format(label))
    for name, aps in summary['label_aps'].items():
        for threshold in THRESHOLDS:
            key = '{}:{}'.format(name, threshold)
            if key not in details:
                raise ValueError('{} missing {}'.format(label, key))
            data = details[key]
            if (len(data['recall']) != 101 or len(data['precision']) != 101 or
                    any(not 0 <= value <= 1 for value in data['precision'])):
                raise ValueError('{} has malformed PR curve {}'.format(label, key))
            computed = calculated_ap(data, cfg['min_recall'], cfg['min_precision'])
            if abs(computed - aps[str(threshold)]) > 1e-8:
                raise ValueError('{} AP mismatch for {}: saved {} versus PR {}'.format(
                    label, key, aps[str(threshold)], computed))
    return {'summary': summary, 'details': details}


def axis(parts, x, y, width, height, min_recall, min_precision):
    def point(recall, precision):
        return x + recall * width, y + (1 - precision) * height
    rect(parts, x, y, min_recall * width, height, '#f2f5f8')
    rect(parts, x, y + (1 - min_precision) * height,
         width, min_precision * height, '#f2f5f8')
    for tick in (0, .25, .5, .75, 1):
        px, py = point(tick, tick)
        line(parts, px, y, px, y + height, '#e2e8ee')
        line(parts, x, py, x + width, py, '#e2e8ee')
        text(parts, px, y + height + 18, '{:.2g}'.format(tick), 'tick', 'middle')
        text(parts, x - 8, py + 4, '{:.2g}'.format(tick), 'tick', 'end')
    line(parts, x + min_recall * width, y,
         x + min_recall * width, y + height, '#a9b4c1', 1, '5 4')
    line(parts, x, y + (1 - min_precision) * height,
         x + width, y + (1 - min_precision) * height, '#a9b4c1', 1, '5 4')
    rect(parts, x, y, width, height, 'none', '#93a4b7')
    return point


def render_class(path, name, models, min_recall, min_precision):
    width, height = 1110, 1050
    parts = svg_start(width, height)
    text(parts, 36, 46, '{}: PETR vs StreamPETR'.format(name.replace('_', ' ').title()),
         'title')
    text(parts, 36, 72, 'Official nuScenes validation PR curves | same class and '
         'center-distance threshold in each panel', 'subtitle')
    line(parts, 39, 99, 76, 99, COLORS['PETR'], 3)
    text(parts, 85, 104, 'PETR', 'legend')
    line(parts, 177, 99, 214, 99, COLORS['StreamPETR'], 3)
    text(parts, 223, 104, 'StreamPETR', 'legend')
    text(parts, 36, 128, 'Gray bands: recall <= {:.0f}% or precision <= {:.0f}% '
         'excluded from AP; labels use the official saved AP.'.format(
             100 * min_recall, 100 * min_precision), 'subtitle')
    for index, threshold in enumerate(THRESHOLDS):
        col, row = index % 2, index // 2
        panel_x, panel_y = 36 + col * 535, 158 + row * 425
        rect(parts, panel_x, panel_y, 516, 402, '#ffffff', '#d6dee6')
        text(parts, panel_x + 16, panel_y + 29,
             '<{} m center distance'.format(threshold), 'panel')
        ap_p = 100 * models['PETR']['summary']['label_aps'][name][str(threshold)]
        ap_s = 100 * models['StreamPETR']['summary']['label_aps'][name][str(threshold)]
        text(parts, panel_x + 16, panel_y + 53,
             'AP {:.1f}%'.format(ap_p), 'legend', color=COLORS['PETR'])
        text(parts, panel_x + 145, panel_y + 53,
             'AP {:.1f}%'.format(ap_s), 'legend', color=COLORS['StreamPETR'])
        text(parts, panel_x + 344, panel_y + 53,
             'Delta {:+.1f} pp'.format(ap_s - ap_p), 'legend')
        x, y, w, h = panel_x + 57, panel_y + 73, 429, 276
        point = axis(parts, x, y, w, h, min_recall, min_precision)
        for label in ('PETR', 'StreamPETR'):
            data = models[label]['details']['{}:{}'.format(name, threshold)]
            polyline(parts, [point(r, p) for r, p in zip(
                data['recall'], data['precision'])], COLORS[label])
        text(parts, x + w / 2, y + h + 43, 'Recall', 'legend', 'middle')
        label_x, label_y = panel_x + 17, y + h / 2
        parts.append('<text x="{:.1f}" y="{:.1f}" class="tick" '
                     'text-anchor="middle" transform="rotate(-90 {:.1f} {:.1f})">'
                     'Precision</text>'.format(label_x, label_y,
                                               label_x, label_y))
    text(parts, 36, height - 19,
         'Curves are devkit-interpolated PR arrays. Beyond maximum achieved '
         'recall, precision is zero. AP also clips precision below the cutoff.',
         'subtitle')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def gain_fill(value):
    # Neutral at zero, progressively darker blue for positive gains.
    if value < 0:
        strength = min(1.0, -value / 25)
        start, end = (255, 244, 241), (190, 50, 46)
    else:
        strength = min(1.0, value / 35)
        start, end = (235, 245, 252), (33, 101, 156)
    rgb = [round(a + strength * (b - a)) for a, b in zip(start, end)]
    return '#{:02x}{:02x}{:02x}'.format(*rgb), strength > .55


def render_heatmap(path, classes, models):
    width, height = 1060, 810
    parts = svg_start(width, height)
    text(parts, 34, 48, 'Where StreamPETR gains AP over PETR', 'title')
    text(parts, 34, 73, 'nuScenes validation | each cell is StreamPETR AP minus '
         'PETR AP, percentage points', 'subtitle')
    x0, y0, cw, ch = 230, 148, 164, 51
    for i, threshold in enumerate(THRESHOLDS):
        text(parts, x0 + (i + .5) * cw, y0 - 22,
             '<{} m'.format(threshold), 'panel', 'middle')
    text(parts, x0 + 4.5 * cw, y0 - 22, 'Mean', 'panel', 'middle')
    all_gains = []
    for j, name in enumerate(classes):
        y = y0 + j * ch
        text(parts, x0 - 13, y + 31, name.replace('_', ' '), 'legend', 'end')
        gains = [100 * (models['StreamPETR']['summary']['label_aps'][name][str(t)] -
                        models['PETR']['summary']['label_aps'][name][str(t)])
                 for t in THRESHOLDS]
        all_gains.append(gains)
        for i, value in enumerate(gains + [sum(gains) / 4]):
            x = x0 + i * cw
            color, light = gain_fill(value)
            rect(parts, x + 2, y + 2, cw - 4, ch - 4, color)
            text(parts, x + cw / 2, y + 31, '{:+.1f}'.format(value),
                 'cell', 'middle', '#ffffff' if light else '#263444')
    total_y = y0 + len(classes) * ch + 16
    line(parts, x0 - 190, total_y - 11, x0 + 5 * cw - 2, total_y - 11,
         '#93a4b7', 1)
    text(parts, x0 - 13, total_y + 21, 'All-class mean', 'legend', 'end')
    mean_by_threshold = [sum(g[i] for g in all_gains) / len(all_gains)
                         for i in range(4)]
    for i, value in enumerate(mean_by_threshold +
                              [sum(mean_by_threshold) / 4]):
        x = x0 + i * cw
        rect(parts, x + 2, total_y - 1, cw - 4, ch - 4, '#dfe9f3')
        text(parts, x + cw / 2, total_y + 28, '{:+.1f}'.format(value),
             'cell', 'middle')
    text(parts, 34, height - 33,
         'Official mAP gives every class and matching threshold equal weight. '
         'The bottom-right cell is the mAP difference.', 'subtitle')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def png(svg_path):
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-o', str(svg_path.with_suffix('.png')),
                        str(svg_path)], check=True)
        return True
    try:
        import cairosvg
    except ImportError:
        return False
    cairosvg.svg2png(url=str(svg_path),
                     write_to=str(svg_path.with_suffix('.png')))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--petr-summary', type=Path, default=HERE.parent /
                        'localization_oracle/outputs/baseline/baseline/metrics_summary.json')
    parser.add_argument('--petr-details', type=Path, default=HERE.parent /
                        'localization_oracle/outputs/baseline/baseline/metrics_details.json')
    parser.add_argument('--stream-summary', type=Path, default=DEV_WS /
                        'StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/'
                        'Fri_Oct__9_14_02_06_2026/pts_bbox/metrics_summary.json')
    parser.add_argument('--stream-details', type=Path, default=DEV_WS /
                        'StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/'
                        'Fri_Oct__9_14_02_06_2026/pts_bbox/metrics_details.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs')
    args = parser.parse_args()
    for path in (args.petr_summary, args.petr_details,
                 args.stream_summary, args.stream_details):
        if not path.is_file():
            parser.error('Missing saved nuScenes metrics: {}'.format(path))
    models = {
        'PETR': read_model('PETR', args.petr_summary, args.petr_details),
        'StreamPETR': read_model('StreamPETR', args.stream_summary,
                                args.stream_details),
    }
    config_p = models['PETR']['summary']['cfg']
    config_s = models['StreamPETR']['summary']['cfg']
    if {key: value for key, value in config_p.items() if key != 'mean_ap_weight'} != {
            key: value for key, value in config_s.items() if key != 'mean_ap_weight'}:
        raise ValueError('Model evaluation configurations differ')
    classes = list(models['PETR']['summary']['label_aps'])
    if set(classes) != set(models['StreamPETR']['summary']['label_aps']):
        raise ValueError('Models do not contain the same classes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in classes:
        for threshold in THRESHOLDS:
            ap_p = 100 * models['PETR']['summary']['label_aps'][name][str(threshold)]
            ap_s = 100 * models['StreamPETR']['summary']['label_aps'][name][str(threshold)]
            rows.append({'class': name, 'threshold_m': threshold,
                         'petr_ap_percent': ap_p, 'stream_ap_percent': ap_s,
                         'gain_pp': ap_s - ap_p})
        figure = args.output_dir / ('pr_{}.svg'.format(name))
        render_class(figure, name, models, config_p['min_recall'],
                     config_p['min_precision'])
        png(figure)
    with (args.output_dir / 'ap_by_class_threshold.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    heatmap = args.output_dir / 'ap_gain_heatmap.svg'
    render_heatmap(heatmap, classes, models)
    png(heatmap)
    observed_gain = 100 * (models['StreamPETR']['summary']['mean_ap'] -
                           models['PETR']['summary']['mean_ap'])
    calculated_gain = sum(row['gain_pp'] for row in rows) / len(rows)
    if abs(observed_gain - calculated_gain) > 1e-8:
        raise AssertionError('Heatmap mean does not reproduce mAP gain')
    lines = ['# PETR versus StreamPETR precision–recall curves', '',
             'These are the **official nuScenes validation PR arrays** saved in '
             '`metrics_details.json`; no inference or new matching was run. '
             'The AP values were checked against both saved '
             '`metrics_summary.json` files.', '',
             '![AP gain by class and center-distance threshold](ap_gain_heatmap.svg)', '',
             'PETR mAP: {:.2f}%. StreamPETR mAP: {:.2f}%. Gain: {:+.2f} '
             'percentage points.'.format(
                 100 * models['PETR']['summary']['mean_ap'],
                 100 * models['StreamPETR']['summary']['mean_ap'],
                 observed_gain), '',
             'Each class figure overlays PETR and StreamPETR at 0.5, 1, 2, '
             'and 4 m. AP integrates the curves beyond 10% recall after '
             'subtracting the 10% precision floor.', '']
    for name in classes:
        lines.append('- [{}](pr_{}.svg)'.format(name.replace('_', ' '), name))
    lines += ['', 'See `ap_by_class_threshold.csv` for exact values. '
              'These curves cover the official class-wide evaluation; '
              'distance or visibility slices require separate custom '
              'evaluation definitions.', '']
    (args.output_dir / 'RESULTS.md').write_text('\n'.join(lines))
    print('Wrote {} class PR figures and AP heatmap to {}'.format(
        len(classes), args.output_dir), flush=True)


if __name__ == '__main__':
    main()
