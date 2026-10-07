#!/usr/bin/env python3
"""Plot saved official R1-f car AP without rerunning inference or evaluation."""

import argparse
import html
import json
from pathlib import Path


THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')
NAVY = '#13243a'
MUTED = '#63758a'
GRID = '#dce4ec'
NATIVE = '#167f77'
SWAPPED = '#c24e46'


def esc(value):
    return html.escape(str(value))


def text(x, y, value, size=20, color=NAVY, weight=400, anchor='start'):
    return ('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="{}" '
            'font-weight="{}" text-anchor="{}" '
            'font-family="Arial, Helvetica, sans-serif">{}</text>'.format(
                x, y, color, size, weight, anchor, esc(value)))


def line(x1, y1, x2, y2, color, width=1, dash=None):
    extra = ' stroke-dasharray="{}"'.format(dash) if dash else ''
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="{}"{} />'.format(
                x1, y1, x2, y2, color, width, extra))


def path(points, color, dash=None):
    route = ' '.join(('M' if index == 0 else 'L')+'{:.1f},{:.1f}'.format(x, y)
                     for index, (x, y) in enumerate(points))
    extra = ' stroke-dasharray="{}"'.format(dash) if dash else ''
    return '<path d="{}" fill="none" stroke="{}" stroke-width="3"{} />'.format(
        route, color, extra)


def values(section, condition):
    metrics = section['official_metrics'][condition]
    return [100.0*float(metrics['object/car_ap_dist_'+threshold])
            for threshold in THRESHOLDS]


def panel(x, title, section, ymax):
    native = values(section, 'native')
    swapped = values(section, 'swapped')
    mean_native = sum(native)/4
    mean_swapped = sum(swapped)/4
    elements = [
        '<rect x="{}" y="175" width="720" height="635" rx="22" fill="white" />'.format(x),
        text(x+30, 221, title, 27, NAVY, 700),
        text(x+30, 253, 'Car mean AP (4 thresholds)', 17, MUTED),
        text(x+690, 253, '{:.2f}%  →  {:.2f}%'.format(mean_native, mean_swapped),
             19, NAVY, 700, 'end'),
    ]
    left, right = x+75, x+675
    top, bottom = 292, 560
    for tick in range(5):
        value = ymax*tick/4.0
        y = bottom-(bottom-top)*tick/4.0
        elements.append(line(left, y, right, y, GRID, 1))
        elements.append(text(left-12, y+6, '{:.0f}%'.format(value),
                             15, MUTED, anchor='end'))
    xs = [left+40+i*(right-left-80)/3.0 for i in range(4)]
    for i, threshold in enumerate(THRESHOLDS):
        elements.append(text(xs[i], 594, threshold+' m', 17, MUTED, 500, 'middle'))
    coords = []
    for series, color, dash in ((native, NATIVE, None), (swapped, SWAPPED, '9 6')):
        points = [(xs[i], bottom-(bottom-top)*value/ymax)
                  for i, value in enumerate(series)]
        coords.append(points)
        elements.append(path(points, color, dash))
        for px, py in points:
            elements.append('<circle cx="{:.1f}" cy="{:.1f}" r="5.5" '
                            'fill="{}" stroke="white" stroke-width="2" />'.format(
                                px, py, color))
    elements.append(text(x+35, 639, 'Threshold', 16, MUTED, 700))
    elements.append(text(x+265, 639, 'Native', 16, NATIVE, 700, 'end'))
    elements.append(text(x+435, 639, 'Swapped', 16, SWAPPED, 700, 'end'))
    elements.append(text(x+685, 639, 'Δ swap − native', 16, MUTED, 700, 'end'))
    for i, threshold in enumerate(THRESHOLDS):
        y = 679+i*31
        delta = swapped[i]-native[i]
        elements.append(line(x+30, y-22, x+690, y-22, GRID, 1))
        elements.append(text(x+35, y, threshold+' m', 17, NAVY, 600))
        elements.append(text(x+265, y, '{:.2f}%'.format(native[i]), 17,
                             NAVY, 600, 'end'))
        elements.append(text(x+435, y, '{:.2f}%'.format(swapped[i]), 17,
                             NAVY, 600, 'end'))
        elements.append(text(x+685, y, '{:+.2f} pp'.format(delta), 17,
                             NATIVE if delta > 0 else SWAPPED if delta < 0 else MUTED,
                             700, 'end'))
    return elements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path(
        'experiments/cross_rig_attention/output/regression_head_swap_val/summary.json'))
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    with args.summary.open() as handle:
        summary = json.load(handle)
    target = summary['R1-f']
    for host in ('r1', 'r1f'):
        if 'official_metrics' not in target[host]:
            raise ValueError('Missing official R1-f AP metrics; run --evaluate first')
    frames = target['r1']['processed_frames']
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="900" viewBox="0 0 1560 900">',
        '<rect width="1560" height="900" fill="#f3f6fa" />',
        text(45, 68, 'Does swapping the regression head recover car localization?',
             31, NAVY, 700),
        text(45, 102, 'R1-f validation · {} frames · official car AP at center-distance thresholds'.format(
            frames), 18, MUTED),
        line(48, 137, 96, 137, NATIVE, 4),
        text(107, 143, 'Native head', 18, NAVY, 600),
        line(285, 137, 333, 137, SWAPPED, 4, '9 6'),
        text(344, 143, 'Other checkpoint’s head', 18, NAVY, 600),
    ]
    elements.extend(panel(45, 'R1 decoder states', target['r1'], 16))
    elements.extend(panel(795, 'R1-f decoder states', target['r1f'], 65))
    elements.append(text(45, 852,
                         'Panels use different AP scales; use the exact values and Δ in each table for comparison.',
                         17, MUTED))
    elements.append('</svg>')
    output = args.output or args.summary.parent / 'regression_head_swap_r1f_car_ap.svg'
    output.write_text('\n'.join(elements))
    print('Saved:', output)


if __name__ == '__main__':
    main()
