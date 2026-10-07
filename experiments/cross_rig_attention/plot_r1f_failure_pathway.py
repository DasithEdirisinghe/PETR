#!/usr/bin/env python3
"""Focused R1-f-only evidence figure from saved diagnostics and AP metrics."""

import argparse
import json
from pathlib import Path

from plot_regression_head_swap_ap import line, path, text

NAVY = '#13243a'
MUTED = '#64758a'
GRID = '#dce4ec'
R1 = '#c24e46'
R1F = '#167f77'


def rect(x, y, w, h, fill='white', radius=20):
    return '<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="{}" />'.format(
        x, y, w, h, radius, fill)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--four-run-summary', type=Path, default=Path(
        'experiments/cross_rig_attention/output/four_run_val/summary.json'))
    parser.add_argument('--swap-summary', type=Path, default=Path(
        'experiments/cross_rig_attention/output/regression_head_swap_val/summary.json'))
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    with args.four_run_summary.open() as handle:
        four = json.load(handle)
    with args.swap_summary.open() as handle:
        swap = json.load(handle)
    r1 = four['conditions']['R1-f_r1']
    r1f = four['conditions']['R1-f_r1f']
    target = swap['R1-f']
    ap = {(host, mode): 100*target[host]['official_metrics'][mode][
        'object/car_ap_dist_1.0'] for host in ('r1', 'r1f')
        for mode in ('native', 'swapped')}
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1080" viewBox="0 0 1600 1080">',
        rect(0, 0, 1600, 1080, '#f3f6fa', 0),
        text(50, 69, 'Where does R1 PETR lose localization on R1-f frames?',
             33, NAVY, 700),
        text(50, 108, 'Same R1-f images and calibration · red = R1 checkpoint · teal = R1-f checkpoint',
             19, MUTED),
        rect(50, 150, 720, 213), rect(800, 150, 750, 213),
        text(76, 190, '1   Car-region features are distinguishable', 24, NAVY, 700),
        text(76, 218, 'GT-box rays vs other same-camera rays · cosine AUROC · 8 cars', 16, MUTED),
        text(826, 190, '2   Successful model does not start closer', 24, NAVY, 700),
        text(826, 218, 'Selected-query reference-to-GT distance · 8 paired cars', 16, MUTED),
    ]
    for i, (label, value, color) in enumerate([
            ('R1', r1['feature_auc']['input_proj'], R1),
            ('R1-f', r1f['feature_auc']['input_proj'], R1F)]):
        y = 261+i*54
        elements += [text(76, y+7, label, 19, NAVY, 700),
                     rect(163, y-13, 485, 22, '#e9eef3', 9),
                     rect(163, y-13, 485*value, 22, color, 9),
                     text(738, y+7, '{:.3f}'.format(value), 20, color, 700, 'end')]
    for i, (label, value, color) in enumerate([
            ('R1', r1['reference_error_m'], R1),
            ('R1-f', r1f['reference_error_m'], R1F)]):
        y = 261+i*54
        elements += [text(826, y+7, label, 19, NAVY, 700),
                     rect(920, y-13, 480, 22, '#e9eef3', 9),
                     rect(920, y-13, 480*value/10.0, 22, color, 9),
                     text(1520, y+7, '{:.2f} m'.format(value), 20, color, 700, 'end')]
    elements += [rect(50, 390, 920, 505), rect(1000, 390, 550, 505),
                 text(77, 434, '3   Localization gap appears at the first decoded box',
                      24, NAVY, 700),
                 text(77, 463, 'Median BEV center error by decoder layer · same 8 cars',
                      17, MUTED),
                 text(1028, 434, '4   Head swap does not restore AP@1 m',
                      23, NAVY, 700),
                 text(1028, 463, 'Official car AP · 1,020 R1-f validation frames',
                      17, MUTED)]
    x0, x1, y0, y1 = 130, 916, 505, 787
    for value in range(6):
        y = y1-(y1-y0)*value/5
        elements += [line(x0, y, x1, y, GRID),
                     text(x0-15, y+5, '{} m'.format(value), 15, MUTED,
                          anchor='end')]
    xs = [x0+50+i*(x1-x0-100)/5 for i in range(6)]
    for i, x in enumerate(xs):
        elements.append(text(x, 818, 'L{}'.format(i+1), 17, MUTED,
                             anchor='middle'))
    for entry, color, label in ((r1, R1, 'R1'), (r1f, R1F, 'R1-f')):
        values = entry['decoder_center_error_m']
        points = [(x, y1-(y1-y0)*value/5) for x, value in zip(xs, values)]
        elements.append(path(points, color))
        for x, y in points:
            elements.append('<circle cx="{:.1f}" cy="{:.1f}" r="6" fill="{}" '
                            'stroke="white" stroke-width="2" />'.format(
                                x, y, color))
        elements.append(text(points[0][0]+7, points[0][1]-13,
                             '{} {:.2f} m'.format(label, values[0]), 17,
                             color, 700))
        elements.append(text(points[-1][0]-5, points[-1][1]-13,
                             '{:.2f} m'.format(values[-1]), 17,
                             color, 700, 'end'))
    elements.append(text(77, 866,
                         'L1 GT-ray attention enrichment: R1 1.13 · R1-f 1.12 (similar)',
                         16, MUTED))
    for i, (host, name, color) in enumerate([
            ('r1', 'R1 decoder state', R1),
            ('r1f', 'R1-f decoder state', R1F)]):
        y = 524+i*164
        elements.append(text(1028, y, name, 20, NAVY, 700))
        for j, (mode, label, bar_color) in enumerate([
                ('native', 'Native head', color),
                ('swapped', 'Other head', '#929da9')]):
            value = ap[(host, mode)]
            by = y+34+j*47
            elements += [text(1028, by+5, label, 17, MUTED),
                         rect(1170, by-12, 280, 22, '#e9eef3', 9),
                         rect(1170, by-12, max(2, 280*value/25), 22,
                              bar_color, 9),
                         text(1522, by+7, '{:.2f}%'.format(value), 18,
                              bar_color, 700, 'end')]
    elements += [
        rect(50, 920, 1500, 114, '#e6edf4', 17),
        text(77, 959,
             'Observed: this feature-separability test stays high; reference distance alone does not explain success,',
             20, NAVY, 700),
        text(77, 989,
             'and the R1 box is wrong from L1. A transplanted head does not fix strict AP.',
             20, NAVY, 700),
        text(77, 1018,
             'Not yet isolated: whether the decoder state, its regression mapping, or their interaction causes the gap.',
             17, MUTED),
        '</svg>',
    ]
    output = args.output or args.four_run_summary.parent / 'r1f_failure_pathway.svg'
    output.write_text('\n'.join(elements))
    print('Saved:', output)


if __name__ == '__main__':
    main()
