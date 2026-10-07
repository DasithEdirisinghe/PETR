#!/usr/bin/env python3
"""Focused paired plot of GT attention and reference-to-box recovery."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/cross_rig_attention/output/r1f_val/common_tp_attention'
NAVY, MUTED, GRID = '#183047', '#53687b', '#dce6ef'
COLORS = {'R1': '#cf584d', 'R1-f': '#148577'}


def rect(x, y, w, h, fill='white', radius=12):
    return ('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
            'rx="{}" fill="{}"/>').format(x, y, w, h, radius, fill)


def label(x, y, value, size=18, color=NAVY, bold=False, anchor='start'):
    return ('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="{}" '
            'font-weight="{}" text-anchor="{}" '
            'font-family="Arial, Helvetica, sans-serif">{}</text>').format(
                x, y, color, size, 700 if bold else 400, anchor,
                escape(str(value)))


def line(x1, y1, x2, y2, color=GRID, width=2):
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="{}"/>').format(x1, y1, x2, y2,
                                                           color, width)


def dot(x, y, color, radius=9):
    return ('<circle cx="{:.1f}" cy="{:.1f}" r="{}" fill="{}" '
            'stroke="white" stroke-width="2"/>').format(x, y, radius, color)


def required_number(value, name):
    if value is None or not math.isfinite(float(value)):
        raise ValueError('Missing or invalid {} in full common-TP summary'.format(name))
    return float(value)


def plot(summary_path, output_path):
    with Path(summary_path).open() as handle:
        s = json.load(handle)
    count = s['common_tp_total']
    if s['paired_traced'] != count or count != 2583:
        raise ValueError('Figure requires the complete 2,583-car shared-TP cohort')
    names = ('R1', 'R1-f')
    data = s['models']
    for name in names:
        if data[name]['n'] != count or not data[name]['n_attention_valid']:
            raise ValueError('Incomplete model cohort/attention: {}'.format(name))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1070" '
        'viewBox="0 0 1500 1070">',
        rect(0, 0, 1500, 1070, '#f3f6fa', 0),
        label(50, 62, 'Same 2,583 cars detected by both models', 31, NAVY, True),
        label(50, 94, 'R1-f frames · official car TP at BEV-center distance <4 m · paired by GT identity',
              18, MUTED),
        rect(45, 125, 1410, 260),
        label(70, 165, '1  Last-layer attention mass on projected GT-box region',
              23, NAVY, True),
        label(70, 195, 'Mean over heads; summed across all valid feature cells and cameras. Range: 0–100%.',
              16, MUTED),
    ]
    for i, name in enumerate(names):
        item, y = data[name], 225+i*60
        value = required_number(item['median_gt_attention_mass'], name+' attention')
        fraction = required_number(item['median_gt_cell_fraction'], name+' GT cells')
        parts += [label(75, y+18, name, 20, COLORS[name], True),
                  rect(195, y, 900, 23, GRID, 7),
                  rect(195, y, max(2, 900*value), 23, COLORS[name], 7),
                  label(1120, y+19, '{:.1f}%'.format(100*value), 21,
                        COLORS[name], True),
                  label(1245, y+19, 'GT cells {:.1f}%'.format(100*fraction),
                        16, MUTED)]
    delta = required_number(s['paired_median_r1f_minus_r1_attention_mass'],
                            'paired attention delta')
    n_attn = s['paired_attention_comparisons']
    parts += [label(75, 357, 'Median paired R1-f − R1 attention mass: {:+.1f} percentage points (n={:,})'.format(
        100*delta, n_attn), 18, NAVY, True),
              rect(45, 405, 1410, 300),
              label(70, 446, '2  Query reference → final recovered BEV center',
                    23, NAVY, True),
              label(70, 475, 'Median distance to the same GT center; lower is better.',
                    17, MUTED)]
    maximum = max(required_number(data[name][field], name+' '+field)
                  for name in names for field in
                  ('median_reference_error_m', 'median_final_error_m'))
    scale = max(4, math.ceil(maximum*1.1))
    x0, x1 = 310, 1250
    coord = lambda value: x0+(x1-x0)*value/scale
    parts += [line(x0, 510, x1, 510, NAVY, 2)]
    for tick in range(0, scale+1, max(1, math.ceil(scale/8))):
        x = coord(tick)
        parts += [line(x, 505, x, 515, NAVY, 1),
                  label(x, 531, '{} m'.format(tick), 14, MUTED,
                        anchor='middle')]
    threshold_x = coord(4)
    parts += [line(threshold_x, 505, threshold_x, 662, '#a7b8c9', 1),
              label(threshold_x+8, 552, '4 m TP threshold', 14, MUTED)]
    for i, name in enumerate(names):
        item, y = data[name], 580+i*65
        ref = item['median_reference_error_m']
        final = item['median_final_error_m']
        parts += [label(75, y+6, name, 20, COLORS[name], True),
                  line(coord(final), y, coord(ref), y, COLORS[name], 5),
                  dot(coord(ref), y, COLORS[name]),
                  dot(coord(final), y, COLORS[name]),
                  label(1280, y-4, 'ref {:.2f} m'.format(ref), 17, NAVY, True),
                  label(1280, y+19, 'final {:.2f} m'.format(final), 17,
                        COLORS[name], True)]
    parts += [label(315, 684, 'Each endpoint is a cohort median; the per-car recovery is measured separately below.',
                    15, MUTED),
              rect(45, 725, 1410, 245),
              label(70, 765, '3  Reference-to-final recovery margin', 23, NAVY, True),
              label(70, 793, 'Per car: reference distance − final box distance. Positive means the decoder moved closer to GT.',
                    16, MUTED)]
    magnitudes = [abs(value) for name in names for value in
                  data[name]['recovery_margin_iqr_m']+
                  [data[name]['median_recovery_margin_m']]]
    limit = max(1, math.ceil(max(magnitudes)*1.15))
    mid, half = 730, 430
    xmargin = lambda value: mid+half*value/limit
    parts += [line(mid, 817, mid, 930, '#9eafc0', 2),
              label(mid, 810, '0 m', 14, MUTED, anchor='middle')]
    for i, name in enumerate(names):
        item, y = data[name], 850+i*63
        low, high = item['recovery_margin_iqr_m']
        median = item['median_recovery_margin_m']
        parts += [label(75, y+7, name, 20, COLORS[name], True),
                  line(xmargin(low), y, xmargin(high), y, COLORS[name], 8),
                  dot(xmargin(median), y, COLORS[name], 10),
                  label(1190, y-4, 'median {:+.2f} m'.format(median), 17,
                        COLORS[name], True),
                  label(1190, y+19, '{:.0f}% improve'.format(
                      100*item['fraction_positive_recovery']), 15, MUTED)]
    margin_delta = required_number(
        s['paired_median_r1f_minus_r1_recovery_margin_m'], 'paired recovery delta')
    parts += [label(75, 952, 'Bars show IQR; dots show medians. Median paired R1-f − R1 margin: {:+.2f} m.'.format(
        margin_delta), 16, NAVY),
              rect(45, 992, 1410, 55, '#e6edf4', 10),
              label(70, 1020, 'Shared TPs are a success-only cohort: this comparison cannot explain the 857 native-only cars or ranked FPs.',
                    17, NAVY, True),
              label(70, 1040, 'Attention mass describes where this query weights features, not proof of object recognition or causal use.',
                    14, MUTED), '</svg>']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(parts))
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-o', str(output_path.with_suffix('.png')),
                        str(output_path)], check=True)
    print('Saved', output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT / 'summary.json')
    parser.add_argument('--output', type=Path,
                        default=DEFAULT / 'common_tp_geometry_attention.svg')
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == '__main__':
    main()
