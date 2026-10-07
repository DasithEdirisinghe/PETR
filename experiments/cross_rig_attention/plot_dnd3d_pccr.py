#!/usr/bin/env python3
"""Render the paired GT/error transitions and ranked-FP context for 3D DnD."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/cross_rig_attention/output/r1f_val/dnd3d_car_4m'
NAVY, MUTED, GRID = '#183047', '#53687b', '#e4ebf1'
COLORS = {'both_tp': '#168176', 'native_only': '#d59a27',
          'cross_only': '#6a83bd', 'neither': '#9aaabb',
          'Cls': '#805db8', 'Loc': '#d59a27', 'Both': '#5482be',
          'Miss': '#8799ab', 'Competition': '#445b70'}


def rect(x, y, width, height, fill='white', radius=12):
    return ('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
            'rx="{}" fill="{}"/>').format(x, y, width, height, radius, fill)


def label(x, y, value, size=17, color=NAVY, bold=False, anchor='start'):
    return ('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="{}" '
            'font-weight="{}" text-anchor="{}" '
            'font-family="Arial, Helvetica, sans-serif">{}</text>').format(
                x, y, color, size, 700 if bold else 400, anchor,
                escape(str(value)))


def horizontal(parts, x, y, label_text, count, maximum, color,
               width=330, value_x=770):
    parts.append(label(x, y+16, label_text, 17))
    parts.append(rect(x+230, y, width, 23, GRID, 6))
    parts.append(rect(x+230, y, max(2, width*count/max(maximum, 1)), 23,
                      color, 6))
    parts.append(label(value_x, y+18, '{:,}'.format(count), 18, NAVY, True,
                       'end'))


def plot(summary_path, output_path):
    with Path(summary_path).open() as handle:
        s = json.load(handle)
    car = s['gt_counts_by_class_and_transition']['car']
    total = sum(car.values())
    errors = s['native_only_cross_error_candidates'].get('car', {})
    neither = s['neither_error_pair_matrix'].get('car', {})
    native_fp = s['car_ranked_fp']['R1-f']
    cross_fp = s['car_ranked_fp']['R1']
    ap_native = 100*s['ap_validation']['R1-f']['car_ap']
    ap_cross = 100*s['ap_validation']['R1']['car_ap']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1050" '
             'viewBox="0 0 1600 1050">', rect(0, 0, 1600, 1050, '#f3f6fa', 0),
             label(45, 56, '3D Differences in Detection · car @ 4 m on R1-f frames',
                   30, NAVY, True),
             label(45, 87, 'Official AP replay verified for all 9 classes · R1-f native vs R1 cross-rig',
                   18, MUTED),
             rect(45, 118, 1510, 80),
             label(72, 166, 'Native R1-f AP: {:.2f}%'.format(ap_native),
                   24, '#168176', True),
             label(570, 166, 'Cross R1 AP: {:.2f}%'.format(ap_cross),
                   24, '#c9574c', True),
             label(1100, 166, 'Gap: {:.2f} pp'.format(ap_native-ap_cross),
                   24, NAVY, True),
             rect(45, 220, 735, 343), rect(805, 220, 750, 343),
             label(70, 257, 'GT-car outcomes ({:,} total)'.format(total),
                   23, NAVY, True),
             label(830, 257, 'R1 candidates for native-only cars',
                   23, NAVY, True),
             label(830, 285, 'One-to-one free-box assignment; diagnostic radius {:.0f} m'.format(
                 s['diagnostic_candidate_radius_m']), 16, MUTED)]
    names = [('both_tp', 'Both TP'), ('native_only', 'R1-f only'),
             ('cross_only', 'R1 only'), ('neither', 'Neither')]
    for index, (key, name) in enumerate(names):
        horizontal(parts, 70, 294+index*62, name, car.get(key, 0), total,
                   COLORS[key], width=325, value_x=730)
    error_total = car.get('native_only', 0)
    for index, key in enumerate(('Cls', 'Loc', 'Both', 'Miss', 'Competition')):
        horizontal(parts, 830, 310+index*46, key, errors.get(key, 0),
                   error_total, COLORS[key], width=335, value_x=1510)
    parts.extend([rect(45, 585, 735, 355), rect(805, 585, 750, 355),
                  label(70, 622, 'Shared misses: paired candidate matrix',
                        22, NAVY, True),
                  label(70, 647, 'Rows: R1-f · columns: R1 · {:,} GT cars'.format(
                      car.get('neither', 0)), 16, MUTED),
                  label(830, 622, 'Ranked car false positives', 22, NAVY, True),
                  label(830, 647, 'Each model stops at its own recall checkpoint',
                        16, MUTED)])
    kinds = ('Cls', 'Loc', 'Both', 'Miss', 'Competition')
    cell, x0, y0 = 66, 292, 690
    maximum = max([0]+list(neither.values()))
    for index, kind in enumerate(kinds):
        parts.append(label(x0+index*cell+cell/2, 677, kind, 14, MUTED,
                           anchor='middle'))
        parts.append(label(273, y0+index*43+28, kind, 15, MUTED,
                           anchor='end'))
    for row, rkind in enumerate(kinds):
        for col, ckind in enumerate(kinds):
            count = neither.get(rkind+'__'+ckind, 0)
            intensity = count/max(maximum, 1)
            fill = '#dfe9f4' if intensity < .2 else (
                '#a8c8dc' if intensity < .5 else '#4f8ab1')
            x, y = x0+col*cell, y0+row*43
            parts.append(rect(x, y, 58, 38, fill, 5))
            parts.append(label(x+29, y+26, count, 16,
                               'white' if intensity >= .5 else NAVY,
                               bool(count), 'middle'))
    for index, target in enumerate(('0.25', '0.5')):
        a, b = native_fp[target], cross_fp[target]
        y = 689+index*119
        parts.append(label(830, y, '{}% recall · {:,} TP each'.format(
            int(float(target)*100), a['tp']), 19, NAVY, True))
        parts.append(label(830, y+35, 'R1-f: {:,} FP · {:.1f}% precision'.format(
            a['fp'], 100*a['precision']), 17, '#168176'))
        parts.append(label(830, y+66, 'R1: {:,} FP · {:.1f}% precision'.format(
            b['fp'], 100*b['precision']), 17, '#c9574c'))
    parts.extend([rect(45, 962, 1510, 60, '#e6edf4', 10),
                  label(70, 988, 'Cls/Loc/Both are geometric candidate labels, not proven causes; candidate boxes may also be FPs.',
                        16, NAVY, True),
                  label(70, 1010, 'The GT partition and FP rank ledger are complementary, not additive decompositions of the AP gap.',
                        15, MUTED), '</svg>'])
    output_path = Path(output_path)
    output_path.write_text('\n'.join(parts))
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-o', str(output_path.with_suffix('.png')),
                        str(output_path)], check=True)
    print('Saved', output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT / 'summary.json')
    parser.add_argument('--output', type=Path, default=DEFAULT / 'dnd3d_car_summary.svg')
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == '__main__':
    main()
