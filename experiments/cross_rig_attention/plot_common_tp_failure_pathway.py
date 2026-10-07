#!/usr/bin/env python3
"""Four-panel common-TP comparison with true GT-attention mass in panel 1."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from plot_regression_head_swap_ap import line, path, text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_val/common_tp_attention'
SWAP = ROOT / 'experiments/cross_rig_attention/output/regression_head_swap_val/summary_R1f.json'
NAVY = '#13243a'
MUTED = '#64758a'
GRID = '#dce4ec'
R1 = '#c24e46'
R1F = '#167f77'


def rect(x, y, width, height, fill='white', radius=18):
    return '<rect x="{}" y="{}" width="{:.1f}" height="{:.1f}" rx="{}" fill="{}"/>'.format(
        x, y, width, height, radius, fill)


def bar(elements, x, y, width, value, maximum, color):
    elements.append(rect(x, y, width, 22, '#e9eef3', 9))
    elements.append(rect(x, y, max(2, width*value/maximum), 22, color, 9))


def plot(summary_path, output_path, swap_path=SWAP):
    with Path(summary_path).open() as handle:
        summary = json.load(handle)
    if summary['paired_traced'] != summary['common_tp_total']:
        raise ValueError('Full figure requires every common TP; use summary for smoke tests')
    if min(summary['models'][name]['n_attention_valid'] for name in ('R1', 'R1-f')) < 1:
        raise ValueError('No valid attention measurements')
    with Path(swap_path).open() as handle:
        swap = json.load(handle)
    ap = {(model, mode): 100*swap['r1' if model == 'R1' else 'r1f'][
        'official_metrics'][mode]['object/car_ap_dist_4.0']
        for model in ('R1', 'R1-f') for mode in ('native', 'swapped')}
    n = summary['paired_traced']
    r1, r1f = summary['models']['R1'], summary['models']['R1-f']
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1080" viewBox="0 0 1600 1080">',
        rect(0, 0, 1600, 1080, '#f3f6fa', 0),
        text(50, 67, 'PETR on R1-f frames: cars both models detect', 32, NAVY, 700),
        text(50, 105, '{} common car TPs at official 4 m · same frames and calibration'.format(n),
             19, MUTED),
        rect(50, 145, 720, 233), rect(800, 145, 750, 233),
        text(76, 185, '1   Attention on the GT region', 24, NAVY, 700),
        text(76, 214, 'Last-layer TP query · all cameras · GT box + {:.1f} m'.format(
            summary['box_expansion_m']), 16, MUTED),
        text(826, 185, '2   Query reference distance', 24, NAVY, 700),
        text(826, 214, 'Median reference-to-GT BEV distance · actual TP queries', 16, MUTED),
    ]
    for index, (model, record, color) in enumerate((('R1', r1, R1), ('R1-f', r1f, R1F))):
        y = 245+index*57
        value = record['median_gt_attention_mass']
        parts.append(text(76, y+16, model, 19, NAVY, 700))
        bar(parts, 165, y, 455, value, 1, color)
        parts.append(text(745, y+18, '{:.1f}%'.format(100*value), 20,
                          color, 700, 'end'))
        ref = record['median_reference_error_m']
        parts.append(text(826, y+16, model, 19, NAVY, 700))
        bar(parts, 921, y, 450, min(ref, 15), 15, color)
        parts.append(text(1522, y+18, '{:.2f} m'.format(ref), 20,
                          color, 700, 'end'))
    delta = summary['paired_median_r1f_minus_r1_attention_mass']
    parts.append(text(76, 346, 'Paired median R1-f − R1: {:+.1f} pp (n={:,})'.format(
        100*delta, summary['paired_attention_comparisons']), 15, MUTED))
    parts.append(text(76, 368, 'Median GT-cell fraction: R1 {:.1f}% · R1-f {:.1f}%'.format(
        100*r1['median_gt_cell_fraction'], 100*r1f['median_gt_cell_fraction']),
        15, MUTED))
    parts.append(text(826, 361, 'References are not final predicted box centers.', 15, MUTED))
    parts.extend([
        rect(50, 400, 920, 480), rect(1000, 400, 550, 480),
        text(77, 443, '3   Decoder box accuracy on the same cars', 23, NAVY, 700),
        text(77, 471, 'Median TP-query BEV center error by decoder layer', 16, MUTED),
        text(1028, 443, '4   Full-validation AP@4 m', 23, NAVY, 700),
        text(1028, 471, 'All R1-f val frames · separate from the TP cohort', 16, MUTED),
    ])
    x0, x1, y0, y1 = 135, 923, 510, 788
    values = r1['median_layer_error_m'] + r1f['median_layer_error_m']
    ymax = max(1, int(max(values)*1.25+0.999))
    for tick in range(6):
        height = ymax*tick/5
        y = y1-(y1-y0)*tick/5
        parts.append(line(x0, y, x1, y, GRID))
        parts.append(text(x0-13, y+5, '{:.1f} m'.format(height), 15, MUTED,
                          anchor='end'))
    xs = [x0+48+i*(x1-x0-96)/5 for i in range(6)]
    for index, x in enumerate(xs):
        parts.append(text(x, 817, 'L{}'.format(index+1), 17, MUTED,
                          anchor='middle'))
    for model, record, color in (('R1', r1, R1), ('R1-f', r1f, R1F)):
        points = [(x, y1-(y1-y0)*value/ymax) for x, value in zip(
            xs, record['median_layer_error_m'])]
        parts.append(path(points, color))
        for x, y in points:
            parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="6" fill="{}" '
                         'stroke="white" stroke-width="2"/>'.format(x, y, color))
        parts.append(text(points[-1][0]-5, points[-1][1]-12,
                          '{} {:.2f} m'.format(model, record['median_layer_error_m'][-1]),
                          16, color, 700, 'end'))
    parts.append(text(77, 853, 'Final error is <4 m by cohort definition; compare within-TP precision only.',
                      15, MUTED))
    for index, (model, color) in enumerate((('R1', R1), ('R1-f', R1F))):
        y = 520+index*166
        parts.append(text(1028, y, '{} decoder state'.format(model), 20, NAVY, 700))
        for offset, (mode, label, bar_color) in enumerate((
                ('native', 'Native head', color), ('swapped', 'Other head', '#929da9'))):
            value = ap[model, mode]
            by = y+33+offset*47
            parts.append(text(1028, by+17, label, 17, MUTED))
            bar(parts, 1170, by, 280, value, 65, bar_color)
            parts.append(text(1522, by+18, '{:.2f}%'.format(value), 18,
                              bar_color, 700, 'end'))
    parts.extend([
        rect(50, 907, 1500, 126, '#e6edf4', 17),
        text(77, 946, 'Panel 1 measures GT-region attention mass, not feature separability or cross-model map overlap.',
             19, NAVY, 700),
        text(77, 976, 'Panels 1–3 condition on both models already detecting the car; they cannot explain missed cars or early FPs.',
             18, NAVY),
        text(77, 1005, 'Panel 4 uses full-validation AP, not AP recalculated on common TPs. Head swaps are diagnostic only.',
             16, MUTED),
        '</svg>',
    ])
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
    parser.add_argument('--summary', type=Path, default=DEFAULT_INPUT / 'summary.json')
    parser.add_argument('--swap-summary', type=Path, default=SWAP)
    parser.add_argument('--output', type=Path,
                        default=DEFAULT_INPUT / 'common_tp_failure_pathway.svg')
    args = parser.parse_args()
    plot(args.summary, args.output, args.swap_summary)


if __name__ == '__main__':
    main()
