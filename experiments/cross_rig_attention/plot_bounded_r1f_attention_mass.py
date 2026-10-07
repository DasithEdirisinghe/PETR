#!/usr/bin/env python3
"""One-score SVG explainer: percent of L6 attention on each GT-car region."""

import argparse
import csv
import shutil
import statistics
import subprocess
from pathlib import Path

from plot_bounded_r1f_attention import box, label, NAVY, TEAL, MUTED

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_val'


def values(input_dir):
    path = input_dir / 'bounded_tide3d_attention/paired_attention_l6.csv'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3591:
        raise ValueError('Expected 3,591 paired GT-car attention rows')
    r1 = [100*float(row['r1_gt_mass']) for row in rows]
    r1f = [100*float(row['r1f_gt_mass']) for row in rows]
    return statistics.median(r1), statistics.median(r1f), len(rows)


def render(input_dir=DEFAULT_INPUT, output_dir=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir / 'bounded_tide3d_attention')
    output_dir.mkdir(parents=True, exist_ok=True)
    r1, r1f, n = values(input_dir)
    width, height = 1100, 750
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
             'viewBox="0 0 {} {}">'.format(width, height, width, height),
             box(0, 0, width, height, '#f8fafb')]
    parts += [label(55, 56, 'What percentage of attention lands on the GT-car region?', 27, NAVY, 'bold'),
              label(55, 84, 'Same R1-f frames · 3,591 GT cars · PETR decoder layer 6 · all camera views pooled', 14, MUTED)]

    parts.append(box(55, 111, 990, 240, '#eaf2f4', 13))
    parts.append(label(81, 146, 'ONE SCORE: GT-car attention share', 15, NAVY, 'bold'))
    parts.append(label(81, 199, 'S (%) = 100 × Σ(i ∈ M) A(i)', 29, NAVY, 'bold'))
    parts.append(label(81, 236, 'M = feature cells whose rays intersect this GT car, across all cameras.', 16, NAVY))
    parts.append(label(81, 264, 'A(i) = final-layer attention at cell i, averaged across heads; all valid A(i) sum to 1.', 15, NAVY))
    parts.append(label(81, 315, 'Example: S = 3% means 3% of the query’s total image attention falls on that car region.', 16, TEAL, 'bold'))

    parts.append(label(55, 398, 'Scale: 0% to 100%', 20, NAVY, 'bold'))
    parts.append(label(55, 430, '0% = no attention on the car region.  100% = all attention on that region.', 16, MUTED))
    parts.append(label(55, 459, 'The same GT car has the same region mask for both models, so their percentages are directly comparable.', 15, MUTED))

    parts.append(label(55, 518, 'Median attention share over the same GT cars', 20, NAVY, 'bold'))
    parts.append(label(55, 545, 'Bars zoomed to 0–5% so the difference is visible; the score itself is always bounded 0–100%.', 14, MUTED))
    x0, bar_width, cap = 206, 680, 5.0
    for tick in range(6):
        x = x0+bar_width*tick/cap
        parts.append('<line x1="{:.1f}" y1="572" x2="{:.1f}" y2="666" '
                     'stroke="#d7e2e7" stroke-width="1"/>'.format(x, x))
        parts.append(label(x, 566, '{}%'.format(tick), 11, MUTED, anchor='middle'))
    for y, name, value, color in ((581, 'R1', r1, NAVY), (629, 'R1-f', r1f, TEAL)):
        parts.append(label(55, y+23, name, 19, color, 'bold'))
        parts.append(box(x0, y, round(bar_width*min(value, cap)/cap, 1), 30, color, 5))
        parts.append(label(946, y+23, '{:.2f}%'.format(value), 21, color, 'bold'))
    parts.append(label(55, 703, 'R1-f assigns a larger median share, but this does not prove identical attention locations or explain the AP drop.', 13, MUTED))
    parts.append(label(55, 730, 'GT region is a projected 3D-box ray mask, not a pixel-perfect visible-car mask; GT associates queries after standard inference.', 11, MUTED))
    parts.append('</svg>')

    path = output_dir / 'attention_mass_explainer.svg'
    path.write_text('\n'.join(parts))
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-o', str(path.with_suffix('.png')),
                        str(path)], check=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    print(render(args.input_dir, args.output_dir))


if __name__ == '__main__':
    main()
