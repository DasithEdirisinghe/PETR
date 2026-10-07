#!/usr/bin/env python3
"""One-score SVG explainer: GT-car attention enrichment, R1 versus R1-f."""

import argparse
import csv
import html
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_val'
NAVY = '#243e61'
TEAL = '#087f78'
MUTED = '#526578'


def label(x, y, value, size=17, color=NAVY, weight='normal', anchor='start'):
    return ('<text x="{}" y="{}" fill="{}" font-family="Arial, sans-serif" '
            'font-size="{}" font-weight="{}" text-anchor="{}">{}</text>').format(
                x, y, color, size, weight, anchor, html.escape(str(value)))


def box(x, y, width, height, color, radius=0):
    return ('<rect x="{}" y="{}" width="{}" height="{}" rx="{}" '
            'fill="{}"/>').format(x, y, width, height, radius, color)


def scores(input_dir):
    path = input_dir / 'query_scorecard/per_car_query.csv'
    rows = list(csv.DictReader(path.open(newline='')))
    paired = {}
    for row in rows:
        paired.setdefault((row['sample_token'], row['gt_index']), {})[
            row['model']] = row
    if len(paired) != 3591 or any(set(pair) != {'r1', 'r1f'} for pair in paired.values()):
        raise ValueError('Expected exactly 3,591 GT cars scored by both models')
    return {model: statistics.median(float(pair[model]['attention_lift_l6'])
                                     for pair in paired.values())
            for model in ('r1', 'r1f')}, len(paired)


def render(input_dir=DEFAULT_INPUT, output_dir=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir / 'bounded_tide3d_attention')
    output_dir.mkdir(parents=True, exist_ok=True)
    medians, n = scores(input_dir)
    width, height = 1100, 760
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
             'viewBox="0 0 {} {}">'.format(width, height, width, height),
             box(0, 0, width, height, '#f8fafb')]
    parts += [label(55, 55, 'How strongly does each model attend to the GT-car region?', 27, NAVY, 'bold'),
              label(55, 83, 'Same R1-f frames · 3,591 GT cars · PETR decoder layer 6 · all camera views pooled', 14, MUTED)]

    # The equation and its scale come before the numbers.
    parts.append(box(55, 112, 990, 247, '#eaf2f4', 13))
    parts.append(label(81, 146, 'ONE SCORE: GT-car attention enrichment', 15, NAVY, 'bold'))
    parts.append(label(81, 193, 'S = [ Σ(i ∈ M) A(i) ]  /  [ |M| / |V| ]', 27, NAVY, 'bold'))
    parts.append(label(81, 226, 'M = feature cells whose rays intersect this GT car;  V = all valid cells from all cameras.', 15, NAVY))
    parts.append(label(81, 251, 'A(i) = last-layer attention at cell i, averaged across heads;  Σ(i ∈ V) A(i) = 1.', 15, NAVY))
    parts.append(label(81, 290, 'Numerator = attention fraction on the car.    Denominator = fraction of cells belonging to the car.', 15, MUTED))
    parts.append(label(81, 327, 'Example: car region is 1% of cells, receives 5% of attention  →  S = 5×.', 16, TEAL, 'bold'))

    parts.append(label(55, 403, 'How to read the scale', 19, NAVY, 'bold'))
    parts.append(label(55, 431, '0× means no attention on the GT-car region.', 15, MUTED))
    parts.append(label(55, 457, '1× means uniform attention: the region gets attention proportional to its size.', 15, MUTED))
    parts.append(label(55, 483, 'Above 1× means attention is concentrated on the region.', 15, MUTED))
    parts.append(label(55, 509, 'There is no fixed upper limit; it depends on how many feature cells the car occupies.', 13, MUTED))

    parts.append(label(55, 558, 'Median score across the same GT cars', 20, NAVY, 'bold'))
    x0, full, xmax = 205, 700, 8.0
    scale = lambda value: full*value/xmax
    baseline_x = x0+scale(1)
    parts.append('<line x1="{:.1f}" y1="575" x2="{:.1f}" y2="683" stroke="#97a7b2" '
                 'stroke-width="2" stroke-dasharray="5,4"/>'.format(baseline_x, baseline_x))
    parts.append(label(baseline_x+6, 586, 'uniform = 1×', 12, MUTED))
    for y, model, name, color in ((598, 'r1', 'R1', NAVY),
                                  (647, 'r1f', 'R1-f', TEAL)):
        parts.append(label(55, y+24, name, 19, color, 'bold'))
        parts.append(box(x0, y, round(scale(medians[model]), 1), 31, color, 5))
        parts.append(label(955, y+24, '{:.2f}×'.format(medians[model]), 21,
                           color, 'bold'))
    parts.append(label(55, 719, 'R1-f has higher median car-region focus. This score does not show identical cells or prove why AP drops.', 14, MUTED))
    parts.append(label(55, 745, 'GT region is a projected 3D-box ray mask, not a pixel-perfect visible-car mask; GT associates queries only after standard inference.', 11, MUTED))
    parts.append('</svg>')
    path = output_dir / 'attention_region_explainer.svg'
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
