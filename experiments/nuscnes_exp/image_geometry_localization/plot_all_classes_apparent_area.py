#!/usr/bin/env python3
"""Plot pooled-visibility apparent-area quartiles for every class in one PNG.

Reads Experiment 2's area_quartiles.csv; no nuScenes loading or matching.
The overall panel pools class-specific quartiles using 4 m TP counts as weights.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


CLASSES = ('car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')
COLORS = ('#3178B8', '#E69F00', '#009E73', '#CC6677', '#7855A6')


def read_summary(path):
    records = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['visibility'] != 'all':
                continue
            records.append({
                'class': row['class'],
                'range': row['range_bin_m'],
                'quartile': row['area_quartile'],
                'tp': int(row['tp_at_4m']),
                'mean_error': (float(row['mean_center_error_m'])
                               if row['mean_center_error_m'] else None),
            })
    if not records:
        raise ValueError('No visibility=all rows found in {}'.format(path))
    return records


def overall_records(records):
    grouped = defaultdict(lambda: [0, 0.0])
    for row in records:
        if row['mean_error'] is None or row['tp'] == 0:
            continue
        key = row['range'], row['quartile']
        grouped[key][0] += row['tp']
        grouped[key][1] += row['tp'] * row['mean_error']
    return [{'class': 'overall', 'range': distance, 'quartile': quartile,
             'tp': count, 'mean_error': weighted_sum / count}
            for (distance, quartile), (count, weighted_sum) in grouped.items()
            if count]


def plot(summary_path, output_path, model, min_tp):
    records = read_summary(summary_path)
    records += overall_records(records)
    available = {row['class'] for row in records}
    missing = [name for name in CLASSES if name not in available]
    if missing:
        raise ValueError('Missing classes in summary: {}'.format(', '.join(missing)))
    distances = sorted({row['range'] for row in records},
                       key=lambda value: int(value.split('-')[0]))
    if len(distances) > len(COLORS):
        raise ValueError('Expected at most {} distance bins'.format(len(COLORS)))
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'bold',
    })
    figure, axes = plt.subplots(6, 2, figsize=(17, 23), sharex=False,
                               sharey=False)
    figure.subplots_adjust(left=0.075, right=0.97, top=0.885,
                           bottom=0.065, wspace=0.18, hspace=0.49)
    figure.suptitle('{} | Apparent size and center localization'.format(model),
                    fontsize=22, fontweight='bold', y=0.982)
    figure.text(0.5, 0.957,
                'Mean BEV center error among original 4 m true positives; '
                'lower is better',
                ha='center', va='center', fontsize=12, color='#475569')
    handles = [Line2D([0], [0], color=color, marker='o', linewidth=2,
                      label='{} m'.format(distance))
               for color, distance in zip(COLORS, distances)]
    figure.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 0.932), ncol=len(handles),
                  frameon=False, fontsize=11, title='GT distance from ego')
    for ax, name in zip(axes.flat, ('overall',) + CLASSES):
        subset = [row for row in records if row['class'] == name]
        lookup = {(row['range'], row['quartile']): row for row in subset}
        total_tp = sum(row['tp'] for row in subset)
        for color, distance in zip(COLORS, distances):
            ys = []
            for quartile in range(1, 5):
                row = lookup.get((distance, 'Q{}'.format(quartile)))
                if row and row['tp'] >= min_tp and row['mean_error'] is not None:
                    ys.append(row['mean_error'])
                else:
                    ys.append(math.nan)
            if any(math.isfinite(value) for value in ys):
                ax.plot((1, 2, 3, 4), ys, '-o', color=color,
                        linewidth=2, markersize=4.5)
        ax.set_title('{}   |   {:,} TPs at 4 m'.format(
            name.replace('_', ' ').title(), total_tp),
            fontsize=12, loc='left', pad=11)
        ax.xaxis.set_major_locator(FixedLocator((1, 2, 3, 4)))
        ax.xaxis.set_major_formatter(FixedFormatter(
            ('Q1\nsmallest', 'Q2', 'Q3', 'Q4\nlargest')))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.85, 4.15)
        ax.tick_params(axis='x', labelbottom=True, labelsize=9, pad=5)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylabel('Mean center error (m)', fontsize=10)
    guide = axes.flat[-1]
    guide.axis('off')
    guide.text(0.02, 0.92, 'How to read this figure', fontsize=14,
               fontweight='bold', va='top', transform=guide.transAxes)
    guide.text(0.02, 0.72,
               'Q1 → Q4: smaller → larger projected GT box area.\n'
               'Quartiles are formed separately for each class and range.\n'
               'Each line is one GT distance bin; lower error is better.\n'
               'Overall pools class-specific quartiles, weighted by TP count.\n'
               'Missing points have fewer than {} TPs.'.format(min_tp),
               fontsize=11, linespacing=1.7, va='top',
               transform=guide.transAxes)
    figure.text(0.5, 0.025,
                'Apparent size is the projected GT box area in pixels²; '
                'it is not a count of visible object pixels.',
                ha='center', va='center', fontsize=10, color='#475569')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + '.tmp.png')
    figure.savefig(temporary_path, dpi=160, facecolor='white')
    plt.close(figure)
    temporary_path.replace(output_path)
    print(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', required=True, type=Path,
                        help='Experiment 2 area_quartiles.csv')
    parser.add_argument('--output', required=True, type=Path,
                        help='Combined PNG path')
    parser.add_argument('--model', default='PETR')
    parser.add_argument('--min-tp', type=int, default=10,
                        help='Minimum 4 m TPs per plotted point (default: 10)')
    args = parser.parse_args()
    plot(args.summary_csv, args.output, args.model, args.min_tp)


if __name__ == '__main__':
    main()
