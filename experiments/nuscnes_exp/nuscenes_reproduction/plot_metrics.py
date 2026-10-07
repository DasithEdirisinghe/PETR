#!/usr/bin/env python3
"""Export tables and plots from the official nuScenes metrics_summary.json."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True, type=Path,
                        help='metrics_summary.json or a directory containing it.')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--dpi', type=int, default=160)
    return parser.parse_args()


def resolve_metrics(path):
    if path.is_file():
        return path
    matches = sorted(path.rglob('metrics_summary.json'))
    if len(matches) != 1:
        raise RuntimeError('Expected one metrics_summary.json below {}, found {}'.format(
            path, len(matches)))
    return matches[0]


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else float('nan')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis):
    axis.grid(axis='y', alpha=0.25, zorder=0)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def annotate_bars(axis, bars, value_format, padding_fraction=0.012,
                  fontsize=9):
    """Matplotlib-compatible replacement for Axes.bar_label (>= 3.4)."""
    lower, upper = axis.get_ylim()
    padding = (upper - lower) * padding_fraction
    for bar in bars:
        value = bar.get_height()
        if not math.isfinite(float(value)):
            continue
        axis.text(bar.get_x() + bar.get_width() / 2.0, value + padding,
                  value_format.format(value), ha='center', va='bottom',
                  fontsize=fontsize, fontweight='bold')


def plot_overall(metrics, output, dpi):
    labels = ['mAP', 'NDS']
    values = [100 * finite(metrics['mean_ap']), 100 * finite(metrics['nd_score'])]
    figure, axis = plt.subplots(figsize=(6.5, 5), dpi=dpi)
    bars = axis.bar(labels, values, color=['#3478c9', '#ef8a3a'], zorder=2)
    axis.set_ylim(0, 100)
    axis.set_ylabel('Score (%)')
    axis.set_title('PETR nuScenes validation reproduction')
    style_axis(axis)
    annotate_bars(axis, bars, '{:.2f}%')
    figure.tight_layout()
    figure.savefig(output / 'overall_map_nds.png', bbox_inches='tight')
    plt.close(figure)


def plot_class_ap(rows, output, dpi):
    names = [row['class'] for row in rows]
    values = [100 * row['mean_ap'] for row in rows]
    figure, axis = plt.subplots(figsize=(11, 6), dpi=dpi)
    bars = axis.bar(names, values, color='#3478c9', zorder=2)
    axis.set_ylabel('Mean AP (%)')
    axis.set_title('PETR per-class AP (mean over nuScenes distance thresholds)')
    axis.tick_params(axis='x', rotation=35)
    axis.set_ylim(0, max(10, max(values) * 1.18))
    style_axis(axis)
    annotate_bars(axis, bars, '{:.1f}', fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'per_class_ap.png', bbox_inches='tight')
    plt.close(figure)


def plot_class_errors(rows, error_names, output, dpi):
    classes = [row['class'] for row in rows]
    columns = 2
    plot_rows = int(math.ceil(len(error_names) / columns))
    figure, axes = plt.subplots(plot_rows, columns,
                                figsize=(14, 4.2 * plot_rows), dpi=dpi)
    axes = np.asarray(axes).reshape(-1)
    colors = ['#3478c9', '#ef8a3a', '#49a078', '#a45dbb', '#d94f4f']
    for position, error_name in enumerate(error_names):
        axis = axes[position]
        values = [row.get(error_name, float('nan')) for row in rows]
        axis.bar(classes, values, color=colors[position % len(colors)], zorder=2)
        axis.set_title(error_name)
        axis.tick_params(axis='x', rotation=40, labelsize=8)
        style_axis(axis)
    for axis in axes[len(error_names):]:
        axis.set_visible(False)
    figure.suptitle('PETR per-class nuScenes true-positive errors', fontweight='bold')
    figure.tight_layout()
    figure.savefig(output / 'per_class_tp_errors.png', bbox_inches='tight')
    plt.close(figure)


def main():
    args = parse_args()
    metrics_path = resolve_metrics(args.metrics)
    with metrics_path.open('r') as handle:
        metrics = json.load(handle)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_aps = metrics['label_aps']
    label_errors = metrics.get('label_tp_errors', {})
    thresholds = sorted({str(threshold) for values in label_aps.values()
                         for threshold in values}, key=float)
    error_names = sorted({name for values in label_errors.values()
                          for name in values})
    rows = []
    for class_name, aps in label_aps.items():
        row = {'class': class_name}
        ap_values = []
        for threshold in thresholds:
            value = finite(aps.get(threshold, float('nan')))
            row['AP_dist_' + threshold] = value
            ap_values.append(value)
        row['mean_ap'] = float(np.nanmean(ap_values))
        for error_name in error_names:
            row[error_name] = finite(
                label_errors.get(class_name, {}).get(error_name, float('nan')))
        rows.append(row)

    overall = [{
        'mAP': finite(metrics['mean_ap']),
        'NDS': finite(metrics['nd_score']),
        **{name: finite(value) for name, value in metrics.get('tp_errors', {}).items()},
    }]
    write_csv(args.output_dir / 'overall_metrics.csv', overall)
    write_csv(args.output_dir / 'per_class_metrics.csv', rows)
    with (args.output_dir / 'metrics_source.json').open('w') as handle:
        json.dump({'source': str(metrics_path.resolve()),
                   'distance_thresholds': thresholds,
                   'classes': [row['class'] for row in rows]}, handle, indent=2)
    plot_overall(metrics, args.output_dir, args.dpi)
    plot_class_ap(rows, args.output_dir, args.dpi)
    if error_names:
        plot_class_errors(rows, error_names, args.output_dir, args.dpi)
    print('mAP={:.4f} NDS={:.4f}'.format(
        metrics['mean_ap'], metrics['nd_score']))
    print('Tables and plots:', args.output_dir)


if __name__ == '__main__':
    main()
