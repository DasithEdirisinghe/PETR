#!/usr/bin/env python3
"""Compare PETR oracle adapters from saved validation and test metrics."""

import argparse
import ast
import csv
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODES = ('query', 'key', 'query_key', 'reference', 'output')
LABELS = {
    'frozen_r1': 'Frozen R1',
    'target_r1f': 'R1-f target model',
    'query': 'Query',
    'key': 'Key',
    'query_key': 'Query + Key',
    'reference': 'Reference',
    'output': 'Output',
}
COLORS = {
    'frozen_r1': '#6B7280',
    'target_r1f': '#111827',
    'query': '#2563EB',
    'key': '#EA580C',
    'query_key': '#7C3AED',
    'reference': '#059669',
    'output': '#DC2626',
}
TEXT = '#253047'
MUTED = '#657087'
GRID = '#D9DEE8'
PANEL = '#F8FAFC'
CLASSES = (
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'adult', 'child',
    'traffic_light', 'traffic_sign')
THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline-json')
    parser.add_argument(
        '--target-json',
        help='Standardized results for vanilla PETR trained on R1-f.')
    return parser.parse_args()


def metric_dict_from_test_log(path):
    if not os.path.isfile(path):
        return None
    candidate = None
    with open(path, 'r', errors='replace') as handle:
        for line in handle:
            start = line.find("{'object/")
            if start < 0 or "'object/map'" not in line:
                continue
            try:
                parsed = ast.literal_eval(line[start:].strip())
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict) and 'object/map' in parsed:
                candidate = parsed
    return candidate


def validation_history(mode_dir):
    candidates = sorted(glob(os.path.join(mode_dir, '*.log.json')))
    best_history = []
    for path in candidates:
        history = []
        with open(path, 'r', errors='replace') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if record.get('mode') == 'val' and 'object/map' in record:
                    history.append(record)
        if len(history) > len(best_history):
            best_history = history
    return best_history


def baseline_metrics(path, rig):
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r') as handle:
        payload = json.load(handle)
    try:
        return payload['results'][rig]['metrics']['raw']
    except KeyError:
        return None


def mean_threshold_ap(metrics, threshold):
    values = [
        metrics.get('object/{}_ap_dist_{}'.format(name, threshold))
        for name in CLASSES]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else float('nan')


def summary_row(mode, target, source):
    row = {
        'mode': mode,
        'label': LABELS[mode],
        'target_mAP': target.get('object/map', float('nan')),
        'target_mATE': target.get('object/mATE', float('nan')),
        'source_mAP': (source or {}).get('object/map', float('nan')),
        'source_mATE': (source or {}).get('object/mATE', float('nan')),
    }
    for threshold in THRESHOLDS:
        row['target_AP@{}m'.format(threshold)] = mean_threshold_ap(
            target, threshold)
        row['car_AP@{}m'.format(threshold)] = target.get(
            'object/car_ap_dist_{}'.format(threshold), float('nan'))
    return row


def save_summary(rows, output_dir):
    csv_path = os.path.join(output_dir, 'oracle_comparison.csv')
    fields = ['mode', 'label', 'target_mAP', 'target_mATE']
    fields += ['target_AP@{}m'.format(t) for t in THRESHOLDS]
    fields += ['car_AP@{}m'.format(t) for t in THRESHOLDS]
    fields += ['source_mAP', 'source_mATE']
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(output_dir, 'oracle_comparison.json'), 'w') as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)

    markdown_path = os.path.join(output_dir, 'oracle_comparison.md')
    with open(markdown_path, 'w') as handle:
        handle.write('# Oracle adapter test comparison\n\n')
        handle.write('| Mode | R1-f mAP | R1-f mATE | AP@0.5m | AP@1m | AP@2m | AP@4m | R1 mAP |\n')
        handle.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for row in rows:
            values = [
                row['label'], row['target_mAP'], row['target_mATE'],
                row['target_AP@0.5m'], row['target_AP@1.0m'],
                row['target_AP@2.0m'], row['target_AP@4.0m'],
                row['source_mAP']]
            handle.write('| {} | {} |\n'.format(
                values[0], ' | '.join(
                    'N/A' if not np.isfinite(value) else '{:.4f}'.format(value)
                    for value in values[1:])))
    return csv_path


def style_axis(axis, title, ylabel):
    axis.set_title(title, fontsize=13, fontweight='bold', color=TEXT,
                   loc='left', pad=12)
    axis.set_ylabel(ylabel)
    axis.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_color(GRID)
    axis.spines['bottom'].set_color(GRID)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)


def annotate_bars(axis, bars, decimals=2, rotation=0):
    finite = [bar.get_height() for bar in bars if np.isfinite(bar.get_height())]
    offset = (max(finite) * 0.018) if finite and max(finite) > 0 else 0.002
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            ('{:.' + str(decimals) + 'f}').format(value),
            ha='center', va='bottom', fontsize=8, rotation=rotation,
            fontweight='bold')


def plot_overview(rows, output_dir):
    names = [row['label'] for row in rows]
    colors = [COLORS[row['mode']] for row in rows]
    target = np.asarray([100.0 * row['target_mAP'] for row in rows])
    source = np.asarray([100.0 * row['source_mAP'] for row in rows])
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.3),
                             gridspec_kw={'width_ratios': [1.55, 1]})
    fig.patch.set_facecolor('white')

    axes[0].set_facecolor(PANEL)
    bars = axes[0].barh(y, target, color=colors, height=0.56, zorder=3)
    axes[0].invert_yaxis()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontweight='semibold', color=TEXT)
    axes[0].set_xlabel('R1-f test mAP (%)  ·  higher is better')
    axes[0].grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_axisbelow(True)
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    axes[0].tick_params(axis='y', length=0)
    axes[0].tick_params(axis='x', colors=MUTED)
    axes[0].xaxis.label.set_color(MUTED)
    axes[0].set_title('Target-rig recovery', loc='left', fontsize=14,
                      fontweight='bold', color=TEXT, pad=12)
    limit = max(target) * 1.22 if np.isfinite(target).any() else 1.0
    axes[0].set_xlim(0, limit)
    for bar, value in zip(bars, target):
        if np.isfinite(value):
            axes[0].text(value + limit * .015,
                         bar.get_y() + bar.get_height() / 2,
                         '{:.2f}%'.format(value), va='center', ha='left',
                         color=TEXT, fontsize=11, fontweight='bold')

    axes[1].set_facecolor('#FAF7F2')
    axes[1].hlines(y, 0, source, color=GRID, linewidth=3, zorder=1)
    for index, (value, color) in enumerate(zip(source, colors)):
        if np.isfinite(value):
            axes[1].scatter(value, index, s=90, facecolor='white',
                            edgecolor=color, linewidth=2.8, zorder=3)
            axes[1].text(value + max(source) * .035, index,
                         '{:.2f}%'.format(value), va='center', color=TEXT,
                         fontsize=10, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, color=TEXT)
    axes[1].set_xlabel('R1 test mAP (%)')
    axes[1].set_title('Source-rig retention', loc='left', fontsize=14,
                      fontweight='bold', color=TEXT, pad=12)
    axes[1].grid(axis='x', color=GRID, linewidth=.8, zorder=0)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    axes[1].tick_params(axis='y', length=0)
    axes[1].tick_params(axis='x', colors=MUTED)
    axes[1].xaxis.label.set_color(MUTED)
    if np.isfinite(source).any():
        axes[1].set_xlim(0, max(source) * 1.30)

    fig.suptitle('How much cross-rig performance does each oracle recover?',
                 x=.065, y=.98, ha='left', fontsize=19,
                 fontweight='bold', color=TEXT)
    fig.text(.065, .925,
             'PETR trained on R1 and evaluated on R1-f  |  '
             'R1-f target model = vanilla PETR trained from scratch on R1-f',
             ha='left', fontsize=11, color=MUTED)
    fig.tight_layout(rect=(.04, .04, .98, .88), w_pad=4)
    save_figure(fig, output_dir, 'oracle_overview')


def grouped_threshold_plot(rows, output_dir, car_only=False):
    x = np.arange(len(THRESHOLDS))
    fig, axis = plt.subplots(figsize=(13.2, 7.2))
    fig.patch.set_facecolor('white')
    axis.set_facecolor('white')
    label_offsets = (-20, -9, 4, 17, 30, 43, 56)
    for row_index, row in enumerate(rows):
        prefix = 'car_AP@' if car_only else 'target_AP@'
        values = np.asarray(
            [100.0 * row[prefix + threshold + 'm']
             for threshold in THRESHOLDS])
        is_anchor = row['mode'] in ('frozen_r1', 'target_r1f')
        axis.plot(x, values, label=row['label'], color=COLORS[row['mode']],
                  linewidth=3.0 if is_anchor else 2.35,
                  linestyle='--' if row['mode'] == 'frozen_r1' else '-',
                  marker='o', markersize=8, markerfacecolor='white',
                  markeredgewidth=2.3, zorder=4)
        for xi, value in zip(x, values):
            if np.isfinite(value):
                axis.annotate('{:.2f}'.format(value), (xi, value),
                              xytext=(0, label_offsets[row_index]),
                              textcoords='offset points',
                              ha='center', va='bottom', fontsize=8.5,
                              color=COLORS[row['mode']], fontweight='bold',
                              bbox=dict(boxstyle='round,pad=.16', fc='white',
                                        ec='none', alpha=.82))
    axis.set_xticks(x)
    axis.set_xticklabels(['{} m'.format(t) for t in THRESHOLDS])
    axis.set_xlabel('Center-distance matching threshold')
    title = ('Car localization across distance thresholds' if car_only else
             'Localization quality across distance thresholds')
    ylabel = ('Car AP (%)' if car_only else 'Class-averaged mAP (%)')
    style_axis(axis, title, ylabel)
    axis.axvspan(-.18, 1.18, color='#FFF4E8', alpha=.62, zorder=0)
    axis.text(.5, .975, 'STRICT LOCALIZATION', transform=axis.get_xaxis_transform(),
              ha='center', va='top', fontsize=9, color='#A85A12',
              fontweight='bold')
    axis.legend(frameon=False, ncol=min(5, len(rows)), loc='upper left',
                bbox_to_anchor=(0, 1.18), borderaxespad=0,
                labelcolor=TEXT, handlelength=2.5, columnspacing=1.6)
    map_text = '  |  '.join(
        '{} {:.2f}%'.format(row['label'], 100.0 * row['target_mAP'])
        for row in rows if np.isfinite(row['target_mAP']))
    fig.suptitle('PETR oracle-adapter comparison on the R1-f target rig',
                 x=.075, y=.985, ha='left', fontsize=18,
                 fontweight='bold', color=TEXT)
    fig.text(.075, .94, 'Overall R1-f mAP: ' + map_text,
             ha='left', fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(.055, .055, .985, .84))
    stem = 'oracle_car_strict_ap' if car_only else 'oracle_strict_ap'
    save_figure(fig, output_dir, stem)


def plot_validation(histories, output_dir):
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.patch.set_facecolor('white')
    for mode, history in histories.items():
        epochs = [record['epoch'] for record in history]
        axes[0].plot(epochs, [100.0 * record['object/map'] for record in history],
                     marker='o', markerfacecolor='white', markeredgewidth=1.8,
                     linewidth=2.2, label=LABELS[mode], color=COLORS[mode])
        axes[1].plot(epochs, [record['object/mATE'] for record in history],
                     marker='o', label=LABELS[mode], color=COLORS[mode])
    style_axis(axes[0], 'Validation mAP trajectory', 'R1-f validation mAP (%)')
    style_axis(axes[1], 'Validation mATE trajectory', 'R1-f validation mATE')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.legend(frameon=False, labelcolor=TEXT)
    fig.suptitle('Oracle-adapter optimization on R1-f', x=.065, y=.98,
                 ha='left', fontsize=18, fontweight='bold', color=TEXT)
    fig.text(.065, .92, 'Validation trajectories; markers show evaluated epochs',
             fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(.04, .04, .99, .87), w_pad=3)
    save_figure(fig, output_dir, 'oracle_validation_curves')


def save_figure(fig, output_dir, stem):
    fig.savefig(os.path.join(output_dir, stem + '.png'), dpi=220,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, stem + '.svg'),
                bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    histories = {}

    baseline_target = baseline_metrics(args.baseline_json, 'R1-f')
    baseline_source = baseline_metrics(args.baseline_json, 'R1')
    if baseline_target:
        rows.append(summary_row('frozen_r1', baseline_target, baseline_source))

    for mode in MODES:
        mode_dir = os.path.join(args.root, mode)
        history = validation_history(mode_dir)
        if history:
            histories[mode] = history
        target = metric_dict_from_test_log(os.path.join(
            mode_dir, 'evaluation', 'R1-f_metrics.log'))
        source = metric_dict_from_test_log(os.path.join(
            mode_dir, 'evaluation', 'R1_metrics.log'))
        if target:
            rows.append(summary_row(mode, target, source))

    target_target = baseline_metrics(args.target_json, 'R1-f')
    target_source = baseline_metrics(args.target_json, 'R1')
    if target_target:
        rows.append(summary_row('target_r1f', target_target, target_source))

    if not rows:
        raise RuntimeError('No baseline or adapter test metrics were found')

    csv_path = save_summary(rows, args.output_dir)
    plot_overview(rows, args.output_dir)
    grouped_threshold_plot(rows, args.output_dir)
    grouped_threshold_plot(rows, args.output_dir, car_only=True)
    plot_validation(histories, args.output_dir)
    print('Compared: {}'.format(', '.join(row['label'] for row in rows)))
    print('Summary: {}'.format(csv_path))


if __name__ == '__main__':
    main()
