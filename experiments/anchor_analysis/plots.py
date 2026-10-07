"""Plotting functions for the PETR anchor experiment."""

from __future__ import print_function

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = {
    'navy': '#23395B',
    'blue': '#2F6BFF',
    'cyan': '#12A4A6',
    'orange': '#F28E2B',
    'red': '#D94B4B',
    'gray': '#718096',
}


def values(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def style_axis(axis, title, xlabel, ylabel):
    axis.set_title(title, loc='left', fontsize=12, fontweight='bold')
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color='#E2E8F0', linewidth=0.8, alpha=0.8)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def make_anchor_map(anchors, final, output_dir):
    figure, axis = plt.subplots(figsize=(9, 9))
    axis.scatter(anchors[:, 0], anchors[:, 1], s=8, alpha=0.30,
                 color=COLORS['gray'], label='learned anchors')
    usage = np.zeros(anchors.shape[0], dtype=np.int64)
    for row in final:
        usage[int(row['query_index'])] += 1
    used = usage > 0
    if used.any():
        usage_plot = axis.scatter(
            anchors[used, 0], anchors[used, 1], c=np.log1p(usage[used]),
            cmap='plasma', s=18, alpha=0.85, label='matched anchors')
        colorbar = figure.colorbar(usage_plot, ax=axis, shrink=0.72)
        ticks = colorbar.get_ticks()
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([
            str(int(round(np.expm1(tick)))) for tick in ticks])
        colorbar.set_label('final Hungarian assignments')
    # Limit arrows to a readable deterministic subset.
    shown = final[:min(250, len(final))]
    if shown:
        ax = values(shown, 'anchor_x')
        ay = values(shown, 'anchor_y')
        px = values(shown, 'prediction_x')
        py = values(shown, 'prediction_y')
        gx = values(shown, 'gt_x')
        gy = values(shown, 'gt_y')
        axis.quiver(ax, ay, px - ax, py - ay, angles='xy', scale_units='xy',
                    scale=1, width=0.0022, alpha=0.28,
                    color=COLORS['blue'], label='anchor to prediction')
        axis.scatter(gx, gy, s=16, alpha=0.65, color=COLORS['orange'],
                     label='matched GT')
    axis.axhline(0, color='#CBD5E0', linewidth=0.8)
    axis.axvline(0, color='#CBD5E0', linewidth=0.8)
    axis.set_aspect('equal', adjustable='box')
    style_axis(axis, 'Learned anchor map and decoder corrections',
               'ego x (m)', 'ego y (m)')
    axis.legend(frameon=False, loc='upper right')
    save_figure(figure, output_dir / 'anchor_bev_and_corrections.png')


def make_scatter_plots(final, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    anchor_distance = values(final, 'anchor_distance_bev')
    final_error = values(final, 'final_error_bev')
    correction = values(final, 'correction_distance_bev')
    score = values(final, 'matched_class_score')
    scatter = axes[0].scatter(anchor_distance, final_error, c=score,
                              cmap='viridis', s=12, alpha=0.55,
                              vmin=0.0, vmax=1.0)
    style_axis(axes[0], 'Does a distant anchor reduce localization quality?',
               'selected anchor to GT distance (m)',
               'final prediction to GT error (m)')
    figure.colorbar(scatter, ax=axes[0], label='matched-class score')
    axes[1].scatter(anchor_distance, correction, color=COLORS['cyan'],
                    s=12, alpha=0.45)
    maximum = max(np.percentile(anchor_distance, 99), 1.0)
    axes[1].plot([0, maximum], [0, maximum], '--', color=COLORS['gray'],
                 linewidth=1, label='correction = anchor distance')
    axes[1].legend(frameon=False)
    style_axis(axes[1], 'How far does the decoder correct?',
               'selected anchor to GT distance (m)',
               'anchor to prediction correction (m)')
    save_figure(figure, output_dir / 'distance_relationships.png')


def make_layer_plot(records, stability, output_dir):
    layers = sorted(set(row['decoder_layer'] for row in records))
    med_error = []
    med_correction = []
    mean_score = []
    stable_fraction = []
    for layer in layers:
        layer_rows = [r for r in records if r['decoder_layer'] == layer]
        med_error.append(np.median(values(layer_rows, 'final_error_bev')))
        med_correction.append(np.median(values(
            layer_rows, 'correction_distance_bev')))
        mean_score.append(np.mean(values(layer_rows, 'matched_class_score')))
        layer_stability = [r['fraction_same_query_as_final']
                           for r in stability if r['decoder_layer'] == layer]
        stable_fraction.append(np.mean(layer_stability))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(layers, med_error, 'o-', color=COLORS['red'],
                 label='median final error')
    axes[0].plot(layers, med_correction, 'o-', color=COLORS['blue'],
                 label='median anchor correction')
    axes[0].legend(frameon=False)
    style_axis(axes[0], 'Box-center progression through decoder',
               'decoder layer', 'BEV distance (m)')
    axes[1].plot(layers, mean_score, 'o-', color=COLORS['orange'],
                 label='matched-class score')
    axes[1].plot(layers, stable_fraction, 'o-', color=COLORS['cyan'],
                 label='same query as final assignment')
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(frameon=False)
    style_axis(axes[1], 'Confidence and assignment stabilization',
               'decoder layer', 'fraction / probability')
    save_figure(figure, output_dir / 'decoder_progression.png')


def make_coverage_plot(coverage, final, output_dir, score_threshold):
    bins = np.asarray([0.0, 2.0, 5.0, 10.0, 20.0, np.inf])
    labels = ['0-2', '2-5', '5-10', '10-20', '>20']
    thresholds = (0.5, 1.0, 2.0, 4.0)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    axes[0].hist(values(coverage, 'nearest_anchor_distance_bev'), bins=35,
                 color=COLORS['navy'], alpha=0.85)
    style_axis(axes[0], 'Geometric coverage of all GT objects',
               'nearest learned-anchor distance (m)', 'GT count')

    selected = values(final, 'anchor_distance_bev')
    for threshold in thresholds:
        rates = []
        for left, right in zip(bins[:-1], bins[1:]):
            rows = [row for row in final
                    if left <= row['anchor_distance_bev'] < right]
            if not rows:
                rates.append(np.nan)
                continue
            success = [row['class_correct'] and
                       row['matched_class_score'] >= score_threshold and
                       row['final_error_bev'] <= threshold for row in rows]
            rates.append(float(np.mean(success)))
        axes[1].plot(range(len(labels)), rates, 'o-',
                     label='error <= {} m'.format(threshold))
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1], 'Query success versus selected-anchor distance',
               'anchor-to-GT distance bin (m)', 'query success fraction')
    save_figure(figure, output_dir / 'coverage_and_query_success.png')


def make_class_plot(final, output_dir):
    class_names = sorted(set(row['gt_class'] for row in final))
    errors = []
    distances = []
    for name in class_names:
        rows = [row for row in final if row['gt_class'] == name]
        errors.append(np.median(values(rows, 'final_error_bev')))
        distances.append(np.median(values(rows, 'anchor_distance_bev')))
    y = np.arange(len(class_names))
    figure, axis = plt.subplots(figsize=(9, max(4.8, len(class_names) * 0.48)))
    width = 0.38
    axis.barh(y - width / 2, distances, height=width,
              color=COLORS['blue'], label='selected anchor to GT')
    axis.barh(y + width / 2, errors, height=width,
              color=COLORS['orange'], label='final prediction error')
    axis.set_yticks(y)
    axis.set_yticklabels([name.replace('_', ' ') for name in class_names])
    axis.invert_yaxis()
    axis.legend(frameon=False)
    style_axis(axis, 'Median distance by object class',
               'BEV distance (m)', '')
    save_figure(figure, output_dir / 'per_class_distances.png')


def make_all_plots(records, coverage, stability, anchors, output_dir,
                   score_threshold):
    final_layer = max(row['decoder_layer'] for row in records)
    final = [row for row in records if row['decoder_layer'] == final_layer]
    make_anchor_map(anchors, final, output_dir)
    make_scatter_plots(final, output_dir)
    make_layer_plot(records, stability, output_dir)
    make_coverage_plot(coverage, final, output_dir, score_threshold)
    make_class_plot(final, output_dir)
