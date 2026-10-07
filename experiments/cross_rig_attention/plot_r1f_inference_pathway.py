#!/usr/bin/env python3
"""Summarize object-aligned PETR inference interventions without training."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'r1': '#c24e46', 'r1f': '#167f77'}
LABELS = {'r1': 'R1 checkpoint', 'r1f': 'R1-f checkpoint'}
KEY_MODES = ('key_x', 'key_g3d', 'key_gmv')


def median(values):
    values = [float(value) for value in values if value is not None and
              np.isfinite(float(value))]
    return float(np.median(values)) if values else None


def mean(values):
    values = [float(value) for value in values if value is not None and
              np.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def make_report(output, selected, layers, strengths):
    cases = {'r1': [], 'r1f': []}
    for model in cases:
        for item in selected:
            path = output / 'cases' / model / '{}_gt{:03d}.json'.format(
                item['sample_token'], int(item['tracer_gt_index']))
            with path.open() as handle:
                case = json.load(handle)
            cases[model].append(case)
    strength = max(strengths)
    if strength == 0:
        raise ValueError('Need at least one positive intervention strength for plot')
    observed_rows, intervention_rows, stage_rows = [], [], []
    for model, examples in cases.items():
        for pair, case in enumerate(examples):
            for layer, observation in enumerate(case['observed_layers'], 1):
                reference = np.asarray(case['reference_xy'], dtype=float)
                required = np.asarray(case['gt_center_xy'], dtype=float)-reference
                predicted = np.asarray(
                    case['baseline']['layer_center_xy'][layer-1], dtype=float)-reference
                required_norm = np.linalg.norm(required)
                predicted_norm = np.linalg.norm(predicted)
                alignment = (float(np.dot(required, predicted) /
                                   (required_norm*predicted_norm))
                             if required_norm > .1 and predicted_norm > .1 else None)
                observed_rows.append(dict(
                    model=model, pair=pair, token=case['sample_token'],
                    gt_index=case['gt_index'], query_index=case['query_index'],
                    reference_error_m=case['reference_error_m'],
                    center_error_m=case['baseline']['layer_center_error_m'][layer-1],
                    car_confidence=case['baseline']['layer_car_confidence'][layer-1],
                    correction_alignment=alignment,
                    correction_length_ratio=(float(predicted_norm/required_norm)
                                             if required_norm > .1 else None),
                    **observation))
                stages = case['query_state_stages'][layer-1]
                for operation in ('self_attention', 'cross_attention', 'ffn'):
                    stage = stages[operation]
                    stage_rows.append(dict(
                        model=model, pair=pair, layer=layer, operation=operation,
                        update_norm=stage['update_norm'],
                        relative_update_norm=stage['update_norm']/max(
                            stage['before_norm'], 1e-8)))
            for row in case['interventions']:
                intervention_rows.append(dict(
                    model=model, pair=pair, token=case['sample_token'],
                    gt_index=case['gt_index'], query_index=case['query_index'],
                    native_final_error_m=case['baseline']['layer_center_error_m'][-1],
                    **row))
    with (output / 'per_car_observations.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observed_rows[0]))
        writer.writeheader()
        writer.writerows(observed_rows)
    with (output / 'per_car_interventions.csv').open('w', newline='') as handle:
        columns = sorted(set().union(*(row.keys() for row in intervention_rows)))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(intervention_rows)
    with (output / 'per_car_state_updates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)

    summary = {'num_cars': len(selected), 'plotted_strength': strength,
               'layers': layers, 'models': {}}
    for model, examples in cases.items():
        summary['models'][model] = {
            'median_reference_error_m': median(
                case['reference_error_m'] for case in examples),
            'median_layer_center_error_m': [median(
                case['baseline']['layer_center_error_m'][layer-1]
                for case in examples) for layer in range(1, 7)],
            'median_car_attention_enrichment': [median(
                case['observed_layers'][layer-1]['car_attention_enrichment']
                for case in examples) for layer in range(1, 7)],
            'median_car_to_total_update_norm_ratio': [median(
                case['observed_layers'][layer-1][
                    'car_to_total_update_norm_ratio'] for case in examples)
                for layer in range(1, 7)],
            'median_correction_alignment': [median(
                row['correction_alignment'] for row in observed_rows
                if row['model'] == model and row['layer'] == layer)
                for layer in range(1, 7)],
            'median_correction_length_ratio': [median(
                row['correction_length_ratio'] for row in observed_rows
                if row['model'] == model and row['layer'] == layer)
                for layer in range(1, 7)],
            'median_relative_state_update': {
                operation: [median(row['relative_update_norm'] for row in stage_rows
                                   if row['model'] == model and row['layer'] == layer
                                   and row['operation'] == operation)
                            for layer in range(1, 7)]
                for operation in ('self_attention', 'cross_attention', 'ffn')},
            'intervention': {},
        }
        for layer in layers:
            effects = {mode: [] for mode in KEY_MODES}
            specificity = []
            car_effects = []
            control_effects = []
            for case in examples:
                rows = [row for row in case['interventions']
                        if row['layer'] == layer and row['strength'] == strength]
                car = next(row['delta_final_error_m'] for row in rows
                           if row['mode'] == 'value_car')
                control = mean(row['delta_final_error_m'] for row in rows
                               if row['mode'] == 'value_control')
                car_effects.append(car)
                control_effects.append(control)
                specificity.append(car-control)
                for mode in KEY_MODES:
                    effects[mode].append(next(row['delta_final_error_m']
                                              for row in rows if row['mode'] == mode))
            summary['models'][model]['intervention'][str(layer)] = {
                'value_car_median_delta_error_m': median(car_effects),
                'value_control_median_delta_error_m': median(control_effects),
                'value_car_minus_control_median_delta_error_m': median(specificity),
                'value_specificity_positive_fraction': mean(
                    effect > 0 for effect in specificity),
                'key_median_delta_error_m': {
                    mode: median(values) for mode, values in effects.items()},
                'key_fraction_improved': {
                    mode: mean(value < 0 for value in values)
                    for mode, values in effects.items()},
            }
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)

    # Figure 1: observation. Do not conflate attention mass with useful values.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    fields = [('median_layer_center_error_m', 'BEV center error (m)'),
              ('median_car_attention_enrichment',
               'Attention enrichment on GT-box rays'),
              ('median_car_to_total_update_norm_ratio',
               'Car-ray update norm / total update norm')]
    for axis, (field, ylabel) in zip(axes, fields):
        for model in ('r1', 'r1f'):
            values = summary['models'][model][field]
            axis.plot(range(1, 7), values, '-o', linewidth=2.4,
                      color=COLORS[model], label=LABELS[model])
            axis.annotate('{:.2f}'.format(values[-1]), (6, values[-1]),
                          xytext=(-8, 10), textcoords='offset points',
                          ha='right', fontsize=9, color=COLORS[model])
        axis.set_xticks(range(1, 7))
        axis.set_xlabel('Decoder layer')
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('Same R1-f frames: where the selected car query diverges',
                 x=.03, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output / 'observed_pathway.png', dpi=220)
    fig.savefig(output / 'observed_pathway.svg')
    plt.close(fig)

    # Figure 2: compare the actual center correction with the one GT requires.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    correction_fields = [
        ('median_correction_alignment', 'Correction alignment with GT',
         '1 = toward car; 0 = sideways; -1 = away'),
        ('median_correction_length_ratio', 'Correction length / required length',
         '1 = correct distance from reference'),
        ('median_layer_center_error_m', 'BEV center error (m)', 'Lower is better')]
    for axis, (field, title, subtitle) in zip(axes, correction_fields):
        for model in ('r1', 'r1f'):
            values = summary['models'][model][field]
            axis.plot(range(1, 7), values, '-o', linewidth=2.4,
                      color=COLORS[model], label=LABELS[model])
            if values[-1] is not None:
                axis.annotate('{:.2f}'.format(values[-1]), (6, values[-1]),
                              xytext=(-8, 10), textcoords='offset points',
                              ha='right', fontsize=9, color=COLORS[model])
        axis.set_xticks(range(1, 7))
        axis.set_xlabel('Decoder layer')
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(subtitle)
        axis.grid(alpha=.2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(0, color='#82909e', lw=.8)
    axes[1].axhline(1, color='#82909e', lw=.8, ls='--')
    axes[0].legend(frameon=False)
    fig.suptitle('What each frozen model learned to do with its car query',
                 x=.03, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output / 'correction_trajectory.png', dpi=220)
    fig.savefig(output / 'correction_trajectory.svg')
    plt.close(fig)

    # Figure 3: state update size is descriptive, not localization accuracy.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.1))
    for axis, operation in zip(axes,
                               ('self_attention', 'cross_attention', 'ffn')):
        for model in ('r1', 'r1f'):
            axis.plot(range(1, 7), summary['models'][model][
                'median_relative_state_update'][operation], '-o', lw=2.4,
                color=COLORS[model], label=LABELS[model])
        axis.set_title(operation.replace('_', ' ').title())
        axis.set_xlabel('Decoder layer')
        axis.set_ylabel('State update norm / incoming state norm')
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=.2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('Selected query: size of each decoder operation',
                 x=.03, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(output / 'query_state_updates.png', dpi=220)
    fig.savefig(output / 'query_state_updates.svg')
    plt.close(fig)

    # Figure 4: causal interventions. Each cell is delta final GT-car error.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    magnitude = max(.25, max(abs(row['delta_final_error_m'])
                             for row in intervention_rows
                             if row['strength'] == strength))
    magnitude = min(magnitude, 8.0)
    for column, model in enumerate(('r1', 'r1f')):
        section = summary['models'][model]['intervention']
        value_effects = [section[str(layer)][
            'value_car_minus_control_median_delta_error_m'] for layer in layers]
        axis = axes[0, column]
        bars = axis.bar(range(len(layers)), value_effects,
                        color=[COLORS[model] if value >= 0 else '#7c8a9b'
                               for value in value_effects], width=.67)
        axis.axhline(0, color='#283e52', linewidth=1)
        axis.set_xticks(range(len(layers)))
        axis.set_xticklabels(['L{}'.format(layer) for layer in layers])
        axis.set_ylabel('Δ final error: car-value removal − control (m)')
        axis.set_title('{} · car-value specificity'.format(LABELS[model]))
        axis.grid(axis='y', alpha=.2)
        for bar, value in zip(bars, value_effects):
            axis.annotate('{:+.2f}'.format(value),
                          (bar.get_x()+bar.get_width()/2, value),
                          xytext=(0, 4 if value >= 0 else -14),
                          textcoords='offset points', ha='center', fontsize=9)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        matrix = np.asarray([[section[str(layer)][
            'key_median_delta_error_m'][mode] for layer in layers]
            for mode in KEY_MODES])
        axis = axes[1, column]
        image = axis.imshow(matrix, cmap='RdBu_r', vmin=-magnitude,
                            vmax=magnitude, aspect='auto')
        axis.set_xticks(range(len(layers)))
        axis.set_xticklabels(['L{}'.format(layer) for layer in layers])
        axis.set_yticks(range(3))
        axis.set_yticklabels(['X-key', 'G3D-key', 'GMV-key'])
        axis.set_title('{} · attenuate one key-logit term'.format(LABELS[model]))
        for row in range(3):
            for col in range(len(layers)):
                axis.text(col, row, '{:+.2f}'.format(matrix[row, col]),
                          ha='center', va='center', fontsize=9,
                          color='white' if abs(matrix[row, col]) > .55*magnitude
                          else '#15243a')
    colorbar = fig.colorbar(image, ax=axes[1, :].tolist(), fraction=.025, pad=.03)
    colorbar.set_label('Δ final BEV center error (m): positive = worse')
    fig.suptitle('Frozen-model interventions on one car query per frame · strength {}'.format(
        strength), x=.035, ha='left', fontsize=16, fontweight='bold')
    fig.subplots_adjust(top=.88, bottom=.09, left=.11, right=.91,
                        wspace=.26, hspace=.42)
    fig.savefig(output / 'causal_interventions.png', dpi=220)
    fig.savefig(output / 'causal_interventions.svg')
    plt.close(fig)

    lines = [
        '# R1-f inference-only PETR pathway diagnosis', '',
        '{} selected GT cars, both frozen checkpoints, identical R1-f frames '
        'and calibration. Intervention strength shown: {}.'.format(
            len(selected), strength),
        '', 'Positive intervention Δerror means removing that contribution '
        'made localization worse; negative means it improved.',
        '', '| Model | Reference median (m) | L1 error (m) | L6 error (m) | '
        'L6 correction alignment | L6 correction length ratio | '
        'L1 car-ray enrichment | L6 car-ray enrichment |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model in ('r1', 'r1f'):
        entry = summary['models'][model]
        lines.append('| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | '
                     '{:.2f} | {:.2f} |'.format(
            LABELS[model], entry['median_reference_error_m'],
            entry['median_layer_center_error_m'][0],
            entry['median_layer_center_error_m'][-1],
            entry['median_correction_alignment'][-1],
            entry['median_correction_length_ratio'][-1],
            entry['median_car_attention_enrichment'][0],
            entry['median_car_attention_enrichment'][-1]))
    lines += ['', '## Intervention effects by layer', '',
              'Values are medians across cars, at fixed native query. '
              'Car-value specificity subtracts the mean effect of matched '
              'same-camera control tokens **within each car**.', '',
              '| Model | Layer | Car-value specificity Δerror (m) | '
              'X-key Δerror | G3D-key Δerror | GMV-key Δerror |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in ('r1', 'r1f'):
        for layer in layers:
            entry = summary['models'][model]['intervention'][str(layer)]
            key = entry['key_median_delta_error_m']
            lines.append('| {} | {} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.2f} |'.format(
                LABELS[model], layer,
                entry['value_car_minus_control_median_delta_error_m'],
                key['key_x'], key['key_g3d'], key['key_gmv']))
    lines += ['', '## How to read the figures', '',
              '`correction_trajectory` compares each model\'s own reference-to-box '
              'vector against its reference-to-GT vector. Alignment measures '
              'direction; length ratio measures distance. The state-update plot '
              'shows operation size only, not correctness. Car-ray attention and '
              'value-update ratios are descriptive; interventions measure '
              'within-model sensitivity. A positive car-minus-control value effect '
              'is evidence that those car-ray values matter specifically for final '
              'localization at that layer.',
              '', '## Interpretation boundary', '',
              'This experiment establishes *within-checkpoint* functional '
              'sensitivity, not equality of hidden vectors across checkpoints. '
              'The two models may use different query IDs and latent bases. '
              'GT defines diagnostic car-ray groups; the baseline forward '
              'does not receive GT. Removing car-ray values or key-logit terms '
              'is an out-of-distribution intervention, so inspect both '
              'strengths and matched controls before attributing cause. '
              'The selected cohort is not full-validation AP.', '']
    with (output / 'diagnostic_report.md').open('w') as handle:
        handle.write('\n'.join(lines))
    print('Saved pathway figures and report:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--layers', nargs='+', type=int, default=list(range(1, 7)))
    parser.add_argument('--strengths', nargs='+', type=float, default=[0.5])
    args = parser.parse_args()
    with args.selection.open() as handle:
        selected = json.load(handle)[:args.max_examples]
    make_report(args.output_dir, selected, args.layers, args.strengths)


if __name__ == '__main__':
    main()
