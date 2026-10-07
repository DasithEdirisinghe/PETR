#!/usr/bin/env python3
"""Summarize the first measurable four-run gap, not a unique causal root."""

import argparse
import json
from pathlib import Path

STAGES = ('backbone_0', 'backbone_1', 'fpn_0', 'fpn_1', 'input_proj')


def fmt(value):
    return 'n/a' if value is None else '{:.3f}'.format(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with (args.output_dir / 'summary.json').open() as handle:
        summary = json.load(handle)
    conditions = summary['conditions']
    failed = conditions['R1-f_r1']
    reference = conditions['R1-f_r1f']
    source = conditions['R1_r1']
    reciprocal = conditions['R1_r1f']
    lines = [
        '# Four-run PETR pipeline comparison', '',
        '{} paired physical GT cars; the same two checkpoints were run on both rigs.'.format(
            summary['num_paired_cars']),
        'R1-f/R1 is the failure run; R1-f/R1-f is the successful reference. '
        'GT is used for pairing and scoring only.', '',
        '## Upstream car-feature evidence', '',
        'Held-out GT-ray versus same-camera background AUROC. The feature '
        'vectors themselves are **not** compared across separately trained models.',
        '',
        '| Stage | R1 / R1 model | R1 / R1-f model | R1-f / R1 model | R1-f / R1-f model | Target advantage |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    first_upstream = None
    for stage in STAGES:
        values = [conditions[key]['feature_auc'][stage] for key in
                  ('R1_r1', 'R1_r1f', 'R1-f_r1', 'R1-f_r1f')]
        advantage = (values[3]-values[2] if values[2] is not None and
                     values[3] is not None else None)
        if first_upstream is None and advantage is not None and advantage >= .05:
            first_upstream = stage
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            stage, *(fmt(v) for v in values), fmt(advantage)))
    lines += ['', '## Query and decoder', '',
              '| Measure | R1 / R1 model | R1 / R1-f model | R1-f / R1 model | R1-f / R1-f model |',
              '| --- | ---: | ---: | ---: | ---: |']
    for field, label in [('reference_error_m', 'Selected query reference-to-GT error (m)'),
                         ('final_car_confidence', 'Final selected-query car confidence')]:
        values = [conditions[key][field] for key in
                  ('R1_r1', 'R1_r1f', 'R1-f_r1', 'R1-f_r1f')]
        lines.append('| {} | {} | {} | {} | {} |'.format(
            label, *(fmt(v) for v in values)))
    lines += ['', '| Decoder layer | R1 / R1 model center (m) | R1 / R1-f model center (m) | R1-f / R1 model center (m) | R1-f / R1-f model center (m) |',
              '| --- | ---: | ---: | ---: | ---: |']
    first_decoder = None
    for layer in range(6):
        values = [conditions[key]['decoder_center_error_m'][layer] for key in
                  ('R1_r1', 'R1_r1f', 'R1-f_r1', 'R1-f_r1f')]
        if first_decoder is None and values[2] is not None and values[3] is not None and (
                values[2]-values[3] >= .5):
            first_decoder = layer+1
        lines.append('| {} | {} | {} | {} | {} |'.format(
            layer+1, *(fmt(v) for v in values)))
    lines += ['', 'Final decoder GT-ray enrichment and G3D attention utility:', '',
              '| Measure | R1 / R1 | R1 / R1-f | R1-f / R1 | R1-f / R1-f |',
              '| --- | ---: | ---: | ---: | ---: |']
    for field, label in [('gt_ray_enrichment', 'GT-ray enrichment'),
                         ('geometry_utility', 'G3D leave-term-out utility'),
                         ('appearance_utility', 'X leave-term-out utility')]:
        values = [conditions[key][field][-1] for key in
                  ('R1_r1', 'R1_r1f', 'R1-f_r1', 'R1-f_r1f')]
        lines.append('| {} | {} | {} | {} | {} |'.format(
            label, *(fmt(v) for v in values)))
    lines += ['', '## First measured gap', '',
              'At the tested thresholds (target-model AUROC advantage ≥0.05, '
              'or target-model center-error advantage ≥0.5 m), the first '
              'upstream feature stage is **{}**; the first decoder output '
              'is **{}**.'.format(first_upstream or 'none',
                                  'layer {}'.format(first_decoder) if first_decoder else 'none'),
              '', 'These thresholds are descriptive, not significance tests. '
              'A feature AUROC gap does not establish that the backbone caused '
              'the box error; a decoder gap may have originated upstream. '
              'The selected GT-associated queries can have different IDs and '
              'reference points across checkpoints. The R1-f checkpoint is a '
              'successful behavioral reference, **not tensor ground truth**. '
              'Use `feature_evidence.csv`, `decoder_trajectory.csv`, and the '
              'saved tensors to inspect car-level exceptions before a causal '
              'intervention or full-validation AP test.', '']
    path = args.output_dir / 'diagnostic_report.md'
    path.write_text('\n'.join(lines))
    print('Saved:', path)


if __name__ == '__main__':
    main()
