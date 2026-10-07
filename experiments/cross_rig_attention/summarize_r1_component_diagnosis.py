#!/usr/bin/env python3
"""Combine observational and controlled-probe results without overstating cause."""

import argparse
import json
from pathlib import Path


def number(value, digits=3):
    return 'n/a' if value is None else ('{:.%sf}' % digits).format(value)


def final(summary, condition, metric):
    return summary['median_by_layer'][condition][metric][-1]


def best_probe(summary, rig, branch):
    values = summary['conditions'][rig][branch]
    candidates = [(float(alpha), row) for alpha, row in values.items()
                  if float(alpha) != 1.0]
    return min(candidates, key=lambda entry: entry[1]['mean_delta_center_error_m'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--probe-dir', type=Path, required=True)
    args = parser.parse_args()
    with (args.trace_dir / 'summary.json').open() as handle:
        trace = json.load(handle)
    with (args.probe_dir / 'summary.json').open() as handle:
        probe = json.load(handle)
    if trace['num_paired_cars'] != probe['num_paired_cars']:
        raise ValueError('Trace and probe cohort sizes differ')
    evidence = trace['image_feature_evidence']
    lines = [
        '# Frozen R1 PETR: paired-R1/R1-f diagnostic report', '',
        '{} matched GT cars; {} changed independently assigned query.'.format(
            trace['num_paired_cars'], trace['query_changed_count']),
        'The model, weights, and input images are unchanged. GT was used only for pairing and scoring.',
        '', '## Observed pathway', '',
        '| Measure | R1 frames | R1-f frames, fixed R1 query |',
        '| --- | ---: | ---: |',
    ]
    for metric, title in [('bev_error_m', 'Final median BEV center error (m)'),
                          ('radial_ego_m', 'Final median signed radial error (m)'),
                          ('gt_ray_enrichment', 'Final GT-ray attention enrichment'),
                          ('geometry_utility', 'Final G3D directional utility'),
                          ('appearance_utility', 'Final X directional utility')]:
        lines.append('| {} | {} | {} |'.format(
            title, number(final(trace, 'r1_native', metric)),
            number(final(trace, 'r1f_same_query', metric))))
    lines += ['', 'R1-car prototype discrimination (AUROC): R1 {}, R1-f {}. '
              'Cross-rig car-feature cosine: {}.'.format(
                  number(evidence['source_template_auc_r1_median']),
                  number(evidence['source_template_auc_r1f_median']),
                  number(evidence['object_feature_cosine_median'])),
              '', '## Controlled key-component probes', '',
              'Each row shows the best tested non-1.0 scale on R1-f. '
              'Negative Δ means lower final center error than the original model.',
              '', '| Branch | Scale | R1-f Δerror (m) | Cars improved | R1 Δerror (m) at same scale |',
              '| --- | ---: | ---: | ---: | ---: |']
    for branch in probe['branches']:
        alpha, target = best_probe(probe, 'R1-f', branch)
        source = probe['conditions']['R1'][branch][str(alpha)]
        lines.append('| {} | {} | {} | {}% | {} |'.format(
            branch, number(alpha, 2),
            number(target['mean_delta_center_error_m']),
            number(100*target['fraction_cars_improved'], 1),
            number(source['mean_delta_center_error_m'])))
    lines += ['', '## Reading the result', '',
              'A drop in feature AUROC suggests lost image evidence. Preserved feature '
              'evidence with early GT-ray attention loss suggests a key-query routing '
              'problem. A selective branch intervention that improves R1-f without '
              'material R1 damage strengthens that hypothesis; similar benefit from '
              '`all_key` may instead reflect attention temperature. Later-layer error '
              'growth with preserved early attention points toward value aggregation '
              'or box refinement. These are hypotheses to validate per car and with full '
              'detection AP, not unique causal identifications.',
              '', 'Inspect `per_car_per_layer.csv` in both directories and the PNG plots '
              'before choosing an architectural intervention.', '']
    output = args.trace_dir / 'diagnostic_report.md'
    output.write_text('\n'.join(lines))
    print('Saved combined diagnostic report:', output)


if __name__ == '__main__':
    main()
