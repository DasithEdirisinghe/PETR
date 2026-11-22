#!/usr/bin/env python3
"""Add 4 m TP retention and candidate-factor comparisons from saved audit CSVs."""

import argparse
from pathlib import Path

from audit import (factor_outcome_rates, read_csv, report,
                   transition_factors, write_csv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    source = args.output_dir
    details = read_csv(source / 'gt_transitions.csv')
    corrected = 0
    for row in details:
        if (row['pred4_status_at_0.5m'] == 'matched_other_gt' and
                row['pred4_id'] == row['gt05_pred_id']):
            row['pred4_status_at_0.5m'] = 'matched_same_gt'
            corrected += 1
    if corrected:
        write_csv(source / 'gt_transitions.csv', details, list(details[0]))
    summary = read_csv(source / 'class_range_summary.csv')
    predictions = read_csv(source / 'prediction_transitions.csv')
    factors = transition_factors(details, predictions)
    write_csv(source / 'tp_transition_factors.csv', factors, list(factors[0]))
    rates = factor_outcome_rates(factors)
    write_csv(source / 'tp_factor_outcome_rates.csv', rates, list(rates[0]))
    report(source / 'RESULTS.md', summary, predictions, factors)
    print('Updated transition summaries in {}'.format(source))


if __name__ == '__main__':
    main()
