#!/usr/bin/env python3
"""Render-contract checks; GPU tracing is validated in the PETR environment."""

import json
import tempfile
import unittest
from pathlib import Path

from plot_common_tp_geometry_attention import plot


class CommonTPPlotTests(unittest.TestCase):
    def test_full_paired_summary_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            summary = {
                'common_tp_total': 2583, 'paired_traced': 2583,
                'paired_attention_comparisons': 2578,
                'paired_median_r1f_minus_r1_attention_mass': .015,
                'paired_median_r1f_minus_r1_recovery_margin_m': .4,
                'models': {
                    name: {'n': 2583, 'n_attention_valid': 2578,
                           'median_gt_attention_mass': mass,
                           'median_gt_cell_fraction': .04,
                           'median_reference_error_m': ref,
                           'median_final_error_m': final,
                           'median_recovery_margin_m': ref-final,
                           'recovery_margin_iqr_m': [ref-final-1,
                                                     ref-final+1],
                           'fraction_positive_recovery': .8}
                    for name, mass, ref, final in
                    (('R1', .12, 9.0, 2.0), ('R1-f', .15, 10.0, 1.8))}}
            input_path, output_path = folder/'summary.json', folder/'plot.svg'
            input_path.write_text(json.dumps(summary))
            plot(input_path, output_path)
            svg = output_path.read_text()
            self.assertIn('Same 2,583 cars', svg)
            self.assertIn('Reference-to-final recovery margin', svg)
            self.assertIn('1.5 percentage points', svg)

    def test_rejects_incomplete_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder/'summary.json'
            source.write_text(json.dumps({'common_tp_total': 2583,
                                          'paired_traced': 5}))
            with self.assertRaises(ValueError):
                plot(source, folder/'plot.svg')


if __name__ == '__main__':
    unittest.main()
