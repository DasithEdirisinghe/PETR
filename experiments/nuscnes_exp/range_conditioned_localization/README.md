# Baseline PETR localization by GT ego range

This experiment groups the official 4 m matched predictions from
`center_error_decomposition` by their ground-truth ego range. Run from the
PETR repository root:

```bash
python3 experiments/nuscnes_exp/range_conditioned_localization/analyze_gt_range.py
```

Outputs in `outputs/baseline/` include a Markdown table, JSON and CSV summary,
and a per-class CSV. Percentages use the number of 4 m matched predictions
within each GT range bin as the denominator. The `Share of 4 m matches` column
uses all 4 m matches and sums to 100% across the range bins. In the per-class
CSV, that share uses all 4 m matches for the class. The four displayed error categories are disjoint
`[0, 0.5)`, `[0.5, 1)`, `[1, 2)`, and `[2, 4)` m, so they sum to 100% within
each populated GT range row. This conditional analysis does not measure the fraction
of all GT objects detected or AP at each range. The `>50 m` row is expected to
be empty with the official nuScenes detection-range filters.

For the classwise Markdown table with eligible GT and matched prediction
counts, run `python3 experiments/nuscnes_exp/classwise_report.py`; see
`outputs/baseline/gt_range_by_class.md`.
