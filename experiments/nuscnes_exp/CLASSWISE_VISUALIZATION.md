# Classwise localization dashboard

After the center-error, GT-coverage, and range-conditioned scripts have produced
their CSV files, run from the PETR repository root:

```bash
python3 experiments/nuscnes_exp/classwise_report.py
```

This regenerates the three classwise Markdown tables and
`experiments/nuscnes_exp/classwise_localization_dashboard.svg`. The SVG places
all detection classes and GT ego-range bins in one figure. Each cell shows the
eligible GT count, prediction-to-GT matches within 4 m, the fraction of GT in
four distance-threshold bands plus no match, and mean absolute radial and
tangential error for the matched subset. The final column summarizes each
class across all ranges.

The stacked threshold bands are GT coverage counts from independent matching
at each threshold. They are not AP values: AP also depends on confidence order
and false positives.
