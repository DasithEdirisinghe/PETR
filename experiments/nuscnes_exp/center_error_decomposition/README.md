# Baseline PETR center-error decomposition

Run from the PETR repository root in the `petr` environment:

```bash
python experiments/nuscnes_exp/center_error_decomposition/analyze_center_error.py \
  --data-root data/nuscenes \
  --results-json experiments/detection_error_analysis/outputs/petr_car_ap_aligned/formatted/pts_bbox/results_nusc.json \
  --output-dir experiments/nuscnes_exp/center_error_decomposition/outputs/baseline
```

This reuses saved epoch-24 baseline PETR predictions; no inference is required.
It uses the official evaluator's filtering and confidence-ordered, same-class
matching at 4 m. Matches are grouped into `[0, 0.5)`, `[0.5, 1)`, `[1, 2)`,
and `[2, 4)` metres, following the evaluator's strict upper bound. Radial
direction points from the ego position to the GT center in the BEV plane;
tangential direction is perpendicular to it. The summary reports signed radial
bias and absolute radial/tangential errors so opposite signed errors do not
cancel. Counts include a percentage of all 4 m true positives; percentages in
`center_error_by_class.csv` use that class's 4 m true positives as the denominator.
`center_error_matches.csv` retains every matched prediction. This analysis covers
4 m true positives only; it is not a decomposition of AP loss or all false
positives.

For a classwise Markdown table with eligible GT and matched prediction counts,
run `python3 experiments/nuscnes_exp/classwise_report.py`; see
`outputs/baseline/center_error_by_class.md`.

Plot the saved summary with Matplotlib:

```bash
python experiments/nuscnes_exp/center_error_decomposition/plot_center_error_summary.py
```

The plotter writes PNG and SVG files to `outputs/baseline/plots/` in this
experiment directory.

For an illustrated explanation of radial and tangential error alongside the
measured bin means, run:

```bash
python experiments/nuscnes_exp/center_error_decomposition/plot_radial_tangential.py
```

This writes `outputs/baseline/plots/radial_tangential.svg`.
The GT-to-prediction center distance is Euclidean distance in the horizontal
BEV plane: `sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2)`. Height does not enter
the nuScenes distance-threshold match.
