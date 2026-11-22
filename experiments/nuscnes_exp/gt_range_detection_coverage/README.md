# GT detection coverage by ego range

This experiment counts **all evaluator-eligible GT objects** in each ego-range
bin, including objects that baseline PETR did not match. It reuses saved
epoch-24 predictions and replays official class-specific, confidence-ordered
matching independently at 0.5, 1, 2, and 4 m.

Run from the PETR repository root in the `petr` environment:

```bash
python experiments/nuscnes_exp/gt_range_detection_coverage/analyze_gt_coverage.py \
  --data-root data/nuscenes \
  --results-json experiments/detection_error_analysis/outputs/petr_car_ap_aligned/formatted/pts_bbox/results_nusc.json
```

The output Markdown report and CSV/JSON tables are written to
`outputs/baseline/` in this experiment directory. The first table partitions
all GT into four disjoint localization bands and a no-match group. The second
table gives cumulative GT recall at each threshold, using **all GT in the
range bin** as denominator. Both tables include a total row, and a per-class
CSV is saved separately. When the previous center-error summary is available
for the same prediction file, the script also verifies that its 4 m match
count agrees. GT matched at a wider threshold may be claimed by a different
prediction, so the bands describe GT coverage rather than a fixed prediction's
error distribution.

For the classwise Markdown table, run
`python3 experiments/nuscnes_exp/classwise_report.py`; see
`outputs/baseline/gt_coverage_by_class.md`.
