# Baseline PETR range-sliced AP heatmap

Run from the PETR repository root in the cluster `petr` environment:

```bash
python experiments/nuscnes_exp/range_sliced_ap/analyze_range_ap.py
python experiments/nuscnes_exp/range_sliced_ap/plot_range_ap.py
```

The first command loads the saved PETR validation predictions and evaluator-
filtered nuScenes GT. It puts each GT in an ego-range bin using its GT center
and each prediction in a bin using its predicted center. For every class, range
bin, and center-distance threshold (0.5, 1, 2, 4 m), it uses the nuScenes
`accumulate` and `calc_ap` functions, counts eligible GT, filtered submitted
predictions, and matched GT, and writes `outputs/baseline/range_sliced_ap.csv`.

The second command creates `outputs/baseline/range_sliced_ap_heatmap.svg` and
`range_sliced_ap.md`. Heatmap color encodes AP; each cell prints AP, TP/GT, and
prediction count. `All ranges` reproduces each class's official unsliced AP.
The `all classes` panel averages class AP in each slice, excluding classes
without GT there. A range slice is an analysis extension of the official
nuScenes metric, not a leaderboard metric. Near bin boundaries, GT and
predictions can fall in different bins.
