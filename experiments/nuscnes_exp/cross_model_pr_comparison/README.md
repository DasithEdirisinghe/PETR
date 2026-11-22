# PETR versus StreamPETR precision–recall curves

This plotter reads the already saved nuScenes validation `metrics_details.json`
and `metrics_summary.json` files. It does not run inference or rematch boxes.

From `dev_ws/PETR`:

```bash
python3 experiments/nuscnes_exp/cross_model_pr_comparison/plot_pr_curves.py
```

The output directory contains `pr_<class>.svg` and `.png` for each of the ten
nuScenes detection classes, an `ap_gain_heatmap.svg` and `.png`, exact values
in `ap_by_class_threshold.csv`, and a linked `RESULTS.md` index. PNG export
uses `rsvg-convert` when available or Python `cairosvg`; SVGs are always saved.

Every class figure compares PETR and StreamPETR at center-distance matching
thresholds 0.5, 1, 2, and 4 m. The plotter reconstructs AP from each saved PR
array using the official nuScenes recall and precision cutoffs and checks it
against the corresponding saved AP before drawing. It also verifies that the
40 class × threshold AP differences average to the official mAP difference.

The curves cover the official class-wide validation evaluation. They do not
separate GT by range or visibility. The heatmap provides an exact decomposition
of official mAP across classes and thresholds; a range- or visibility-sliced
AP plot would be a separate custom diagnostic with explicit false-positive
assignment rules.
