# PETR nuScenes validation reproduction

This experiment runs the official nuScenes evaluator on the complete
validation split. It preserves predictions, the evaluator log, official raw
metrics, per-class tables, and plots.

## Run locally/on an interactive GPU node

From the PETR repository root in the `petr` environment:

```bash
bash experiments/nuscenes_reproduction/run_eval.sh
```

Explicit paths and output directory can be supplied:

```bash
bash experiments/nuscenes_reproduction/run_eval.sh \
  projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_local.py \
  ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth \
  experiments/nuscenes_reproduction/outputs
```

For multiple GPUs, set `NUM_GPUS`, for example `NUM_GPUS=2 bash ...`.

## PBS/A100

```bash
qsub experiments/nuscenes_reproduction/run_eval_a100.pbs
```

## Outputs

- `predictions.pkl`: reusable model predictions, avoiding another inference run;
- `eval.log`: complete official evaluator output;
- `formatted/**/metrics_summary.json`: official nuScenes metrics source;
- `plots/overall_metrics.csv`: mAP, NDS, mATE, mASE, mAOE, mAVE, and mAAE;
- `plots/per_class_metrics.csv`: AP at every distance threshold, mean class AP,
  and available per-class TP errors;
- `plots/overall_map_nds.png`;
- `plots/per_class_ap.png`;
- `plots/per_class_ap_by_threshold.svg` and `.png`: official AP at 0.5, 1,
  2, and 4 m for each class, plus the mean AP across thresholds for each class;
- `plots/per_class_ap_by_threshold.csv`: the exact plotted values in percent;
- `plots/per_class_tp_errors.png`.

To regenerate the class-by-threshold figure from the saved official metrics:

```bash
python3 experiments/nuscnes_exp/nuscenes_reproduction/plot_per_class_threshold_ap.py
```

## Validation GT visibility by class and distance

From the PETR repository root, run:

```bash
python3 experiments/nuscnes_exp/nuscenes_reproduction/plot_gt_visibility_by_distance.py
```

This reads the saved evaluator-eligible GT records in
`../petr_ap_gap_audit/outputs/gt_transitions.csv`; it does not run inference or
reload the nuScenes dataset. The output files are
`outputs/plots/eligible_gt_visibility_by_class_distance.png` (and `.svg` and
`.csv`). The PNG requires `rsvg-convert`; the SVG and CSV are always written.
Each class × GT-distance cell shows the four visibility-token counts and their
within-cell percentages. Gray cells lie outside that class's evaluation range.
Counts are per keyframe and total 121,861 eligible validation GTs. These are
not all raw validation annotations, and the plot contains no PETR predictions.

For the raw training annotations and a directly comparable raw validation
plot, run:

```bash
python3 experiments/nuscnes_exp/nuscenes_reproduction/plot_gt_visibility_by_distance.py --population raw-train
python3 experiments/nuscnes_exp/nuscenes_reproduction/plot_gt_visibility_by_distance.py --population raw-val
```

These use `../gt_distance_distribution/GT_VISIBILITY_BY_DISTANCE.csv` and write
`outputs/plots/raw_train_gt_visibility_by_class_distance.*` and
`outputs/plots/raw_val_gt_visibility_by_class_distance.*`. Both include the
50 m+ bin so that all raw annotations are counted: 944,881 training and
187,528 validation GTs. The raw training plot is not a count of PETR's
post-filter training targets; compare raw train with raw val when studying
dataset composition, and use the eligible validation plot for AP analysis.

## Trained model comparison

Run the threshold comparison from this directory with:

```bash
python3 plot_threshold_ap.py
```

It reads the final (epoch 24) validation records from the baseline PETR,
URoPE multiview PETR, and PETR without 3DPE training logs in `results/`.
`outputs/plots/threshold_ap_comparison.svg` plots AP at 0.5, 1, 2, and 4 m
beside full mAP. `outputs/plots/threshold_ap_comparison.csv` records the
values and source log for each model. Each threshold AP is averaged over the
10 nuScenes detection classes; full mAP is the logged metric averaged over
all classes and thresholds.

Use the unchanged official checkpoint and validation config to compare against
the PETR paper. Record the exact config, checkpoint checksum, GPU/software
environment, and evaluator version alongside reported numbers.
