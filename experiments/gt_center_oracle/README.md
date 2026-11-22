# GT-centered nearest-reference oracle

This diagnostic exposes nuScenes validation GT centers during inference. For
each frame, GT gravity centers and the model's 900 learned reference points
are matched one-to-one by Hungarian center-distance assignment. Matched
references are moved exactly to their GT centers; unmatched references remain
unchanged. This is an oracle experiment and is not a deployable validation
result.

Baseline PETR:

```bash
python experiments/gt_center_oracle/eval_gt_center_oracle.py \
  projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py \
  results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/epoch_24.pth \
  --data-root data/nuscenes/ \
  --ann-file data/nuscenes/nuscenes_infos_temporal_val.pkl \
  --output-dir experiments/gt_center_oracle/outputs/petr_gt_center_3d
```

PETR + URoPE:

```bash
python experiments/gt_center_oracle/eval_gt_center_oracle.py \
  projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_urope_multiview.py \
  results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_urope_multiview/epoch_24.pth \
  --data-root data/nuscenes/ \
  --ann-file data/nuscenes/nuscenes_infos_temporal_val.pkl \
  --output-dir experiments/gt_center_oracle/outputs/urope_gt_center_3d
```

Use `--distance bev` to select references using XY distance while still
moving the selected references to the full XYZ GT gravity centers. Each run
saves predictions, official nuScenes metrics, and the complete per-frame
GT-to-query assignment audit.

## Four-variant visualization

After both oracle evaluations finish, generate a comparison plot from the two
original epoch-24 validation logs and the two oracle `metrics_summary.json`
files:

```bash
python experiments/gt_center_oracle/plot_four_variant_metrics.py
```

The default output is
`experiments/gt_center_oracle/outputs/four_variant_comparison.svg`.
The plot compares NDS, mAP, mATE, car AP at all four nuScenes center-distance
thresholds, and class-wise AP averaged over those thresholds. Its inputs can
be overridden with `--petr-original`, `--urope-original`, `--petr-oracle`,
`--urope-oracle`, and `--output`. The plotter uses only Python's standard
library and does not rerun inference or evaluation. A PNG of the current
results is also saved next to the SVG.

The original log's detailed AP values are rounded to four decimals, while
the oracle `metrics_summary.json` values have full precision. This difference
is too small to affect the visual comparison but should be considered when
reporting very small per-class deltas.
