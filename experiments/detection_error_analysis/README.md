# AP-aligned detection error analysis

`analyze_detections.py` runs normal PETR inference on the complete nuScenes
validation split and analyzes one target class. It does **not** add a confidence
threshold. PETR's normal top-300 query-class predictions are formatted by the
dataset and evaluated by the official nuScenes devkit. The script then replays
the devkit's class-specific, score-ranked matching at every configured AP
distance threshold. It verifies that recalculated target-class AP equals the
official AP before reporting match counts.

For baseline PETR and cars, run from the PETR repository root:

```bash
python -u experiments/detection_error_analysis/analyze_detections.py \
  projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py \
  results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/epoch_24.pth \
  --target-class car \
  --data-root data/nuscenes/ \
  --ann-file data/nuscenes/nuscenes_infos_temporal_val.pkl \
  --output-dir experiments/detection_error_analysis/outputs/petr_car_ap_aligned
```

For PETR+URoPE, substitute its config and checkpoint and use a separate output
directory. The same `--target-class` argument supports other nuScenes classes.

Outputs include the official `metrics_summary.json`, formatted nuScenes
predictions, `ap_aligned_summary.json`, and `gt_match_analysis.csv`. For each
distance threshold, the terminal table shows target-class AP and the numbers
of true positives, false negatives, and false positives. The CSV records the
status of every evaluator-filtered GT object.

For missed GT objects, additional spatial evidence is labelled as possible
class confusion, possible localization error, mixed evidence, a nearby
same-class prediction already consumed by another match, or no nearby
prediction. These evidence groups are **not** an additive or causal breakdown
of AP loss. Nearby predictions of a different class may belong to another
object. A low-scored but correctly located target-class prediction is still
included in AP; its effect is determined by its place in the score ranking.

The script intentionally requires a complete validation run because official
nuScenes AP uses the full validation split. The earlier score-thresholded
diagnostic output directories remain untouched but should not be interpreted
as official AP error counts.
