# Paired PETR and StreamPETR validation GT comparison

This analysis uses the saved PETR and StreamPETR nuScenes validation prediction
JSON files and the official evaluator-filtered GT. It does not run inference.
Each eligible GT is the common unit of comparison, identified by its nuScenes
annotation token. Visibility is read from that annotation for the same frame.

Run from `dev_ws/PETR` in a Python environment with `nuscenes-devkit`, NumPy,
and the dependencies used by the other nuScenes experiments:

```bash
python experiments/nuscnes_exp/paired_visibility_comparison/compare.py
```

The defaults point to `dev_ws/StreamPETR/data/nuscenes`, PETR's saved baseline
results under `localization_oracle`, and the saved StreamPETR validation results.
Use `--data-root`, `--petr-results`, `--stream-results`, or `--output-dir` to
override them.

Outputs in `outputs/`:

- `paired_gt.csv`: one row for every eligible GT, with class, range, visibility,
  annotation and instance tokens, both models' fixed `<4 m` match errors and
  scores, and paired match status at 0.5, 1, 2, and 4 m.
- `summary.csv`: GT counts and recall for all classes, each class, each range
  band, each visibility band, and their intersections at every threshold.
  `stream_only` and `petr_only` count GT recovered exclusively by that model.
- `RESULTS.md`: compact overview of the 4 m results by visibility band.

Matching follows the nuScenes detection evaluator: predictions are ranked by
score within class, each is matched to its nearest unused same-class GT if the
BEV center distance is strictly below the threshold. Matching is rerun at each
threshold. Therefore a GT's assigned prediction can change between thresholds;
the two error columns in `paired_gt.csv` refer specifically to the 4 m match.

The output measures GT coverage and localization, not AP. It does not count
false positives or account for score ranking in AP. Compare visibility bands
within each class and distance range before attributing model gains to
occlusion; nuScenes visibility is a four-level estimate across all six cameras.

## Video comparison of the same PETR TP cases

Run `compare.py` above first. Then, from `dev_ws/PETR`, render a small pilot:

```bash
python experiments/nuscnes_exp/paired_visibility_comparison/render_petr_tp_comparison.py \
  --classes pedestrian --distance-bins 0-10 --visibility-tokens 4 \
  --examples-per-band 6
```

To render every available class, 10 m GT-range bin and visibility level, omit
those filters:

```bash
python experiments/nuscnes_exp/paired_visibility_comparison/render_petr_tp_comparison.py \
  --examples-per-band 8
```

Visibility tokens are `1` = 0–40%, `2` = 40–60%, `3` = 60–80%, and `4` =
80–100%. Videos are saved as `outputs/petr_tp_videos/<class>/<visibility>/<range>m.mp4`.
They retain the original four PETR error-band columns. Each selected GT uses
the same camera frame and pixel crop twice: GT/PETR above GT/StreamPETR. The
shared BEV below shows ego context and a GT-centered 12 m zoom. A missing
StreamPETR 4 m match is labeled explicitly; it does not assert that the model
made no nearby prediction.

The script also writes:

- `petr_tp_transitions.csv`: every selected-cohort PETR 4 m TP, including both
  errors and the outcome. This is the complete cohort, not video sampling.
- `transition_summary.csv`: counts and percentages grouped by class, range,
  visibility, and original PETR error band. Better/same/worse refer to the
  **error band**; continuous error change is reported separately. Tighter
  threshold counts keep each model's original 4 m assigned prediction fixed.
- `selected_examples.csv`: the video sample and exact annotation tokens.
- `manifest.csv`: video paths and counts.

The `--metadata-only` flag writes the two full transition CSVs without loading
the detection evaluators or images. Video examples cycle through better-band,
same-band, worse-band and unmatched cases where each exists. Empty error-band
columns remain empty. The statistics use all PETR TPs even when no camera view
is suitable for a particular video example.

## Shared-GT visibility, localization, and confidence

To analyze only the GT annotations matched by **both** models within 4 m:

```bash
python3 experiments/nuscnes_exp/paired_visibility_comparison/analyze_shared_gt.py
```

This reads `outputs/paired_gt.csv` and writes
`outputs/shared_gt_analysis/RESULTS.md`, two SVG/PNG figures, and two CSVs.
The error-band heatmap has one 4×4 transition matrix for each GT visibility
band. Rows are PETR's assigned 4 m TP error bands; columns are StreamPETR's
assigned 4 m TP error bands. Each cell shows a GT count and its percentage
within the PETR row. Left-of-diagonal cells have a lower StreamPETR error band.

The confidence figure shows raw-score distributions and within-class score
percentiles for the same shared GTs. Raw scores are **not calibrated between
models**. Percentile ranks are computed within each model and class among only
these shared GT TPs; they are a matched-case ranking diagnostic, not AP or an
assessment of how false positives rank. `shared_gt_visibility_summary.csv`
includes an `all` class row and class-by-visibility rows with localization and
confidence statistics. `shared_gt_error_transitions.csv` contains exact matrix
counts.
