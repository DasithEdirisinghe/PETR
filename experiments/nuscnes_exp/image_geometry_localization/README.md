# Image geometry and 3D center localization

This experiment joins nuScenes validation GT camera projections to the original
4 m true positives of a saved detector result. It runs on PETR and StreamPETR
without model inference. A checkpoint path is stored in `run.json` as provenance;
the predictions actually analyzed come from `--results-json`.

Run in an environment with the nuScenes devkit, NumPy, pyquaternion, and
Matplotlib. From the PETR repository root:

```bash
DATA=/Volumes/n12893251/dasith_ws/dev_ws/RoboBEV/data/sets/nuscenes
SCRIPT=experiments/nuscnes_exp/image_geometry_localization/analyze.py

python "$SCRIPT" \
  --model PETR \
  --checkpoint ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth \
  --results-json experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json \
  --data-root "$DATA" \
  --output-dir experiments/nuscnes_exp/image_geometry_localization/outputs/petr

python "$SCRIPT" \
  --model StreamPETR \
  --checkpoint /path/to/streampetr_checkpoint.pth \
  --results-json ../StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json \
  --data-root "$DATA" \
  --output-dir experiments/nuscnes_exp/image_geometry_localization/outputs/streampetr
```

`--checkpoint` may be omitted when its path is unknown. OPEN can be analyzed
later with the same script by changing `--model` and `--results-json`.

Outputs:

- `camera_views.csv`: one row per evaluator-eligible GT and camera whose
  projected box intersects the image, with depth, clipped projected width and
  height, axial viewing angle, off-axis angle, and 4 m match details.
- `gt_geometry.csv`: one row per eligible GT. Its camera features come from the
  view with the largest clipped projected box area, preventing objects seen by
  two cameras from being counted twice.
- `geometry_summary.csv`: GT count, 4 m TP count and coverage, strict-threshold
  retention among those same original 4 m TPs, and mean absolute radial and
  tangential errors, split by class, GT range, and image footprint, viewing
  angle, or annotation visibility.
- `summary.md`: compact per-class range results; `plots/`: footprint and angle
  heatmaps for 0.5 m retention and mean absolute radial error.
- `apparent_size_tertiles.csv`, `apparent_size_contrasts.csv`,
  `apparent_size_summary.md`, and `apparent_size_retention_delta.svg`: apparent
  size comparisons within each class and 5 m GT-range bin. Small, medium, and
  large groups contain approximately equal numbers of GT, ranked by projected
  image size. The heatmap shows the large-minus-small 0.5 m retention gap and
  greys out comparisons with fewer than 30 original 4 m TPs in either group.
- `run.json`: exact input paths and counts.
- `experiment2/`: projected-area quartiles within class, GT range and nuScenes
  visibility. `objects.csv` joins projected width/height/area, orientation,
  physical GT length/width/height and center/radial/tangential errors. The
  summary CSV includes all-GT counts, 4 m TP coverage and mean errors by
  quartile; `q4_vs_q1.csv` reports Q4−Q1 differences with scene-cluster
  intervals. `adjusted_area_association.csv` reports the per-class center-error
  association per doubling of image area after adjusting for exact range,
  physical dimensions, orientation, visibility, and camera. `plots/` contains
  per-class line plots of mean center error
  against Q1–Q4, with distance groups as separate lines and visibility
  categories as separate figures.

Footprint is `sqrt(clipped projected width × height)` and uses <32, 32–96,
and ≥96 pixel bins. Viewing angle is the absolute angle between the GT box
heading and camera-to-GT ray in the ground plane: 0° is an axial view and 90°
is side-on. Camera depth is optical-axis Z in the camera frame. GT range is
planar ego-to-GT distance. Each GT is matched once using the devkit's filtered
boxes, confidence order, and closest available same-class GT at 4 m.

The projection test measures geometric camera-field overlap. It does not prove
that an object is unoccluded in that particular camera. nuScenes
`visibility_token` is an annotation-level visibility category across images,
so it is reported separately and should not be read as a camera-specific mask.
Retention is not AP; official AP can change through confidence ranking and
independent matching at each threshold.

To regenerate only the apparent-size analysis from an existing `gt_geometry.csv`
without repeating nuScenes matching or projection:

```bash
python experiments/nuscnes_exp/image_geometry_localization/apparent_size_analysis.py \
  --input experiments/nuscnes_exp/image_geometry_localization/outputs/petr/gt_geometry.csv \
  --model PETR
```

For an existing `gt_geometry.csv` produced before scene IDs and physical box
dimensions were added, Experiment 2 can backfill those fields from nuScenes
metadata without repeating projection or matching:

```bash
python experiments/nuscnes_exp/image_geometry_localization/experiment2_apparent_area.py \
  --input experiments/nuscnes_exp/image_geometry_localization/outputs/petr/gt_geometry.csv \
  --data-root /Volumes/n12893251/dasith_ws/dev_ws/RoboBEV/data/sets/nuscenes \
  --model PETR \
  --output-dir experiments/nuscnes_exp/image_geometry_localization/outputs/petr/experiment2
```

Quartiles use all eligible GT within each class/range/visibility stratum. Mean
center error is conditional on an original 4 m match; the report includes 4 m
coverage and strict GT success to expose this selection. The scene-cluster
interval describes uncertainty in the observed Q4−Q1 difference, not a causal
effect of adding pixels. Use `--range-width 5` as a narrower-range sensitivity
check; the default is 10 m, matching the requested 30–40 m example.
