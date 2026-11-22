# Distance-sliced error-band comparison videos

`compare_error_bands.py` is a separate visualization workflow. The existing
chronological `pipeline.py` is unchanged. Both workflows use the same saved
PETR `results_nusc.json`, nuScenes validation data, evaluator filtering, and
strict `<4 m` same-class center-distance TP matching. No model inference is
run. Use the PETR environment with `nuscenes-devkit`, NumPy, `pyquaternion`,
OpenCV, and `ffmpeg` with `libx264`.

## Pilot

From this directory:

```bash
python compare_error_bands.py --scenes scene-0103 --classes car \
  --distance-bins 20-30 --examples-per-bin 6 \
  --output-dir outputs/comparison_pilot
```

Inspect `outputs/comparison_pilot/car/20-30m.png` first, then
`outputs/comparison_pilot/car/20-30m.mp4`.

To compare **up to 100 distinct 0–10 m car TPs from all validation scenes**,
use 25 examples in each of the four error columns:

```bash
python compare_error_bands.py --all-val --classes car \
  --distance-bins 0-10 --examples-per-bin 25 \
  --output-dir outputs/comparison_all_val_car_0_10
```

The resulting video is `outputs/comparison_all_val_car_0_10/car/0-10m.mp4`.
`--examples-per-bin` is a per-error-column limit; a column can contain fewer
examples when its matched-TP pool is smaller or no camera view is available.

## Full selected-scene run

```bash
python compare_error_bands.py --scene-count 10 \
  --output-dir outputs/error_band_comparisons
```

## High-visibility-only videos

From the PETR repository root, run:

```bash
python3 experiments/nuscnes_exp/petr_localization_visualization/compare_high_visibility_error_bands.py \
  --examples-per-bin 50
```

This separate entry point defaults to **all validation scenes**, all ten
classes, and all five class-eligible distance bins. It keeps only original
`<4 m` TPs whose **assigned GT has nuScenes visibility token `4`** (80–100%
visible). Each class/distance video still has four error-band columns and the
same crop, full-camera view, BEV, labels, and selection strategy. The filtered
pool counts are shown in each frame and in `metadata/video_manifest.csv`.
Videos are written under `outputs/high_visibility_error_band_comparisons/`,
so the existing unfiltered videos are preserved. `metadata/selection_config.json`
records the active filter and selection settings.

To start with pedestrian 20–30 m only:

```bash
python3 experiments/nuscnes_exp/petr_localization_visualization/compare_high_visibility_error_bands.py \
  --classes pedestrian --distance-bins 20-30 --examples-per-bin 50
```

The script reuses saved predictions; it does not run PETR inference. Passing
`--scenes` or `--scene-count` overrides its all-validation-scene default.

Use `--all-val` to draw examples from all validation scenes, or `--scenes`
with comma-separated validation scene names. `--classes` and `--distance-bins`
restrict the outputs. `--examples-per-bin` controls the maximum example sets
per distance-bin video. `--hold-frames` and `--fps` control how long each set
is displayed. `--cameras` restricts which camera can be chosen for each GT.

Each video covers **one class and one GT ego-distance bin**. Its four columns
correspond to the disjoint error bands `[0,0.5)`, `[0.5,1)`, `[1,2)`, and
`[2,4)` m. Every displayed GT–prediction pair is a fixed original 4 m TP.
The columns are independent keyframes, not synchronized views. Each column
shows a fixed 800×450 source-pixel camera crop, the full camera image with
the crop outline, and a local BEV fitted to show ego, GT, and prediction. The
best camera is selected by the clipped projected GT box area.

Examples are selected near shared exact-distance targets across the four
error columns. The sampler prefers new GT instances and a mix of scenes.
It never creates artificial matches. A blank column means no further
camera-visible example was selected from that band. Counts and percentages
in the video describe the **selected scenes' complete matched-TP pool**,
which may differ from the full-validation heatmap. Empty class/distance bins
are skipped, including bins beyond a class's evaluation range.

`metadata/selected_scenes.json`, `metadata/selected_examples.csv`, and
`metadata/video_manifest.csv` record the source scenes, examples, and output
counts for reproducibility. The PNG beside each MP4 is its first frame.

These videos support qualitative comparisons of appearance, visibility,
viewpoint, and center displacement. They do not prove which image features
PETR used internally. The original scene videos remain the source for
chronological context.
