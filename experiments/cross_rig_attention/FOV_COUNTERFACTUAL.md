# FoV counterfactual: locating PETR's cross-rig failure

This experiment keeps the **R1-trained PETR weights frozen**. On R1-f validation
frames it compares four conditions:

| Mode | Image features | Geometry supplied to PETR | What it probes |
| --- | --- | --- | --- |
| `raw` | Original R1-f images | Actual R1-f calibration | Reproduce the cross-rig failure |
| `image_warp` | R1-f images rectified into the virtual R1 pixel grid **before** the backbone | Calibration transformed by the same homography | Can valid FoV normalization restore performance? |
| `feature_warp` | Backbone features from original R1-f images, resampled into the virtual R1 grid | Same transformed calibration | Approximate separation of backbone scale sensitivity from decoder/token geometry |
| `source_view` | Actual synchronized R1 images | Virtual R1 calibration expressed in the target LiDAR frame | Paired-view upper bound; requires source-rig images and is **not** a deployable target-only method |

The homography comes from the **post-resize/post-crop `lidar2img` matrices**, not
raw camera intrinsics. For each camera, the script verifies equal global camera
pose and that its full 3×4 projection matrices are related by a homography.
It aborts if those checks fail. No GT is used to construct the warp. Out-of-view
areas are zero-filled in normalized image/feature space; `geometry_audit.json`
records the image-overlap fraction, so improvements must be interpreted with
coverage loss in mind. It also records the mean absolute difference between
the warped and actual R1 images on overlap (in normalized pixel units), as a
sanity check that the supposedly FoV-only views really align. Feature warping
is approximate because backbone receptive
fields have already mixed nearby pixels.

## Run

From the PETR root in the `petr` environment on an interactive GPU:

```bash
# Quick geometry/inference smoke test on eight frames; no official AP.
python experiments/cross_rig_attention/run_fov_counterfactual.py \
  --source-rig R1 --target-rig R1-f --split val \
  --max-frames 8 \
  --output-dir experiments/cross_rig_attention/output/fov_counterfactual/smoke

# Full validation, all modes. Results/progress are saved per mode every 25 frames.
python experiments/cross_rig_attention/run_fov_counterfactual.py \
  --source-rig R1 --target-rig R1-f --split val

# Summarize official car AP, detection coverage by distance, and radial errors.
python experiments/cross_rig_attention/analyze_fov_counterfactual.py \
  --experiment-dir experiments/cross_rig_attention/output/fov_counterfactual/r1f_to_r1_val \
  --ann-file data/pccr/R1-f/R1-f_infos_val.pkl

# Separately analyze the existing paired eight-car traces, including attention
# mass on feature rays that intersect each car's GT 3D box.
python experiments/cross_rig_attention/analyze_fov_traces.py --split val
```

The full runner writes `raw/`, `image_warp/`, `feature_warp/`, and
`source_view/` directories.
Each has `progress.pkl` (resumable output), `geometry_audit.json`, and official
`formatted/results_pccr.json` plus `formatted/metrics_summary.json`. Its `raw`
car AP is checked against the already cached R1-on-R1-f validation evaluation,
when available. The comparison script writes `comparison.json` and
`comparison.png`; `paired_with_raw` compares center and radial errors on the
**same** cars detected in both conditions. Partial/smoke runs intentionally do
not report official AP.

The paired-trace analysis writes
`output/r1_val/traces/fov_diagnostic/paired_layer_metrics.csv`, a plot, and a
JSON summary. It reports center error, signed ego-radial error, confidence, and
GT-ray attention enrichment at each decoder layer for both checkpoints on the
same eight physical cars across the two rigs. GT enters this **offline
diagnostic**, not the counterfactual inference.

## Reading the outcome

- `image_warp` substantially recovers strict AP and radial error: FoV-induced
  input geometry/appearance mismatch is sufficient to explain much of the
  failure; this does **not** yet isolate one PETR parameter block.
- `image_warp` helps much more than `feature_warp`: evidence that correcting
  image scale before backbone extraction matters. The feature condition is an
  approximation, so this is not a formal causal proof.
- Both help similarly: stronger evidence that the decoder benefits from
  calibrated canonical token geometry, rather than requiring a source-scale
  backbone representation.
- Neither helps: inspect overlap, the baseline fidelity check, and paired
  GT-ray attention before concluding that FoV normalization is insufficient.
- `source_view` recovers in-domain behavior but `image_warp` does not: the
  target image lacks information or has residual appearance/visibility changes
  that simple rectification cannot correct. `source_view` is an experimental
  upper bound, not an inference-time solution.

The next controlled test, if needed, is activation patching within the **same
checkpoint** on aligned grids to distinguish the image-feature (`X`) and 3DPE
(`G3D`) contributions. A large leave-one-term-out attention percentage alone
is not a causal result.
