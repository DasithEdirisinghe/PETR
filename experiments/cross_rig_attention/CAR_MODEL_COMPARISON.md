# R1 versus R1-f PETR: car-only analysis on R1-f

For the requested **unwarped, same-R1-checkpoint internal comparison**, see
[R1_PAIRED_RIG_COMPONENT_TRACE.md](R1_PAIRED_RIG_COMPONENT_TRACE.md). The separate
FoV-normalization intervention in [FOV_COUNTERFACTUAL.md](FOV_COUNTERFACTUAL.md)
is optional and is not needed for that diagnostic.

## Reciprocal R1 validation comparison

The launcher accepts a target rig as its second argument and keeps results in
separate named directories. Your renamed R1-f validation directory is
`output/r1f_val`; the reciprocal run goes to `output/r1_val`.

```bash
# First R1 validation run: normal inference once per checkpoint, then analysis.
bash experiments/cross_rig_attention/run_car_model_comparison.sh val R1

# Rebuild R1 validation plots from cached detections without inference.
CAR_COMPARE_ALLOW_INFERENCE=0 \
  bash experiments/cross_rig_attention/run_car_model_comparison.sh val R1

# Optional selected-car decoder trace on R1 validation frames.
python experiments/cross_rig_attention/trace_car_model_comparison.py \
  --split val --target-rig R1 --max-examples 8 --device cuda:0

# Trace the SAME eight physical cars previously selected on R1-f, now seen
# through R1 cameras. This is the direct reciprocal paired-object comparison.
python experiments/cross_rig_attention/trace_car_model_comparison.py \
  --split val --selection-rig R1-f --target-rig R1 \
  --max-examples 8 --device cuda:0

# Existing R1-f validation results and trace use the other target rig.
CAR_COMPARE_ALLOW_INFERENCE=0 \
  bash experiments/cross_rig_attention/run_car_model_comparison.sh val R1-f
```

The analysis validates exact annotation/result sample-token equality and will
stop if saved test detections are mistaken for validation detections. The
existing saved R1 evaluation files contain 1,800 samples and are test results;
R1 validation requires a fresh inference pass unless a separate validation
cache already exists under `output/r1_val`.
For the paired-object trace, scene name and zero-based frame order identify the
synchronized frame; nearest GT car center in global coordinates must then be
within 0.25 m. A missing scene/frame or mismatched GT fails instead of
silently tracing a different car. The mapping and distances are written to
`output/r1_val/traces/paired_selection.json`. The trace uses the R1-f selection
file, so `--max-examples 8` means the same first eight selected R1-f cars.

Both vanilla PETR checkpoints receive the **same R1-f images and calibration**.
They run normal inference; no GT is supplied to either model. GT is used only
after inference to analyze car detections. This experiment does not train,
adapt, swap, or modify model weights.

## Saved-result audit

The existing formatted detections for both checkpoints are under
`results/petr_r50dcn_gridmask_p4_800x320_pccr/{R1,R1-f}/cross_rig/evaluations/R1-f/formatted/`.
Those files cover `R1-f_infos_test.pkl` (1,800 sample tokens), **not**
`R1-f_infos_val.pkl`. We checked a known validation token and it is absent from
the saved test detections. The launcher checks exact token coverage again
before comparing models, so a split mismatch stops with an error.

## Commands

In the PETR environment, from the repository root:

```bash
# Immediate analysis of already saved R1-f test detections; no inference.
CAR_COMPARE_ALLOW_INFERENCE=0 \
  bash experiments/cross_rig_attention/run_car_model_comparison.sh test

# Validation experiment requested for diagnosis. Each missing checkpoint's
# normal inference runs once and its formatted detections are then cached.
bash experiments/cross_rig_attention/run_car_model_comparison.sh val

# Subsequent validation analyses use the cache and run no inference.
CAR_COMPARE_ALLOW_INFERENCE=0 \
  bash experiments/cross_rig_attention/run_car_model_comparison.sh val
```

The `val` command needs an interactive GPU shell for its first run. It runs
`tools/test.py` for each checkpoint, with `data.test` set to R1-f validation,
`--eval bbox`, and `jsonfile_prefix` inside this experiment's `output/r1f_val`.
It stores `predictions.pkl`, `inference.log`, `results_pccr.json`, and
`metrics_summary.json` per checkpoint. Existing complete formatted results
are reused; partial results do not count as a cache hit.

Set `PYTHON_BIN` if the PETR Python executable is not called `python`.
Set `CAR_COMPARE_OUTPUT` to change the root output directory,
`CAR_COMPARE_SCORE_THRESHOLD` (default `0.35`) for diagnostic matching, or
`CAR_COMPARE_ASSOCIATION_DISTANCE` (default `4.0` metres). These diagnostic
parameters **do not change official AP**.

## Analysis outputs

`experiments/cross_rig_attention/output/<split>/analysis/` contains:

- `car_per_gt.csv`: one row per GT car, with both models' matched detections,
  center errors, scores, nearest car candidates, and signed ego-radial/lateral
  errors;
- `car_summary.json`: official car AP at 0.5/1/2/4 m from the saved evaluator,
  detection coverage groups, and center-error statistics;
- `car_coverage_and_error.png`/`.svg`: matched-car counts and conditional
  center-error distributions;
- `car_signed_errors.png`/`.svg`: signed radial and lateral distributions for
  cars matched by both models;
- `car_coverage_by_distance.png`/`.svg`: coverage by GT distance;
- `trace_candidates.json`: examples where R1-f succeeds and R1 fails or where
  both detect but R1 has much larger center error.

The diagnostic matcher uses **final car detections** with score at least the
configured threshold and greedy one-to-one association within 4 m. It does
not reproduce the evaluator's confidence-integrated AP. Official AP is read
directly from each model's `metrics_summary.json`. Error distributions include
only associated detections; the group counts show the unmatched cars so this
selection is visible. `nearest_*` columns probe whether a lower-confidence car
candidate lies near each GT, but formatted final detections cannot expose
queries discarded earlier by PETR. The signed radial direction is from the ego
origin toward the GT car, which is a BEV diagnostic and not an exact camera ray.

## Decoder and attention follow-up

The existing `experiments/query_trace/trace_query.py` captures per-query
decoder predictions and attention terms, but the saved trace tensors currently
cover selected frames only. They cannot provide whole-validation layer
statistics. After inspecting `trace_candidates.json`, run an optional GPU pass
for selected cars with **each checkpoint on the same R1-f frame**:

```bash
python experiments/cross_rig_attention/trace_car_model_comparison.py \
  --split val --max-examples 8 --device cuda:0
```

The wrapper computes the tracer's filtered GT index, validates that the traced
GT car center matches the selected car, and caches traces under
`output/r1f_val/traces/{r1,r1f}/`. It writes `layer_car_comparison.csv` and a
median layer-trajectory plot. Each model's matched query can differ. Its
attention plots and term breakdowns are **diagnostics**, not official inference.
The wrapper's selected cases are deliberately a sample, not whole-validation
attention statistics.
# Reading the validation figures

The clearest starting point is `output/r1f_val/analysis/car_insights.md`, followed by
`car_coverage_by_distance.png` (where detections disappear),
`car_coverage_and_error.png` (matched-car outcomes versus official car AP), and
`car_signed_errors.png` (center error and ego-radial bias on **the same cars**
detected by both models). The plots print exact percentages or mean errors.
Re-run the shell command after a plotting update; it reuses cached validation
detections and does not repeat inference.
