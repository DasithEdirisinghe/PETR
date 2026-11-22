# Car reference-point recovery on R1-f validation

This experiment measures **how far each checkpoint's final and intermediate
car prediction moves toward GT from its selected query's initial reference
point**. It runs the frozen R1 and R1-f checkpoints on every R1-f validation
frame, using the same images and calibration. GT is used only after inference
to assign a predicted query to each car and measure errors; it is not input to
PETR.

The quantity for car \(i\), query \(q\), and decoder layer \(l\) is

\[
e_{0,i}=\lVert r_q-g_i\rVert_2,\quad
e_{l,i}=\lVert \hat c_{q,l}-g_i\rVert_2,\quad
\mathrm{recovery}_{l,i}=e_{0,i}-e_{l,i}.
\]

All three use BEV \(x,y\) coordinates in metres. Positive recovery means the
predicted center is closer to GT than the initial reference; negative recovery
means it is farther away. We report both the **mean of per-car recoveries** and
the median, not just a difference of aggregate medians. A normalized fraction
is also reported only for cars whose reference starts at least 1 m from GT,
because division by a nearly zero reference error is unstable. Each decoder
layer uses the *same* initial reference point in this PETR implementation.

## Run

From the PETR root on an interactive GPU shell:

```bash
# Quick smoke test; uses only the first 20 validation frames.
bash experiments/cross_rig_attention/run_car_reference_recovery.sh --max-frames 20

# Full R1-f validation set, both checkpoints.
bash experiments/cross_rig_attention/run_car_reference_recovery.sh
```

Use `--device cuda:1` to select a different GPU, `--score-threshold 0.35` to
change the diagnostic detection-like subset, or `--output-dir PATH` to choose
another destination. Per-frame JSON files are cached separately for each
checkpoint, so rerunning after interruption skips completed frames. The full
run after a smoke test reuses those frames and processes the rest. This does
not rerun the official AP evaluator.

## Query association and cohorts

For each checkpoint independently, PETR's existing Hungarian training
assigner matches final-layer predictions to all GT objects. We then retain
the GT **car** matches. This supplies one query per matched car and avoids
choosing a query based only on a near-GT center. We report two cohorts:

- `assigned`: every GT car that received a Hungarian query, regardless of
  score or predicted class. This is the main recovery analysis and includes
  poor detections.
- `eligible`: assigned cars whose final predicted class is car and whose car
  score exceeds `--score-threshold`. This is detection-like but conditioned
  on success; do not use its error mean alone to compare models.

Unassigned GT cars and both cohort sizes are included in `summary.json`.
Hungarian matching is an *analysis choice*, not the postprocessed inference
decision, and the selected query can differ between checkpoints. The `paired`
section restricts comparisons to GT cars assigned by **both** checkpoints.

## Outputs

Default directory:
`experiments/cross_rig_attention/output/r1f_val/reference_recovery/`

- `frames/r1/<token>.json`, `frames/r1f/<token>.json`: one resumable result
  per frame, including each car's query ID, reference and predicted centers,
  layer errors, recovery, progress along the required correction direction,
  final car score, and class-correct flag.
- `per_car_r1.json`, `per_car_r1f.json`: flattened car records for analysis.
- `summary.json`: car counts, mean/median reference error, mean/median error
  and recovery after every decoder layer, fraction improved, strict 0.5/1/2 m
  center-hit fractions, fraction whose layer-6 prediction is worse than layer
  1, and same-car paired comparisons.
- `car_reference_recovery.png`/`.svg`: two model curves, showing mean distance
  to GT and mean recovery from reference at each layer.

The result measures the combined **decoder-state plus regression-head**
correction. It does not isolate the regression MLP from its input state; that
would require an additional controlled readout or swap experiment. Likewise,
these center-hit fractions are not mAP: confidence ranking, duplicate
predictions, and official matching rules are not included.
