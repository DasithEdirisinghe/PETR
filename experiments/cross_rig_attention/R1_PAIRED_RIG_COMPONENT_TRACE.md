# R1 PETR paired-rig component trace

This is the **first** component-diagnostic experiment. There is no image or
feature warping, no altered calibration, no finetuning, and no checkpoint swap.
The same R1 checkpoint sees paired R1 and R1-f validation frames of the same
physical cars. GT selects and evaluates cars after normal inference; it is not
fed to the detector.

Six of the eight existing paired cars were assigned different final-layer
queries across the rigs (their GT 3D centers match exactly). Comparing only
object-matched queries would mix two effects. This pipeline therefore compares:

1. R1 image, R1 model, its object-matched query (cached trace).
2. R1-f image, **same** R1 model and **same query index** as condition 1
   (new fixed-query trace).
3. R1-f image, same R1 model, its independently object-matched query
   (cached trace).

The new trace uses PETR's normal R1-f images/calibration; it merely records
what the original R1 query does on that frame. It does not force that query
to be a final detection or alter model computation.

## Run

From the PETR root in the `petr` environment on an interactive GPU:

```bash
bash experiments/cross_rig_attention/run_r1_paired_rig_component_trace.sh
```

Only the eight missing fixed-query R1-f traces require GPU forward passes.
Existing R1 and R1-f object-matched traces are reused. The command caches
fixed-query traces under `output/r1f_val/traces/r1_fixed_source_query/` and
writes analysis to `output/r1f_val/traces/r1_unwarped_internal/`.

To replot from saved traces without inference:

```bash
bash experiments/cross_rig_attention/run_r1_paired_rig_component_trace.sh --analyze-only
```

The outputs are `paired_cases.json` (query IDs and pairing),
`per_car_per_layer.csv`, `summary.json`, and `internal_divergence.png`/`.svg`.
For each decoder layer, the plot shows BEV center error, signed ego-radial
error, car confidence, GT-ray attention enrichment, and the directional
attention-mass utility of image-feature (`X`) and 3DPE (`G3D`) terms. GT-ray
enrichment divides attention mass on rays intersecting the GT 3D box by the
fraction of valid tokens representing those rays; this adjusts for object
size/FoV in the token grid. Individual H-X, E-X, H-G3D, and E-G3D values are
also in the CSV.

## Interpretation

Compare **R1 native versus R1-f same query first**. If a gap appears in
GT-ray attention at layer 1, cross-attention retrieval is a candidate early
failure point. If attention remains well localized but box centers diverge,
look at value aggregation and regression. Then compare the same-query and
object-query R1-f curves: a large gap indicates query reassignment is a
separate failure mode. The R1 model's query reference is fixed for the
same-query comparison, so reference-point drift cannot explain its difference.

This is an observational localization of the failure, **not** proof that X or
G3D causes it. Each leave-one-logit-term-out utility answers how that term
changes attention to GT rays in the *current* network; it does not measure the
AP change from fixing that component. A later causal intervention should be
chosen based on the earliest divergence found here.
