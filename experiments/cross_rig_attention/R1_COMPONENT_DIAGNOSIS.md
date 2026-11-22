# One pipeline: where does frozen R1 PETR lose the car across FoVs?

This pipeline uses **unchanged synchronized R1 and R1-f images**, each rig's
real calibration, the **same frozen R1 checkpoint**, and the same physical GT
cars. It does not train, warp images/features, swap checkpoint weights, or use
GT as PETR input. GT is used only to identify paired cars, define car-supporting
camera rays, and measure errors. Start on the eight previously paired cars;
expand the paired-trace cohort before claiming population-level behavior.

```bash
# From the PETR root in the petr environment on an interactive GPU.
bash experiments/cross_rig_attention/run_r1_component_diagnosis.sh

# Larger, seeded distance × outcome cohort (slower; creates missing traces).
bash experiments/cross_rig_attention/run_r1_component_diagnosis.sh balanced

# Re-analyze saved traces without GPU, then separately replot/check the probe
# by reading its saved CSV and summary.
bash experiments/cross_rig_attention/run_r1_paired_rig_component_trace.sh --analyze-only
```

Set `MAX_EXAMPLES` and `DEVICE` for a different paired-trace count or GPU.
`PER_STRATUM` controls the balanced-cohort sampling cap (default 5). Balanced
mode saves `paired_selection.json` and `cohort_summary.json` under
`output/r1f_val/traces/cohorts/balanced_5/`. Its `MAX_EXAMPLES` default is all
selected cars; the default mode uses eight.
The script first completes fixed-R1-query tracing on R1-f (reusing cached R1
and R1-f object-query traces), then runs the component sensitivity probe. The
probe caches its result and will not repeat it unless called directly with
`--force`. Its default scales are 0.75, 1.0, and 1.25; 1.0 must exactly replay
the unmodified model or the experiment stops. It also checks that its R1
baseline reproduces the saved R1 trace's final center.

## Outputs and what they mean

`output/r1f_val/traces/r1_unwarped_internal/` (or
`output/r1f_val/traces/cohorts/balanced_5/internal/`):

- `paired_cases.json`: scene/frame, paired sample tokens, and query IDs. Six of
  the original eight cars switch assigned queries across rigs.
- `image_feature_car_evidence.csv`: source-car-prototype discrimination of GT
  car rays versus same-camera background for R1 and R1-f. This measures the
  decoder-input image feature (X) after backbone and input projection, not
  a pure backbone-layer attribution.
- `per_car_per_layer.csv`: GT-ray attention mass/enrichment, (H/E) × (X/G3D)
  term discrimination and directional utility, center/radial errors and
  confidence by decoder layer.
- `internal_divergence.png` and `summary.json`: median trajectories for the
  R1 native query, the *same* R1 query on R1-f, and the independently assigned
  R1-f query.
- `diagnostic_report.md`: combined synthesis of trace and intervention results.

`output/r1f_val/traces/r1_component_sensitivity/` (or
`output/r1f_val/traces/cohorts/balanced_5/sensitivity/`):

- `per_car_per_layer.csv`: every component/scale/rig/car/layer outcome, with
  distance and detection-outcome stratum where available.
- `component_sensitivity.png` and `summary.json`: mean final center and radial
  errors and car confidence for the same fixed queries on both rigs.

The three interventions are performed **inside** the frozen R1 model:

| Probe | Changed | Held fixed |
| --- | --- | --- |
| `x_key` | Scales image-content (X) entering attention keys | Image input, real calibration, 3DPE, and value (V) |
| `g3d_key` | Scales learned 3DPE entering keys | (X), (V), query/reference, real calibration |
| `all_key` | Scales both key content and key PE | (V) and model weights; coarse attention-sharpness control |

The reported `mean_delta_center_error_m` is relative to **the same car and
query at scale 1.0**; negative means improved localization. A useful result is
one that improves R1-f consistently without comparably harming R1, and is
stronger than the `all_key` control. Examine per-car effects and distance
strata, not just an eight-car mean. Scaling a term is a sensitivity test, not
the final architecture or proof that the term is inherently wrong.

## Decision logic

1. If the car's (X) evidence is already weak on R1-f, investigate backbone
   scale/appearance sensitivity before modifying 3DPE.
2. If (X) evidence remains but the fixed query loses GT-ray attention at
   layer 1, compare the signed (X) and (G3D) term discrimination/utility.
3. If selectively changing one term improves final center error and radial
   bias on R1-f while retaining R1, that branch is a plausible causal
   contributor. If `all_key` helps equally, the effect may just be attention
   sharpness. If neither selective probe helps, inspect value aggregation and
   box refinement instead of claiming an (X/G3D) cause.

This pipeline can identify a **dominant failure pathway**. It cannot prove a
unique faulty component from only two FoVs, because image features and 3DPE
change jointly and interact in attention. A later full-validation AP run is
needed before adopting any probe as a model modification.
