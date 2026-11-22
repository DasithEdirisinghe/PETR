# Four-run PETR pipeline comparison

This experiment asks **where the R1-trained PETR model first loses the same
physical car when the input switches to the R1-f rig**. It runs both frozen
checkpoints on both paired frames:

| | R1 checkpoint | R1-f checkpoint |
| --- | --- | --- |
| R1 image/calibration | Source reference | Reciprocal transfer |
| R1-f image/calibration | Failure under study | Successful target reference |

The R1-f checkpoint is a *behavioral reference*, not ground truth for hidden
tensors. Different trained models may use different channels and query IDs.
Ground-truth car boxes are used only to pair the same object and score its
features, attention, and predictions. No GT enters model inference.

## Run

From the PETR root in the `petr` GPU environment:

```bash
bash experiments/cross_rig_attention/run_four_run_pipeline.sh
```

This uses the eight existing cars in
`output/r1_val/traces/paired_selection.json`. It reuses any existing
`trace_summary.json` **and** `trace_tensors.pt`; missing traces are generated.
Older paired-selection files without `source_tracer_gt_index` are supported:
the index is resolved from the R1-f annotations and the world-space GT-car
center is checked before running that case.
Upstream stage captures are cached per frame and checkpoint. A larger paired
selection can be supplied, for example:

```bash
bash experiments/cross_rig_attention/run_four_run_pipeline.sh \
  --selection experiments/cross_rig_attention/output/r1f_val/traces/cohorts/balanced_5/paired_selection.json \
  --max-examples 100
```

Use `--device cuda:1` for another GPU, `--split test` for matching test
annotations, `--output-dir PATH` for a different result directory, and
`--force` to recalculate all trace and stage captures. The default output is
`experiments/cross_rig_attention/output/four_run_val/`. If a subprocess
fails, its `.log` is kept next to the expected tensor/trace file.

## What is captured and plotted

- `tensors/R1|R1f/r1|r1f/<token>.pt` stores full fp16 outputs of the two
  ResNet feature stages, two FPN levels, input projection, learned 3DPE,
  multiview 2DPE, all 900 query reference points, and all decoder-layer
  scores/boxes. Existing per-car trace files store cross-attention tensors
  and component logits for the selected query.
- `feature_evidence.csv` gives, at each backbone/FPN/input-projection stage,
  a **within-run held-out GT-box-ray prototype AUROC**. Calibrated feature-cell
  rays intersecting the GT box expanded by 0.5 m are positive; other rays in
  the same cameras are sampled as negatives. Half the positive cells make a
  mean L2-normalized feature prototype; held-out positive and negative cells
  are scored by cosine similarity. AUROC is the probability a held-out
  positive scores above a sampled negative, with ties counted as half. This
  does not require raw channels to match between checkpoints. It is **not** a
  PETR car-classification score or a test of metric-depth information. Rays
  may intersect an occluded box, and nearby cells can be highly correlated.
- `decoder_trajectory.csv` records selected-query reference error, car score,
  center error, GT-ray attention/enrichment, and X/G3D attention-term utility
  across all six decoder layers for each of the four conditions.
- `four_run_pipeline.png` presents the four-condition upstream AUROC, query
  reference location, decoder center error, GT-ray attention enrichment, and
  G3D/X attention utility side by side.
- `diagnostic_report.md` tables the values and flags the first *measured*
  upstream/decoder gap using explicit descriptive thresholds.
- `r1f_failure_pathway.svg`/`.png` is a focused R1-f-only evidence figure:
  car-feature separability, reference distance, per-layer center error, and
  the later full-validation AP@1 m head-swap control. The first three use
  eight selected cars; the AP panel uses the full validation set. Regenerate
  the SVG from saved summaries without GPU with
  `python experiments/cross_rig_attention/plot_r1f_failure_pathway.py`.

The report's “first measured gap” is **not automatically the root cause**.
Feature AUROC does not test metric depth; PE is saved but not treated as an
appearance classifier; and two checkpoints' hidden vectors or query indices
must not be compared numerically. To establish causality, follow an observed
gap with a controlled component intervention and evaluate full car AP on both
rigs. This eight-car cohort is a diagnostic sample, not a population estimate.
