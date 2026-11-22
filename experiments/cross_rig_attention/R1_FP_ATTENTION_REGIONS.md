# High-confidence R1 car-FP attention-region audit

This first-stage diagnostic traces only *saved final R1 car false positives*
with sigmoid score at least 0.60 on R1-f validation or test frames back to their
producing queries. FP decisions come from the existing official AP@4m rank
ledger; inference and AP evaluation are not modified. TPs and R1-f outputs
are intentionally excluded and can be studied later with the same metrics.

## Run on the A100 environment

From the PETR repository root, with the existing PETR environment activated:

```bash
python experiments/cross_rig_attention/audit_r1_fp_attention_regions.py \
  --device cuda:0 --max-frames 20
```

If the smoke test succeeds, run the full cohort. Per-frame cases are cached and
the full run reuses completed smoke-test frames:

```bash
python experiments/cross_rig_attention/audit_r1_fp_attention_regions.py \
  --device cuda:0
```

The default is `--test-set r1f_val`. For the R1-f **test** FP cohort produced
by `audit_r1f_test_car_ap_fp.py`, run:

```bash
python experiments/cross_rig_attention/audit_r1_fp_attention_regions.py \
  --test-set r1f_test --device cuda:0 --max-frames 20
```

The test run uses a separate cache and output directory.

Defaults: saved car sigmoid score >= 0.60; 1 m expansion on every side of each
annotated 3D GT box. This selects **875 R1 FPs** in the full R1-f validation
rank ledger; no TP query is traced.

Outputs under `output/<test-set>/r1_fp_attention_regions/`:

- `per_prediction.csv`: one row per selected final prediction, including
  official-FP query identity, confidence, optional DnD FP proximity
  category, per-camera attention mass, and all region metrics.
- `summary.json`: cohort counts, metric definitions, AUROC/mass medians for
  all selected R1 FPs, and medians by prior FP-proximity category.
- `r1_fp_attention_regions.png` and `.svg`: overall three-region comparison.
- `r1_fp_category_attention_regions.png` and `.svg`: R1 FP attention by the
  five previously audited GT-proximity categories, when that DnD ledger exists.
- `cases/r1/<sample-token>.json`: resumable per-frame cache. If the
  checkpoint, rank ledger, threshold, or expansion changes, use a new output
  directory or `--force`.

## Region and metric definitions

Each image-feature cell is converted to a calibrated 3D camera ray. A ray is
labelled by its *nearest positive intersection* with an annotated 3D GT box
(expanded by 1 m): `car`, `other_object`, or `no_annotated_gt`. Thus the three
labels are mutually exclusive and partition all valid attention cells. The
fraction of rays intersecting multiple expanded boxes is recorded. This
geometric tie-break is not an occlusion or instance-segmentation oracle.

For a query and region R, using final-decoder-layer attention averaged over
heads and normalized over all valid cells in all cameras:

- `attention_mass`: sum of weights in R / sum of all valid weights. The three
  region masses sum to 1.
- `attention_auroc`: probability that a randomly selected R cell has higher
  weight than a randomly selected non-R cell, with ties worth 0.5. Its range
  is 0 to 1; 0.5 means no rank preference. Null if either side is empty.
- `cell_fraction`: R's share of valid feature cells, for interpreting mass.

The raw info annotations, including boxes excluded by AP eligibility rules,
are used to avoid calling a ray empty merely because an annotated object is
outside the evaluator's class/range subset. Nevertheless,
`no_annotated_gt` is **not** proof of empty background. Feature cells have
receptive fields, projected boxes are not pixel masks, and attention is
descriptive rather than causal.
