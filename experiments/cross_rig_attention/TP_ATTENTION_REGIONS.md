# Configurable car-TP attention-region audit

This separate script uses the official car AP@4m rank ledger to select true
positives, then reruns standard PETR inference only to identify their queries
and capture last-decoder-layer attention. It uses the same 1 m-expanded GT-ray
regions and one-vs-rest AUROC as the R1-FP audit. It does not change predictions
or AP matching.

Run from the PETR repository root in the working PETR GPU environment:

```bash
python experiments/cross_rig_attention/audit_tp_attention_regions.py \
  --model r1 --test-set r1f_val --score-threshold 0.60 \
  --device cuda:0 --max-frames 20
```

Then omit `--max-frames 20` for the full run. Completed frames are cached.
Set `--model` to `r1` or `r1f`; set `--test-set` to `r1_val`, `r1f_val`, or
`r1f_test`. The R1-f test ledger is produced by
`audit_r1f_test_car_ap_fp.py` and replay-validated against official AP@4m.
`r1_test` is also accepted once its labelled rank ledger and saved final
predictions are present. The script fails clearly if either is absent; it
will not invent a TP matching rule. The default score threshold is 0, so
**all official car TPs** are selected.

Default output location:

```text
experiments/cross_rig_attention/output/<test-set>/tp_attention_regions/<model>/
```

`per_prediction.csv` contains query identity, GT match identity/distance,
score, camera attention masses, and each region's AUROC, attention mass and
cell fraction. `summary.json` has cohort counts and medians, and
`tp_attention_regions.png`/`.svg` shows the three-region comparison. Per-frame
JSON files under `cases/` allow resuming. Use a different `--output-dir` or
`--force` when changing the checkpoint, threshold, or GT-box expansion.

AUROC 0.5 means no preference in attention-weight ranking; 1 means every
region cell ranks above every non-region cell. It is null if a region or its
complement has no valid cells. `no_annotated_gt` does not prove empty
background, and attention is descriptive rather than causal.
