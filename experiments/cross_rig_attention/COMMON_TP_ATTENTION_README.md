# Common-TP attention and decoder pathway on R1-f frames

This replaces the eight-car interpretation of `r1f_failure_pathway.png` with
the **official AP@4m common-car-TP cohort** on R1-f validation frames (2,583
cars in the currently saved results). The cohort comes from the existing
`car_ap_rank_audit/ranked_car_predictions.csv`; it is not defined by Hungarian
training assignment or a diagnostic score threshold.
When the DnD `paired_gt.csv` exists, the script also verifies that its 2,583
paired final-prediction indices match the rank-ledger cohort exactly.

The GPU pipeline reruns ordinary, frozen PETR inference. It mirrors the
unchanged `NMSFreeCoder` top-300 query–class selection and post-center filter
plus the PCCR dataset's class-specific range filtering to recover each saved
evaluator-TP prediction's query ID. It associates saved final detections to
the traced query–class candidates one-to-one by class, score, and global BEV
center, failing on absent or ambiguous identities; it does not assume the two
lists have identical order. Because a fresh GPU pass can differ slightly from
the saved inference, the identity check permits at most 0.02 score and 0.05 m
center discrepancy and writes the *actual* discrepancies for every TP to the
CSV and summary. It checks the retraced final center against the saved 4 m TP
with a 0.05 m numerical allowance before measuring
the **last decoder layer's** attention on GT-box-intersecting feature-cell
rays. Attention is averaged over heads and summed over all valid cells from
all cameras. The reported mass is this GT-region sum divided by total valid
attention; its range is 0–1. The default GT box expansion is 0.5 m to account
for feature-grid sampling, and the figure states this explicitly. This is
not a measure of cross-model attention-map overlap or causal feature use.

From the PETR root, in the usual PETR GPU environment:

```bash
# Small validation run; output is resumable and no full-cohort figure is drawn.
python experiments/cross_rig_attention/audit_common_tp_attention.py \
  --device cuda:0 --max-frames 5

# Complete 2,583-car cohort, both checkpoints; resumes cached cases.
python experiments/cross_rig_attention/audit_common_tp_attention.py \
  --device cuda:0
```

On the PBS cluster, submit the equivalent single-A100 job instead:

```bash
# Quick batch validation; useful when interactive allocations land on H100.
qsub -v SMOKE_ONLY=1 experiments/cross_rig_attention/run_common_tp_attention_a100.pbs

# After the smoke job exits successfully, run the full 2,583-car cohort.
qsub experiments/cross_rig_attention/run_common_tp_attention_a100.pbs
```

The job checks required inputs, runs the five-frame smoke test, then the
resumable full run. Its default `PETR_DIR` matches the existing A100 job
scripts. Use `qsub -v PETR_DIR=/path/to/PETR` if your cluster path differs.

The full run writes `output/r1f_val/common_tp_attention/per_common_tp.csv`,
`summary.json`, and `common_tp_geometry_attention.svg` (plus PNG when
`rsvg-convert` is installed). Cases are cached per model/frame. Use `--force`
only to recompute cases; checkpoint file identity and rank-ledger identity are
checked against cache metadata. To redraw without a GPU:

```bash
python experiments/cross_rig_attention/plot_common_tp_geometry_attention.py
```

The focused plot reports *the same 2,583 common-TP cars*: median GT-region
attention mass, median initial-query-reference and final-box BEV center error,
and the median **per-car recovery margin** (reference error minus final error,
with its interquartile range). It also reports paired R1-f minus R1 medians;
these are medians of per-GT differences, not differences of model medians.
The plot identifies how many attention measurements were valid. A GT region
with no intersecting valid feature cells is excluded from its attention
median, not silently assigned zero. Last-layer attention mass is descriptive:
it cannot prove object recognition or causal use of those features.

The earlier four-panel `plot_common_tp_failure_pathway.py` remains available
for optional reproduction of the head-swap AP context if its swap summary is
present. The new focused plot does not depend on that analysis.

The existing DnD-style car failure visualization is
`output/r1f_val/dnd3d_car_4m/dnd3d_car_summary.svg`; its upper-right panel
plots the R1 candidate categories for all **857 native-only car TPs**. That
cohort is deliberately separate from the 2,583 common TPs in this trace.
