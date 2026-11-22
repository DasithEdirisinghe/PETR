# PETR regression-head swap: decoder state or output mapping?

This experiment tests both trained PETR checkpoints on the **same R1-f frames**
and, as a reciprocal control, on R1 frames. It crosses the model that produces
the decoder state with the checkpoint that supplies the six regression MLPs:

| Decoder state and query reference | Regression MLPs | Interpretation |
| --- | --- | --- |
| R1 | R1 | Native R1 baseline |
| R1 | R1-f | Does the R1-f head help R1 states? |
| R1-f | R1 | Can the R1 head use R1-f states? |
| R1-f | R1-f | Native R1-f baseline |

The **host** model supplies images/features, calibration-conditioned PE,
queries, decoder state, *initial reference points*, and classification scores.
Only the six post-decoder regression MLPs are replaced. Thus the comparison
within a host leaves its car scores unchanged but can change its 3D boxes and
which query is associated with a GT car. There is no training and no GT enters
the forward pass. The script first recomputes boxes using the host's own MLPs
and aborts unless they reproduce PETR's original boxes within `1e-4`.

## Commands

From the PETR root in the `petr` interactive GPU environment:

```bash
# Smoke test on both rigs.
bash experiments/cross_rig_attention/run_regression_head_swap.sh --max-frames 20

# All validation frames on R1-f and R1, with official AP evaluation.
bash experiments/cross_rig_attention/run_regression_head_swap.sh --evaluate

# R1-f only, if you want to start with the target rig.
bash experiments/cross_rig_attention/run_regression_head_swap.sh \
  --rigs R1-f --evaluate
```

`--device cuda:1` selects another GPU. `--split test` changes the annotation
split; `--output-dir PATH` changes output location. Per-frame results are
cached, and reruns skip completed frames. Cache metadata checks checkpoint
file identity (path, size and modification time); a changed checkpoint
requires a different output directory. `--evaluate` requires the full split,
not `--max-frames`; it runs the repository's official dataset evaluator for
native and swapped predictions. It can take considerably longer than the
forward pass. Without `--evaluate`, the script still reports GT-car center
errors, but those are **not mAP**.

## Outputs

Default directory: `experiments/cross_rig_attention/output/regression_head_swap_val/`.

- `R1f/r1/<token>.pt`, `R1f/r1f/<token>.pt`, and the analogous `R1/` files:
  resumable per-frame GT-car records plus both sets of postprocessed detections.
- `summary.json` and per-rig `summary_R1f.json`, `summary_R1.json`:
  car counts, native/swapped mean center error on the **same host query**,
  fraction of cars improved, query-reassignment frequency, and error after
  allowing Hungarian reassignment. With `--evaluate`, these also contain
  official AP metrics for the full validation set.
- `regression_head_swap.png`/`.svg`: mean change in same-query GT-car center
  error when the other checkpoint's head is used. Negative values mean the
  swap helped.
- `regression_head_swap_r1f_car_ap.svg`: official **car AP by 0.5, 1, 2,
  and 4 m threshold** on R1-f frames, comparing native and swapped heads
  for both decoder-state models. It also gives exact values and percentage-
  point changes. A PNG is generated when `rsvg-convert` is available.

To regenerate the AP plot from saved `summary.json` without GPU or evaluation:

```bash
python experiments/cross_rig_attention/plot_regression_head_swap_ap.py
```

Hungarian assignment uses the final output and GT **only for analysis**. The
`swapped_fixed_query` metric scores the donor head on the same native query,
avoiding a query-selection confound. The `swapped_reassigned` metric shows
what changes when the swapped boxes are associated independently. Neither
GT-matched metric substitutes for AP; use `--evaluate` for detection-level
evidence.

## How to interpret it

If the R1-f head improves R1 states on R1-f while the R1 head worsens R1-f
states, the output mapping is a plausible contributor. If both swaps fail,
that is **not** proof that the head is innocent: independently trained decoder
channels may be incompatible with the other model's MLP. If a swap helps
R1-f but harms R1, it does not solve cross-rig retention. Compare strict car
AP thresholds and source-rig AP before choosing an architecture.
