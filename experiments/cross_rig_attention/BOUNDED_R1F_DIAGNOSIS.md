# Bounded R1-f car AP and attention diagnosis

Run from the PETR root with any Python 3 interpreter (no third-party package is
needed by this *analysis* script):

```bash
python3 experiments/cross_rig_attention/run_bounded_r1f_diagnosis.py
```

The script reads saved R1-f validation predictions from both checkpoints and
saved attention scorecard CSVs. It does **not** rerun or alter PETR inference.
The required scorecard was created by `score_r1f_car_queries.py` using the normal
forward pass plus passive hooks; GT is used only afterward to associate queries
and define car-intersecting camera rays. The output is
`experiments/cross_rig_attention/output/r1f_val/bounded_tide3d_attention/`.
Use `--output-dir` for an independent run and `--near-radius` (default 8 m)
for a diagnostic sensitivity check. The official car AP threshold remains 4 m.

## AP audit

The script first reconstructs the official evaluator-eligible GT and checks
that the replayed R1 and R1-f car AP@4 m agrees with the saved official AP to
`1e-4`. If this fails, it stops. All-class GT and final predictions are used
to investigate car errors. The mutually exclusive prediction-side labels are:

- `Cls`: unmatched car output within 4 m of one other-class GT not already
  detected by its own class;
- `Loc`: unmatched car output 4–8 m from an unmatched GT car;
- `Both`: unmatched car output 4–8 m from an other-class GT;
- `Dupe`: unmatched car output within 4 m of an already-matched GT car;
- `Bkgd`: no evaluator-eligible GT within 8 m;
- `Ambiguous`: competing GTs, detected other-class GT, or other conflicting
  associations.

For missed GT cars, a free other-class prediction within 4 m gives a `Cls`
candidate; at 4–8 m it gives `Both`. A linked displaced car FP gives `Loc`.
Without a free candidate within 8 m it is `Missed`. Ambiguous cases remain
explicit. These are TIDE-*inspired* center-distance categories, not TIDE's
original IoU categories and not proofs of internal network causation. In
particular, `Bkgd` means **no evaluator-eligible GT nearby**, not visually
verified empty background. The six TIDE types include `Missed`, a false
negative rather than a false positive.

`summary.json` gives full-list counts and counts before 25% recall. Isolated
oracle APs modify one type at a time and replay car AP: delete the respective
car FPs, relabel/relocate free missed-car candidates, or (for `Missed`) reduce
the GT-positive denominator without changing the ranked TP/FP sequence, as in
TIDE. These are diagnostic hypothetical changes, not feasible PETR results.
Their AP gains **do not add** and must not be interpreted as exact attribution
of the 47.96-point gap. The complete per-FP and per-FN CSVs preserve identities
and reasons for manual inspection. `bounded_diagnosis.svg` summarizes the two
models and the attention comparison.

## Attention

The primary attention score is last-decoder-layer GT-ray AUROC over **all
cameras pooled**: the chance a GT-car-intersecting feature ray receives more
weight than a non-intersecting valid ray. Both models are measured on the same
3,591 GT cars, with a paired R1-f-minus-R1 difference and a scene-clustered
bootstrap confidence interval (2,000 fixed-seed replicates). For reference,
the CSV also gives each model's normalized attention mass on GT rays,
whole-map overlap, joint GT-ray attention mass, and within-GT-ray overlap.
AUROC for each model does *not* measure identical cell selection; spatial
overlap does. None of these scores alone proves classification or localization
failure. The GT ray mask is an oriented-box projection proxy, not a visible
object segmentation mask. The compared query for each GT is the model's
post-hoc GT-assigned query; this need not be an emitted AP detection.

`attention_region_explainer.svg` (and `.png` where `rsvg-convert` is installed)
is a one-score explanation of last-layer, all-camera GT-car attention
enrichment. It presents the equation and scale before the two model medians.
To regenerate just this plot without rerunning the AP audit:

```bash
python3 experiments/cross_rig_attention/plot_bounded_r1f_attention.py
```

`attention_mass_explainer.svg` (and `.png` where available) gives the bounded
alternative: the **percentage** of final-layer attention on the GT-car region,
from 0% to 100%. It uses the same cars and masks. To regenerate it:

```bash
python3 experiments/cross_rig_attention/plot_bounded_r1f_attention_mass.py
```
