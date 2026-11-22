# 3D Differences in Detection on PCCR

`analyze_dnd3d_pccr.py` compares saved final PETR detections from the R1-f
native and R1 cross-rig checkpoints on the same R1-f validation frames. It is
a PCCR/BEV-center-distance adaptation of the GT-centred *Differences in
Detection* (DnD) idea, with a separate ranked-FP ledger. It does not change
PETR inference, scores, labels, boxes, or the official evaluator.

From the PETR repository root, run:

```bash
python3 experiments/cross_rig_attention/analyze_dnd3d_pccr.py
```

This uses existing `output/r1f_val/{R1-f,R1}/formatted/results_pccr.json`
and PCCR annotation tables. No GPU or new inference is needed. It checks its
replay of AP@4 m against the stored evaluator metric for **all nine classes**.

Outputs in `output/r1f_val/dnd3d_car_4m/`:

- `summary.json`: all-class paired GT transitions, diagnostic candidate
  categories, car FP checkpoints, and AP-replay validation.
- `paired_gt.csv`: one row per evaluator-eligible GT (11,621 here), with
  native/cross official TP state and nearby-box diagnostic details.
- `R1_car_fp_before_50pct.csv` and `R1f_car_fp_before_50pct.csv`: each
  model's car FPs through its own 50% recall checkpoint, including rank,
  score, proximity category, and possible missed-GT candidate link.
- `dnd3d_car_summary.svg` and, if `rsvg-convert` is installed, PNG.

The `both_tp`, `native_only`, `cross_only`, `neither` partition uses only
official class-specific, descending-score greedy matching with strict
BEV-center distance `<4 m`. For cars it is **2,583 / 857 / 47 / 104**.

For unmatched GTs, the script searches final-output boxes within 8 m by
default. Official TPs cannot be reused. Remaining boxes are assigned
one-to-one across **all nine GT classes**, by nearest distance then higher
score. Candidate labels are:

| Label | Nearby-box diagnostic |
| --- | --- |
| `Cls` | Different class, distance <4 m |
| `Loc` | Same class, distance 4–8 m |
| `Both` | Different class, distance 4–8 m |
| `Miss` | No final-output box within 8 m |
| `Competition` | Nearby box is an official TP elsewhere or was assigned to another GT |

These are geometric candidate labels, **not proven object identities or
causal errors**. A nearby truck prediction does not prove a car was
misclassified as a truck. Assignment is global across classes, unlike the
older car-only TIDE audit, so the category counts need not match it.
`--near-radius` changes only the diagnostic window, never official AP.

The ranked-FP ledger is prediction-centred. A box may be both a car FP and a
candidate for an unmatched GT. GT partitions and FP counts are complementary,
not additive decompositions of the AP gap. At each recall checkpoint, both
models have the same TP count but can reach it at very different ranks.

To vary the diagnostic window without overwriting the default output:

```bash
python3 experiments/cross_rig_attention/analyze_dnd3d_pccr.py \
  --near-radius 6 --output-dir experiments/cross_rig_attention/output/r1f_val/dnd3d_car_4m_radius6
```

This is an offline PCCR-native adaptation, not a modification of the cloned
2D COCO/TIDE evaluator. COCO IoU matching/AP would not represent PETR/PCCR's
center-distance AP and would make the comparison invalid.
