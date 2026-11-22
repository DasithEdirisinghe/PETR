# Car AP rank audit

This script replays the repository's official PCCR car AP matching from saved
`results_pccr.json` outputs. It uses the existing analysis CSV for evaluator-
valid GT car centers. No inference, decoding, detection score, or box is
changed. The script checks that its replayed AP equals `metrics_summary.json`
before writing a report.

From the PETR root:

```bash
python3 experiments/cross_rig_attention/audit_car_ap_ranking.py
```

Default: R1-f validation frames, both R1 and R1-f checkpoints, 4 m car AP.
The output directory is `output/r1f_val/car_ap_rank_audit/`:

- `ranked_car_predictions.csv`: every car-labelled final prediction in the
  exact score order used by AP, with TP/FP status, score, nearest GT distances,
  cumulative precision, and recall.
- `summary.json`: official and replayed AP, TP/FP totals, median TP scores,
  score-cutoff diagnostics, and FP reasons at 10%, 25%, 50%, and 70% recall.
- `precision_recall.svg`: official-equivalent interpolated car precision–recall
  curves at 4 m, with TP/FP counts at 25% and 50% recall for both models.
- `r1_unassociated_before_25pct_recall.csv`: the early R1 car false positives
  with no GT object within 4 m, including their score, BEV coordinates,
  nearest GT car and distance, and whether native R1-f detected that car.
- `frame_44b18378388647449563a98b86c8c3d6_top10_car_bev.svg`/`.png`:
  one example R1-f frame with the ten highest-score R1 car outputs and
  evaluator-valid GT car boxes, annotated with official TP/FP decisions and
  the ego position/heading from the synchronized keyframe pose.

For the reverse comparison on R1 validation frames, using the same official
AP replay and producing `output/r1_val/car_ap_rank_audit/precision_recall.svg`:

```bash
python3 experiments/cross_rig_attention/audit_car_ap_ranking.py \
  --input-dir experiments/cross_rig_attention/output/r1_val \
  --rig R1 --threshold 4.0
```

To plot another frame after running the rank audit:

```bash
python3 experiments/cross_rig_attention/plot_car_frame_bev.py \
  --sample-token FRAME_TOKEN
```

To inspect the same GT and top-ten prediction boxes in all eight **original,
uncropped** R1-f camera images, using the synchronized ego pose and each
camera's stored extrinsic/intrinsic calibration:

```bash
python3 experiments/cross_rig_attention/plot_car_multiview_projection.py \
  --sample-token FRAME_TOKEN
```

To compare both checkpoints on the **same frame** in one canvas (shared-scale
BEV above the eight original camera views for each model), run:

```bash
python3 experiments/cross_rig_attention/plot_car_frame_model_comparison.py \
  --sample-token FRAME_TOKEN
```

The combined output is `frame_FRAME_TOKEN_R1_vs_R1f_bev_and_cameras.svg`.
The command also writes separate `R1` and `R1f` BEV/camera SVGs so either side
can be inspected at full size. These plots use saved final PETR predictions;
they do not rerun or modify inference. Green boxes are evaluator-valid GT cars;
blue/red boxes are the respective model's official TP/FP car outputs. Camera
visibility changes what appears in each panel; TP/FP still comes from the
official global-BEV 4 m evaluation.

The AP match is exact: highest-scored car prediction first, nearest unmatched
GT car in the same frame, strict center distance less than the chosen
threshold. The FP reason labels are **post hoc diagnostics**:

- `duplicate_or_competing_car`: within threshold of a GT car already taken;
- `near_other_class_gt`: no car GT within threshold, but near another GT class;
- `no_gt_within_threshold`: near neither a car nor another GT class.

Other-class proximity uses the raw annotation table with nonzero point count,
so it is descriptive rather than a formal evaluator assignment. A box with no
nearby GT could be background, a severely misplaced car box, or another
unresolved prediction. The rank ledger makes these cases inspectable.

## TIDE-inspired high-ranked FP audit

To compare **R1 cross-rig** against **R1-f native** on the same R1-f frames,
run the rank audit above, then:

```bash
python3 experiments/cross_rig_attention/audit_tide3d_car_fp.py
```

This validates both replayed car AP@4 m values against the official metrics,
then writes `car_ap_rank_audit/tide3d_car_fp/summary.json`,
`early_fp_categories.svg`, and one `*_early_fp_25pct.csv` per checkpoint.
The CSVs contain every car FP before that model first reaches 25% recall,
including prediction identity, score, distances, and proximity category.

Categories partition FPs into duplicates, within-4-m proximity to another GT
class, nearest car/other GT at 4–8 m or 8–20 m, and no considered GT within
20 m. The 8 m and 20 m bands are diagnostic choices, **not** PETR evaluation
thresholds; proximity does not prove object identity or causal error type.
Other-class GT proximity uses nonzero-point raw annotations, not the official
car-GT list. The script also computes independent **FP suppression oracles**:
the car-AP change if one FP category were deleted from saved final detections,
either before 25% recall or across all ranks. These are post-hoc upper-bound
diagnostics, not changed inference or model scores. Their AP gains are not
additive. This is a TIDE-inspired FP audit, **not** a full 3D implementation of
TIDE classification, localization, and missed-GT oracles.

## Paired all-class car failure audit

For the GT-centred complement and the reverse car-FP class confusion direction:

```bash
python3 experiments/cross_rig_attention/audit_tide3d_car_full.py
```

This reads the two saved `results_pccr.json` outputs, the R1-f evaluation
tables, and the existing car GT ledger. It rebuilds evaluator-valid GT with
the exact nonzero-point and class-range filters, then requires **every class's
AP@4 m** to match `metrics_summary.json` for both models before reporting.
Outputs under `car_ap_rank_audit/tide3d_car_full/` are:

- `paired_car_gt.csv`: one row per GT car and model, including the paired
  native/cross TP transition, chosen unmatched-GT candidate, its class,
  distance, score, and an association-ambiguity check. `gt_index` matches the
  earlier car ledger; `gt_list_index` records the evaluator's annotation-list
  index used internally by this audit.
- `R1_car_fp_before_25pct.csv` and `R1f_car_fp_before_25pct.csv`: evaluator
  car false positives before 25% recall, classified against all eligible GT
  classes. In particular, `car_label_on_other_class_gt_candidate` records the
  nearest non-car GT's class.
- `summary.json`: paired transitions, candidate-category counts, early FP
  class confusion counts, and independently recomputed post-hoc AP scenarios.

The default diagnostic association radius is 8 m; use `--near-radius` to test
sensitivity. Only 4 m is an official matching threshold. Candidate categories
are evidence from final outputs, **not** proof of model intent. The oracle
scenarios repair one selected candidate per missed GT, or relabel car FPs near
other-class GT; they preserve saved scores and rerun the exact score-ranked AP
calculation. They are not model results, full TIDE oracles, or an additive
decomposition of the AP gap. The saved final boxes do not contain query IDs or
all pre-top-300 class logits.

To render the paired GT and early-FP categories together:

```bash
python3 experiments/cross_rig_attention/plot_tide3d_car_summary.py
```

This writes `paired_car_failure_summary.svg` beside `summary.json`. The figure
labels the default 8 m diagnostic radius; rerun the plot only for that radius.

Use `--threshold 0.5`, `1.0`, or `2.0` for the other official car AP
thresholds. For R1 validation, pass `--rig R1 --input-dir
experiments/cross_rig_attention/output/r1_val`.
