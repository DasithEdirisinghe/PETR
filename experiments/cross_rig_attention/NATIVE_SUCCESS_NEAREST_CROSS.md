# Native-success GT cars: nearest cross-rig output

This is the paired experiment requested for classification versus localization.
It reads the four saved **standard PETR validation detections**. It does not
change inference, decoder top-k, scores, boxes, or official AP.
GT car centers and indices come from the existing
`output/{r1_val,r1f_val}/analysis/car_per_gt.csv` files produced by
`run_car_model_comparison.sh`; those files must be present.

Run from the PETR root in the `petr` Python environment:

```bash
python experiments/cross_rig_attention/native_success_nearest_cross.py
```

For each validation rig, car-labelled native predictions are processed in
descending confidence, each matching the closest as-yet-unmatched GT car if
its BEV center distance is strictly less than 4 m. This mirrors the car AP
evaluator's true-positive matching at 4 m and fixes the GT cohort before the
cross-rig outputs are examined.

For each selected GT car, the script finds the **closest final cross-rig box of
any class**, even if it is more than 4 m away. Equal-distance boxes are broken
by higher output score, then result order. If no box exists in that frame, the
cross-output fields are blank. A single cross-rig box may be the nearest to
multiple selected GT cars; `cross_nearest_shared_by_gt_count` exposes this.

Outputs in `output/native_success_nearest_cross/`:

- `per_gt_nearest_cross.csv`: GT location, native matched box score/distance,
  nearest cross-rig box coordinates, class, score, distance, and 1/4 m flags.
  It also records a separate **one-to-one car-labelled cross-rig match within
  4 m**, if one exists.
- `summary.json`: native-success cohort sizes and the distribution of nearest
  cross-rig classes and distances for each target rig. Paired native/cross
  mean center errors use only GT cars matched by both models under the
  one-to-one car rule.

The nearest box is an observational proxy for what the cross-rig model output
for the car. If it is distant or the frame is crowded, it may belong to a
different object. The per-GT CSV retains the actual distance and shared-box
count so those cases can be inspected. **Do not interpret the mean distance to
an unrestricted nearest box as localization error for the same object.** Use
the paired one-to-one car matches for that comparison. These diagnostic counts do not replace
the unchanged official AP calculation.
