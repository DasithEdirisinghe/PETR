# Native-to-cross-rig car transitions

This audit reads the **ordinary PETR formatted validation detections** and the
**official saved car AP** for both checkpoints on both rigs. It does not change
inference, decoding, scores, predictions, or the evaluator.

From the PETR repository root, after the four validation evaluations exist:

```bash
python experiments/cross_rig_attention/audit_car_transitions.py
```

The default output is `experiments/cross_rig_attention/output/car_transition_audit/`:

- `summary.json`: counts for each rig and AP threshold (0.5, 1, 2, 4 m),
  plus official native/cross-rig car AP read from `metrics_summary.json`.
- `per_car_transitions.csv`: one row for every GT car correctly detected by
  the native model at that threshold, with its cross-rig outcome and nearby
  final-output scores and distances.

"Correct" uses confidence-ordered, same-class, nearest-unmatched-GT matching
at the evaluator's *strict* center-distance threshold. The transition cohort
is fixed to native-correct GT cars on the same rig and frames. For a cross-rig
miss, each final output within 4 m is assigned to its nearest GT car **for
diagnosis only**; this assignment is separate from official AP.

Categories: `still_correct`; `localization` (car output within 4 m but not
within the chosen threshold); `classification` (only an other-class output
within the threshold); `both` (only other-class output, between the threshold
and 4 m); `competing_outputs` (both car and other-class outputs, or a car
output within the threshold that lost one-to-one matching); and
`no_output_within_4m`. The categories use all saved output scores; there is
no added 0.35 cutoff. No-output and competing-output cases remain unresolved.

These counts are not an additive decomposition of the AP gap. AP also depends
on false positives and score ranking, and a nearby other-class output is only
diagnostic evidence of a possible classification failure. The per-car CSV
supports checking ambiguous or crowded scenes before making causal claims.
