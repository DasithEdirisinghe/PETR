# R1 validation: car diagnostic

Normal PETR inference from two independently trained checkpoints on the same R1 frames.

- 3378 GT cars. At score ≥ 0.35 and BEV center < 4.0 m, R1 matches 2295 (67.9%), while R1-f matches 1091 (32.3%).
- At 20–30 m: R1 matches 550/699 (78.7%), R1-f matches 97/699 (13.9%).
- On the 1007 cars matched by both: mean center error R1 0.57 m vs R1-f 1.68 m; mean signed ego-radial error R1 +0.12 m vs R1-f -1.33 m.

Interpretation: compare missed cars by distance and signed BEV center bias on cars found by both models. This is observational: it does not isolate camera-ray depth, the attention mechanism, or a causal model component.

Diagnostic matching is not official AP. Official AP uses the evaluator’s ranking and matching rules; see car_coverage_and_error.png for AP at each threshold.
