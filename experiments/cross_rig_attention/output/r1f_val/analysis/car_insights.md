# R1-f validation: car diagnostic

Normal PETR inference from two independently trained checkpoints on the same R1-f frames.

- 3591 GT cars. At score ≥ 0.35 and BEV center < 4.0 m, R1 matches 953 (26.5%), while R1-f matches 2181 (60.7%).
- At 20–30 m: R1 matches 54/762 (7.1%), R1-f matches 547/762 (71.8%).
- On the 816 cars matched by both: mean center error R1 1.46 m vs R1-f 0.55 m; mean signed ego-radial error R1 +1.03 m vs R1-f -0.07 m.

Interpretation: the cross-rig loss includes both missed cars (especially beyond 20 m) and an outward BEV center bias on cars found by both models. This is observational: it does not isolate camera-ray depth, the attention mechanism, or a causal model component.

Diagnostic matching is not official AP. Official AP uses the evaluator’s ranking and matching rules; see car_coverage_and_error.png for AP at each threshold.
