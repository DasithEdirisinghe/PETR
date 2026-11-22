# Full-validation R1-f car failure-mode breakdown

This diagnostic answers: **for every R1-f validation GT car, especially those the R1-f checkpoint detects, what does the R1 checkpoint output on that same R1-f frame?** It uses the already saved post-processed predictions from both checkpoints and the R1-f GT file. It does **not** run inference or evaluate again.

```bash
bash experiments/cross_rig_attention/run_r1f_car_failure_modes.sh
```

Default inputs:

- `data/pccr/R1-f/R1-f_infos_val.pkl`
- `experiments/cross_rig_attention/output/r1f_val/{R1,R1-f}/formatted/results_pccr.json`
- The corresponding saved `metrics_summary.json` files for **official** car AP.

Results: `experiments/cross_rig_attention/output/r1f_val/failure_modes/`.

To test sensitivity to diagnostic choices:

```bash
bash experiments/cross_rig_attention/run_r1f_car_failure_modes.sh \
  --score-threshold 0.35 --candidate-score-floor 0.05 --nearby-radius 8
```

The car score cutoff (default 0.35) separates high-score from low-score car candidates. The candidate floor (default 0.05) ignores weaker saved outputs when describing nearby candidates. The radius (default 8 m) limits which outputs can be tentatively associated with a GT car; it must exceed the largest AP center threshold, 4 m. Set `--output-dir` to compare parameter choices without replacing prior plots. Input paths can be overridden with `--ann-file`, `--r1-results`, `--r1f-results`, `--r1-metrics`, and `--r1f-metrics`.

## Method

The GT-car set is the same one used by the earlier car comparison: valid R1-f cars within 50 m in ego coordinates, excluding zero-point objects when point counts exist. For each frame, every saved output box above the candidate floor is *owned by its nearest GT car* if its global BEV center is within the radius. This prevents one candidate from being counted for two cars, but it is not a query-identity match and can be ambiguous in crowded scenes.

At each 0.5, 1, 2, and 4 m threshold, the R1-f-detected cohort is formed with **one-to-one, confidence-ordered, car-only** matching at car score ≥ 0.35. We report outcomes for both that cohort and **all** GT cars. For R1, each car receives one exclusive category, in this priority order:

1. High-score car output with center error below threshold: **car correct**.
2. High-score car output within the candidate radius, but outside threshold: **car score passes, center fails**.
3. Low-score car output with center below threshold: **center passes, car score low**.
4. Low-score car output outside threshold: **both score and center fail**.
5. Only other-class saved outputs within **2 m**: **nearby other-class candidate only**. The tighter limit avoids attributing unrelated signs or people inside the broader candidate radius to a car.
6. No qualifying car output inside the radius or other-class output within 2 m: **no qualifying nearby output**.

The priority makes the categories non-overlapping, but a car may have several competing outputs (e.g. a high-score mislocalized car and a low-score correctly localized one). The per-car CSV retains the chosen candidate and nearest car candidate; examine it before making a claim about such cases.

## Read the outputs

- `outcome_overview.png/.svg`: stacked percentages and counts for every threshold, separately for **all GT cars** and **R1-f-detected GT cars**. This is the main answer to “what did R1 do?”
- `ap_and_coverage.png/.svg`: official car AP alongside separate fixed-score diagnostic GT coverage. These are intentionally distinct metrics.
- `distance_breakdown_1m.png/.svg`: R1 categories at 1 m for R1-f-detected cars, split by ego distance.
- `summary.json` and `README_RESULTS.md`: exact counts, percentages, official AP values, and interpretation boundaries.
- `per_car_threshold_outcomes.csv`: each physical GT car, four thresholds, R1/R1-f diagnostic hit flags, R1 category, and candidate score/distance/class for follow-up.

**Important limit:** post-processed JSON stores box class labels and scores, not each query's full class-logit vector. A nearby other-class box is *not proof* that a particular query meant for the car was misclassified. Similarly, a car-labelled box 5 m away could belong to a neighboring car; nearest-GT ownership is only a controlled diagnostic association. The categories can distinguish observed class-labelled candidate and localization patterns, but cannot by themselves prove the internal failure mechanism. Official AP includes false positives and score ranking; these per-GT categories do not sum to AP.
