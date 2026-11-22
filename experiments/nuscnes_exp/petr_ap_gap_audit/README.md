# PETR AP threshold gap audit

Run from the PETR repository root:

```bash
python3 experiments/nuscnes_exp/petr_ap_gap_audit/audit.py
```

This reads the existing, evaluator-filtered PETR validation predictions,
nuScenes metadata, the prior GT-level match replay, and official saved PETR
and oracle metrics. It does not run model inference and uses Python's standard
library only. It validates replayed GT matches at 0.5, 1, 2, and 4 m against
the existing independent replay before writing results.

Outputs in `outputs/`:

- `RESULTS.md`: official class AP gaps and localization oracle ceilings.
- `class_range_summary.csv`: class × 10 m GT-range counts and characteristics
  of matches lost when the tolerance tightens.
- `gt_transitions.csv`: each eligible GT, its match transition, visibility,
  confidence, radial/tangential errors, and nearest-other-GT diagnostic.
- `prediction_transitions.csv`: all predictions by TP/FP transition and
  confidence, including duplicate/competition flags.
- `tp_transition_factors.csv`: category counts and within-outcome percentages
  for retained and lost original 4 m TPs, by class. Categories include GT
  visibility, distance, nearby same-class GTs, radial direction and dominance,
  score rank, and original center-error band.
- `tp_factor_outcome_rates.csv`: retained/lost counts and loss rate among all
  original 4 m TPs in each class × factor category.

After the full audit has run once, refresh only the transition tables and
report from its saved CSVs:

```bash
python3 experiments/nuscnes_exp/petr_ap_gap_audit/summarize_transitions.py
```

To compare original 4 m TP predictions associated with 80–100% visible GTs
that remain TPs versus become FPs at 0.5 m, using the saved audit CSV:

```bash
python3 experiments/nuscnes_exp/petr_ap_gap_audit/compare_visible_outcomes.py
```

This writes `outputs/high_visibility_comparison.md` and
`outputs/high_visibility_outcomes_by_class_range.csv` without running inference.

Distance-bin summaries use GT range. FPs without an assigned GT have no GT
range and appear in the class-level prediction table. The AP figures are the
official full-validation-set class AP values, not sums of per-bin counts.
Visibility and nearby-GT indicators describe associations, not causes.
