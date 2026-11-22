# R1-f test: car AP and false-positive audit

Run from the PETR root (no GPU or model inference required):

```bash
python3 experiments/cross_rig_attention/audit_r1f_test_car_ap_fp.py
```

The script reads the already saved R1 and R1-f final predictions under
`results/petr_r50dcn_gridmask_p4_800x320_pccr/<model>/cross_rig/evaluations/R1-f/`
and the R1-f **test** annotation tables. It reproduces official car AP@4m
from the full confidence-ranked car predictions and refuses to report FP
results if the AP or TP identities do not agree.

The output is `experiments/cross_rig_attention/output/r1f_test/car_ap_fp_audit/`:

- `summary.json`: official car AP at 0.5/1/2/4 m, their car-class mean,
  AP@4m precision/recall checkpoints, maximum recall, and FP proximity counts.
- `precision_recall.svg`: the AP@4m precision–recall comparison.
- `ranked_car_predictions.csv`: every car prediction, its official TP/FP
  decision, rank, confidence and nearest GT diagnostics. This is the TP/FP
  ledger used by the test attention scripts.
- `R1_high_score_car_fps.csv` and `R1f_high_score_car_fps.csv`: FPs above
  `--score-threshold` (default 0.60) with TIDE-inspired GT proximity labels.
- `R1_fp_before_25pct_recall.csv` and `R1f_fp_before_25pct_recall.csv`:
  high-ranked FP prefixes, independent of the score threshold.

The FP proximity labels use evaluator GT boxes across classes. They are
descriptive, not proof that a prediction belongs to a particular object.
This script does not reconstruct query attention; use
`audit_r1_fp_attention_regions.py --test-set r1f_test` for that GPU step.
