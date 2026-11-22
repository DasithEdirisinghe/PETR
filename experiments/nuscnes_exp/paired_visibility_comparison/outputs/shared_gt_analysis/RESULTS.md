# PETR and StreamPETR on the same <4 m matched GT

This analysis keeps **only GT annotations matched by both models within 4 m**. The denominator is 114,014 GT annotations across validation frames. Each model's score-ordered 4 m match is used; predictions are not directly paired by query identity.

![Error-band transitions by GT visibility](shared_gt_error_transitions.svg)

![Matched-detection confidence distributions](shared_gt_scores.svg)

| GT visibility | Shared GT | Mean center error PETR → StreamPETR | Stream lower error | Median raw score PETR → StreamPETR | Median within-class rank PETR → StreamPETR |
|---|---:|---:|---:|---:|---:|
| 0-40% | 19,943 | 1.858 → 1.461 m | 63.0% | 0.311 → 0.350 | 21.6 → 23.9 percentile |
| 40-60% | 11,863 | 1.544 → 1.234 m | 62.5% | 0.395 → 0.451 | 33.5 → 34.0 percentile |
| 60-80% | 16,662 | 1.356 → 0.995 m | 65.7% | 0.509 → 0.623 | 44.5 → 46.4 percentile |
| 80-100% | 65,546 | 1.012 → 0.768 m | 64.9% | 0.819 → 0.800 | 63.8 → 63.1 percentile |

Raw score magnitudes are **not directly calibrated across models**. The within-class percentiles compare each detection's relative priority among **shared matched TPs only**; neither plot accounts for false positives. The localization comparison is conditional on both models matching a GT within 4 m and does not describe GT missed by either model.

See `shared_gt_visibility_summary.csv` for class × visibility results and `shared_gt_error_transitions.csv` for the exact 4×4 matrices.
