# Paired PETR–StreamPETR GT comparison

Uses the same official evaluator-filtered nuScenes validation GT and saved model predictions. No inference was run. Matching is replayed independently at each threshold in descending score order.

These are GT recall and center-error comparisons, **not AP**. AP also depends on false positives and ranking. Visibility is the nuScenes six-camera panorama annotation band.

Total eligible GT: 121,861. At <4 m: PETR 116,212, StreamPETR 117,456, both 114,014, PETR only 2,198, StreamPETR only 3,442.

| Visibility | GT | PETR <4 m | StreamPETR <4 m | Gain (pp) | Stream only | PETR only |
|---|---:|---:|---:|---:|---:|---:|
| 0-40% | 23,437 | 88.8% | 91.5% | +2.6 | 1,496 | 878 |
| 40-60% | 12,779 | 95.2% | 95.9% | +0.7 | 386 | 297 |
| 60-80% | 17,815 | 95.3% | 96.1% | +0.8 | 466 | 320 |
| 80-100% | 67,830 | 97.7% | 98.2% | +0.6 | 1,094 | 703 |

See `summary.csv` for class × distance × visibility at all thresholds and `paired_gt.csv` for each GT and the two fixed <4 m center errors. Differences within the “both” cohort are descriptive; cohort membership changes at tighter thresholds.
