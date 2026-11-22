# PETR localization oracles

Fixed original 4 m greedy matches: 116,212. Only global BEV XY centers of matched predictions were changed. Scores, classes, sizes, heights, and rotations were preserved.

| Oracle | AP@0.5m | AP@1m | AP@2m | AP@4m | mAP | ΔmAP |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5.08% | 21.92% | 45.06% | 59.07% | 32.78% | +0.00 pp |
| radial | 32.77% | 48.67% | 56.76% | 59.09% | 49.32% | +16.54 pp |
| tangential | 8.63% | 26.26% | 47.88% | 59.06% | 35.46% | +2.68 pp |
| full_center | 59.07% | 59.07% | 59.07% | 59.07% | 59.07% | +26.29 pp |

Official evaluator rematches each corrected submission. Unmatched predictions are unchanged. This is an oracle for the original 4 m matched subset, not a claim about an achievable learned correction.
