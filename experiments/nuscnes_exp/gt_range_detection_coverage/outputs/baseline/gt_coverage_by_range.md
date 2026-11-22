# Baseline PETR GT detection coverage by ego range

| GT ego range | Total GT | Matched <4 m | No match <4 m | Error <0.5 m | Error 0.5–1 m | Error 1–2 m | Error 2–4 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–10 m | 15,769 | 15,491 | 278 | 10,315 | 3,414 | 1,431 | 331 |
| 10–20 m | 36,649 | 35,503 | 1,146 | 17,130 | 10,983 | 5,861 | 1,529 |
| 20–30 m | 34,418 | 32,571 | 1,847 | 9,591 | 11,051 | 9,332 | 2,597 |
| 30–40 m | 22,737 | 21,273 | 1,464 | 4,557 | 6,807 | 7,674 | 2,235 |
| 40–50 m | 12,288 | 11,374 | 914 | 1,793 | 3,270 | 4,804 | 1,507 |
| >50 m | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Total | 121,861 | 116,212 | 5,649 | 43,386 | 35,525 | 29,102 | 8,199 |

| GT ego range | GT recall <0.5 m | GT recall <1 m | GT recall <2 m | GT recall <4 m |
|---|---:|---:|---:|---:|
| 0–10 m | 65.4% | 87.1% | 96.1% | 98.2% |
| 10–20 m | 46.7% | 76.7% | 92.7% | 96.9% |
| 20–30 m | 27.9% | 60.0% | 87.1% | 94.6% |
| 30–40 m | 20.0% | 50.0% | 83.7% | 93.6% |
| 40–50 m | 14.6% | 41.2% | 80.3% | 92.6% |
| >50 m | — | — | — | — |
| Total | 35.6% | 64.8% | 88.6% | 95.4% |

GT and predictions use the official nuScenes evaluation filters. The four localization bands plus the no-match group partition all GT in each range bin. Recall percentages use all eligible GT in that range bin as denominator. Matching is replayed independently at each threshold.
