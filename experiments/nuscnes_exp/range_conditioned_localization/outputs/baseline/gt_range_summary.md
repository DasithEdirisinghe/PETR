# Baseline PETR localization by GT ego range

| GT ego range | Matched predictions | Share of 4 m matches | Error <0.5 m | Error 0.5–1 m | Error 1–2 m | Error 2–4 m | Mean abs radial | Mean abs tangential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–10 m | 15,491 | 13.3% | 58.7% | 23.2% | 11.6% | 6.5% | 0.50 m | 0.33 m |
| 10–20 m | 35,503 | 30.6% | 38.5% | 27.5% | 21.4% | 12.6% | 0.79 m | 0.39 m |
| 20–30 m | 32,571 | 28.0% | 19.4% | 24.3% | 31.2% | 25.1% | 1.16 m | 0.53 m |
| 30–40 m | 21,273 | 18.3% | 12.2% | 18.4% | 32.1% | 37.3% | 1.44 m | 0.63 m |
| 40–50 m | 11,374 | 9.8% | 8.2% | 14.4% | 31.5% | 45.9% | 1.57 m | 0.79 m |
| >50 m | 0 | 0.0% | — | — | — | — | — | — |

The range-share percentages sum to 100% across rows and use all 4 m matched predictions as the denominator. The four error-band percentages are disjoint and sum to 100% within each populated range bin. The displayed final percentage absorbs rounding to one decimal; JSON and CSV retain unrounded values. Radial and tangential errors are in ego-centered BEV.
