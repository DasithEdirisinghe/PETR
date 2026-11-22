# PETR AP threshold gap audit

Official nuScenes AP is read from saved evaluator metrics. The prediction and GT transition counts are from a replay of the evaluator-filtered predictions, validated against the existing independent GT match replay at all four thresholds. No inference was run.

| Class | AP@0.5 | AP@4 | Gap | 4 m GT matches | Lost GT at 0.5 | Radial oracle AP@0.5 | Full-center oracle AP@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| barrier | 3.4% | 68.7% | 65.3 pp | 14,768 | 9,664 | 25.1% | 68.7% |
| bicycle | 5.2% | 53.4% | 48.1 pp | 1,558 | 944 | 38.8% | 53.4% |
| bus | 2.1% | 69.0% | 66.9 pp | 1,835 | 1,406 | 36.4% | 69.0% |
| car | 12.7% | 81.9% | 69.2 pp | 52,177 | 32,608 | 59.1% | 81.9% |
| construction_vehicle | 0.0% | 19.5% | 19.5 pp | 1,036 | 918 | 2.3% | 19.5% |
| motorcycle | 3.6% | 56.0% | 52.4 pp | 1,545 | 931 | 36.9% | 56.0% |
| pedestrian | 6.3% | 72.2% | 65.8 pp | 22,615 | 13,310 | 44.0% | 72.2% |
| traffic_cone | 16.2% | 77.2% | 61.0 pp | 10,201 | 4,899 | 51.5% | 77.2% |
| trailer | 0.0% | 35.7% | 35.7 pp | 1,912 | 1,671 | 2.2% | 35.7% |
| truck | 1.2% | 57.3% | 56.1 pp | 8,565 | 6,475 | 31.4% | 57.3% |

## Characteristics of GT matches lost at 0.5 m

Percentages in this table use lost GT matches as the denominator. The radial and tangential columns are mean absolute error on the original 4 m matched prediction.

| Class | Lost / 4 m matched | 0–40% visible | Other GT closer | Abs radial | Abs tangential |
|---|---:|---:|---:|---:|---:|
| barrier | 65.4% | 20.4% | 20.3% | 1.06 m | 0.76 m |
| bicycle | 60.6% | 21.5% | 14.7% | 1.45 m | 0.54 m |
| bus | 76.6% | 6.3% | 0.0% | 1.45 m | 0.62 m |
| car | 62.5% | 26.0% | 4.3% | 1.37 m | 0.59 m |
| construction_vehicle | 88.6% | 21.7% | 0.5% | 1.64 m | 0.92 m |
| motorcycle | 60.3% | 22.3% | 15.0% | 1.55 m | 0.58 m |
| pedestrian | 58.9% | 25.9% | 32.5% | 1.47 m | 0.67 m |
| traffic_cone | 48.0% | 14.4% | 25.4% | 1.28 m | 0.65 m |
| trailer | 87.4% | 36.9% | 6.8% | 1.53 m | 1.22 m |
| truck | 75.6% | 15.5% | 1.6% | 1.43 m | 0.68 m |

## Where 4 m GT matches lose their 0.5 m match

Each cell is lost GT / GT matched at 4 m, with the lost count in parentheses. A dash means no eligible GT in that range.

| Class | 0-10m | 10-20m | 20-30m | 30-40m | 40-50m |
|---|---:|---:|---:|---:|---:|
| barrier | 55.2% (1,720) | 62.1% (3,976) | 75.6% (3,968) | — | — |
| bicycle | 48.2% (108) | 49.7% (298) | 72.1% (351) | 75.4% (187) | — |
| bus | 55.0% (66) | 57.9% (237) | 80.2% (365) | 85.2% (413) | 88.8% (325) |
| car | 18.7% (1,030) | 42.9% (5,173) | 69.5% (9,598) | 79.3% (9,613) | 83.1% (7,194) |
| construction_vehicle | 73.3% (33) | 84.8% (251) | 91.0% (212) | 88.4% (190) | 93.9% (232) |
| motorcycle | 37.0% (78) | 53.8% (257) | 65.9% (348) | 75.6% (248) | — |
| pedestrian | 34.2% (981) | 52.4% (4,252) | 65.9% (4,326) | 74.1% (3,751) | — |
| traffic_cone | 28.9% (781) | 49.7% (2,494) | 65.5% (1,624) | — | — |
| trailer | 72.0% (18) | 75.9% (240) | 88.6% (523) | 89.3% (528) | 92.8% (362) |
| truck | 54.9% (361) | 66.2% (1,195) | 76.8% (1,665) | 80.5% (1,786) | 85.5% (1,468) |

## Prediction transitions

These counts include all evaluator-filtered predictions, not only predictions matched at 4 m. A 4 m TP that fails at 0.5 m usually has no eligible GT within 0.5 m; a small subset is blocked by an earlier, higher-confidence match.

Retention percentages use original 4 m TPs as denominator. Competition percentage uses 4 m TP → 0.5 m FP as denominator.

| Class | FP at 4 m | TP at 4 m | TP4 → TP0.5 | TP4 → FP0.5 | Competition among lost |
|---|---:|---:|---:|---:|---:|
| barrier | 34,041 | 14,768 | 3,888 (26.3%) | 10,880 (73.7%) | 48 (0.4%) |
| bicycle | 54,040 | 1,558 | 438 (28.1%) | 1,120 (71.9%) | 36 (3.2%) |
| bus | 20,311 | 1,835 | 379 (20.7%) | 1,456 (79.3%) | 0 (0.0%) |
| car | 424,123 | 52,177 | 16,877 (32.3%) | 35,300 (67.7%) | 24 (0.1%) |
| construction_vehicle | 28,338 | 1,036 | 62 (6.0%) | 974 (94.0%) | 0 (0.0%) |
| motorcycle | 34,341 | 1,545 | 403 (26.1%) | 1,142 (73.9%) | 15 (1.3%) |
| pedestrian | 327,419 | 22,615 | 6,323 (28.0%) | 16,292 (72.0%) | 558 (3.4%) |
| traffic_cone | 72,026 | 10,201 | 4,058 (39.8%) | 6,143 (60.2%) | 206 (3.4%) |
| trailer | 28,674 | 1,912 | 102 (5.3%) | 1,810 (94.7%) | 0 (0.0%) |
| truck | 146,743 | 8,565 | 1,713 (20.0%) | 6,852 (80.0%) | 1 (0.0%) |

## Candidate indicators in retained and lost 4 m TPs

Each percentage uses its own outcome group as denominator: TP4 → TP0.5 or TP4 → FP0.5. Indicators overlap and are descriptive, so their percentages must not be added. Indicators describe the original 4 m assigned GT and prediction. A retained prediction can match a different GT at 0.5 m. Confidence rank is among all evaluator-filtered predictions of that class. Full category counts are in `tp_transition_factors.csv`; within-category loss rates are in `tp_factor_outcome_rates.csv`.

| Class | 0–40% visible: retained | 0–40% visible: lost |
|---|---:|---:|
| barrier | 400 (10.3%) | 2,040 (18.8%) |
| bicycle | 28 (6.4%) | 225 (20.1%) |
| bus | 2 (0.5%) | 92 (6.3%) |
| car | 943 (5.6%) | 9,116 (25.8%) |
| construction_vehicle | 8 (12.9%) | 210 (21.6%) |
| motorcycle | 47 (11.7%) | 227 (19.9%) |
| pedestrian | 815 (12.9%) | 3,879 (23.8%) |
| traffic_cone | 167 (4.1%) | 805 (13.1%) |
| trailer | 15 (14.7%) | 651 (36.0%) |
| truck | 100 (5.8%) | 1,051 (15.3%) |

| Class | GT ≥30 m: retained | GT ≥30 m: lost |
|---|---:|---:|
| barrier | 0 (0.0%) | 0 (0.0%) |
| bicycle | 22 (5.0%) | 226 (20.2%) |
| bus | 75 (19.8%) | 776 (53.3%) |
| car | 2,476 (14.7%) | 18,301 (51.8%) |
| construction_vehicle | 16 (25.8%) | 446 (45.8%) |
| motorcycle | 33 (8.2%) | 295 (25.8%) |
| pedestrian | 647 (10.2%) | 4,418 (27.1%) |
| traffic_cone | 0 (0.0%) | 0 (0.0%) |
| trailer | 32 (31.4%) | 949 (52.4%) |
| truck | 421 (24.6%) | 3,514 (51.3%) |

| Class | Other GT closer: retained | Other GT closer: lost |
|---|---:|---:|
| barrier | 338 (8.7%) | 2,010 (18.5%) |
| bicycle | 15 (3.4%) | 148 (13.2%) |
| bus | 0 (0.0%) | 0 (0.0%) |
| car | 103 (0.6%) | 1,547 (4.4%) |
| construction_vehicle | 2 (3.2%) | 4 (0.4%) |
| motorcycle | 20 (5.0%) | 143 (12.5%) |
| pedestrian | 973 (15.4%) | 4,381 (26.9%) |
| traffic_cone | 380 (9.4%) | 1,343 (21.9%) |
| trailer | 5 (4.9%) | 120 (6.6%) |
| truck | 4 (0.2%) | 110 (1.6%) |

| Class | Radial dominant: retained | Radial dominant: lost |
|---|---:|---:|
| barrier | 2,122 (54.6%) | 7,052 (64.8%) |
| bicycle | 240 (54.8%) | 926 (82.7%) |
| bus | 235 (62.0%) | 1,194 (82.0%) |
| car | 11,606 (68.8%) | 29,090 (82.4%) |
| construction_vehicle | 32 (51.6%) | 713 (73.2%) |
| motorcycle | 244 (60.5%) | 971 (85.0%) |
| pedestrian | 3,801 (60.1%) | 13,030 (80.0%) |
| traffic_cone | 2,539 (62.6%) | 4,841 (78.8%) |
| trailer | 54 (52.9%) | 1,131 (62.5%) |
| truck | 1,047 (61.1%) | 5,352 (78.1%) |

| Class | Top 10% score rank: retained | Top 10% score rank: lost |
|---|---:|---:|
| barrier | 1,414 (36.4%) | 2,818 (25.9%) |
| bicycle | 415 (94.7%) | 895 (79.9%) |
| bus | 360 (95.0%) | 1,122 (77.1%) |
| car | 15,873 (94.1%) | 23,864 (67.6%) |
| construction_vehicle | 45 (72.6%) | 594 (61.0%) |
| motorcycle | 378 (93.8%) | 899 (78.7%) |
| pedestrian | 5,918 (93.6%) | 12,824 (78.7%) |
| traffic_cone | 3,370 (83.0%) | 3,571 (58.1%) |
| trailer | 82 (80.4%) | 1,176 (65.0%) |
| truck | 1,578 (92.1%) | 5,088 (74.3%) |


## Reading the outputs

`class_range_summary.csv` contains class × GT-range counts, visibility, nearby-GT, confidence, and radial/tangential diagnostics. `gt_transitions.csv` contains each eligible GT. `prediction_transitions.csv` counts every evaluator-filtered prediction, including FPs and duplicate/competition flags. `tp_transition_factors.csv` has all category counts for both 4 m TP outcomes.

A 4 m TP can become an FP at 0.5 m while its GT becomes unmatched; these are two views of the same event, not additive causes. The duplicate flag means another eligible GT lies within the threshold but was already taken in confidence order; it does not prove object identity confusion. Visibility and nearby-GT rates are associations, not causal explanations. Official AP gaps cannot be apportioned by adding GT counts or oracle gains.

Distance bins are GT-defined, so FP predictions without a matched GT are summarized by class rather than forced into a GT bin. A separate bin-restricted AP would define a different evaluation population and is not presented as a contribution to global AP. Saved PETR query features are not linked to every evaluated prediction in these artifacts, so query-level claims are outside this run.
