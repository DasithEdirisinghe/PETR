# PETR image geometry and localization

The analysis replays official 4 m matching on saved validation predictions. Every evaluator-eligible GT contributes once to `gt_geometry.csv`; every geometrically visible GT-camera pair contributes once to `camera_views.csv`. The camera with the largest clipped GT box area supplies each GT-level image feature.

Retention columns use the original 4 m TPs as denominator. Coverage uses all evaluator-eligible GT as denominator. These values are not AP. An empty cell means no matched 4 m TP.

| Class | GT range | GT | 4 m TPs | <0.5 m / 4 m TP | <1 m / 4 m TP | Mean |radial| (m) | Mean |tangential| (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| all classes | 0-10 m | 15,769 | 15,491 | 58.7% | 81.9% | 0.50 | 0.33 |
| all classes | 10-20 m | 36,649 | 35,503 | 38.5% | 66.0% | 0.79 | 0.39 |
| all classes | 20-30 m | 34,418 | 32,571 | 19.4% | 43.7% | 1.16 | 0.53 |
| all classes | 30-40 m | 22,737 | 21,273 | 12.2% | 30.6% | 1.44 | 0.63 |
| all classes | 40-50 m | 12,288 | 11,374 | 8.2% | 22.6% | 1.57 | 0.79 |
| barrier | 0-10 m | 3,185 | 3,117 | 33.6% | 66.1% | 0.65 | 0.56 |
| barrier | 10-20 m | 6,955 | 6,401 | 27.0% | 56.7% | 0.81 | 0.58 |
| barrier | 20-30 m | 5,763 | 5,250 | 14.8% | 40.9% | 1.04 | 0.70 |
| bicycle | 0-10 m | 239 | 224 | 47.3% | 72.8% | 0.60 | 0.32 |
| bicycle | 10-20 m | 697 | 599 | 38.1% | 62.4% | 0.91 | 0.37 |
| bicycle | 20-30 m | 581 | 487 | 14.2% | 39.2% | 1.35 | 0.45 |
| bicycle | 30-40 m | 307 | 248 | 8.5% | 30.2% | 1.57 | 0.52 |
| bus | 0-10 m | 120 | 120 | 43.3% | 70.8% | 0.84 | 0.35 |
| bus | 10-20 m | 420 | 409 | 41.8% | 73.6% | 0.70 | 0.28 |
| bus | 20-30 m | 481 | 455 | 17.8% | 40.4% | 1.21 | 0.48 |
| bus | 30-40 m | 595 | 485 | 11.3% | 27.8% | 1.44 | 0.63 |
| bus | 40-50 m | 446 | 366 | 5.5% | 19.7% | 1.67 | 0.73 |
| car | 0-10 m | 5,536 | 5,518 | 80.3% | 95.2% | 0.30 | 0.15 |
| car | 10-20 m | 12,159 | 12,070 | 53.9% | 80.8% | 0.60 | 0.23 |
| car | 20-30 m | 14,059 | 13,812 | 24.7% | 51.3% | 1.04 | 0.42 |
| car | 30-40 m | 12,409 | 12,123 | 13.7% | 32.6% | 1.39 | 0.57 |
| car | 40-50 m | 8,880 | 8,654 | 8.9% | 24.1% | 1.56 | 0.74 |
| construction_vehicle | 0-10 m | 59 | 45 | 22.2% | 53.3% | 0.86 | 0.65 |
| construction_vehicle | 10-20 m | 342 | 296 | 8.4% | 34.5% | 1.27 | 0.66 |
| construction_vehicle | 20-30 m | 375 | 233 | 3.9% | 20.2% | 1.73 | 0.73 |
| construction_vehicle | 30-40 m | 318 | 215 | 5.6% | 17.2% | 1.68 | 0.99 |
| construction_vehicle | 40-50 m | 419 | 247 | 1.6% | 9.3% | 1.87 | 1.27 |
| motorcycle | 0-10 m | 226 | 211 | 53.6% | 80.6% | 0.75 | 0.28 |
| motorcycle | 10-20 m | 520 | 478 | 31.4% | 59.8% | 0.94 | 0.40 |
| motorcycle | 20-30 m | 659 | 528 | 17.0% | 37.1% | 1.33 | 0.48 |
| motorcycle | 30-40 m | 450 | 328 | 9.1% | 22.3% | 1.71 | 0.56 |
| pedestrian | 0-10 m | 2,907 | 2,868 | 51.3% | 78.0% | 0.57 | 0.38 |
| pedestrian | 10-20 m | 8,222 | 8,115 | 30.2% | 57.8% | 0.93 | 0.44 |
| pedestrian | 20-30 m | 6,887 | 6,567 | 15.5% | 36.6% | 1.38 | 0.53 |
| pedestrian | 30-40 m | 5,484 | 5,065 | 9.9% | 28.0% | 1.53 | 0.68 |
| traffic_cone | 0-10 m | 2,798 | 2,706 | 58.1% | 78.8% | 0.58 | 0.36 |
| traffic_cone | 10-20 m | 5,128 | 5,015 | 35.7% | 63.3% | 0.85 | 0.42 |
| traffic_cone | 20-30 m | 2,644 | 2,480 | 16.9% | 39.5% | 1.26 | 0.62 |
| trailer | 0-10 m | 26 | 25 | 24.0% | 76.0% | 0.51 | 0.66 |
| trailer | 10-20 m | 343 | 316 | 8.5% | 18.4% | 1.36 | 0.83 |
| trailer | 20-30 m | 652 | 590 | 5.4% | 15.9% | 1.49 | 1.08 |
| trailer | 30-40 m | 768 | 591 | 3.2% | 10.8% | 1.63 | 1.33 |
| trailer | 40-50 m | 542 | 390 | 3.3% | 13.1% | 1.50 | 1.27 |
| truck | 0-10 m | 673 | 657 | 43.2% | 82.0% | 0.54 | 0.36 |
| truck | 10-20 m | 1,863 | 1,804 | 32.5% | 58.7% | 0.87 | 0.36 |
| truck | 20-30 m | 2,317 | 2,169 | 19.4% | 42.3% | 1.23 | 0.53 |
| truck | 30-40 m | 2,406 | 2,218 | 13.2% | 34.0% | 1.39 | 0.66 |
| truck | 40-50 m | 2,001 | 1,717 | 7.3% | 19.9% | 1.60 | 0.83 |

See `geometry_summary.csv` for the same class/range breakdown by footprint, viewing angle, and annotation visibility. Heatmaps in `plots/` show retention and radial error by range and geometry bin.
