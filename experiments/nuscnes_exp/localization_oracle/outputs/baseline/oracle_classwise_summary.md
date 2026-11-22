# PETR localization oracle: classwise AP

AP values are percentages from the official nuScenes evaluator. Class mAP is the mean of AP at 0.5, 1, 2, and 4 m. Gains are percentage points relative to the same-run baseline.

| Class | Eligible GT | Original <4 m matches | Baseline mAP | Radial mAP | Radial gain | Tangential mAP | Tangential gain | Full-center mAP | Full-center gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 53,043 | 52,177 | 50.71% | 73.72% | +23.01 pp | 53.43% | +2.72 pp | 81.88% | +31.17 pp |
| truck | 9,260 | 8,565 | 28.73% | 48.23% | +19.50 pp | 31.25% | +2.52 pp | 57.35% | +28.61 pp |
| bus | 2,062 | 1,835 | 34.89% | 58.02% | +23.13 pp | 37.48% | +2.59 pp | 69.00% | +34.11 pp |
| trailer | 2,331 | 1,912 | 11.83% | 19.38% | +7.55 pp | 14.76% | +2.93 pp | 35.68% | +23.85 pp |
| construction_vehicle | 1,513 | 1,036 | 6.92% | 12.62% | +5.70 pp | 7.72% | +0.79 pp | 19.45% | +12.53 pp |
| pedestrian | 23,500 | 22,615 | 41.25% | 62.23% | +20.97 pp | 44.21% | +2.96 pp | 72.15% | +30.90 pp |
| motorcycle | 1,855 | 1,545 | 30.74% | 49.20% | +18.46 pp | 32.39% | +1.64 pp | 55.99% | +25.24 pp |
| bicycle | 1,824 | 1,558 | 30.43% | 48.92% | +18.48 pp | 31.97% | +1.53 pp | 53.35% | +22.92 pp |
| traffic_cone | 10,570 | 10,201 | 52.02% | 68.00% | +15.98 pp | 55.37% | +3.35 pp | 77.16% | +25.14 pp |
| barrier | 15,903 | 14,768 | 40.29% | 52.91% | +12.62 pp | 46.01% | +5.72 pp | 68.70% | +28.41 pp |

## Main observations

- Radial correction improves class mAP more than tangential correction for every class.
- The largest radial gain is bus (+23.13 pp); the largest tangential gain is barrier (+5.72 pp).
- Full-center class mAP equals its original AP@4 m because the corrected set consists of the original <4 m matches. Classes with low AP@4 m retain a low oracle ceiling.


## AP by threshold

| Class | Oracle | AP@0.5 m | AP@1 m | AP@2 m | AP@4 m | Class mAP |
|---|---|---:|---:|---:|---:|---:|
| car | baseline | 12.71% | 39.06% | 69.19% | 81.88% | 50.71% |
| car | radial | 59.05% | 73.82% | 80.15% | 81.87% | 73.72% |
| car | tangential | 16.82% | 43.82% | 71.19% | 81.88% | 53.43% |
| car | full_center | 81.88% | 81.88% | 81.88% | 81.88% | 81.88% |
| truck | baseline | 1.25% | 14.55% | 41.79% | 57.35% | 28.73% |
| truck | radial | 31.42% | 48.52% | 55.64% | 57.35% | 48.23% |
| truck | tangential | 4.02% | 19.21% | 44.44% | 57.34% | 31.25% |
| truck | full_center | 57.35% | 57.35% | 57.35% | 57.35% | 57.35% |
| bus | baseline | 2.09% | 18.17% | 50.29% | 69.00% | 34.89% |
| bus | radial | 36.43% | 59.33% | 67.30% | 69.00% | 58.02% |
| bus | tangential | 5.13% | 22.91% | 52.88% | 69.00% | 37.48% |
| bus | full_center | 69.00% | 69.00% | 69.00% | 69.00% | 69.00% |
| trailer | baseline | 0.00% | 0.08% | 11.56% | 35.68% | 11.83% |
| trailer | radial | 2.16% | 11.13% | 28.53% | 35.70% | 19.38% |
| trailer | tangential | 0.00% | 2.61% | 20.79% | 35.64% | 14.76% |
| trailer | full_center | 35.68% | 35.68% | 35.68% | 35.68% | 35.68% |
| construction_vehicle | baseline | 0.00% | 0.14% | 8.09% | 19.45% | 6.92% |
| construction_vehicle | radial | 2.25% | 11.09% | 17.69% | 19.45% | 12.62% |
| construction_vehicle | tangential | 0.00% | 1.33% | 10.08% | 19.45% | 7.72% |
| construction_vehicle | full_center | 19.45% | 19.45% | 19.45% | 19.45% | 19.45% |
| pedestrian | baseline | 6.32% | 29.81% | 56.73% | 72.15% | 41.25% |
| pedestrian | radial | 44.04% | 62.41% | 70.18% | 72.28% | 62.23% |
| pedestrian | tangential | 11.34% | 34.30% | 59.06% | 72.14% | 44.21% |
| pedestrian | full_center | 72.15% | 72.15% | 72.15% | 72.15% | 72.15% |
| motorcycle | baseline | 3.61% | 21.08% | 42.30% | 55.99% | 30.74% |
| motorcycle | radial | 36.89% | 49.68% | 54.26% | 55.99% | 49.20% |
| motorcycle | tangential | 6.26% | 23.35% | 43.94% | 55.99% | 32.39% |
| motorcycle | full_center | 55.99% | 55.99% | 55.99% | 55.99% | 55.99% |
| bicycle | baseline | 5.24% | 21.59% | 41.55% | 53.35% | 30.43% |
| bicycle | radial | 38.82% | 50.61% | 52.85% | 53.38% | 48.92% |
| bicycle | tangential | 8.44% | 23.96% | 42.09% | 53.37% | 31.97% |
| bicycle | full_center | 53.35% | 53.35% | 53.35% | 53.35% | 53.35% |
| traffic_cone | baseline | 16.18% | 46.23% | 68.51% | 77.16% | 52.02% |
| traffic_cone | radial | 51.45% | 68.22% | 75.12% | 77.22% | 68.00% |
| traffic_cone | tangential | 22.89% | 50.81% | 70.62% | 77.15% | 55.37% |
| traffic_cone | full_center | 77.16% | 77.16% | 77.16% | 77.16% | 77.16% |
| barrier | baseline | 3.40% | 28.47% | 60.60% | 68.70% | 40.29% |
| barrier | radial | 25.13% | 51.93% | 65.89% | 68.71% | 52.91% |
| barrier | tangential | 11.45% | 40.27% | 63.66% | 68.67% | 46.01% |
| barrier | full_center | 68.70% | 68.70% | 68.70% | 68.70% | 68.70% |

Only original predictions matched to GT within 4 m had their global XY centers changed. Classes, scores, sizes, and orientations were preserved. Official evaluation rematched each corrected submission. Eligible GT and match counts come from the baseline GT-coverage analysis when available.
Class mAP gains cannot be added across classes; overall mAP is their unweighted mean. Radial and tangential gains are separate oracle runs and are not additive.
