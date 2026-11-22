# PETR: 80–100% visible GTs, retained versus lost 4 m TPs

Only original 4 m TP predictions assigned to visibility-token-4 GTs are included. Retained means the same prediction is still a TP after independent 0.5 m matching; it may match a different GT. Lost means that prediction is an FP at 0.5 m.

## All GT distances

| Class | Retained / lost | Median GT range (retained / lost) | Median score (retained / lost) | Another GT closer (retained / lost) |
|---|---:|---:|---:|---:|
| barrier | 2,632 / 6,278 | 13.5 / 17.7 m | 0.870 / 0.794 | 5.8% / 14.4% |
| bicycle | 339 / 585 | 13.9 / 23.7 m | 0.729 / 0.450 | 2.9% / 9.1% |
| bus | 316 / 875 | 18.6 / 28.0 m | 0.883 / 0.800 | 0.0% / 0.0% |
| car | 12,755 / 15,450 | 14.1 / 28.3 m | 0.947 / 0.814 | 0.1% / 1.6% |
| construction_vehicle | 27 / 363 | 15.3 / 21.4 m | 0.733 / 0.434 | 0.0% / 0.8% |
| motorcycle | 285 / 592 | 12.7 / 23.9 m | 0.792 / 0.433 | 0.4% / 5.1% |
| pedestrian | 4,102 / 8,112 | 13.7 / 22.3 m | 0.886 / 0.544 | 9.9% / 22.2% |
| traffic_cone | 3,571 / 4,630 | 11.1 / 16.5 m | 0.791 / 0.505 | 8.6% / 18.0% |
| trailer | 61 / 617 | 23.0 / 28.5 m | 0.533 / 0.386 | 1.6% / 2.3% |
| truck | 1,259 / 3,400 | 17.5 / 26.3 m | 0.780 / 0.640 | 0.2% / 0.6% |

## How to interpret this comparison

- Compare rows within the **same class and GT distance bin** in `high_visibility_outcomes_by_class_range.csv` before attributing a difference to score or scene context. GT distance can affect both score and localization.
- Radial and tangential errors are provided for diagnosis, but the lost group necessarily has larger center error by definition. That difference alone does not explain the failure.
- Another-GT-closer and nearby-GT flags are geometric context, not evidence that PETR confused object identities.
- An 80–100% nuScenes visibility token describes the GT across six cameras; it does not mean each camera image shows the entire object.

For visual inspection, compare retained and lost examples from the **same class, 10 m range bin, and similar score**. Show all six images with projected GT/prediction centers, plus a BEV crop containing nearby same-class GTs. This avoids mistaking a distance or score difference for a visual feature difference.
