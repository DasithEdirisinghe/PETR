# PETR EVA02-L epoch 16 car hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 53,043; predictions: 415,605.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 52,367 | 53,043 GTs |
| Unmatched GTs | 676 | 53,043 GTs |
| Matched GTs with a different best-3D candidate | 16,827 | 52,367 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 7,976 | 16,827 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 7,649 | 16,827 comparable GTs |
| Geometric cases whose alternative has lower confidence | 7,895 | 7,976 geometric cases |
| Best-3D candidate already TP for another GT | 746 | 16,827 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another car; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 418 (61.8%) | 73 (10.8%) | 64 (9.5%) | 121 (17.9%) |
| Different best-3D candidate | 4,846 (28.8%) | 2,663 (15.8%) | 2,989 (17.8%) | 6,329 (37.6%) |
| Geometrically better | 2,384 (29.9%) | 1,298 (16.3%) | 1,435 (18.0%) | 2,859 (35.8%) |
| Geometrically better, alternative unmatched | 2,262 (29.6%) | 1,233 (16.1%) | 1,364 (17.8%) | 2,790 (36.5%) |
