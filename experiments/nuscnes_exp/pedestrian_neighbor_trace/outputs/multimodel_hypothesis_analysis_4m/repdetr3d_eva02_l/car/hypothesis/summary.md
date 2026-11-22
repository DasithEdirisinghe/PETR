# RepDETR3D EVA02-L car hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 53,043; predictions: 482,445.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 51,757 | 53,043 GTs |
| Unmatched GTs | 1,286 | 53,043 GTs |
| Matched GTs with a different best-3D candidate | 13,431 | 51,757 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 6,516 | 13,431 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 6,398 | 13,431 comparable GTs |
| Geometric cases whose alternative has lower confidence | 6,498 | 6,516 geometric cases |
| Best-3D candidate already TP for another GT | 230 | 13,431 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another car; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 688 (53.5%) | 147 (11.4%) | 97 (7.5%) | 354 (27.5%) |
| Different best-3D candidate | 4,541 (33.8%) | 2,154 (16.0%) | 2,094 (15.6%) | 4,642 (34.6%) |
| Geometrically better | 2,449 (37.6%) | 1,055 (16.2%) | 969 (14.9%) | 2,043 (31.4%) |
| Geometrically better, alternative unmatched | 2,378 (37.2%) | 1,036 (16.2%) | 958 (15.0%) | 2,026 (31.7%) |
