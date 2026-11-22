# PETR motorcycle hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 1,855; predictions: 35,886.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 1,545 | 1,855 GTs |
| Unmatched GTs | 310 | 1,855 GTs |
| Matched GTs with a different best-3D candidate | 754 | 1,545 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 338 | 754 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 299 | 754 comparable GTs |
| Geometric cases whose alternative has lower confidence | 330 | 338 geometric cases |
| Best-3D candidate already TP for another GT | 97 | 754 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another motorcycle; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 146 (47.1%) | 23 (7.4%) | 38 (12.3%) | 103 (33.2%) |
| Different best-3D candidate | 146 (19.4%) | 58 (7.7%) | 161 (21.4%) | 389 (51.6%) |
| Geometrically better | 62 (18.3%) | 23 (6.8%) | 71 (21.0%) | 182 (53.8%) |
| Geometrically better, alternative unmatched | 53 (17.7%) | 20 (6.7%) | 57 (19.1%) | 169 (56.5%) |
