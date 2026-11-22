# PETR R50 car hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 53,043; predictions: 476,300.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 52,177 | 53,043 GTs |
| Unmatched GTs | 866 | 53,043 GTs |
| Matched GTs with a different best-3D candidate | 14,761 | 52,177 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 6,998 | 14,761 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 6,645 | 14,761 comparable GTs |
| Geometric cases whose alternative has lower confidence | 6,928 | 6,998 geometric cases |
| Best-3D candidate already TP for another GT | 840 | 14,761 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another car; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 474 (54.7%) | 107 (12.4%) | 80 (9.2%) | 205 (23.7%) |
| Different best-3D candidate | 5,140 (34.8%) | 2,533 (17.2%) | 2,722 (18.4%) | 4,366 (29.6%) |
| Geometrically better | 2,580 (36.9%) | 1,244 (17.8%) | 1,217 (17.4%) | 1,957 (28.0%) |
| Geometrically better, alternative unmatched | 2,452 (36.9%) | 1,176 (17.7%) | 1,135 (17.1%) | 1,882 (28.3%) |
