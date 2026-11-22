# PETR R50 bicycle hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 1,824; predictions: 55,598.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 1,558 | 1,824 GTs |
| Unmatched GTs | 266 | 1,824 GTs |
| Matched GTs with a different best-3D candidate | 785 | 1,558 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 236 | 785 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 207 | 785 comparable GTs |
| Geometric cases whose alternative has lower confidence | 226 | 236 geometric cases |
| Best-3D candidate already TP for another GT | 112 | 785 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another bicycle; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 117 (44.0%) | 33 (12.4%) | 30 (11.3%) | 86 (32.3%) |
| Different best-3D candidate | 134 (17.1%) | 97 (12.4%) | 118 (15.0%) | 436 (55.5%) |
| Geometrically better | 34 (14.4%) | 24 (10.2%) | 37 (15.7%) | 141 (59.7%) |
| Geometrically better, alternative unmatched | 27 (13.0%) | 22 (10.6%) | 30 (14.5%) | 128 (61.8%) |
