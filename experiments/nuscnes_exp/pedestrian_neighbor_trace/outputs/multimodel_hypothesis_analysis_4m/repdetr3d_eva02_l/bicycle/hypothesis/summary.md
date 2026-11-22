# RepDETR3D EVA02-L bicycle hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 1,824; predictions: 67,249.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 1,662 | 1,824 GTs |
| Unmatched GTs | 162 | 1,824 GTs |
| Matched GTs with a different best-3D candidate | 837 | 1,662 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 295 | 837 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 267 | 837 comparable GTs |
| Geometric cases whose alternative has lower confidence | 286 | 295 geometric cases |
| Best-3D candidate already TP for another GT | 57 | 837 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another bicycle; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 76 (46.9%) | 23 (14.2%) | 9 (5.6%) | 54 (33.3%) |
| Different best-3D candidate | 174 (20.8%) | 119 (14.2%) | 109 (13.0%) | 435 (52.0%) |
| Geometrically better | 70 (23.7%) | 49 (16.6%) | 29 (9.8%) | 147 (49.8%) |
| Geometrically better, alternative unmatched | 63 (23.6%) | 43 (16.1%) | 25 (9.4%) | 136 (50.9%) |
