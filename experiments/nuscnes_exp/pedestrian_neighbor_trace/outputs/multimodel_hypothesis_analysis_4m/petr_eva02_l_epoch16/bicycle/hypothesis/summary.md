# PETR EVA02-L epoch 16 bicycle hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 1,824; predictions: 73,116.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 1,695 | 1,824 GTs |
| Unmatched GTs | 129 | 1,824 GTs |
| Matched GTs with a different best-3D candidate | 841 | 1,695 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 332 | 841 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 295 | 841 comparable GTs |
| Geometric cases whose alternative has lower confidence | 319 | 332 geometric cases |
| Best-3D candidate already TP for another GT | 86 | 841 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another bicycle; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 80 (62.0%) | 16 (12.4%) | 9 (7.0%) | 24 (18.6%) |
| Different best-3D candidate | 160 (19.0%) | 112 (13.3%) | 135 (16.1%) | 434 (51.6%) |
| Geometrically better | 51 (15.4%) | 34 (10.2%) | 59 (17.8%) | 188 (56.6%) |
| Geometrically better, alternative unmatched | 41 (13.9%) | 31 (10.5%) | 55 (18.6%) | 168 (56.9%) |
