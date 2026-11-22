# StreamPETR R50 bicycle hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 1,824; predictions: 42,618.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 1,625 | 1,824 GTs |
| Unmatched GTs | 199 | 1,824 GTs |
| Matched GTs with a different best-3D candidate | 849 | 1,625 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 301 | 849 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 280 | 849 comparable GTs |
| Geometric cases whose alternative has lower confidence | 298 | 301 geometric cases |
| Best-3D candidate already TP for another GT | 70 | 849 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another bicycle; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 85 (42.7%) | 38 (19.1%) | 13 (6.5%) | 63 (31.7%) |
| Different best-3D candidate | 162 (19.1%) | 97 (11.4%) | 123 (14.5%) | 467 (55.0%) |
| Geometrically better | 50 (16.6%) | 36 (12.0%) | 30 (10.0%) | 185 (61.5%) |
| Geometrically better, alternative unmatched | 47 (16.8%) | 33 (11.8%) | 27 (9.6%) | 173 (61.8%) |
