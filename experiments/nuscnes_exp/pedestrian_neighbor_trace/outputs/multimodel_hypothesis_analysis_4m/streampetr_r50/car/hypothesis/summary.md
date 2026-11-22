# StreamPETR R50 car hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 53,043; predictions: 495,252.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 52,161 | 53,043 GTs |
| Unmatched GTs | 882 | 53,043 GTs |
| Matched GTs with a different best-3D candidate | 14,300 | 52,161 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 6,977 | 14,300 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 6,773 | 14,300 comparable GTs |
| Geometric cases whose alternative has lower confidence | 6,966 | 6,977 geometric cases |
| Best-3D candidate already TP for another GT | 420 | 14,300 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another car; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 460 (52.2%) | 108 (12.2%) | 58 (6.6%) | 256 (29.0%) |
| Different best-3D candidate | 4,727 (33.1%) | 2,506 (17.5%) | 2,394 (16.7%) | 4,673 (32.7%) |
| Geometrically better | 2,553 (36.6%) | 1,218 (17.5%) | 1,120 (16.1%) | 2,086 (29.9%) |
| Geometrically better, alternative unmatched | 2,438 (36.0%) | 1,189 (17.6%) | 1,094 (16.2%) | 2,052 (30.3%) |
