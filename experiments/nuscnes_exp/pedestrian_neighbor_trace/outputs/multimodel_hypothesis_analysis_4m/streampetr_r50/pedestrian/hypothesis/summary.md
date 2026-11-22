# StreamPETR R50 pedestrian hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 23,500; predictions: 298,631.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 22,841 | 23,500 GTs |
| Unmatched GTs | 659 | 23,500 GTs |
| Matched GTs with a different best-3D candidate | 14,059 | 22,841 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 5,953 | 14,059 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 4,827 | 14,059 comparable GTs |
| Geometric cases whose alternative has lower confidence | 5,693 | 5,953 geometric cases |
| Best-3D candidate already TP for another GT | 2,801 | 14,059 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another pedestrian; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 353 (53.6%) | 108 (16.4%) | 69 (10.5%) | 129 (19.6%) |
| Different best-3D candidate | 3,341 (23.8%) | 1,927 (13.7%) | 1,951 (13.9%) | 6,840 (48.7%) |
| Geometrically better | 1,389 (23.3%) | 845 (14.2%) | 882 (14.8%) | 2,837 (47.7%) |
| Geometrically better, alternative unmatched | 1,115 (23.1%) | 686 (14.2%) | 676 (14.0%) | 2,350 (48.7%) |
