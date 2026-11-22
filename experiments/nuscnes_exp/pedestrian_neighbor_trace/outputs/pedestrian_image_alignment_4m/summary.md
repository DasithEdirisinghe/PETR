# PETR pedestrian hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 23,500; predictions: 350,034.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 22,615 | 23,500 GTs |
| Unmatched GTs | 885 | 23,500 GTs |
| Matched GTs with a different best-3D candidate | 13,304 | 22,615 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 4,619 | 13,304 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 3,288 | 13,304 comparable GTs |
| Geometric cases whose alternative has lower confidence | 4,295 | 4,619 geometric cases |
| Best-3D candidate already TP for another GT | 4,060 | 13,304 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another pedestrian; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 485 (54.8%) | 110 (12.4%) | 95 (10.7%) | 195 (22.0%) |
| Different best-3D candidate | 3,071 (23.1%) | 1,747 (13.1%) | 1,787 (13.4%) | 6,699 (50.4%) |
| Geometrically better | 904 (19.6%) | 607 (13.1%) | 640 (13.9%) | 2,468 (53.4%) |
| Geometrically better, alternative unmatched | 606 (18.4%) | 426 (13.0%) | 438 (13.3%) | 1,818 (55.3%) |
