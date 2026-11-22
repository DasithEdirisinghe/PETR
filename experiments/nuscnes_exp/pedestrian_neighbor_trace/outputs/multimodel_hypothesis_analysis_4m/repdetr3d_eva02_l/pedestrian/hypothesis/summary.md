# RepDETR3D EVA02-L pedestrian hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 23,500; predictions: 348,354.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 22,741 | 23,500 GTs |
| Unmatched GTs | 759 | 23,500 GTs |
| Matched GTs with a different best-3D candidate | 13,521 | 22,741 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 6,141 | 13,521 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 5,335 | 13,521 comparable GTs |
| Geometric cases whose alternative has lower confidence | 5,985 | 6,141 geometric cases |
| Best-3D candidate already TP for another GT | 1,896 | 13,521 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another pedestrian; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 411 (54.2%) | 121 (15.9%) | 99 (13.0%) | 128 (16.9%) |
| Different best-3D candidate | 3,299 (24.4%) | 1,865 (13.8%) | 1,870 (13.8%) | 6,487 (48.0%) |
| Geometrically better | 1,549 (25.2%) | 853 (13.9%) | 870 (14.2%) | 2,869 (46.7%) |
| Geometrically better, alternative unmatched | 1,345 (25.2%) | 734 (13.8%) | 701 (13.1%) | 2,555 (47.9%) |
