# PETR EVA02-L epoch 16 pedestrian hypothesis alignment, validation set

Matching threshold: 4 m. GTs: 23,500; predictions: 310,845.

| Measure | Count | Denominator |
|---|---:|---:|
| Officially matched GTs | 22,970 | 23,500 GTs |
| Unmatched GTs | 530 | 23,500 GTs |
| Matched GTs with a different best-3D candidate | 13,575 | 22,970 matched GTs |
| Better 3D IoU, projected 2D IoU, and absolute radial error | 4,800 | 13,575 comparable GTs |
| Same geometric pattern with alternative officially unmatched | 3,532 | 13,575 comparable GTs |
| Geometric cases whose alternative has lower confidence | 4,468 | 4,800 geometric cases |
| Best-3D candidate already TP for another GT | 3,704 | 13,575 compared GTs |


Confidence is reported descriptively and is not part of the geometric case definition.

The geometric pattern is a GT-informed diagnosis. Projected 2D IoU compares image-clipped rectangles around projected 3D boxes in one camera selected from the GT projection alone. It does not measure visual feature correspondence. The alternative may depict another pedestrian; inspect such cases before treating them as missed ranking opportunities.

See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the geometric pattern, and `paired_iou_deltas.png` for the distribution.

## Visibility distribution

| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |
|---|---:|---:|---:|---:|
| Official unmatched GT | 331 (62.5%) | 54 (10.2%) | 63 (11.9%) | 82 (15.5%) |
| Different best-3D candidate | 3,119 (23.0%) | 1,850 (13.6%) | 1,778 (13.1%) | 6,828 (50.3%) |
| Geometrically better | 928 (19.3%) | 626 (13.0%) | 646 (13.5%) | 2,600 (54.2%) |
| Geometrically better, alternative unmatched | 635 (18.0%) | 455 (12.9%) | 447 (12.7%) | 1,995 (56.5%) |
