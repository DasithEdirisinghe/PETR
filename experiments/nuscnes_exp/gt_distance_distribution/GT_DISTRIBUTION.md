# nuScenes GT distribution by ego distance

Counts cover all annotated keyframes in the nuScenes v1.0-trainval tables: **700 training scenes** (28,130 keyframes) and **150 validation scenes** (6,019 keyframes).

Each cell is **GT count (share of that class within the split)**. Distance is the horizontal Euclidean distance between the GT box center and the LIDAR_TOP ego pose at that sample. Bins are left inclusive and right exclusive; 50+ includes all remaining GT.

## Training set

| Class | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50+ | Total |
|:--|--:|--:|--:|--:|--:|--:|--:|
| car | 30,293 (7.3%) | 70,350 (17.0%) | 81,874 (19.8%) | 75,865 (18.4%) | 58,837 (14.2%) | 96,099 (23.3%) | 413,318 |
| truck | 4,178 (5.7%) | 9,678 (13.3%) | 11,453 (15.7%) | 11,856 (16.3%) | 10,454 (14.4%) | 25,196 (34.6%) | 72,815 |
| construction vehicle | 520 (4.3%) | 1,806 (15.1%) | 2,031 (16.9%) | 2,305 (19.2%) | 1,972 (16.4%) | 3,359 (28.0%) | 11,993 |
| bus | 530 (4.0%) | 1,609 (12.2%) | 1,842 (14.0%) | 1,756 (13.3%) | 1,646 (12.5%) | 5,780 (43.9%) | 13,163 |
| trailer | 651 (3.1%) | 1,922 (9.3%) | 3,427 (16.6%) | 3,791 (18.3%) | 2,760 (13.3%) | 8,150 (39.4%) | 20,701 |
| barrier | 18,153 (14.5%) | 34,724 (27.8%) | 28,823 (23.0%) | 20,869 (16.7%) | 12,328 (9.9%) | 10,198 (8.2%) | 125,095 |
| motorcycle | 1,265 (12.5%) | 2,700 (26.7%) | 2,514 (24.9%) | 1,570 (15.5%) | 1,142 (11.3%) | 918 (9.1%) | 10,109 |
| bicycle | 1,081 (11.4%) | 2,794 (29.5%) | 2,826 (29.8%) | 1,415 (14.9%) | 821 (8.7%) | 541 (5.7%) | 9,478 |
| pedestrian | 12,725 (6.8%) | 44,758 (24.1%) | 45,552 (24.5%) | 35,944 (19.3%) | 23,120 (12.4%) | 23,748 (12.8%) | 185,847 |
| traffic cone | 15,499 (18.8%) | 26,455 (32.1%) | 19,665 (23.9%) | 11,336 (13.8%) | 5,611 (6.8%) | 3,796 (4.6%) | 82,362 |
| All classes | 84,895 (9.0%) | 196,796 (20.8%) | 200,007 (21.2%) | 166,707 (17.6%) | 118,691 (12.6%) | 177,785 (18.8%) | 944,881 |

## Validation set

| Class | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50+ | Total |
|:--|--:|--:|--:|--:|--:|--:|--:|
| car | 5,541 (6.9%) | 12,384 (15.5%) | 15,142 (18.9%) | 14,877 (18.6%) | 11,578 (14.5%) | 20,482 (25.6%) | 80,004 |
| truck | 673 (4.3%) | 1,881 (12.0%) | 2,430 (15.5%) | 2,541 (16.2%) | 2,239 (14.3%) | 5,940 (37.8%) | 15,704 |
| construction vehicle | 59 (2.2%) | 344 (12.8%) | 396 (14.8%) | 348 (13.0%) | 452 (16.9%) | 1,079 (40.3%) | 2,678 |
| bus | 120 (3.8%) | 420 (13.3%) | 486 (15.4%) | 604 (19.1%) | 455 (14.4%) | 1,073 (34.0%) | 3,158 |
| trailer | 26 (0.6%) | 344 (8.3%) | 681 (16.4%) | 807 (19.4%) | 588 (14.1%) | 1,713 (41.2%) | 4,159 |
| barrier | 3,211 (11.9%) | 7,222 (26.8%) | 6,478 (24.0%) | 4,418 (16.4%) | 2,413 (8.9%) | 3,250 (12.0%) | 26,992 |
| motorcycle | 226 (9.0%) | 546 (21.8%) | 707 (28.2%) | 530 (21.1%) | 305 (12.2%) | 194 (7.7%) | 2,508 |
| bicycle | 245 (10.3%) | 735 (30.9%) | 665 (27.9%) | 415 (17.4%) | 193 (8.1%) | 128 (5.4%) | 2,381 |
| pedestrian | 2,943 (8.6%) | 8,838 (25.7%) | 7,634 (22.2%) | 6,263 (18.2%) | 3,844 (11.2%) | 4,825 (14.0%) | 34,347 |
| traffic cone | 2,895 (18.6%) | 5,973 (38.3%) | 3,623 (23.2%) | 1,954 (12.5%) | 747 (4.8%) | 405 (2.6%) | 15,597 |
| All classes | 15,939 (8.5%) | 38,687 (20.6%) | 38,242 (20.4%) | 32,757 (17.5%) | 22,814 (12.2%) | 39,089 (20.8%) | 187,528 |

## GT visibility by distance

The [visibility overview](visibility_plots/gt_visibility_bands_overview.png) and [CSV counts](GT_VISIBILITY_BY_DISTANCE.csv) show the four nuScenes visibility categories within every split × detection class × distance bin. Each cell in the plot shows a 100% stacked bar, the exact category counts, and percentages of its GT total. There are also larger per-class plots in `visibility_plots/`. Dark red means less visible; light blue-gray means more visible.

## Counting rules

- Only the ten nuScenes detection classes are shown. Pedestrian subtypes and rigid/bendy buses are merged using the official detection class mapping. Other general categories are excluded.
- These are raw annotated GT counts. No detection evaluation class-range, point-count, or bike-rack filters were applied. Counts can therefore differ from evaluator-eligible GT or 4 m TP counts in the PETR plots.
- A GT annotation at a different keyframe counts again. This is an annotation distribution, not a unique-instance count.
- The validation scene names come from the official split saved by `compare_error_bands.py --all-val`; the remaining scenes in `v1.0-trainval` are training scenes.
- Regenerate with `python3 experiments/nuscnes_exp/gt_distance_distribution/summarize_gt_distribution.py` from the PETR repository root.

Official class mapping: [nuScenes detection README](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md).
