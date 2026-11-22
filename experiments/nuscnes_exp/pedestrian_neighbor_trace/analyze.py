#!/usr/bin/env python3
"""Analyze one crowded pedestrian group per nuScenes validation keyframe."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def center_distance(a, b):
    return math.hypot(a.translation[0] - b.translation[0],
                      a.translation[1] - b.translation[1])


def corners(box):
    w, qx, qy, qz = box.rotation
    yaw = math.atan2(2 * (w*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
    x, y = box.translation[:2]
    width, length = box.size[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + u*c - v*s, y + u*s + v*c)
            for u, v in ((-length/2, -width/2), (length/2, -width/2),
                         (length/2, width/2), (-length/2, width/2))]


def polygon_area(poly):
    return abs(sum(poly[i][0]*poly[(i+1) % len(poly)][1] -
                   poly[(i+1) % len(poly)][0]*poly[i][1]
                   for i in range(len(poly)))) / 2 if len(poly) >= 3 else 0.0


def intersection(subject, clip):
    """Convex polygon clipping; both rectangles are counterclockwise."""
    for i, a in enumerate(clip):
        b = clip[(i + 1) % len(clip)]
        def side(p):
            return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        def cross(p, q):
            sp, sq = side(p), side(q)
            t = sp / (sp-sq)
            return (p[0] + t*(q[0]-p[0]), p[1] + t*(q[1]-p[1]))
        output = []
        if not subject:
            break
        previous = subject[-1]
        for current in subject:
            inside, previous_inside = side(current) >= -1e-10, side(previous) >= -1e-10
            if inside != previous_inside:
                output.append(cross(previous, current))
            if inside:
                output.append(current)
            previous = current
        subject = output
    return subject


def overlaps(a, b):
    area_a, area_b = polygon_area(corners(a)), polygon_area(corners(b))
    area_i = polygon_area(intersection(corners(a), corners(b)))
    bev = area_i / (area_a + area_b - area_i) if area_a + area_b > area_i else 0.0
    low_a, high_a = a.translation[2]-a.size[2]/2, a.translation[2]+a.size[2]/2
    low_b, high_b = b.translation[2]-b.size[2]/2, b.translation[2]+b.size[2]/2
    volume_i = area_i * max(0.0, min(high_a, high_b)-max(low_a, low_b))
    volume_a, volume_b = area_a*a.size[2], area_b*b.size[2]
    iou3d = volume_i / (volume_a + volume_b - volume_i) if volume_a + volume_b > volume_i else 0.0
    return bev, iou3d


def replay(evaluator, threshold):
    """Mirror detection.algo.accumulate score order and GT tie handling."""
    predictions = [(token, index, box)
                   for token in evaluator.pred_boxes.sample_tokens
                   for index, box in enumerate(evaluator.pred_boxes[token])
                   if box.detection_name == 'pedestrian']
    order = [i for _, i in sorted((box.detection_score, i)
             for i, (_, _, box) in enumerate(predictions))][::-1]
    taken, assignments, ranks, available = set(), {}, {}, {}
    for rank, position in enumerate(order, 1):
        token, index, pred = predictions[position]
        candidates = [(center_distance(pred, gt), gi)
                      for gi, gt in enumerate(evaluator.gt_boxes[token])
                      if gt.detection_name == 'pedestrian' and (token, gi) not in taken]
        distance, gi = min(candidates, default=(float('inf'), None))
        key = (token, index)
        ranks[key] = rank
        available[key] = (gi, distance)
        if distance < threshold:
            assignments[key] = gi
            taken.add((token, gi))
    return assignments, ranks, available


def gt_identity(nusc, gt):
    sample = nusc.get('sample', gt.sample_token)
    matches = []
    for token in sample['anns']:
        annotation = nusc.get('sample_annotation', token)
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(
            annotation['translation'], gt.translation)))
        if distance < 1e-4:
            matches.append(annotation)
    if len(matches) != 1:
        raise ValueError('Cannot resolve GT annotation for sample ' + gt.sample_token)
    return {'annotation_token': matches[0]['token'],
            'instance_token': matches[0]['instance_token']}


def crowded_components(gts, radius):
    """Connected GT groups under a BEV center-distance edge."""
    remaining = set(range(len(gts)))
    groups = []
    while remaining:
        group = {remaining.pop()}
        frontier = list(group)
        while frontier:
            i = frontier.pop()
            neighbors = {j for j in remaining
                         if center_distance(gts[i][1], gts[j][1]) <= radius}
            group.update(neighbors)
            frontier.extend(neighbors)
            remaining.difference_update(neighbors)
        if len(group) >= 2:
            groups.append(sorted(group))
    return groups


def diagnostic_assignment(pairs, gt_indices, pred_indices, threshold, mode, min_iou):
    """Max-cardinality local matching, then best distance or overlap."""
    from scipy.optimize import linear_sum_assignment
    if not gt_indices or not pred_indices:
        return []
    n_gt, n_pred = len(gt_indices), len(pred_indices)
    # One additional valid match must outweigh every possible cost difference.
    unmatched_cost = (n_gt + 1) * max(threshold, 1.0) + 1.0
    forbidden_cost = unmatched_cost * (n_gt + 1)
    costs = [[forbidden_cost] * n_pred + [unmatched_cost] * n_gt
             for _ in gt_indices]
    for row, gi in enumerate(gt_indices):
        for col, pi in enumerate(pred_indices):
            pair = pairs[pi, gi]
            if pair['center_distance_m'] >= threshold:
                continue
            if mode == 'bev_iou' and pair['bev_iou'] < min_iou:
                continue
            costs[row][col] = (pair['center_distance_m'] if mode == 'center'
                               else 1.0 - pair['bev_iou'])
    rows, cols = linear_sum_assignment(costs)
    return [{'gt_index': gt_indices[row], 'prediction_index': pred_indices[col]}
            for row, col in zip(rows, cols)
            if col < n_pred and costs[row][col] < unmatched_cost]


def analyze_groups(evaluator, nusc, args):
    """One crowded pedestrian group per keyframe, held fixed across thresholds."""
    thresholds = tuple(sorted(evaluator.cfg.dist_ths))
    matches = {threshold: replay(evaluator, threshold) for threshold in thresholds}
    selected = 0
    counts = Counter()
    fields = ('sample_token', 'group_gt_count', 'local_prediction_count',
              'threshold_m', 'official_tp_gt_count', 'official_fn_gt_count',
              'local_prediction_tp_in_group', 'local_prediction_matched_elsewhere',
              'local_prediction_fp', 'center_counterfactual_tp_count',
              'bev_counterfactual_tp_count', 'diagnostic_pattern')
    with (args.output_dir / 'groups.jsonl').open('w') as detail_handle, \
            (args.output_dir / 'groups.csv').open('w', newline='') as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        for token in evaluator.gt_boxes.sample_tokens:
            if args.sample_token and token != args.sample_token:
                continue
            gts = [(i, box) for i, box in enumerate(evaluator.gt_boxes[token])
                   if box.detection_name == 'pedestrian']
            components = crowded_components(gts, args.cluster_distance)
            if not components:
                continue
            # Pick the largest group; break ties by its mean pairwise distance.
            def priority(component):
                distances = [center_distance(gts[i][1], gts[j][1])
                             for position, i in enumerate(component)
                             for j in component[position + 1:]]
                return (-len(component), sum(distances) / len(distances),
                        tuple(gts[i][0] for i in component))
            group = min(components, key=priority)
            chosen_gts = [gts[i] for i in group]
            gt_indices = [i for i, _ in chosen_gts]
            group_set = set(gt_indices)
            all_preds = [(i, box) for i, box in enumerate(evaluator.pred_boxes[token])
                         if box.detection_name == 'pedestrian']
            pred_indices = [i for i, box in all_preds
                            if any(center_distance(box, gt) <= args.prediction_radius
                                   for _, gt in chosen_gts)
                            or any(assignments.get((token, i)) in group_set
                                   for assignments, _, _ in matches.values())]
            pred_by_index = dict(all_preds)
            pair_lookup = {}
            pair_rows = []
            for pi in pred_indices:
                for gi, gt in chosen_gts:
                    pred = pred_by_index[pi]
                    bev, iou3d = overlaps(pred, gt)
                    row = {'prediction_index': pi, 'gt_index': gi,
                           'center_distance_m': center_distance(pred, gt),
                           'bev_iou': bev, 'iou_3d': iou3d}
                    pair_lookup[pi, gi] = row
                    pair_rows.append(row)
            gt_rows = []
            for gi, gt in chosen_gts:
                gt_rows.append({'gt_index': gi, **gt_identity(nusc, gt),
                                'center_global': list(gt.translation),
                                'size_wlh': list(gt.size),
                                'rotation_wxyz': list(gt.rotation)})
            pred_rows = []
            local_order = sorted(pred_indices, key=lambda pi:
                                 matches[thresholds[-1]][1][token, pi])
            for local_rank, pi in enumerate(local_order, 1):
                pred = pred_by_index[pi]
                pred_rows.append({'prediction_index': pi, 'prediction_id': token + ':' + str(pi),
                                  'score': pred.detection_score,
                                  'global_class_rank': matches[thresholds[-1]][1][token, pi],
                                  'local_score_rank': local_rank,
                                  'center_global': list(pred.translation),
                                  'size_wlh': list(pred.size),
                                  'rotation_wxyz': list(pred.rotation)})
            outcomes = {}
            for threshold in thresholds:
                assignments, _, _ = matches[threshold]
                gt_owner = {assignments[token, pi]: pi for pi in pred_indices
                            if assignments.get((token, pi)) in group_set}
                tp_in_group = sum(assignments.get((token, pi)) in group_set
                                  for pi in pred_indices)
                matched_elsewhere = sum((token, pi) in assignments and
                                       assignments[token, pi] not in group_set
                                       for pi in pred_indices)
                fp = len(pred_indices) - tp_in_group - matched_elsewhere
                center_match = diagnostic_assignment(pair_lookup, gt_indices,
                                                      pred_indices, threshold, 'center',
                                                      args.min_bev_iou)
                bev_match = diagnostic_assignment(pair_lookup, gt_indices,
                                                   pred_indices, threshold, 'bev_iou',
                                                   args.min_bev_iou)
                if len(center_match) > len(gt_owner):
                    pattern = 'assignment_sensitive'
                elif len(center_match) < len(gt_indices):
                    pattern = 'insufficient_close_predictions'
                elif len(bev_match) < len(gt_indices):
                    pattern = 'center_covered_but_poor_overlap'
                else:
                    pattern = 'covered_at_threshold'
                counts[str(threshold) + ':' + pattern] += 1
                row = {'sample_token': token, 'group_gt_count': len(gt_indices),
                       'local_prediction_count': len(pred_indices), 'threshold_m': threshold,
                       'official_tp_gt_count': len(gt_owner),
                       'official_fn_gt_count': len(gt_indices) - len(gt_owner),
                       'local_prediction_tp_in_group': tp_in_group,
                       'local_prediction_matched_elsewhere': matched_elsewhere,
                       'local_prediction_fp': fp,
                       'center_counterfactual_tp_count': len(center_match),
                       'bev_counterfactual_tp_count': len(bev_match),
                       'diagnostic_pattern': pattern}
                writer.writerow(row)
                outcomes[str(threshold)] = {
                    'summary': row,
                    'official_gt_to_prediction': {str(gi): token + ':' + str(pi)
                                                  for gi, pi in gt_owner.items()},
                    'official_prediction_to_gt': {str(pi): assignments.get((token, pi))
                                                  for pi in pred_indices},
                    'center_counterfactual': center_match,
                    'bev_counterfactual': bev_match}
            detail_handle.write(json.dumps({'sample_token': token,
                                            'ground_truth': gt_rows,
                                            'predictions': pred_rows,
                                            'pair_matrix': pair_rows,
                                            'thresholds': outcomes}) + '\n')
            selected += 1
    summary = {'selected_keyframes': selected, 'one_group_per_keyframe': True,
               'thresholds_m': thresholds, 'patterns_by_threshold': dict(counts),
               'cluster_distance_m': args.cluster_distance,
               'prediction_radius_m': args.prediction_radius,
               'minimum_bev_iou_for_counterfactual': args.min_bev_iou}
    (args.output_dir / 'group_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent / 'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs')
    parser.add_argument('--sample-token', help='Analyze one sample for a quick inspection')
    parser.add_argument('--prediction-radius', type=float, default=6.0)
    parser.add_argument('--cluster-distance', type=float, default=4.0,
                        help='Connect pedestrian GT centers this close in one group')
    parser.add_argument('--min-bev-iou', type=float, default=0.1,
                        help='Minimum IoU for the diagnostic overlap assignment')
    args = parser.parse_args()
    if args.cluster_distance <= 0 or args.prediction_radius <= 0 or not 0 <= args.min_bev_iou <= 1:
        parser.error('Distances must be positive and min-bev-iou must be in [0, 1]')
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved predictions or nuScenes trainval metadata missing')
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = NuScenesEval(nusc, config_factory('detection_cvpr_2019'),
                            str(args.results_json), 'val', str(args.output_dir / 'devkit_work'),
                            verbose=False)
    summary = analyze_groups(evaluator, nusc, args)
    print(json.dumps(summary, indent=2))
    from interpret import interpret
    interpretation = interpret(args.output_dir / 'groups.jsonl', args.output_dir)
    print('Interpretation written for {} thresholds.'.format(
        len(interpretation['gt_counts_by_threshold'])))


if __name__ == '__main__':
    main()
