#!/usr/bin/env python3
"""Pedestrian localization, official matching, and GT-informed score-order experiment.

The same nuScenes-filtered boxes are used throughout. An oracle score order is
constructed independently for each center-distance threshold. Its AP is a
diagnostic GT-informed score-order result, not official PETR performance or a
deployable score function.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from analyze import overlaps
from localization_capacity import THRESHOLDS, OnlineCoverage, analyze, load_filtered_pedestrians


def distances_for_frame(predictions, ground_truth):
    return [[math.hypot(pred.translation[0] - gt.translation[0],
                        pred.translation[1] - gt.translation[1])
             for gt in ground_truth] for pred in predictions]


def selected_matches(gt_by_sample, pred_list, threshold):
    """Return pred index -> matched GT index from maximum-cardinality matching."""
    by_frame = {}
    for index, pred in enumerate(pred_list):
        by_frame.setdefault(pred.sample_token, []).append(index)
    selected = {}
    for token, indices in by_frame.items():
        gt = gt_by_sample[token]
        distances = distances_for_frame([pred_list[i] for i in indices], gt)
        state = OnlineCoverage()
        for row in distances:
            neighbors = sorted((gi for gi, d in enumerate(row) if d < threshold),
                               key=lambda gi: (row[gi], gi))
            state.add(token, neighbors)
        for (_, gi), local_index in state.owner.items():
            selected[indices[local_index]] = (gi, distances[local_index][gi],
                                               sum(d < threshold for d in distances[local_index]))
    return selected


def minimum_distance_maximum_matches(gt_by_sample, pred_list, threshold):
    """Lexicographic oracle: maximize matches, then minimize total XY error.

    Dummy GT assignments have a penalty larger than the maximum possible sum
    of valid distances in one frame, so the assignment solver prioritizes
    cardinality before distance. A prediction can be used only once.
    """
    from scipy.optimize import linear_sum_assignment

    by_frame = {}
    for index, pred in enumerate(pred_list):
        by_frame.setdefault(pred.sample_token, []).append(index)
    selected = {}
    for token, indices in by_frame.items():
        gt = gt_by_sample[token]
        if not gt:
            continue
        distances = distances_for_frame([pred_list[i] for i in indices], gt)
        candidates = [pi for pi, row in enumerate(distances)
                      if any(distance < threshold for distance in row)]
        if not candidates:
            continue
        n_gt = len(gt)
        penalty = n_gt * threshold + 1.0
        costs = [[float('inf')] * len(candidates) + [penalty] * n_gt
                 for _ in range(n_gt)]
        for col, local_pi in enumerate(candidates):
            for gi, distance in enumerate(distances[local_pi]):
                if distance < threshold:
                    costs[gi][col] = distance
        gt_rows, pred_cols = linear_sum_assignment(costs)
        for gi, col in zip(gt_rows, pred_cols):
            if col < len(candidates):
                local_pi = candidates[col]
                distance = distances[local_pi][gi]
                if not distance < threshold:
                    raise AssertionError('Invalid minimum-cost match')
                selected[indices[local_pi]] = (
                    gi, distance,
                    sum(d < threshold for d in distances[local_pi]))
    return selected


def order_from_selected(pred_list, selected):
    # Put selected candidates first. Within that group, prioritize constrained
    # candidates, then their matched distance. Retain PETR order as final tie break.
    priority = sorted(selected, key=lambda i: (selected[i][2], selected[i][1], i))
    priority.extend(i for i in range(len(pred_list)) if i not in selected)
    return priority, selected


def oracle_order(gt_by_sample, pred_list, threshold):
    return order_from_selected(pred_list,
                               selected_matches(gt_by_sample, pred_list, threshold))


def distance_oracle_order(gt_by_sample, pred_list, threshold):
    return order_from_selected(pred_list,
                               minimum_distance_maximum_matches(
                                   gt_by_sample, pred_list, threshold))


def maximum_iou3d_maximum_matches(gt_by_sample, pred_list, threshold):
    """Prefer positive-overlap pairs, then IoU, then zero-overlap coverage.

    Among zero-IoU edges, the smaller XY center distance wins. The assignment
    is one-to-one. Positive-overlap count takes priority over filling an
    additional zero-overlap GT, so the latter cannot displace a positive pair.
    """
    from scipy.optimize import linear_sum_assignment

    by_frame = {}
    for index, pred in enumerate(pred_list):
        by_frame.setdefault(pred.sample_token, []).append(index)
    selected = {}
    for token, indices in by_frame.items():
        gt = gt_by_sample[token]
        if not gt:
            continue
        distances = distances_for_frame([pred_list[i] for i in indices], gt)
        candidates = [pi for pi, row in enumerate(distances)
                      if any(distance < threshold for distance in row)]
        if not candidates:
            continue
        n_gt = len(gt)
        ious = {}
        for local_pi in candidates:
            for gi, distance in enumerate(distances[local_pi]):
                if distance < threshold:
                    ious[local_pi, gi] = overlaps(pred_list[indices[local_pi]], gt[gi])[1]
        used_gt, used_pred = set(), set()
        positive = [pi for pi in candidates
                    if any(ious.get((pi, gi), 0) > 1e-9 for gi in range(n_gt))]
        if positive:
            # Stage 1: maximum number of positive-overlap pairs, then maximum
            # total IoU. Dummies allow GTs without a positive edge to wait.
            penalty = n_gt + 1.0
            costs = [[float('inf')] * len(positive) + [penalty] * n_gt
                     for _ in range(n_gt)]
            for col, pi in enumerate(positive):
                for gi in range(n_gt):
                    if ious.get((pi, gi), 0) > 1e-9:
                        costs[gi][col] = 1.0 - ious[pi, gi]
            gt_rows, pred_cols = linear_sum_assignment(costs)
            for gi, col in zip(gt_rows, pred_cols):
                if col >= len(positive):
                    continue
                local_pi = positive[col]
                used_gt.add(gi); used_pred.add(local_pi)
                selected[indices[local_pi]] = (
                    gi, distances[local_pi][gi],
                    sum(d < threshold for d in distances[local_pi]),
                    ious[local_pi, gi])
        # Stage 2: only unused boxes and GTs may receive zero-overlap pairs.
        # Maximize their count, then minimize total XY center distance.
        remaining_gt = [gi for gi in range(n_gt) if gi not in used_gt]
        remaining_pred = [pi for pi in candidates if pi not in used_pred and
                          any((pi, gi) in ious and ious[pi, gi] <= 1e-9
                              for gi in remaining_gt)]
        if remaining_gt and remaining_pred:
            penalty = len(remaining_gt) * threshold + 1.0
            costs = [[float('inf')] * len(remaining_pred) +
                     [penalty] * len(remaining_gt) for _ in remaining_gt]
            for row, gi in enumerate(remaining_gt):
                for col, pi in enumerate(remaining_pred):
                    if (pi, gi) in ious and ious[pi, gi] <= 1e-9:
                        costs[row][col] = distances[pi][gi]
            gt_rows, pred_cols = linear_sum_assignment(costs)
            for row, col in zip(gt_rows, pred_cols):
                if col >= len(remaining_pred) or not math.isfinite(costs[row][col]):
                    continue
                gi, local_pi = remaining_gt[row], remaining_pred[col]
                selected[indices[local_pi]] = (
                    gi, distances[local_pi][gi],
                    sum(d < threshold for d in distances[local_pi]), 0.0)
    return selected


def iou3d_oracle_order(gt_by_sample, pred_list, threshold):
    selected = maximum_iou3d_maximum_matches(gt_by_sample, pred_list, threshold)
    priority = sorted(selected, key=lambda i: (selected[i][3] <= 1e-9,
                                                -selected[i][3], selected[i][1], i))
    priority.extend(i for i in range(len(pred_list)) if i not in selected)
    return priority, selected


def direct_iou3d_metric_data(selected, order, n_gt, n_predictions,
                             min_recall, min_precision):
    """AP diagnostic from the oracle's actual one-to-one GT assignments.

    This uses the nuScenes 101-point AP interpolation but *not* its nearest-
    center matcher. It must never be reported as official nuScenes AP.
    """
    import numpy as np
    from types import SimpleNamespace

    if len(order) != n_predictions or len(set(order)) != n_predictions:
        raise ValueError('Oracle order must contain every prediction once')
    hits = np.array([int(i in selected) for i in order], dtype=float)
    if not hits.any():
        zeros = np.zeros(101, dtype=float)
        return SimpleNamespace(recall=np.linspace(0, 1, 101), precision=zeros), 0.0
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, n_predictions + 1)
    recall = tp / n_gt
    grid = np.linspace(0, 1, 101)
    interpolated = np.interp(grid, recall, precision, right=0)
    from nuscenes.eval.detection.algo import calc_ap
    data = SimpleNamespace(recall=grid, precision=interpolated)
    return data, calc_ap(data, min_recall, min_precision)


def selected_match_measurements(selected, pred_list, gt_by_sample):
    records = {(pred_list[i].sample_token, match[0]): (i, match[1])
               for i, match in selected.items()}
    return tp_measurements(records, gt_by_sample, pred_list)


def official_ap(gt_by_sample, pred_list, order, cfg, threshold,
                class_name='pedestrian'):
    """Use nuScenes accumulate/calc_ap on unchanged boxes with new scores."""
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.data_classes import DetectionBox

    ground_truth = EvalBoxes()
    prediction = EvalBoxes()
    for token, boxes in gt_by_sample.items():
        ground_truth.add_boxes(token, boxes)
    if order is None:
        for box in pred_list:
            prediction.add_boxes(box.sample_token, [box])
    else:
        n = len(pred_list)
        scores = {index: (n - rank) / (n + 1) for rank, index in enumerate(order)}
        for index, box in enumerate(pred_list):
            clone = DetectionBox.deserialize(box.serialize())
            clone.detection_score = scores[index]
            prediction.add_boxes(clone.sample_token, [clone])
    md = accumulate(ground_truth, prediction, class_name, cfg.dist_fcn_callable,
                    threshold)
    return calc_ap(md, cfg.min_recall, cfg.min_precision)


def greedy_match_records(gt_by_sample, pred_list, order, threshold):
    """Replay official nearest-unmatched matching, retaining the TP pairs."""
    taken = set()
    records = {}
    for i in order:
        pred = pred_list[i]
        token = pred.sample_token
        nearest = None
        nearest_distance = float('inf')
        for gi, gt in enumerate(gt_by_sample[token]):
            if (token, gi) in taken:
                continue
            distance = math.hypot(pred.translation[0] - gt.translation[0],
                                  pred.translation[1] - gt.translation[1])
            if distance < nearest_distance:
                nearest, nearest_distance = gi, distance
        if nearest is not None and nearest_distance < threshold:
            key = (token, nearest)
            taken.add(key)
            records[key] = (i, nearest_distance)
    return records


def tp_measurements(records, gt_by_sample, pred_list):
    """Aggregate original PETR scores and box geometry over matched TPs."""
    scores, distances, ious, ious3d = [], [], [], []
    for (token, gi), (pred_index, distance) in records.items():
        pred = pred_list[pred_index]
        gt = gt_by_sample[token][gi]
        scores.append(pred.detection_score)
        distances.append(distance)
        bev_iou, iou3d = overlaps(pred, gt)
        ious.append(bev_iou)
        ious3d.append(iou3d)
    if not scores:
        return {'mean_original_score': float('nan'),
                'median_original_score': float('nan'),
                'mean_xy_center_error_m': float('nan'),
                'mean_bev_iou': float('nan'),
                'mean_3d_iou': float('nan')}
    return {'mean_original_score': statistics.mean(scores),
            'median_original_score': statistics.median(scores),
            'mean_xy_center_error_m': statistics.mean(distances),
            'mean_bev_iou': statistics.mean(ious),
            'mean_3d_iou': statistics.mean(ious3d)}


def gallery_groups(manifest_path, groups_path):
    if not manifest_path.is_file() or not groups_path.is_file():
        return {}
    with manifest_path.open(newline='') as handle:
        tokens = {row['sample_token'] for row in csv.DictReader(handle)}
    with groups_path.open() as handle:
        return {group['sample_token']: group for group in map(json.loads, handle)
                if group['sample_token'] in tokens}


def gallery_oracle_rows(groups, gt_by_sample, pred_list, order, records):
    """Map full-frame oracle assignments to the gallery's existing IDs."""
    oracle_ranks = {pred_index: rank for rank, pred_index in enumerate(order, 1)}
    pred_owner = {pred_index: (token, gi)
                  for (token, gi), (pred_index, _) in records.items()}
    result = {}
    for token, group in groups.items():
        group_gt_map = {}
        for gi, gt in enumerate(gt_by_sample[token]):
            for entry in group['ground_truth']:
                if max(abs(a-b) for a, b in zip(gt.translation,
                                                 entry['center_global'])) < 1e-4:
                    group_gt_map[gi] = entry['gt_index']
                    break
        if len(group_gt_map) != len(group['ground_truth']):
            raise ValueError('Cannot map all gallery GTs for ' + token)
        local = {}
        for entry in group['predictions']:
            index = entry['global_class_rank'] - 1
            pred = pred_list[index]
            if pred.sample_token != token or max(abs(a-b) for a, b in zip(
                    pred.translation, entry['center_global'])) > 1e-4:
                raise ValueError('Gallery prediction rank disagrees with filtered oracle list')
            owner = pred_owner.get(index)
            local[str(entry['prediction_index'])] = {
                'oracle_global_rank': oracle_ranks[index],
                'assigned_group_gt': group_gt_map.get(owner[1])
                    if owner and owner[0] == token else None,
                'tp_outside_group': bool(owner and owner[0] == token and
                                         owner[1] not in group_gt_map),
            }
        ranked_local = sorted(local, key=lambda pi: local[pi]['oracle_global_rank'])
        for rank, pi in enumerate(ranked_local, 1):
            local[pi]['oracle_local_rank'] = rank
        result[token] = local
    return result


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs' /
                        'reranking_oracle')
    args = parser.parse_args()
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved baseline predictions or nuScenes trainval metadata missing')
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
    except ImportError:
        parser.error('SciPy is required for the minimum-distance localization oracle')

    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    cfg = config_factory('detection_cvpr_2019')
    gt, predictions = load_filtered_pedestrians(args.results_json, args.data_root)
    gallery = gallery_groups(base / 'outputs/visualizations/manifest.csv',
                             base / 'outputs/groups.jsonl')
    gallery_results = {token: {} for token in gallery}
    distance_gallery_results = {token: {} for token in gallery}
    iou3d_gallery_results = {token: {} for token in gallery}
    summaries, _, _ = analyze(gt, predictions)
    gt_count = summaries[0]['gt_count']
    rows = []
    original_order = list(range(len(predictions)))
    for summary in summaries:
        threshold = summary['threshold_m']
        order, selected = oracle_order(gt, predictions, threshold)
        distance_order, distance_selected = distance_oracle_order(
            gt, predictions, threshold)
        iou3d_order, iou3d_selected = iou3d_oracle_order(
            gt, predictions, threshold)
        if len(distance_selected) != len(selected) or len(iou3d_selected) != len(selected):
            raise AssertionError('Geometry oracle changed maximum cardinality')
        oracle_records = greedy_match_records(gt, predictions, order, threshold)
        distance_records = greedy_match_records(gt, predictions, distance_order,
                                                threshold)
        iou3d_records = greedy_match_records(gt, predictions, iou3d_order,
                                              threshold)
        if gallery:
            mapped = gallery_oracle_rows(gallery, gt, predictions, order, oracle_records)
            for token, local in mapped.items():
                gallery_results[token][str(threshold)] = local
            distance_mapped = gallery_oracle_rows(
                gallery, gt, predictions, distance_order, distance_records)
            for token, local in distance_mapped.items():
                distance_gallery_results[token][str(threshold)] = local
            iou3d_mapped = gallery_oracle_rows(
                gallery, gt, predictions, iou3d_order, iou3d_records)
            for token, local in iou3d_mapped.items():
                iou3d_gallery_results[token][str(threshold)] = local
        baseline_records = greedy_match_records(gt, predictions, original_order, threshold)
        oracle_tp = len(oracle_records)
        baseline_tp = len(baseline_records)
        if baseline_tp != summary['official_greedy_full_set_matches']:
            raise AssertionError('Baseline greedy counts disagree')
        if len(selected) != summary['full_set_maximum_one_to_one_matches']:
            raise AssertionError('Maximum matching counts disagree')
        baseline_ap = official_ap(gt, predictions, None, cfg, threshold)
        oracle_ap = official_ap(gt, predictions, order, cfg, threshold)
        distance_ap = official_ap(gt, predictions, distance_order, cfg, threshold)
        iou3d_ap = official_ap(gt, predictions, iou3d_order, cfg, threshold)
        baseline_measurements = tp_measurements(baseline_records, gt, predictions)
        oracle_measurements = tp_measurements(oracle_records, gt, predictions)
        distance_measurements = tp_measurements(distance_records, gt, predictions)
        iou3d_measurements = tp_measurements(iou3d_records, gt, predictions)
        cardinality_geometry = selected_match_measurements(selected,
                                                           predictions, gt)
        lex_geometry = selected_match_measurements(distance_selected,
                                                   predictions, gt)
        iou3d_geometry = selected_match_measurements(iou3d_selected,
                                                     predictions, gt)
        if iou3d_geometry['mean_3d_iou'] + 1e-8 < max(
                cardinality_geometry['mean_3d_iou'], lex_geometry['mean_3d_iou']):
            raise AssertionError('3D IoU oracle did not maximize total 3D IoU')
        cardinality_error = sum(match[1] for match in selected.values())
        minimum_error = sum(match[1] for match in distance_selected.values())
        if minimum_error > cardinality_error + 1e-6:
            raise AssertionError('Minimum-distance oracle increased total XY error')
        shared_gt = baseline_records.keys() & oracle_records.keys()
        baseline_shared = tp_measurements({key: baseline_records[key] for key in shared_gt},
                                          gt, predictions)
        oracle_shared = tp_measurements({key: oracle_records[key] for key in shared_gt},
                                        gt, predictions)
        row = {
            'threshold_m': threshold,
            'gt_count': gt_count,
            'prediction_count': len(predictions),
            'maximum_one_to_one_matches': len(selected),
            'maximum_recall_percent': 100 * len(selected) / gt_count,
            'baseline_full_set_tp': baseline_tp,
            'baseline_full_set_recall_percent': 100 * baseline_tp / gt_count,
            'oracle_rank_full_set_tp': oracle_tp,
            'oracle_rank_full_set_recall_percent': 100 * oracle_tp / gt_count,
            'baseline_ap_percent': 100 * baseline_ap,
            'oracle_rank_ap_percent': 100 * oracle_ap,
            'oracle_ap_gain_percentage_points': 100 * (oracle_ap - baseline_ap),
            'oracle_rank_remaining_matching_gap': len(selected) - oracle_tp,
            'baseline_tp_mean_original_score': baseline_measurements['mean_original_score'],
            'oracle_tp_mean_original_score': oracle_measurements['mean_original_score'],
            'baseline_tp_median_original_score': baseline_measurements['median_original_score'],
            'oracle_tp_median_original_score': oracle_measurements['median_original_score'],
            'baseline_tp_mean_xy_center_error_m': baseline_measurements['mean_xy_center_error_m'],
            'oracle_tp_mean_xy_center_error_m': oracle_measurements['mean_xy_center_error_m'],
            'baseline_tp_mean_bev_iou': baseline_measurements['mean_bev_iou'],
            'oracle_tp_mean_bev_iou': oracle_measurements['mean_bev_iou'],
            'shared_gt_tp_count': len(shared_gt),
            'baseline_shared_gt_mean_xy_center_error_m': baseline_shared['mean_xy_center_error_m'],
            'oracle_shared_gt_mean_xy_center_error_m': oracle_shared['mean_xy_center_error_m'],
            'baseline_shared_gt_mean_bev_iou': baseline_shared['mean_bev_iou'],
            'oracle_shared_gt_mean_bev_iou': oracle_shared['mean_bev_iou'],
            'cardinality_oracle_total_xy_error_m': cardinality_error,
            'cardinality_oracle_mean_original_score': cardinality_geometry['mean_original_score'],
            'cardinality_oracle_mean_xy_error_m': cardinality_geometry['mean_xy_center_error_m'],
            'cardinality_oracle_mean_bev_iou': cardinality_geometry['mean_bev_iou'],
            'min_distance_oracle_total_xy_error_m': minimum_error,
            'min_distance_oracle_mean_original_score': lex_geometry['mean_original_score'],
            'min_distance_oracle_mean_xy_error_m': lex_geometry['mean_xy_center_error_m'],
            'min_distance_oracle_mean_bev_iou': lex_geometry['mean_bev_iou'],
            'min_distance_center_error_reduction_m': cardinality_error - minimum_error,
            'min_distance_rank_full_set_tp': len(distance_records),
            'min_distance_rank_full_set_recall_percent': 100 * len(distance_records) / gt_count,
            'min_distance_rank_ap_percent': 100 * distance_ap,
            'min_distance_rank_ap_gain_percentage_points': 100 * (distance_ap - baseline_ap),
            'min_distance_rank_remaining_matching_gap': len(distance_selected) - len(distance_records),
            'min_distance_rank_tp_mean_original_score': distance_measurements['mean_original_score'],
            'min_distance_rank_tp_mean_xy_center_error_m': distance_measurements['mean_xy_center_error_m'],
            'min_distance_rank_tp_mean_bev_iou': distance_measurements['mean_bev_iou'],
            'baseline_tp_mean_3d_iou': baseline_measurements['mean_3d_iou'],
            'oracle_tp_mean_3d_iou': oracle_measurements['mean_3d_iou'],
            'min_distance_oracle_mean_3d_iou': lex_geometry['mean_3d_iou'],
            'min_distance_rank_tp_mean_3d_iou': distance_measurements['mean_3d_iou'],
            'max_iou3d_oracle_mean_original_score': iou3d_geometry['mean_original_score'],
            'max_iou3d_oracle_mean_xy_error_m': iou3d_geometry['mean_xy_center_error_m'],
            'max_iou3d_oracle_mean_bev_iou': iou3d_geometry['mean_bev_iou'],
            'max_iou3d_oracle_mean_3d_iou': iou3d_geometry['mean_3d_iou'],
            'max_iou3d_rank_full_set_tp': len(iou3d_records),
            'max_iou3d_rank_full_set_recall_percent': 100 * len(iou3d_records) / gt_count,
            'max_iou3d_rank_ap_percent': 100 * iou3d_ap,
            'max_iou3d_rank_tp_mean_original_score': iou3d_measurements['mean_original_score'],
            'max_iou3d_rank_tp_mean_xy_center_error_m': iou3d_measurements['mean_xy_center_error_m'],
            'max_iou3d_rank_tp_mean_bev_iou': iou3d_measurements['mean_bev_iou'],
            'max_iou3d_rank_tp_mean_3d_iou': iou3d_measurements['mean_3d_iou'],
            'max_iou3d_rank_remaining_matching_gap': len(selected) - len(iou3d_records),
        }
        rows.append(row)
        print('{threshold_m:g} m: maximum {maximum_one_to_one_matches}/{gt_count}; '
              'TPs baseline {baseline_full_set_tp} / cardinality-rank {oracle_rank_full_set_tp} '
              '/ min-distance-rank {min_distance_rank_full_set_tp}; '
              'AP {baseline_ap_percent:.2f}% / {oracle_rank_ap_percent:.2f}% '
              '/ {min_distance_rank_ap_percent:.2f}%'.format(**row),
              flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'pedestrian_oracle_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'source.json').write_text(json.dumps({
        'results_json': str(args.results_json.resolve()),
        'data_root': str(args.data_root.resolve()),
        'class': 'pedestrian',
        'thresholds_m': THRESHOLDS,
        'matching_distance': 'XY center distance, strictly below threshold',
        'prediction_set': 'official nuScenes filtered, max boxes per sample enforced',
        'oracle_scores': 'unique positive GT-informed ranks; prediction boxes unchanged',
        'tp_score_measurement': 'mean and median ORIGINAL PETR scores of matched TPs',
        'tp_geometry_measurement': 'mean XY center error and oriented BEV box IoU over matched TPs',
        'minimum_distance_oracle': 'maximize matched GT count, then minimize total XY center distance; one-to-one per frame',
        'minimum_distance_rank': 'selected min-distance pairs prioritized, then official greedy matching and AP',
        'maximum_iou3d_oracle': 'maximize matched GT count, then maximize total oriented 3D IoU over center-distance-eligible pairs',
        'maximum_iou3d_rank': 'selected max-3D-IoU pairs prioritized, then official greedy matching and AP',
        'shared_gt_measurement': 'same GTs matched by both orders; boxes may differ',
        'interpretation': 'diagnostic oracle AP, not official PETR AP',
    }, indent=2) + '\n')
    if gallery:
        (args.output_dir / 'gallery_oracle_assignments.json').write_text(
            json.dumps({'sample_tokens': gallery_results,
                        'distance_sample_tokens': distance_gallery_results,
                        'iou3d_sample_tokens': iou3d_gallery_results,
                        'source_results_json': str(args.results_json.resolve()),
                        'thresholds_m': THRESHOLDS}, indent=2) + '\n')


if __name__ == '__main__':
    main()
