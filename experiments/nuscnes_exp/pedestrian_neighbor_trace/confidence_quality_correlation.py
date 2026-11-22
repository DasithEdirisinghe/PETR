#!/usr/bin/env python3
"""Correlate PETR confidence with RARE-style best same-class 3D IoU.

Uses the official nuScenes validation prediction set after max-box and range
filtering. Quality is computed per prediction, independently of TP assignment.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from analyze import overlaps
from localization_capacity import THRESHOLDS


DETAIL_FIELDS = ('sample_token', 'class', 'prediction_index_in_sample',
                 'original_score', 'best_same_class_3d_iou',
                 'best_gt_index_in_sample', 'best_gt_xy_center_distance_m',
                 'nearest_gt_index_in_sample', 'nearest_gt_xy_center_distance_m') + tuple(
                     'baseline_tp_at_{}m'.format(str(t).replace('.', 'p'))
                     for t in THRESHOLDS) + tuple(
                         'baseline_assigned_gt_at_{}m'.format(str(t).replace('.', 'p'))
                         for t in THRESHOLDS) + tuple(
                             'baseline_assigned_xy_distance_at_{}m'.format(
                                 str(t).replace('.', 'p')) for t in THRESHOLDS)


def threshold_label(threshold):
    return '{}m'.format(str(threshold).replace('.', 'p'))


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    if xx <= 0 or yy <= 0:
        return None
    xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return max(-1.0, min(1.0, xy / math.sqrt(xx * yy)))


def average_ranks(values):
    """Ascending ranks, with average ranks for ties (Spearman convention)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def possible_3d_overlap(pred, gt):
    """Cheap conservative rejection before oriented-box clipping."""
    if abs(pred.translation[2] - gt.translation[2]) >= \
            (pred.size[2] + gt.size[2]) / 2.0:
        return False
    xy_distance = math.hypot(pred.translation[0] - gt.translation[0],
                             pred.translation[1] - gt.translation[1])
    pred_radius = math.hypot(pred.size[0], pred.size[1]) / 2.0
    gt_radius = math.hypot(gt.size[0], gt.size[1]) / 2.0
    return xy_distance < pred_radius + gt_radius


def best_quality(pred, same_class_gt):
    """RARE Eq. 1: maximum 3D IoU over same-class GTs in this frame."""
    best_iou, best_index, best_distance = 0.0, None, None
    for gt_index, gt in same_class_gt:
        if not possible_3d_overlap(pred, gt):
            continue
        iou = overlaps(pred, gt)[1]
        if iou > best_iou:
            best_iou = iou
            best_index = gt_index
            best_distance = math.hypot(pred.translation[0] - gt.translation[0],
                                       pred.translation[1] - gt.translation[1])
    return best_iou, best_index, best_distance


def nearest_center(pred, same_class_gt):
    if not same_class_gt:
        return None, None
    distance, gt_index = min(
        ((math.hypot(pred.translation[0] - gt.translation[0],
                     pred.translation[1] - gt.translation[1]), gi)
         for gi, gt in same_class_gt), key=lambda item: (item[0], item[1]))
    return gt_index, distance


def baseline_tp_indices(predictions, same_class_gt, threshold):
    """Replay confidence-ordered nearest-unmatched nuScenes center matching."""
    matched_gt, matched_predictions = set(), {}
    ordered = sorted(predictions, key=lambda item: (item[1].detection_score,
                                                     item[0]), reverse=True)
    for pi, pred in ordered:
        choices = ((math.hypot(pred.translation[0] - gt.translation[0],
                               pred.translation[1] - gt.translation[1]), gi)
                   for gi, gt in same_class_gt if gi not in matched_gt)
        nearest = min(choices, default=(float('inf'), None))
        if nearest[0] < threshold:
            matched_gt.add(nearest[1])
            matched_predictions[pi] = (nearest[1], nearest[0])
    return matched_predictions


def center_summary(class_name, subset, threshold, pairs, excluded_no_gt):
    scores = [score for score, _ in pairs]
    distances = [distance for _, distance in pairs]
    negative_distances = [-distance for distance in distances]
    return {
        'class': class_name,
        'subset': subset,
        'threshold_m': '' if threshold is None else threshold,
        'prediction_count': len(pairs),
        'excluded_no_same_class_gt_count': excluded_no_gt,
        'mean_xy_center_distance_m': (
            sum(distances) / len(distances) if distances else None),
        'pearson_score_vs_negative_distance': pearson(scores, negative_distances),
        'spearman_score_vs_negative_distance': spearman(scores, negative_distances),
    }


def summarize(class_name, pairs):
    scores = [score for score, _ in pairs]
    qualities = [quality for _, quality in pairs]
    positive = [(score, quality) for score, quality in pairs if quality > 0]
    positive_scores = [score for score, _ in positive]
    positive_qualities = [quality for _, quality in positive]
    n = len(pairs)
    return {
        'class': class_name,
        'prediction_count': n,
        'positive_iou_prediction_count': len(positive),
        'zero_iou_prediction_count': n - len(positive),
        'positive_iou_fraction': len(positive) / n if n else None,
        'mean_original_score': sum(scores) / n if n else None,
        'mean_best_3d_iou': sum(qualities) / n if n else None,
        'pearson_all': pearson(scores, qualities),
        'spearman_all': spearman(scores, qualities),
        'pearson_positive_iou': pearson(positive_scores, positive_qualities),
        'spearman_positive_iou': spearman(positive_scores, positive_qualities),
    }


def load_filtered_boxes(results_json, data_root):
    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                              load_gt, load_prediction)
    from nuscenes.eval.detection.data_classes import DetectionBox

    nusc = NuScenes(version='v1.0-trainval', dataroot=str(data_root), verbose=False)
    cfg = config_factory('detection_cvpr_2019')
    pred_boxes, _ = load_prediction(str(results_json), cfg.max_boxes_per_sample,
                                    DetectionBox, verbose=False)
    gt_boxes = load_gt(nusc, 'val', DetectionBox, verbose=False)
    if set(pred_boxes.sample_tokens) != set(gt_boxes.sample_tokens):
        raise ValueError('Prediction sample tokens differ from validation GT')
    pred_boxes = filter_eval_boxes(nusc, add_center_dist(nusc, pred_boxes),
                                   cfg.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, add_center_dist(nusc, gt_boxes),
                                 cfg.class_range, verbose=False)
    return pred_boxes, gt_boxes, list(cfg.class_names)


def run(pred_boxes, gt_boxes, classes, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_by_class = {name: [] for name in classes}
    center_pairs = {(name, subset, threshold): []
                    for name in classes
                    for subset, threshold in [('all_with_gt', None)] + [
                        (kind, t) for t in THRESHOLDS
                        for kind in ('nearest_within_threshold', 'baseline_tp')]}
    excluded_no_gt = {name: 0 for name in classes}
    detail_path = output_dir / 'prediction_quality.csv'
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for token in pred_boxes.sample_tokens:
            gt_by_class = {name: [] for name in classes}
            for gi, gt in enumerate(gt_boxes[token]):
                if gt.detection_name in gt_by_class:
                    gt_by_class[gt.detection_name].append((gi, gt))
            pred_by_class = {name: [] for name in classes}
            for pi, pred in enumerate(pred_boxes[token]):
                name = pred.detection_name
                if name in pred_by_class:
                    pred_by_class[name].append((pi, pred))
            tp_by_class = {
                name: {t: baseline_tp_indices(pred_by_class[name], gt_by_class[name], t)
                       for t in THRESHOLDS}
                for name in classes
            }
            for pi, pred in enumerate(pred_boxes[token]):
                name = pred.detection_name
                if name not in pairs_by_class:
                    continue
                quality, gt_index, distance = best_quality(pred, gt_by_class[name])
                nearest_index, nearest_distance = nearest_center(pred, gt_by_class[name])
                score = float(pred.detection_score)
                pairs_by_class[name].append((score, quality))
                if nearest_distance is None:
                    excluded_no_gt[name] += 1
                else:
                    pair = (score, nearest_distance)
                    center_pairs[name, 'all_with_gt', None].append(pair)
                    for threshold in THRESHOLDS:
                        if nearest_distance < threshold:
                            center_pairs[name, 'nearest_within_threshold', threshold].append(pair)
                        tp = tp_by_class[name][threshold].get(pi)
                        if tp is not None:
                            center_pairs[name, 'baseline_tp', threshold].append(
                                (score, tp[1]))
                detail = {
                    'sample_token': token,
                    'class': name,
                    'prediction_index_in_sample': pi,
                    'original_score': score,
                    'best_same_class_3d_iou': quality,
                    'best_gt_index_in_sample': '' if gt_index is None else gt_index,
                    'best_gt_xy_center_distance_m': '' if distance is None else distance,
                    'nearest_gt_index_in_sample': (
                        '' if nearest_index is None else nearest_index),
                    'nearest_gt_xy_center_distance_m': (
                        '' if nearest_distance is None else nearest_distance),
                }
                for threshold in THRESHOLDS:
                    tp = tp_by_class[name][threshold].get(pi)
                    label = threshold_label(threshold)
                    detail['baseline_tp_at_' + label] = int(tp is not None)
                    detail['baseline_assigned_gt_at_' + label] = (
                        '' if tp is None else tp[0])
                    detail['baseline_assigned_xy_distance_at_' + label] = (
                        '' if tp is None else tp[1])
                writer.writerow(detail)
    rows = [summarize(name, pairs_by_class[name]) for name in classes]
    summary_path = output_dir / 'class_correlation.csv'
    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    center_rows = [center_summary(name, 'all_with_gt', None,
                                  center_pairs[name, 'all_with_gt', None],
                                  excluded_no_gt[name])
                   for name in classes]
    for name in classes:
        for threshold in THRESHOLDS:
            for subset in ('nearest_within_threshold', 'baseline_tp'):
                center_rows.append(center_summary(
                    name, subset, threshold,
                    center_pairs[name, subset, threshold], excluded_no_gt[name]))
    with (output_dir / 'center_distance_correlation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(center_rows[0]))
        writer.writeheader()
        writer.writerows(center_rows)
    return rows, center_rows


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs' /
                        'confidence_quality_correlation')
    parser.add_argument('--classes', nargs='+',
                        help='Subset of nuScenes detection classes; default: all classes')
    args = parser.parse_args()
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved PETR predictions or nuScenes trainval metadata missing')
    pred_boxes, gt_boxes, all_classes = load_filtered_boxes(
        args.results_json, args.data_root)
    classes = all_classes if args.classes is None else args.classes
    if not classes or len(set(classes)) != len(classes) or set(classes) - set(all_classes):
        parser.error('Choose unique classes from: ' + ', '.join(all_classes))
    rows, center_rows = run(pred_boxes, gt_boxes, classes, args.output_dir)
    (args.output_dir / 'source.json').write_text(json.dumps({
        'results_json': str(args.results_json.resolve()),
        'data_root': str(args.data_root.resolve()),
        'eval_set': 'val',
        'classes': classes,
        'prediction_set': 'nuScenes load_prediction maximum boxes per sample, then official class range filtering',
        'quality_target': 'maximum oriented 3D IoU with any same-class GT in the same frame; zero if none overlaps',
        'iou_assignment': 'none; predictions can share a best GT',
        'pearson': 'linear correlation of original PETR score and best 3D IoU',
        'spearman': 'Pearson correlation of average-tie ranks',
        'positive_subset': 'predictions whose best same-class 3D IoU is strictly greater than zero',
        'center_distance': 'nearest same-class GT in XY for all and eligible subsets; assigned GT XY distance for baseline TPs; undefined when the frame has no same-class GT',
        'center_subsets': 'all predictions in frames with same-class GT; nearest center distance < threshold; baseline confidence-greedy TPs at threshold',
        'center_thresholds_m': THRESHOLDS,
        'negative_distance_quality': 'negative XY center distance; positive correlation means higher PETR confidence accompanies smaller center error',
    }, indent=2) + '\n')
    for row in rows:
        print('{}: n={} positive={} Pearson(all)={} Spearman(all)={} '
              'Pearson(positive)={} Spearman(positive)={}'.format(
                  row['class'], row['prediction_count'],
                  row['positive_iou_prediction_count'],
                  row['pearson_all'], row['spearman_all'],
                  row['pearson_positive_iou'], row['spearman_positive_iou']),
              flush=True)
    for row in center_rows:
        print('{} {} {}: n={} Pearson(score,-distance)={} Spearman(score,-distance)={}'.format(
            row['class'], row['subset'], row['threshold_m'],
            row['prediction_count'], row['pearson_score_vs_negative_distance'],
            row['spearman_score_vs_negative_distance']), flush=True)
    print('Wrote {}'.format(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
