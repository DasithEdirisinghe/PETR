#!/usr/bin/env python3
"""Compute baseline PETR AP by class, ego-range slice, and center threshold."""

import argparse
import csv
import json
from pathlib import Path


RANGES = ((0, 10, '0–10 m'), (10, 20, '10–20 m'),
          (20, 30, '20–30 m'), (30, 40, '30–40 m'),
          (40, 50, '40–50 m'), (50, float('inf'), '>50 m'))
FIELDS = ('class', 'gt_range', 'threshold_m', 'ap_percent', 'gt_count',
          'prediction_count', 'matched_gt_count', 'gt_coverage_percent')


def range_label(distance):
    for lower, upper, label in RANGES:
        if lower <= distance < upper or (upper == 50 and distance == 50):
            return label
    raise ValueError('Invalid ego distance {}'.format(distance))


def count_matches(gt_boxes, pred_boxes, class_name, distance_fn, threshold):
    """Count TPs using the same confidence order and greedy rule as accumulate."""
    predictions = [b for b in pred_boxes.all if b.detection_name == class_name]
    ranked = [i for _, i in sorted((b.detection_score, i)
              for i, b in enumerate(predictions))][::-1]
    taken = set()
    for index in ranked:
        pred = predictions[index]
        nearest_index, nearest_distance = None, float('inf')
        for gt_index, gt in enumerate(gt_boxes[pred.sample_token]):
            if gt.detection_name != class_name or (pred.sample_token, gt_index) in taken:
                continue
            distance = distance_fn(gt, pred)
            if distance < nearest_distance:
                nearest_index, nearest_distance = gt_index, distance
        if nearest_index is not None and nearest_distance < threshold:
            taken.add((pred.sample_token, nearest_index))
    return len(taken)


def slice_boxes(boxes, sample_tokens, name, label, eval_boxes_type):
    result = eval_boxes_type()
    count = 0
    for token in sample_tokens:
        selected = [box for box in boxes[token]
                    if box.detection_name == name and
                    (label == 'All ranges' or range_label(box.ego_dist) == label)]
        result.add_boxes(token, selected)
        count += len(selected)
    return result, count


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--results-json', type=Path, default=Path(
        'experiments/detection_error_analysis/outputs/petr_car_ap_aligned/'
        'formatted/pts_bbox/results_nusc.json'))
    parser.add_argument('--output-dir', type=Path, default=here / 'outputs/baseline')
    args = parser.parse_args()
    if not args.data_root.is_dir() or not args.results_json.is_file():
        parser.error('Both nuScenes data root and PETR results JSON must exist')

    from nuscenes import NuScenes
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    cfg = config_factory('detection_cvpr_2019')
    evaluator = NuScenesEval(nusc, config=cfg, result_path=str(args.results_json),
                             eval_set='val', output_dir=str(args.output_dir / 'evaluator'),
                             verbose=False)
    tokens = evaluator.gt_boxes.sample_tokens
    rows = []
    for name in cfg.class_names:
        for label in [r[2] for r in RANGES] + ['All ranges']:
            gt, n_gt = slice_boxes(evaluator.gt_boxes, tokens, name, label, EvalBoxes)
            pred, n_pred = slice_boxes(evaluator.pred_boxes, tokens, name, label, EvalBoxes)
            print('{} | {} | GT {:,} | predictions {:,}'.format(
                name, label, n_gt, n_pred), flush=True)
            for threshold in cfg.dist_ths:
                matched = count_matches(gt, pred, name, cfg.dist_fcn_callable, threshold)
                ap = (100 * calc_ap(accumulate(gt, pred, name, cfg.dist_fcn_callable,
                                              threshold), cfg.min_recall,
                                   cfg.min_precision) if n_gt else None)
                rows.append({'class': name, 'gt_range': label,
                             'threshold_m': threshold, 'ap_percent': ap,
                             'gt_count': n_gt, 'prediction_count': n_pred,
                             'matched_gt_count': matched,
                             'gt_coverage_percent': 100 * matched / n_gt if n_gt else None})

    # Check unsliced AP against archived official metrics before reporting slices.
    official_path = args.results_json.with_name('metrics_summary.json')
    if official_path.is_file():
        with official_path.open() as handle:
            official = json.load(handle)
        for row in rows:
            if row['gt_range'] == 'All ranges':
                expected = 100 * official['label_aps'][row['class']][str(row['threshold_m'])]
                if abs(row['ap_percent'] - expected) > 1e-6:
                    raise AssertionError('Whole-range AP mismatch for {} @{}: {} vs {}'.format(
                        row['class'], row['threshold_m'], row['ap_percent'], expected))
    with (args.output_dir / 'range_sliced_ap.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / 'method.json').open('w') as handle:
        json.dump({'source': str(args.results_json),
                   'range_rule': 'GT binned by GT ego distance; prediction binned by predicted ego distance; 40–50 includes 50 m',
                   'ap_method': 'nuScenes accumulate and calc_ap with detection_cvpr_2019 config',
                   'scope': 'official filtered validation boxes; class and range sliced before matching',
                   'counts': 'prediction_count is filtered submitted predictions in predicted range; matched_gt_count is threshold TPs'},
                  handle, indent=2)
    print('Saved:', args.output_dir / 'range_sliced_ap.csv')


if __name__ == '__main__':
    main()
