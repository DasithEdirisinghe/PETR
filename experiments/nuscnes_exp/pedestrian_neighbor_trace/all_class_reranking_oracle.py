#!/usr/bin/env python3
"""Run fixed-box GT-informed confidence re-ranking for all nuScenes classes.

Each class and center-distance threshold gets its own GT-informed ordering.
The resulting mean AP is a diagnostic average, not AP from one deployable
submission or a proof of the best possible score order.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

from localization_capacity import THRESHOLDS
from reranking_oracle import (distance_oracle_order, greedy_match_records,
                              iou3d_oracle_order, direct_iou3d_metric_data,
                              official_ap, oracle_order,
                              selected_match_measurements, tp_measurements)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


GEOMETRY_METHODS = (
    ('baseline', 'PETR baseline', '#2563eb', 'baseline'),
    ('cardinality', 'One-to-one re-rank', '#7c3aed', 'oracle_rank'),
    ('distance', 'Distance re-rank', '#059669', 'min_distance_rank'),
    ('iou3d', '3D IoU re-rank', '#ea580c', 'max_iou3d_rank'),
)


def geometry_plot_rows(rows):
    """Long-format data for plots of realized greedy TP assignments."""
    result = []
    for row in rows:
        for method, label, _, prefix in GEOMETRY_METHODS:
            result.append({
                'class': row['class'],
                'threshold_m': row['threshold_m'],
                'method': method,
                'method_label': label,
                'tp_count': row[prefix + '_full_set_tp'],
                'mean_xy_center_error_m': row[prefix + '_tp_mean_xy_center_error_m'],
                'mean_3d_iou': row[prefix + '_tp_mean_3d_iou'],
            })
    return result


def write_geometry_overview(path, class_names, rows):
    """One SVG: each class occupies a row, with both TP metrics side by side."""
    by_key = {(row['class'], float(row['threshold_m']), row['method']): row
              for row in rows}
    errors = [float(row['mean_xy_center_error_m']) for row in rows]
    finite_errors = [value for value in errors if math.isfinite(value)]
    error_axis = max(0.25, math.ceil(max(finite_errors, default=0.0) * 4) / 4)
    width, row_height = 1900, 148
    height = 205 + row_height * len(class_names)
    panels = (('mean_3d_iou', 'Mean 3D IoU ↑', 1.0, 175),
              ('mean_xy_center_error_m', 'Mean XY center error ↓ (m)',
               error_axis, 1035))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
             'viewBox="0 0 {} {}">'.format(width, height, width, height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172033}'
             '.title{font-size:24px;font-weight:bold}.head{font-size:18px;font-weight:bold}'
             '.label{font-size:14px}.small{font-size:12px;fill:#475569}</style>',
             '<text x="35" y="38" class="title">Assigned TP geometry by class and threshold</text>',
             '<text x="35" y="63" class="small">Means are over each method’s realized '
             'greedy TPs; matched GT sets and TP counts may differ.</text>']
    for method_index, (_, label, color, _) in enumerate(GEOMETRY_METHODS):
        x = 40 + method_index * 320
        parts.extend(('<rect x="{}" y="78" width="16" height="16" fill="{}"/>'.format(x, color),
                      '<text x="{}" y="91" class="label">{}</text>'.format(
                          x + 24, escape(label))))
    for field, title, axis_max, panel_x in panels:
        parts.append('<text x="{}" y="125" class="head">{}</text>'.format(
            panel_x, escape(title)))
        parts.append('<text x="{}" y="147" class="small">Scale: 0 to {:.2f}</text>'.format(
            panel_x, axis_max))
    for class_index, class_name in enumerate(class_names):
        row_top = 170 + class_index * row_height
        baseline = row_top + 92
        parts.append('<text x="20" y="{}" class="label">{}</text>'.format(
            row_top + 18, escape(class_name.replace('_', ' ').title())))
        if class_index % 2:
            parts.append('<rect x="165" y="{}" width="1710" height="{}" '
                         'fill="#f8fafc"/>'.format(row_top, row_height - 5))
        for field, _, axis_max, panel_x in panels:
            for threshold_index, threshold in enumerate(THRESHOLDS):
                group_x = panel_x + threshold_index * 205
                parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" '
                             'stroke="#b5c3d0"/>'.format(group_x, baseline,
                                                          group_x + 164, baseline))
                parts.append('<text x="{}" y="{}" text-anchor="middle" '
                             'class="small">{:g} m</text>'.format(
                                 group_x + 82, baseline + 18, threshold))
                for method_index, (method, _, color, _) in enumerate(GEOMETRY_METHODS):
                    row = by_key.get((class_name, float(threshold), method))
                    if row is None:
                        continue
                    value = float(row[field])
                    if not math.isfinite(value):
                        continue
                    bar_height = 65 * min(max(value / axis_max, 0.0), 1.0)
                    x = group_x + 7 + method_index * 39
                    y = baseline - bar_height
                    parts.append('<rect x="{}" y="{:.1f}" width="31" height="{:.1f}" '
                                 'fill="{}"/>'.format(x, y, bar_height, color))
                    parts.append('<text x="{}" y="{:.1f}" text-anchor="middle" '
                                 'class="small">{:.2f}</text>'.format(
                                     x + 15, max(row_top + 15, y - 3), value))
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def class_boxes(evaluator, class_name):
    """Preserve official filtering and confidence tie order for one class."""
    gt_by_sample = {
        token: [box for box in evaluator.gt_boxes[token]
                if box.detection_name == class_name]
        for token in evaluator.gt_boxes.sample_tokens
    }
    original_predictions = [box for box in evaluator.pred_boxes.all
                            if box.detection_name == class_name]
    predictions = [original_predictions[i] for i in sorted(
        range(len(original_predictions)),
        key=lambda i: (original_predictions[i].detection_score, i),
        reverse=True)]
    return gt_by_sample, original_predictions, predictions


def compare_class(evaluator, class_name):
    gt, original_predictions, predictions = class_boxes(evaluator, class_name)
    n_gt = sum(map(len, gt.values()))
    if not n_gt or not predictions:
        raise ValueError('Expected GTs and predictions for ' + class_name)
    baseline_order = list(range(len(predictions)))
    rows = []
    for threshold in THRESHOLDS:
        oracle_rank, selected = oracle_order(gt, predictions, threshold)
        distance_rank, distance_selected = distance_oracle_order(
            gt, predictions, threshold)
        iou3d_rank, iou3d_selected = iou3d_oracle_order(
            gt, predictions, threshold)
        if len(distance_selected) != len(selected):
            raise AssertionError('Distance oracle changed maximum cardinality')
        baseline_records = greedy_match_records(gt, predictions, baseline_order,
                                                threshold)
        oracle_records = greedy_match_records(gt, predictions, oracle_rank,
                                              threshold)
        distance_records = greedy_match_records(gt, predictions, distance_rank,
                                                threshold)
        iou3d_records = {(predictions[i].sample_token, match[0]): (i, match[1])
                         for i, match in iou3d_selected.items()}
        if len(iou3d_records) != len(iou3d_selected):
            raise AssertionError('3D IoU oracle assigned one GT more than once')
        baseline_tp, oracle_tp, distance_tp = (len(baseline_records),
                                               len(oracle_records),
                                               len(distance_records))
        maximum = len(selected)
        if max(baseline_tp, oracle_tp, distance_tp, len(iou3d_records)) > maximum:
            raise AssertionError('Greedy TP count exceeds maximum matching')
        baseline_ap = official_ap(gt, original_predictions, None, evaluator.cfg,
                                  threshold, class_name)
        oracle_ap = official_ap(gt, predictions, oracle_rank, evaluator.cfg,
                                threshold, class_name)
        distance_ap = official_ap(gt, predictions, distance_rank, evaluator.cfg,
                                  threshold, class_name)
        _, iou3d_ap = direct_iou3d_metric_data(
            iou3d_selected, iou3d_rank, n_gt, len(predictions),
            evaluator.cfg.min_recall, evaluator.cfg.min_precision)
        cardinality_geometry = selected_match_measurements(selected, predictions, gt)
        geometry = selected_match_measurements(distance_selected, predictions, gt)
        iou3d_geometry = selected_match_measurements(iou3d_selected, predictions, gt)
        baseline_geometry = tp_measurements(baseline_records, gt, predictions)
        oracle_geometry = tp_measurements(oracle_records, gt, predictions)
        distance_geometry = tp_measurements(distance_records, gt, predictions)
        iou3d_tp_geometry = tp_measurements(iou3d_records, gt, predictions)
        cardinality_error = sum(match[1] for match in selected.values())
        minimum_error = sum(match[1] for match in distance_selected.values())
        if minimum_error > cardinality_error + 1e-6:
            raise AssertionError('Minimum-distance oracle increased total XY error')
        rows.append({
            'class': class_name,
            'threshold_m': threshold,
            'gt_count': n_gt,
            'prediction_count': len(predictions),
            'maximum_one_to_one_matches': maximum,
            'maximum_recall_percent': 100 * maximum / n_gt,
            'baseline_full_set_tp': baseline_tp,
            'baseline_full_set_recall_percent': 100 * baseline_tp / n_gt,
            'oracle_rank_full_set_tp': oracle_tp,
            'oracle_rank_full_set_recall_percent': 100 * oracle_tp / n_gt,
            'baseline_ap_percent': 100 * baseline_ap,
            'oracle_rank_ap_percent': 100 * oracle_ap,
            'baseline_tp_mean_original_score': baseline_geometry['mean_original_score'],
            'baseline_tp_mean_xy_center_error_m': baseline_geometry['mean_xy_center_error_m'],
            'baseline_tp_mean_bev_iou': baseline_geometry['mean_bev_iou'],
            'oracle_rank_tp_mean_original_score': oracle_geometry['mean_original_score'],
            'oracle_rank_tp_mean_xy_center_error_m': oracle_geometry['mean_xy_center_error_m'],
            'oracle_rank_tp_mean_bev_iou': oracle_geometry['mean_bev_iou'],
            'oracle_ap_gain_percentage_points': 100 * (oracle_ap - baseline_ap),
            'oracle_rank_remaining_matching_gap': maximum - oracle_tp,
            'cardinality_oracle_total_xy_error_m': cardinality_error,
            'cardinality_oracle_mean_original_score': cardinality_geometry['mean_original_score'],
            'cardinality_oracle_mean_xy_error_m': cardinality_geometry['mean_xy_center_error_m'],
            'cardinality_oracle_mean_bev_iou': cardinality_geometry['mean_bev_iou'],
            'min_distance_oracle_total_xy_error_m': minimum_error,
            'min_distance_oracle_mean_original_score': geometry['mean_original_score'],
            'min_distance_oracle_mean_xy_error_m': geometry['mean_xy_center_error_m'],
            'min_distance_oracle_mean_bev_iou': geometry['mean_bev_iou'],
            'min_distance_center_error_reduction_m': cardinality_error - minimum_error,
            'min_distance_rank_full_set_tp': distance_tp,
            'min_distance_rank_full_set_recall_percent': 100 * distance_tp / n_gt,
            'min_distance_rank_ap_percent': 100 * distance_ap,
            'min_distance_rank_tp_mean_original_score': distance_geometry['mean_original_score'],
            'min_distance_rank_tp_mean_xy_center_error_m': distance_geometry['mean_xy_center_error_m'],
            'min_distance_rank_tp_mean_bev_iou': distance_geometry['mean_bev_iou'],
            'min_distance_rank_ap_gain_percentage_points': 100 * (distance_ap - baseline_ap),
            'min_distance_rank_remaining_matching_gap': maximum - distance_tp,
            'baseline_tp_mean_3d_iou': baseline_geometry['mean_3d_iou'],
            'oracle_rank_tp_mean_3d_iou': oracle_geometry['mean_3d_iou'],
            'min_distance_oracle_mean_3d_iou': geometry['mean_3d_iou'],
            'min_distance_rank_tp_mean_3d_iou': distance_geometry['mean_3d_iou'],
            'max_iou3d_oracle_mean_original_score': iou3d_geometry['mean_original_score'],
            'max_iou3d_oracle_mean_xy_error_m': iou3d_geometry['mean_xy_center_error_m'],
            'max_iou3d_oracle_mean_bev_iou': iou3d_geometry['mean_bev_iou'],
            'max_iou3d_oracle_mean_3d_iou': iou3d_geometry['mean_3d_iou'],
            'max_iou3d_rank_full_set_tp': len(iou3d_records),
            'max_iou3d_rank_full_set_recall_percent': 100 * len(iou3d_records) / n_gt,
            'max_iou3d_rank_ap_percent': 100 * iou3d_ap,
            'max_iou3d_rank_tp_mean_original_score': iou3d_tp_geometry['mean_original_score'],
            'max_iou3d_rank_tp_mean_xy_center_error_m': iou3d_tp_geometry['mean_xy_center_error_m'],
            'max_iou3d_rank_tp_mean_bev_iou': iou3d_tp_geometry['mean_bev_iou'],
            'max_iou3d_rank_tp_mean_3d_iou': iou3d_tp_geometry['mean_3d_iou'],
            'max_iou3d_rank_remaining_matching_gap': maximum - len(iou3d_records),
        })
        print('{} @ {:g} m: capacity {}/{}; TPs baseline {} / cardinality-rank {} / '
              'min-distance-rank {}; AP {:.2f}% / {:.2f}% / {:.2f}%'.format(
                  class_name, threshold, maximum, n_gt, baseline_tp, oracle_tp,
                  distance_tp, 100 * baseline_ap, 100 * oracle_ap,
                  100 * distance_ap), flush=True)
    return rows


def summary_rows(rows, class_names):
    by_class = []
    for name in class_names:
        subset = [row for row in rows if row['class'] == name]
        baseline = statistics.mean(row['baseline_ap_percent'] for row in subset)
        oracle = statistics.mean(row['oracle_rank_ap_percent'] for row in subset)
        distance = statistics.mean(row['min_distance_rank_ap_percent'] for row in subset)
        iou3d = statistics.mean(row['max_iou3d_rank_ap_percent'] for row in subset)
        by_class.append({'class': name, 'baseline_mean_ap_percent': baseline,
                         'oracle_rank_mean_ap_percent': oracle,
                         'gain_percentage_points': oracle - baseline,
                         'min_distance_rank_mean_ap_percent': distance,
                         'min_distance_rank_gain_percentage_points': distance - baseline,
                         'max_iou3d_rank_mean_ap_percent': iou3d,
                         'max_iou3d_rank_gain_percentage_points': iou3d - baseline})
    by_threshold = []
    for threshold in THRESHOLDS:
        subset = [row for row in rows if row['threshold_m'] == threshold]
        baseline = statistics.mean(row['baseline_ap_percent'] for row in subset)
        oracle = statistics.mean(row['oracle_rank_ap_percent'] for row in subset)
        distance = statistics.mean(row['min_distance_rank_ap_percent'] for row in subset)
        iou3d = statistics.mean(row['max_iou3d_rank_ap_percent'] for row in subset)
        by_threshold.append({'threshold_m': threshold,
                             'class_count': len(class_names),
                             'baseline_mean_ap_percent': baseline,
                             'oracle_rank_mean_ap_percent': oracle,
                             'gain_percentage_points': oracle - baseline,
                             'min_distance_rank_mean_ap_percent': distance,
                             'min_distance_rank_gain_percentage_points': distance - baseline,
                             'max_iou3d_rank_mean_ap_percent': iou3d,
                             'max_iou3d_rank_gain_percentage_points': iou3d - baseline})
    overall = {
        'class_count': len(class_names),
        'thresholds_m': list(THRESHOLDS),
        'baseline_mean_ap_percent': statistics.mean(
            row['baseline_ap_percent'] for row in rows),
        'oracle_rank_mean_ap_percent': statistics.mean(
            row['oracle_rank_ap_percent'] for row in rows),
        'min_distance_rank_mean_ap_percent': statistics.mean(
            row['min_distance_rank_ap_percent'] for row in rows),
        'max_iou3d_rank_mean_ap_percent': statistics.mean(
            row['max_iou3d_rank_ap_percent'] for row in rows),
    }
    overall['gain_percentage_points'] = (overall['oracle_rank_mean_ap_percent'] -
                                          overall['baseline_mean_ap_percent'])
    overall['min_distance_rank_gain_percentage_points'] = (
        overall['min_distance_rank_mean_ap_percent'] -
        overall['baseline_mean_ap_percent'])
    overall['max_iou3d_rank_gain_percentage_points'] = (
        overall['max_iou3d_rank_mean_ap_percent'] -
        overall['baseline_mean_ap_percent'])
    return by_class, by_threshold, overall


def validate_baseline(rows, source_metrics_path):
    if source_metrics_path is None or not source_metrics_path.is_file():
        return 'source metrics_summary.json unavailable; baseline cross-check skipped'
    with source_metrics_path.open() as handle:
        source = json.load(handle)
    largest_difference_pp = 0.0
    for row in rows:
        expected = 100 * source['label_aps'][row['class']][str(row['threshold_m'])]
        difference_pp = abs(row['baseline_ap_percent'] - expected)
        largest_difference_pp = max(largest_difference_pp, difference_pp)
        if difference_pp > 1e-3:
            raise ValueError('Baseline AP differs from saved nuScenes metrics for '
                             '{} @ {:g} m: {:.6f} vs {:.6f}'.format(
                                 row['class'], row['threshold_m'],
                                 row['baseline_ap_percent'], expected))
    return ('all baseline class/threshold AP values agree with saved nuScenes '
            'metrics within 0.001 percentage points; largest difference '
            '{:.6f} pp'.format(largest_difference_pp))


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs' /
                        'all_class_reranking_oracle')
    parser.add_argument('--eval-set', choices=('train', 'val'), default='val')
    parser.add_argument('--source-metrics', type=Path,
                        help='Optional official metrics_summary.json used to cross-check baseline AP')
    parser.add_argument('--restrict-to-result-tokens', action='store_true',
                        help='Analyze only sample tokens present in a strict subset submission')
    parser.add_argument('--classes', nargs='+',
                        help='Optional subset of nuScenes detection classes; default is all classes')
    args = parser.parse_args()
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved baseline predictions or nuScenes trainval metadata missing')
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
    except ImportError:
        parser.error('SciPy is required for the minimum-distance localization oracle')

    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    cfg = config_factory('detection_cvpr_2019')
    classes = list(cfg.class_names) if args.classes is None else args.classes
    if len(set(classes)) != len(classes) or set(classes) - set(cfg.class_names):
        parser.error('Choose unique classes from: ' + ', '.join(cfg.class_names))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    result_path = args.results_json
    analysis_tokens = None
    if args.restrict_to_result_tokens:
        from nuscenes.eval.common.loaders import load_gt
        from nuscenes.eval.detection.data_classes import DetectionBox
        submission = json.loads(args.results_json.read_text())
        analysis_tokens = set(submission['results'])
        expected = set(load_gt(nusc, args.eval_set, DetectionBox,
                               verbose=False).sample_tokens)
        extra = analysis_tokens-expected
        if extra:
            parser.error('{} result tokens are outside the {} split'.format(
                len(extra), args.eval_set))
        completed = dict(submission)
        completed['results'] = dict(submission['results'])
        for token in expected-analysis_tokens:
            completed['results'][token] = []
        source_dir = args.output_dir/'source_eval'
        source_dir.mkdir(parents=True, exist_ok=True)
        result_path = source_dir/'completed_for_loader_results_nusc.json'
        result_path.write_text(json.dumps(completed))
        print('Subset oracle: analyzing {:,} of {:,} {} samples'.format(
            len(analysis_tokens), len(expected), args.eval_set), flush=True)
    evaluator = NuScenesEval(nusc, config=cfg, result_path=str(result_path),
                             eval_set=args.eval_set,
                             output_dir=str(args.output_dir / 'source_eval'),
                             verbose=False)
    if analysis_tokens is not None:
        evaluator.gt_boxes.boxes = {
            token: evaluator.gt_boxes.boxes[token] for token in analysis_tokens}
        evaluator.pred_boxes.boxes = {
            token: evaluator.pred_boxes.boxes[token] for token in analysis_tokens}
    rows = []
    for name in classes:
        rows.extend(compare_class(evaluator, name))
    # Preserve a full run even if the source-metric cross-check finds a problem.
    write_csv(args.output_dir / 'class_threshold_comparison.csv', rows)
    plot_rows = geometry_plot_rows(rows)
    write_csv(args.output_dir / 'tp_geometry_plot_data.csv', plot_rows)
    write_geometry_overview(args.output_dir / 'tp_geometry_by_class.svg',
                            classes, plot_rows)
    by_class, by_threshold, overall = summary_rows(rows, classes)
    write_csv(args.output_dir / 'class_mean_ap_comparison.csv', by_class)
    write_csv(args.output_dir / 'threshold_mean_ap_comparison.csv', by_threshold)
    source_metrics = args.source_metrics
    if source_metrics is None and args.eval_set == 'val' and \
            not args.restrict_to_result_tokens:
        source_metrics = args.results_json.with_name('metrics_summary.json')
    status = validate_baseline(rows, source_metrics)
    (args.output_dir / 'summary.json').write_text(json.dumps({
        **overall,
        'classes': classes,
        'source_results_json': str(args.results_json.resolve()),
        'data_root': str(args.data_root.resolve()),
        'eval_set': args.eval_set,
        'sample_subset': args.restrict_to_result_tokens,
        'analyzed_sample_count': len(analysis_tokens)
        if analysis_tokens is not None else len(evaluator.gt_boxes.sample_tokens),
        'baseline_validation': status,
        'matching': 'official filtered boxes, XY center distance < threshold',
        'oracle': 'class- and threshold-specific GT-informed confidence order; boxes unchanged',
        'minimum_distance_oracle': 'lexicographic maximum cardinality then minimum total XY center error; same fixed boxes',
        'maximum_iou3d_oracle': 'lexicographic maximum cardinality then maximum total oriented 3D IoU; XY distance eligibility unchanged',
        'interpretation': 'diagnostic average over independent oracle rankings; not one submission',
    }, indent=2) + '\n')
    print(status)
    print('Mean AP over {} classes and {} thresholds: baseline {:.2f}%; '
          'cardinality-rank {:.2f}%; min-distance-rank {:.2f}%; '
          '3D-IoU-rank {:.2f}%'.format(
        len(classes), len(THRESHOLDS), overall['baseline_mean_ap_percent'],
        overall['oracle_rank_mean_ap_percent'],
        overall['min_distance_rank_mean_ap_percent'],
        overall['max_iou3d_rank_mean_ap_percent']))
    print('Wrote {} and tp_geometry_by_class.svg'.format(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
