#!/usr/bin/env python3
"""GT-coverability and confidence-order diagnostic for PETR pedestrians.

This is not nuScenes AP. Predictions and GTs are filtered with the official
nuScenes detection evaluator, then pedestrian boxes are processed in PETR
confidence order. At each prefix, an online bipartite matcher finds the
maximum number of pedestrian GTs coverable by those predictions.
"""

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
from pathlib import Path


THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


class OnlineCoverage:
    def __init__(self):
        self.edges = []
        self.owner = {}
        self.count = 0

    def add(self, sample_token, neighbor_gt_indices):
        pred_index = len(self.edges)
        self.edges.append((sample_token, tuple(neighbor_gt_indices)))
        seen = set()

        def augment(index):
            token, neighbors = self.edges[index]
            for gi in neighbors:
                key = (token, gi)
                if key in seen:
                    continue
                seen.add(key)
                previous = self.owner.get(key)
                if previous is None or augment(previous):
                    self.owner[key] = index
                    return True
            return False

        if augment(pred_index):
            self.count += 1
        return self.count


def load_filtered_class(results_json, data_root, class_name='pedestrian',
                        eval_set='val', allow_prediction_subset=False):
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
    predictions, _ = load_prediction(str(results_json), cfg.max_boxes_per_sample,
                                     DetectionBox, verbose=False)
    ground_truth = load_gt(nusc, eval_set, DetectionBox, verbose=False)
    prediction_tokens = set(predictions.sample_tokens)
    ground_truth_tokens = set(ground_truth.sample_tokens)
    if prediction_tokens != ground_truth_tokens:
        if not allow_prediction_subset or not prediction_tokens < ground_truth_tokens:
            raise ValueError('Prediction samples do not equal {} samples'.format(
                eval_set))
        ground_truth.boxes = {
            token: ground_truth.boxes[token] for token in predictions.sample_tokens}
    predictions = filter_eval_boxes(nusc, add_center_dist(nusc, predictions),
                                    cfg.class_range, verbose=False)
    ground_truth = filter_eval_boxes(nusc, add_center_dist(nusc, ground_truth),
                                    cfg.class_range, verbose=False)
    gt_by_sample = {
        token: [box for box in ground_truth[token]
                if box.detection_name == class_name]
        for token in ground_truth.sample_tokens
    }
    pred_list = [box for box in predictions.all
                 if box.detection_name == class_name]
    # Match the tie order in nuscenes.eval.detection.algo.accumulate.
    pred_list = [pred_list[i] for i in
                 sorted(range(len(pred_list)),
                        key=lambda i: (pred_list[i].detection_score, i),
                        reverse=True)]
    return gt_by_sample, pred_list


def load_filtered_pedestrians(results_json, data_root, eval_set='val'):
    """Backward-compatible pedestrian loader."""
    return load_filtered_class(results_json, data_root, 'pedestrian', eval_set)


def analyze(gt_by_sample, pred_list):
    gt_count = sum(map(len, gt_by_sample.values()))
    if not gt_count or not pred_list:
        raise ValueError('Expected nonempty pedestrian GT and predictions')
    states = {threshold: OnlineCoverage() for threshold in THRESHOLDS}
    official_taken = {threshold: set() for threshold in THRESHOLDS}
    capacity_curves = {threshold: [0] for threshold in THRESHOLDS}
    official_curves = {threshold: [0] for threshold in THRESHOLDS}
    candidate_counts = {threshold: 0 for threshold in THRESHOLDS}
    for box in pred_list:
        token = box.sample_token
        gt = gt_by_sample[token]
        distances = [(gi, math.hypot(box.translation[0]-target.translation[0],
                                     box.translation[1]-target.translation[1]))
                     for gi, target in enumerate(gt)]
        for threshold in THRESHOLDS:
            neighbors = [gi for gi, distance in distances if distance < threshold]
            neighbors.sort(key=lambda gi: distances[gi][1])
            candidate_counts[threshold] += bool(neighbors)
            capacity_curves[threshold].append(states[threshold].add(token, neighbors))
            available = [(distance, gi) for gi, distance in distances
                         if (token, gi) not in official_taken[threshold]]
            if available:
                distance, gi = min(available)
                if distance < threshold:
                    official_taken[threshold].add((token, gi))
            official_curves[threshold].append(len(official_taken[threshold]))
    summary = []
    n_pred = len(pred_list)
    budget = min(n_pred, gt_count)
    for threshold in THRESHOLDS:
        curve = capacity_curves[threshold]
        capacity = curve[-1]
        if len(official_taken[threshold]) > capacity:
            raise AssertionError('Official greedy matching exceeds maximum matching')
        observed_area = sum(curve[1:budget+1])
        official_area = sum(official_curves[threshold][1:budget+1])
        ideal_area = sum(min(k, capacity) for k in range(1, budget+1))
        summary.append({
            'threshold_m': threshold,
            'gt_count': gt_count,
            'prediction_count': n_pred,
            'candidate_prediction_count': candidate_counts[threshold],
            'full_set_maximum_one_to_one_matches': capacity,
            'localization_capacity_percent': 100 * capacity / gt_count,
            'official_greedy_full_set_matches': len(official_taken[threshold]),
            'official_greedy_full_set_recall_percent': 100 * len(official_taken[threshold]) / gt_count,
            'full_set_greedy_assignment_gap_gt': capacity - len(official_taken[threshold]),
            'top_gt_count_predictions_maximum_matches': curve[budget],
            'top_gt_count_predictions_coverage_percent': 100 * curve[budget] / gt_count,
            'ranking_efficiency_percent': 100 * observed_area / ideal_area if ideal_area else float('nan'),
            'greedy_assignment_efficiency_percent': 100 * official_area / observed_area if observed_area else float('nan'),
            'ranking_budget_prediction_count': budget,
        })
    return summary, capacity_curves, official_curves


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figure(path, summary, maximum_curves, official_curves):
    gt_count = summary[0]['gt_count']
    n_pred = summary[0]['prediction_count']
    max_budget = min(n_pred, 2*gt_count)
    width, height = 1400, 1040
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
             'viewBox="0 0 {} {}">'.format(width, height, width, height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#17212b}'
             '.title{font-size:27px;font-weight:bold}.sub{font-size:16px;fill:#52606d}'
             '.label{font-size:14px;fill:#52606d}.panel{font-size:20px;font-weight:bold}'
             '</style>',
             '<text x="55" y="54" class="title">PETR pedestrian localization capacity and confidence order</text>',
             '<text x="55" y="83" class="sub">Full nuScenes validation set · same-class, one-to-one center-distance matching · diagnostic, not AP</text>']
    series = (
        ('Best possible score order', '#94a3b8', 'ideal'),
        ('PETR score order, optimal matching', '#0f766e', 'maximum'),
        ('PETR score order, official greedy matching', '#ea580c', 'official'),
    )
    for ti, record in enumerate(summary):
        panel_x = 55 + (ti % 2)*680
        panel_y = 125 + (ti // 2)*415
        parts.append('<rect x="{}" y="{}" width="645" height="382" rx="10" '
                     'fill="#f8fafc" stroke="#cbd5e1"/>'.format(panel_x, panel_y))
        parts.append('<text x="{}" y="{}" class="panel">Pedestrian @ {:g} m</text>'.format(
            panel_x+22, panel_y+36, record['threshold_m']))
        parts.append('<text x="{}" y="{}" class="sub">Capacity {:.1f}% · ranking {:.1f}% · greedy matching {:.1f}%</text>'.format(
            panel_x+22, panel_y+61, record['localization_capacity_percent'],
            record['ranking_efficiency_percent'], record['greedy_assignment_efficiency_percent']))
        left, right = panel_x+62, panel_x+612
        top, bottom = panel_y+88, panel_y+290
        for tick in (0, 25, 50, 75, 100):
            y = bottom-(bottom-top)*tick/100
            parts.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" '
                         'stroke="#e2e8f0"/>'.format(left, y, right, y))
            parts.append('<text x="{}" y="{:.1f}" class="label" text-anchor="end">{}%</text>'.format(
                left-8, y+4, tick))
        for tick in (0, 0.5, 1, 1.5, 2):
            if tick*gt_count > max_budget:
                continue
            x = left+(right-left)*tick/(max_budget/gt_count)
            parts.append('<text x="{:.1f}" y="{}" class="label" text-anchor="middle">{:g}</text>'.format(
                x, bottom+23, tick))
        parts.append('<text x="{}" y="{}" class="label" text-anchor="middle">'
                     'Top K predictions / number of pedestrian GTs</text>'.format(
                         (left+right)//2, bottom+48))
        step = max(1, max_budget//220)
        sampled = sorted(set(range(0, max_budget+1, step)) | {max_budget})
        capacity = maximum_curves[record['threshold_m']][-1]
        curves = {
            'ideal': lambda k: min(k, capacity),
            'maximum': lambda k: maximum_curves[record['threshold_m']][k],
            'official': lambda k: official_curves[record['threshold_m']][k],
        }
        for _, color, name in series:
            points = ' '.join('{:.1f},{:.1f}'.format(
                left+(right-left)*k/max_budget,
                bottom-(bottom-top)*curves[name](k)/gt_count)
                for k in sampled)
            parts.append('<polyline points="{}" fill="none" stroke="{}" '
                         'stroke-width="3"/>'.format(points, color))
    for index, (label, color, _) in enumerate(series):
        x = 66 + index*445
        parts.append('<line x1="{}" y1="961" x2="{}" y2="961" stroke="{}" '
                     'stroke-width="4"/>'.format(x, x+36, color))
        parts.append('<text x="{}" y="967" font-family="Arial" font-size="15">{}</text>'.format(
            x+45, html.escape(label)))
    parts.append('<text x="55" y="1004" class="sub">Capacity uses every retained prediction; ranking efficiency compares the top-K coverage area with the ideal order up to K = min(predictions, GTs).</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts)+'\n')


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs' /
                        'localization_capacity')
    args = parser.parse_args()
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved baseline predictions or nuScenes trainval metadata missing')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt, predictions = load_filtered_pedestrians(args.results_json, args.data_root)
    summary, maximum_curves, official_curves = analyze(gt, predictions)
    write_csv(args.output_dir / 'pedestrian_localization_capacity.csv', summary)
    gt_count = summary[0]['gt_count']
    n_pred = len(predictions)
    max_budget = min(n_pred, 2*gt_count)
    sampled_k = sorted(set([0, n_pred, min(n_pred, gt_count)] +
                           [round(i*max_budget/400) for i in range(401)]))
    curve_rows = []
    for threshold in THRESHOLDS:
        capacity = maximum_curves[threshold][-1]
        for k in sampled_k:
            curve_rows.append({
                'threshold_m': threshold, 'top_k_predictions': k,
                'prediction_budget_per_gt': k/gt_count,
                'maximum_one_to_one_covered_gt': maximum_curves[threshold][k],
                'official_greedy_covered_gt': official_curves[threshold][k],
                'ideal_order_covered_gt': min(k, capacity),
                'maximum_coverage_percent': 100*maximum_curves[threshold][k]/gt_count,
                'official_greedy_coverage_percent': 100*official_curves[threshold][k]/gt_count,
                'ideal_order_coverage_percent': 100*min(k, capacity)/gt_count,
            })
    write_csv(args.output_dir / 'pedestrian_top_k_coverage.csv', curve_rows)
    svg_path = args.output_dir / 'pedestrian_top_k_coverage.svg'
    write_figure(svg_path, summary, maximum_curves, official_curves)
    converter = shutil.which('rsvg-convert')
    if converter:
        with (args.output_dir / 'pedestrian_top_k_coverage.png').open('wb') as handle:
            subprocess.run([converter, '-f', 'png', str(svg_path)],
                           stdout=handle, check=True)
    (args.output_dir / 'source.json').write_text(json.dumps({
        'results_json': str(args.results_json.resolve()),
        'data_root': str(args.data_root.resolve()),
        'class': 'pedestrian',
        'filtering': 'nuScenes detection_cvpr_2019 evaluator filtering',
        'criterion': 'strictly less than center-distance threshold',
        'ranking_efficiency_budget': 'top min(number of predictions, number of pedestrian GTs)',
        'metric_status': 'diagnostic; not official nuScenes AP',
    }, indent=2) + '\n')
    for row in summary:
        print('{threshold_m:g} m: capacity {localization_capacity_percent:.2f}%, '
              'top-budget {top_gt_count_predictions_coverage_percent:.2f}%, '
              'ranking efficiency {ranking_efficiency_percent:.2f}%, '
              'greedy assignment efficiency {greedy_assignment_efficiency_percent:.2f}%'.format(**row))


if __name__ == '__main__':
    main()
