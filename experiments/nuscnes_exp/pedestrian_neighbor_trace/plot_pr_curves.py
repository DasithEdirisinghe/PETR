#!/usr/bin/env python3
"""Plot PETR and GT-informed 3D-IoU re-ranking PR curves for each class.

Uses the same officially filtered boxes and threshold-specific oracle ordering
as all_class_reranking_oracle.py. The re-ranked curve is diagnostic: its scores
use GT information and are independently reassigned at every threshold.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from all_class_reranking_oracle import class_boxes
from localization_capacity import THRESHOLDS
from reranking_oracle import (greedy_match_records, iou3d_oracle_order,
                              direct_iou3d_metric_data)


METHODS = (('baseline', 'PETR baseline', '#2563eb'),
           ('iou3d', 'Direct one-to-one IoU oracle', '#ea580c'))


def metric_data(gt_by_sample, predictions, order, config, class_name, threshold):
    """Use the exact nuScenes AP accumulator with unchanged predicted boxes."""
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.data_classes import DetectionBox

    gt_boxes, pred_boxes = EvalBoxes(), EvalBoxes()
    for token, boxes in gt_by_sample.items():
        gt_boxes.add_boxes(token, boxes)
    if order is None:
        for box in predictions:
            pred_boxes.add_boxes(box.sample_token, [box])
    else:
        count = len(predictions)
        synthetic_scores = {index: (count - rank) / (count + 1)
                            for rank, index in enumerate(order)}
        for index, box in enumerate(predictions):
            clone = DetectionBox.deserialize(box.serialize())
            clone.detection_score = synthetic_scores[index]
            pred_boxes.add_boxes(clone.sample_token, [clone])
    data = accumulate(gt_boxes, pred_boxes, class_name,
                      config.dist_fcn_callable, threshold)
    return data, calc_ap(data, config.min_recall, config.min_precision)


def curve_points(data, max_recall):
    """Remove nuScenes' zero padding beyond attained recall from the drawing."""
    points = [(float(r), float(p)) for r, p in
              zip(data.recall, data.precision) if float(r) <= max_recall + 1e-9]
    if max_recall > 0 and points and points[-1][0] < max_recall - 1e-6:
        points.append((max_recall, points[-1][1]))
    return points or [(0.0, 0.0)]


def write_svg(path, class_name, panels, model_name):
    """Four distance thresholds in one SVG, with both methods in each panel."""
    width, height = 1600, 550
    left, top, panel_w, panel_h, gap = 58, 170, 325, 290, 62
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
              'viewBox="0 0 {} {}">'.format(width, height, width, height),
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font-family:Arial,sans-serif;fill:#172033}'
              '.title{font-size:25px;font-weight:bold}.label{font-size:15px}'
              '.small{font-size:12px;fill:#475569}</style>',
              '<text x="48" y="42" class="title">{}: precision–recall by center-distance threshold</text>'.format(
                  escape(class_name.replace('_', ' ').title())),
              '<text x="48" y="70" class="small">Same filtered {} boxes; oracle uses direct GT-informed one-to-one matches at each threshold.</text>'.format(escape(model_name)),
              '<text x="48" y="91" class="small">Peak P is the maximum interpolated precision at positive recall; final P uses all retained class predictions.</text>']
    for method_index, (_, label, color) in enumerate(METHODS):
        if method_index == 0:
            label = model_name + ' baseline'
        x = 50 + method_index * 225
        pieces.extend(('<line x1="{}" y1="119" x2="{}" y2="119" stroke="{}" stroke-width="4"/>'.format(
            x, x + 25, color),
                       '<text x="{}" y="124" class="label">{}</text>'.format(x + 33, label)))
    pieces.extend(('<line x1="505" y1="119" x2="530" y2="119" stroke="#64748b" stroke-dasharray="5 4" stroke-width="2"/>',
                   '<text x="538" y="124" class="label">Oracle one-to-one GT recall</text>'))
    for panel_index, panel in enumerate(panels):
        x0 = left + panel_index * (panel_w + gap)
        y0 = top
        x1, y1 = x0 + panel_w, y0 + panel_h
        pieces.append('<text x="{}" y="{}" class="label" font-weight="bold">{:g} m</text>'.format(
            x0, y0 - 25, panel['threshold_m']))
        pieces.append('<rect x="{}" y="{}" width="{}" height="{}" fill="#f8fafc" stroke="#cbd5e1"/>'.format(
            x0, y0, panel_w, panel_h))
        for tick in (0.25, 0.5, 0.75):
            tx, ty = x0 + tick * panel_w, y1 - tick * panel_h
            pieces.extend(('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#e2e8f0"/>'.format(tx, y0, y1),
                           '<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#e2e8f0"/>'.format(
                               x0, ty, x1, ty)))
        capacity_x = x0 + panel['capacity_recall'] * panel_w
        pieces.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#64748b" stroke-width="2" stroke-dasharray="5 4"/>'.format(
            capacity_x, y0, y1))
        for method, _, color in METHODS:
            series = panel[method]
            points = ' '.join('{:.1f},{:.1f}'.format(
                x0 + max(0., min(1., r)) * panel_w,
                y1 - max(0., min(1., p)) * panel_h)
                for r, p in series['points'])
            pieces.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="3"/>'.format(
                points, color))
            end_x = x0 + series['max_recall'] * panel_w
            end_y = y1 - series['points'][-1][1] * panel_h
            pieces.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="{}"/>'.format(
                end_x, end_y, color))
        pieces.extend(('<text x="{}" y="{}" class="small">0</text>'.format(x0 - 5, y1 + 16),
                       '<text x="{}" y="{}" class="small">Recall → 1</text>'.format(x1 - 63, y1 + 16),
                       '<text x="{}" y="{}" class="small">1</text>'.format(x0 - 17, y0 + 4)))
        for method_index, (method, label, color) in enumerate(METHODS):
            result = panel[method]
            pieces.append('<text x="{}" y="{}" class="small" fill="{}">{}: AP {:.1f}%, max R {:.1f}%, peak P {:.1f}%</text>'.format(
                x0, y1 + 39 + 19 * method_index, color,
                'Base' if method == 'baseline' else 'IoU',
                100 * result['ap'], 100 * result['max_recall'],
                100 * result['peak_precision']))
        pieces.append('<text x="{}" y="{}" class="small">Capacity R {:.1f}% · final P: base {:.1f}%, IoU {:.1f}%</text>'.format(
            x0, y1 + 80, 100 * panel['capacity_recall'],
            100 * panel['baseline']['full_set_precision'],
            100 * panel['iou3d']['full_set_precision']))
    pieces.append('</svg>')
    path.write_text('\n'.join(pieces) + '\n')


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs' /
                        'all_class_reranking_oracle/pr_curves')
    parser.add_argument('--classes', nargs='+', help='Default: all nuScenes detection classes')
    parser.add_argument('--model-name', default='PETR', help='Name displayed on plots')
    args = parser.parse_args()
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved predictions or nuScenes trainval metadata missing')

    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    config = config_factory('detection_cvpr_2019')
    classes = list(config.class_names) if args.classes is None else args.classes
    if not classes or len(set(classes)) != len(classes) or set(classes) - set(config.class_names):
        parser.error('Choose unique classes from: ' + ', '.join(config.class_names))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    evaluator = NuScenesEval(nusc, config=config, result_path=str(args.results_json),
                             eval_set='val', output_dir=str(args.output_dir / 'source_eval'),
                             verbose=False)
    metrics_path = args.results_json.with_name('metrics_summary.json')
    source_metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    rows, curves = [], {}
    for class_name in classes:
        gt, original_predictions, predictions = class_boxes(evaluator, class_name)
        n_gt = sum(len(boxes) for boxes in gt.values())
        if not n_gt or not predictions:
            raise ValueError('Expected GTs and predictions for ' + class_name)
        panels = []
        for threshold in THRESHOLDS:
            oracle_order, selected = iou3d_oracle_order(gt, predictions, threshold)
            panel = {'threshold_m': threshold,
                     'capacity_recall': len(selected) / n_gt}
            for method, _, _ in METHODS:
                order = None if method == 'baseline' else oracle_order
                source = original_predictions if order is None else predictions
                if method == 'baseline':
                    data, ap = metric_data(gt, source, order, config, class_name, threshold)
                else:
                    data, ap = direct_iou3d_metric_data(
                        selected, oracle_order, n_gt, len(predictions),
                        config.min_recall, config.min_precision)
                if method == 'baseline' and source_metrics is not None:
                    expected = float(source_metrics['label_aps'][class_name][str(threshold)])
                    if abs(float(ap) - expected) > 1e-5:
                        raise ValueError('{} @ {:g} m baseline AP differs from official summary: '
                                         '{:.6f} vs {:.6f}'.format(class_name, threshold, ap,
                                                                    expected))
                match_order = range(len(predictions)) if order is None else order
                tp_count = (len(greedy_match_records(gt, predictions, match_order,
                                                    threshold)) if method == 'baseline'
                            else len(selected))
                max_recall = tp_count / n_gt
                points = curve_points(data, max_recall)
                positive_precisions = [p for r, p in points if r > 0]
                peak_precision = max(positive_precisions, default=0.0)
                full_set_precision = tp_count / len(predictions)
                panel[method] = {'points': points, 'ap': float(ap),
                                 'tp_count': tp_count, 'max_recall': max_recall,
                                 'peak_precision': peak_precision,
                                 'full_set_precision': full_set_precision}
                rows.append({'class': class_name, 'threshold_m': threshold,
                             'method': method, 'gt_count': n_gt,
                             'prediction_count': len(predictions),
                             'tp_count': tp_count, 'ap_percent': 100 * ap,
                             'max_recall_percent': 100 * max_recall,
                             'peak_precision_percent': 100 * peak_precision,
                             'full_set_precision_percent': 100 * full_set_precision,
                             'capacity_recall_percent': 100 * panel['capacity_recall']})
            panels.append(panel)
            print('{} @ {:g} m: AP {:.2f}% → {:.2f}%, max R {:.2f}% → {:.2f}%'.format(
                class_name, threshold, 100 * panel['baseline']['ap'],
                100 * panel['iou3d']['ap'], 100 * panel['baseline']['max_recall'],
                100 * panel['iou3d']['max_recall']), flush=True)
        curves[class_name] = panels
        write_svg(args.output_dir / (class_name + '_pr.svg'), class_name, panels,
                  args.model_name)
    with (args.output_dir / 'pr_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'pr_curves.json').write_text(json.dumps(curves, indent=2) + '\n')
    html = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<title>{} baseline versus 3D IoU re-ranking PR curves</title>'.format(escape(args.model_name)),
            '<style>body{font:16px Arial,sans-serif;max-width:1650px;margin:32px auto;padding:0 24px;color:#172033}'
            'img{display:block;width:100%;margin:16px 0 48px;border:1px solid #cbd5e1}</style>',
            '<h1>{} baseline versus 3D IoU re-ranking PR curves</h1>'.format(escape(args.model_name)),
            '<p>nuScenes validation, official filtered boxes. Re-ranking uses GT information separately '
            'at each center-distance threshold; these are diagnostic curves. '
            '<a href="pr_summary.csv">Download metrics</a>.</p>']
    for class_name in classes:
        html.append('<h2>{}</h2><img src="{}_pr.svg" alt="{} PR curves">'.format(
            escape(class_name.replace('_', ' ').title()), class_name,
            escape(class_name)))
    html.append('</html>')
    (args.output_dir / 'index.html').write_text('\n'.join(html) + '\n')
    print('Wrote {} class plots to {}'.format(len(classes), args.output_dir))


if __name__ == '__main__':
    main()
