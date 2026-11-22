#!/usr/bin/env python3
"""Rule 4: find unmatched, target-corresponding, geometrically better boxes."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from localization_capacity import load_filtered_class
from pedestrian_image_alignment import (
    CLASS_CATEGORIES, EPS, camera_records, center_distance, gt_visibility,
    measurements, preferred_alternative_better, projected_rect, rect_iou,
    reference_camera)
from reranking_oracle import greedy_match_records


NUSCENES_DETECTION_CLASSES = (
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')


def mean(values):
    return sum(values)/len(values) if values else None


def annotation_box(record):
    return SimpleNamespace(translation=record['translation'], size=record['size'],
                           rotation=record['rotation'])


def squared_distance(a, b):
    return sum((a[i]-b[i])**2 for i in range(3))


def serialize_prediction(prediction):
    return json.dumps(prediction.serialize(), separators=(',', ':'))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, args, prediction_count):
    matched = [row for row in rows if row['official_match']]
    rule4 = [row for row in matched if row['rule4_alternative_exists']]
    no_rule4 = [row for row in matched if not row['rule4_alternative_exists']]

    def metric_summary(group, prefix):
        result = {}
        for metric in ('score', 'center_m', 'iou3d', 'projected_iou2d',
                       'absolute_radial_error_m', 'absolute_transverse_error_m'):
            values = [row[prefix+metric] for row in group
                      if row[prefix+metric] not in ('', None)]
            result[metric] = mean(values)
        return result

    def alignment_summary(group, prediction):
        output = []
        for category in ('target_gt', 'other_same_class_gt',
                         'other_class_or_background'):
            selected = [row for row in group
                        if row[prediction+'alignment_category'] == category]
            output.append({
                'prediction_type': prediction.rstrip('_'),
                'alignment_category': category,
                'count': len(selected),
                'percent': 100*len(selected)/len(group) if group else 0,
                'mean_confidence': mean([row[prediction+'score']
                                         for row in selected]),
            })
        return output

    visibility = {}
    for token in range(1, 5):
        selected = [row for row in rule4 if row['visibility_token'] == token]
        visibility[str(token)] = {
            'count': len(selected),
            'percent': 100*len(selected)/len(rule4) if rule4 else 0,
        }
    payload = {
        'rule_version': 4,
        'model_name': args.model_name,
        'class_name': args.class_name,
        'threshold_m': args.threshold,
        'target_margin_iou2d': args.target_margin_iou2d,
        'prediction_count': prediction_count,
        'gt_count': len(rows),
        'matched_gt_count': len(matched),
        'rule4_alternative_count': len(rule4),
        'rule4_alternative_percent_of_matched':
            100*len(rule4)/len(matched) if matched else 0,
        'no_rule4_alternative_count': len(no_rule4),
        'alternative_lower_confidence_count': sum(
            row['alternative_score'] < row['official_tp_score']-EPS
            for row in rule4),
        'alternative_lower_confidence_percent': 100*sum(
            row['alternative_score'] < row['official_tp_score']-EPS
            for row in rule4)/len(rule4) if rule4 else 0,
        'mean_metrics_rule4_cases': {
            'official_tp': metric_summary(rule4, 'official_tp_'),
            'alternative': metric_summary(rule4, 'alternative_'),
        },
        'mean_metrics_no_rule4_cases': {
            'official_tp': metric_summary(no_rule4, 'official_tp_'),
        },
        'alignment_rule4_cases':
            alignment_summary(rule4, 'official_tp_')+
            alignment_summary(rule4, 'alternative_'),
        'alignment_no_rule4_cases':
            alignment_summary(no_rule4, 'official_tp_'),
        'visibility_rule4_cases': visibility,
        'method': ('Rule 4 considers only officially unmatched same-class '
                   'predictions within the evaluation distance. A candidate must '
                   'overlap the target in the selected camera and exceed every '
                   'other projected annotation by the configured 2D-IoU margin. '
                   'It must then improve target projected 2D IoU and improve 3D '
                   'IoU, or center distance when both 3D IoUs are zero. Surviving '
                   'candidates are ranked by target 2D IoU, 3D IoU, center '
                   'distance, then confidence.'),
    }
    return payload


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--class-name', required=True,
                        choices=NUSCENES_DETECTION_CLASSES)
    parser.add_argument('--results-json', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--target-margin-iou2d', type=float, default=0.05)
    parser.add_argument('--max-gt', type=int, default=0)
    parser.add_argument('--eval-set', choices=('train', 'val'), default='val')
    parser.add_argument('--allow-prediction-subset', action='store_true',
                        help='Restrict GT to prediction tokens when predictions are a strict subset of eval-set samples')
    args = parser.parse_args()
    if args.threshold <= 0 or args.target_margin_iou2d < 0:
        parser.error('threshold must be positive and target margin nonnegative')
    if not args.results_json.is_file():
        parser.error('Missing predictions: {}'.format(args.results_json))

    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                    verbose=False)
    gt_by_sample, predictions = load_filtered_class(
        args.results_json, args.data_root, args.class_name, args.eval_set,
        args.allow_prediction_subset)
    visibility = gt_visibility(nusc, gt_by_sample, args.class_name)
    matches = greedy_match_records(gt_by_sample, predictions,
                                   range(len(predictions)), args.threshold)
    pred_tp_owner = {pi: key for key, (pi, _) in matches.items()}
    by_frame = defaultdict(list)
    for pi, prediction in enumerate(predictions):
        by_frame[prediction.sample_token].append(pi)

    category_cache = {}
    raw_annotation_cache = {}
    projection_cache = {}
    prediction_rect_cache = {}

    def category_for_instance(instance_token):
        if instance_token not in category_cache:
            instance = nusc.get('instance', instance_token)
            category_cache[instance_token] = nusc.get(
                'category', instance['category_token'])['name']
        return category_cache[instance_token]

    def raw_annotations(token):
        if token not in raw_annotation_cache:
            records = []
            for annotation_token in nusc.get('sample', token)['anns']:
                annotation = nusc.get('sample_annotation', annotation_token)
                if annotation['num_lidar_pts']+annotation['num_radar_pts'] <= 0:
                    continue
                records.append({
                    'token': annotation_token,
                    'category': category_for_instance(annotation['instance_token']),
                    'translation': annotation['translation'],
                    'size': annotation['size'],
                    'rotation': annotation['rotation'],
                })
            raw_annotation_cache[token] = records
        return raw_annotation_cache[token]

    def projected_annotations(token, camera_name, camera):
        cache_key = (token, camera_name)
        if cache_key not in projection_cache:
            projected = []
            for annotation in raw_annotations(token):
                rect = projected_rect(annotation_box(annotation), camera)
                if rect is not None:
                    projected.append((annotation, rect))
            projection_cache[cache_key] = projected
        return projection_cache[cache_key]

    rows = []
    target_categories = CLASS_CATEGORIES[args.class_name]
    stop = False
    for token, gt_boxes in gt_by_sample.items():
        frame_indices = by_frame[token]
        records = camera_records(nusc, token) if gt_boxes else []
        for gi, gt in enumerate(gt_boxes):
            key = (token, gi)
            camera_name, camera, gt_rect = reference_camera(records, gt)
            selected_pi = matches[key][0] if key in matches else None
            selected_metrics = measurements(
                predictions[selected_pi], gt, camera, gt_rect) \
                if selected_pi is not None else None
            projected = projected_annotations(token, camera_name, camera) \
                if camera is not None else []
            target_candidates = [annotation for annotation in raw_annotations(token)
                                 if annotation['category'] in target_categories]
            target_annotation = min(
                target_candidates,
                key=lambda annotation: squared_distance(
                    annotation['translation'], gt.translation)) \
                if target_candidates else None

            def correspondence(pi):
                if camera is None or target_annotation is None:
                    return 0.0, 0.0, 'other_class_or_background', '', ''
                rect_key = (pi, camera_name)
                if rect_key not in prediction_rect_cache:
                    prediction_rect_cache[rect_key] = projected_rect(
                        predictions[pi], camera)
                prediction_rect = prediction_rect_cache[rect_key]
                overlaps = [(rect_iou(prediction_rect, rect), annotation)
                            for annotation, rect in projected]
                target_iou = next((iou for iou, annotation in overlaps
                                   if annotation['token'] ==
                                   target_annotation['token']), 0.0)
                competitors = [(iou, annotation) for iou, annotation in overlaps
                               if annotation['token'] != target_annotation['token']]
                competitor_iou, competitor = max(
                    competitors, key=lambda item: item[0]) \
                    if competitors else (0.0, None)
                maximum = max([target_iou, competitor_iou])
                if maximum <= EPS:
                    category = 'other_class_or_background'
                elif target_iou >= competitor_iou-EPS:
                    category = 'target_gt'
                elif competitor['category'] in target_categories:
                    category = 'other_same_class_gt'
                else:
                    category = 'other_class_or_background'
                return (target_iou, competitor_iou, category,
                        competitor['token'] if competitor else '',
                        competitor['category'] if competitor else '')

            selected_correspondence = correspondence(selected_pi) \
                if selected_pi is not None else (0.0, 0.0, '', '', '')
            candidates = []
            target_valid_count = 0
            for pi in frame_indices:
                if selected_pi is None or pi == selected_pi or pi in pred_tp_owner:
                    continue
                prediction = predictions[pi]
                if center_distance(prediction, gt) >= args.threshold:
                    continue
                metrics = measurements(prediction, gt, camera, gt_rect)
                target_iou, competitor_iou, category, competitor_token, \
                    competitor_category = correspondence(pi)
                if target_iou <= EPS or target_iou+EPS < \
                        competitor_iou+args.target_margin_iou2d:
                    continue
                target_valid_count += 1
                if preferred_alternative_better(metrics, selected_metrics):
                    candidates.append((pi, metrics, target_iou, competitor_iou,
                                       category, competitor_token,
                                       competitor_category))
            alternative = max(candidates, key=lambda item: (
                item[1]['projected_iou2d'], item[1]['iou3d'],
                -item[1]['center_m'], item[1]['score'])) if candidates else None

            row = {
                'sample_token': token, 'gt_index': gi,
                'official_match': int(selected_pi is not None),
                'camera': camera_name,
                'visibility_token': visibility[token, gi],
                'threshold_m': args.threshold,
                'target_margin_iou2d': args.target_margin_iou2d,
                'unmatched_candidates_within_threshold': sum(
                    pi not in pred_tp_owner and
                    center_distance(predictions[pi], gt) < args.threshold
                    for pi in frame_indices),
                'target_valid_unmatched_candidates': target_valid_count,
                'rule4_alternative_exists': int(alternative is not None),
                'official_tp_prediction_id': selected_pi if selected_pi is not None else '',
                'alternative_prediction_id': alternative[0] if alternative else '',
                'official_tp_alignment_category': selected_correspondence[2],
                'official_tp_target_iou2d': selected_correspondence[0],
                'official_tp_max_competitor_iou2d': selected_correspondence[1],
                'official_tp_box_json': serialize_prediction(predictions[selected_pi])
                if selected_pi is not None else '',
                'alternative_alignment_category': alternative[4] if alternative else '',
                'alternative_target_iou2d': alternative[2] if alternative else '',
                'alternative_max_competitor_iou2d': alternative[3] if alternative else '',
                'alternative_competitor_annotation_token': alternative[5]
                if alternative else '',
                'alternative_competitor_category': alternative[6]
                if alternative else '',
                'alternative_box_json': serialize_prediction(predictions[alternative[0]])
                if alternative else '',
            }
            for prefix, metric_values in (
                    ('official_tp_', selected_metrics),
                    ('alternative_', alternative[1] if alternative else None)):
                for metric in ('score', 'center_m', 'iou3d', 'projected_iou2d',
                               'radial_error_m', 'transverse_error_m'):
                    value = metric_values[metric] if metric_values is not None else ''
                    row[prefix+metric] = value
                row[prefix+'absolute_radial_error_m'] = abs(
                    metric_values['radial_error_m']) \
                    if metric_values is not None and \
                    metric_values['radial_error_m'] is not None else ''
                row[prefix+'absolute_transverse_error_m'] = abs(
                    metric_values['transverse_error_m']) \
                    if metric_values is not None and \
                    metric_values['transverse_error_m'] is not None else ''
            rows.append(row)
            if len(rows) % 2000 == 0:
                print('Processed {:,} GTs'.format(len(rows)), flush=True)
            if args.max_gt and len(rows) >= args.max_gt:
                stop = True
                break
        if stop:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/'all_gt.csv', rows)
    write_csv(args.output_dir/'rule4_cases.csv', [
        row for row in rows if row['rule4_alternative_exists']])
    summary = summarize(rows, args, len(predictions))
    summary['results_json'] = str(args.results_json.resolve())
    (args.output_dir/'summary.json').write_text(
        json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
