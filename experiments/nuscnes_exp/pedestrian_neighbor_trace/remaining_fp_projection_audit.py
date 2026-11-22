#!/usr/bin/env python3
"""Audit 4 m FPs after excluding Rule-4 alternatives.

For same-class projected FPs, geometry is measured against the exact GT whose
image projection has the largest overlap with the prediction. It is never
measured against an independently selected nearest GT.
"""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from build_hypothesis_pair_gallery import CLASS_CATEGORIES
from localization_capacity import load_filtered_class
from pedestrian_image_alignment import (camera_records, projected_rect,
                                         radial_transverse, rect_iou)
from reranking_oracle import greedy_match_records


def annotation_box(record):
    return SimpleNamespace(translation=record['translation'], size=record['size'],
                           rotation=record['rotation'])


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values)/len(values) if values else None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def write_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--rule4-cases', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--class-name', default='pedestrian',
                        choices=tuple(CLASS_CATEGORIES))
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--projection-iou-threshold', type=float, default=0.01)
    parser.add_argument('--max-fp', type=int, default=0,
                        help='Diagnostic limit; zero processes every FP.')
    args = parser.parse_args()
    if not args.results_json.is_file():
        parser.error('Missing predictions: {}'.format(args.results_json))
    if not args.rule4_cases.is_file():
        parser.error('Missing Rule-4 cases: {}'.format(args.rule4_cases))
    if args.threshold <= 0 or args.projection_iou_threshold < 0:
        parser.error('Thresholds must be nonnegative and distance positive')

    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                    verbose=False)
    gt_by_sample, predictions = load_filtered_class(
        args.results_json, args.data_root, args.class_name)
    matches = greedy_match_records(gt_by_sample, predictions,
                                   range(len(predictions)), args.threshold)
    tp_indices = {prediction_index for prediction_index, _ in matches.values()}

    with args.rule4_cases.open(newline='') as handle:
        rule4_rows = list(csv.DictReader(handle))
    rule4_alternatives = {
        int(row['alternative_prediction_id']) for row in rule4_rows
        if row.get('alternative_prediction_id', '') != ''}
    if rule4_alternatives and max(rule4_alternatives) >= len(predictions):
        raise ValueError('Rule-4 prediction IDs do not match the filtered list')
    overlap_with_tp = rule4_alternatives & tp_indices
    if overlap_with_tp:
        raise ValueError('{} Rule-4 alternatives are official TPs'.format(
            len(overlap_with_tp)))

    remaining = [index for index in range(len(predictions))
                 if index not in tp_indices and index not in rule4_alternatives]
    if args.max_fp:
        remaining = remaining[:args.max_fp]

    category_cache = {}
    raw_annotation_cache = {}
    camera_cache = {}
    projected_annotation_cache = {}

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
                    'category': category_for_instance(
                        annotation['instance_token']),
                    'translation': annotation['translation'],
                    'size': annotation['size'],
                    'rotation': annotation['rotation'],
                })
            raw_annotation_cache[token] = records
        return raw_annotation_cache[token]

    def cameras(token):
        if token not in camera_cache:
            camera_cache[token] = camera_records(nusc, token)
        return camera_cache[token]

    def projected_annotations(token, camera_name, camera):
        key = (token, camera_name)
        if key not in projected_annotation_cache:
            projected_annotation_cache[key] = [
                (annotation, rect)
                for annotation in raw_annotations(token)
                for rect in [projected_rect(annotation_box(annotation), camera)]
                if rect is not None]
        return projected_annotation_cache[key]

    target_categories = CLASS_CATEGORIES[args.class_name]
    output_rows = []
    for position, prediction_index in enumerate(remaining, 1):
        prediction = predictions[prediction_index]
        token = prediction.sample_token
        best = None
        for camera_name, camera in cameras(token):
            prediction_rect = projected_rect(prediction, camera)
            if prediction_rect is None:
                continue
            for annotation, annotation_rect in projected_annotations(
                    token, camera_name, camera):
                iou = rect_iou(prediction_rect, annotation_rect)
                candidate = (iou, annotation['token'], camera_name,
                             annotation, camera)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

        best_iou = best[0] if best else 0.0
        if best is None or best_iou <= args.projection_iou_threshold:
            projection_category = 'background'
            correspondence = None
        elif best[3]['category'] in target_categories:
            projection_category = 'interested_class_gt'
            correspondence = best
        else:
            projection_category = 'other_class_gt'
            correspondence = None

        center_error = radial_error = transverse_error = None
        corresponding_token = corresponding_category = corresponding_camera = ''
        if correspondence is not None:
            _, corresponding_token, corresponding_camera, annotation, camera = \
                correspondence
            corresponding_category = annotation['category']
            gt = annotation_box(annotation)
            center_error = math.hypot(
                prediction.translation[0]-gt.translation[0],
                prediction.translation[1]-gt.translation[1])
            radial_error, transverse_error = radial_transverse(
                prediction, gt, camera)

        output_rows.append({
            'prediction_id': prediction_index,
            'sample_token': token,
            'score': prediction.detection_score,
            'projection_category': projection_category,
            'max_projected_iou2d': best_iou,
            'max_overlap_annotation_token': best[1] if best else '',
            'max_overlap_annotation_category': best[3]['category'] if best else '',
            'max_overlap_camera': best[2] if best else '',
            'corresponding_class_gt_token': corresponding_token,
            'corresponding_class_gt_category': corresponding_category,
            'corresponding_camera': corresponding_camera,
            'corresponding_gt_center_error_m': center_error,
            'corresponding_gt_radial_error_m': radial_error,
            'corresponding_gt_absolute_radial_error_m':
                abs(radial_error) if radial_error is not None else None,
            'corresponding_gt_transverse_error_m': transverse_error,
            'corresponding_gt_absolute_transverse_error_m':
                abs(transverse_error) if transverse_error is not None else None,
        })
        if position % 5000 == 0:
            print('Processed {:,}/{:,} remaining FPs'.format(
                position, len(remaining)), flush=True)

    categories = ('interested_class_gt', 'other_class_gt', 'background')
    summary_rows = []
    for category in categories:
        selected = [row for row in output_rows
                    if row['projection_category'] == category]
        summary_rows.append({
            'class': args.class_name,
            'threshold_m': args.threshold,
            'projection_category': category,
            'count': len(selected),
            'percent_of_remaining_fp':
                100*len(selected)/len(output_rows) if output_rows else 0,
            'mean_confidence': mean([row['score'] for row in selected]),
            'median_confidence': median([row['score'] for row in selected]),
            'mean_max_projected_iou2d': mean([
                row['max_projected_iou2d'] for row in selected]),
            'mean_corresponding_gt_center_error_m': mean([
                row['corresponding_gt_center_error_m'] for row in selected]),
            'mean_corresponding_gt_absolute_radial_error_m': mean([
                row['corresponding_gt_absolute_radial_error_m']
                for row in selected]),
            'mean_corresponding_gt_absolute_transverse_error_m': mean([
                row['corresponding_gt_absolute_transverse_error_m']
                for row in selected]),
            'mean_corresponding_gt_signed_radial_error_m': mean([
                row['corresponding_gt_radial_error_m'] for row in selected]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/'remaining_fp_records.csv', output_rows)
    write_csv(args.output_dir/'summary.csv', summary_rows)
    payload = {
        'class': args.class_name,
        'threshold_m': args.threshold,
        'projection_iou_threshold': args.projection_iou_threshold,
        'prediction_count': len(predictions),
        'official_tp_count': len(tp_indices),
        'official_fp_count': len(predictions)-len(tp_indices),
        'excluded_rule4_alternative_count': len(rule4_alternatives),
        'remaining_fp_count': len(output_rows),
        'projection_category_counts': dict(Counter(
            row['projection_category'] for row in output_rows)),
        'summary': summary_rows,
        'geometry_definition': (
            'For interested_class_gt only, errors use the exact annotation with '
            'maximum projected IoU in the selected camera; no nearest-GT '
            'substitution is performed.'),
    }
    (args.output_dir/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
