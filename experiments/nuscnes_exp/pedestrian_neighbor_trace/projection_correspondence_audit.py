#!/usr/bin/env python3
"""Classify TP/alternative projections as target, same-class, or other/background."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from build_hypothesis_pair_gallery import (CLASS_CATEGORIES, load_metadata)
from pedestrian_image_alignment import projected_rect, rect_iou
from visualize import iter_json_array

EPS = 1e-6


def prediction_box(serialized):
    data = json.loads(serialized)
    return SimpleNamespace(translation=data['translation'], size=data['size'],
                           rotation=data['rotation'])


def annotation_box(record):
    return SimpleNamespace(translation=record['translation'], size=record['size'],
                           rotation=record['rotation'])


def squared_center_distance(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def mean(values):
    return sum(values)/len(values) if values else None


def main():
    base = Path(__file__).resolve().parent
    outputs = base/'outputs'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', default='PETR R50')
    parser.add_argument('--class-name', choices=tuple(CLASS_CATEGORIES),
                        default='pedestrian')
    parser.add_argument('--all-gt-csv', type=Path, default=outputs /
                        'multimodel_hypothesis_analysis_4m/petr_r50/pedestrian/hypothesis/all_gt.csv')
    parser.add_argument('--data-root', type=Path,
                        default=base.parents[3]/'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path,
                        default=outputs/'projection_correspondence_audit_4m/petr_r50/pedestrian')
    parser.add_argument('--background-iou-threshold', type=float, default=EPS)
    args = parser.parse_args()
    with args.all_gt_csv.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['official_match'] == '1']
    if not rows or 'preferred_alt_rule_version_3' not in rows[0]:
        parser.error('Expected completed rule-version-3 all_gt.csv')

    metadata = load_metadata(args.data_root, rows, args.class_name)
    target_gt, cameras = metadata
    tokens = {row['sample_token'] for row in rows}
    table = args.data_root/'v1.0-trainval'
    categories = {record['token']: record['name']
                  for record in iter_json_array(table/'category.json')}
    instance_category = {
        record['token']: categories[record['category_token']]
        for record in iter_json_array(table/'instance.json')
    }
    annotations = {token: [] for token in tokens}
    for record in iter_json_array(table/'sample_annotation.json'):
        if record['sample_token'] in tokens and \
                record['num_lidar_pts']+record['num_radar_pts'] > 0:
            annotations[record['sample_token']].append({
                'token': record['token'],
                'category': instance_category[record['instance_token']],
                'translation': record['translation'], 'size': record['size'],
                'rotation': record['rotation']})

    projection_cache = {}

    def projection_camera(token, camera_name):
        camera = cameras[token, camera_name]
        if 'width' not in camera:
            camera = dict(camera)
            camera['width'] = camera['record']['width']
            camera['height'] = camera['record']['height']
        return camera

    def projected_annotations(token, camera_name):
        key = (token, camera_name)
        if key not in projection_cache:
            camera = projection_camera(token, camera_name)
            projected = []
            for ann in annotations[token]:
                rect = projected_rect(annotation_box(ann), camera)
                if rect is not None:
                    projected.append((ann, rect))
            projection_cache[key] = projected
        return projection_cache[key]

    detailed_rows = []
    scores = defaultdict(list)
    combined_scores = defaultdict(list)
    target_categories = CLASS_CATEGORIES[args.class_name]
    for index, row in enumerate(rows, 1):
        token, gi, camera_name = row['sample_token'], int(row['gt_index']), row['camera']
        gt = target_gt[token][gi]
        projected = projected_annotations(token, camera_name)
        target_candidates = [ann for ann in annotations[token]
                             if ann['category'] in target_categories]
        target_ann = min(
            target_candidates,
            key=lambda ann: squared_center_distance(
                ann['translation'], gt['translation']))
        population = ('better_alternative' if row['preferred_alt_better'] == '1'
                      else 'official_tp_best')
        predictions = [('official_tp', row['selected_box_json'],
                        float(row['selected_score']))]
        if population == 'better_alternative':
            predictions.append(('alternative', row['preferred_alt_box_json'],
                                float(row['preferred_alt_score'])))

        for prediction_type, serialized, score in predictions:
            pred_rect = projected_rect(
                prediction_box(serialized),
                projection_camera(token, camera_name))
            overlaps = []
            if pred_rect is not None:
                overlaps = [(rect_iou(pred_rect, rect), ann)
                            for ann, rect in projected]
            max_iou = max((item[0] for item in overlaps), default=0.0)
            if max_iou <= args.background_iou_threshold:
                category = 'background'
                winner = None
            else:
                tied = [ann for iou, ann in overlaps
                        if abs(iou-max_iou) <= EPS]
                target_tied = [ann for ann in tied
                               if ann['token'] == target_ann['token']]
                same_tied = [ann for ann in tied
                             if ann['category'] in target_categories and
                             ann['token'] != target_ann['token']]
                if target_tied:
                    category, winner = 'target_gt', target_tied[0]
                elif same_tied:
                    category, winner = 'other_same_class_gt', same_tied[0]
                else:
                    category, winner = 'other_class_gt', tied[0]
            combined = ('other_class_or_background'
                        if category in ('other_class_gt', 'background') else category)
            scores[population, prediction_type, category].append(score)
            combined_scores[population, prediction_type, combined].append(score)
            detailed_rows.append({
                'sample_token': token, 'gt_index': gi,
                'camera': camera_name, 'visibility_token': row['visibility_token'],
                'population': population, 'prediction_type': prediction_type,
                'alignment_category': category,
                'combined_category': combined,
                'confidence': score, 'maximum_projected_iou2d': max_iou,
                'winning_annotation_token': winner['token'] if winner else '',
                'winning_annotation_category': winner['category'] if winner else '',
                'target_annotation_token': target_ann['token'],
            })
        if index % 2000 == 0:
            print('Processed {:,}/{:,} matched GTs'.format(index, len(rows)),
                  flush=True)

    def summary_rows(source):
        output = []
        populations = ('better_alternative', 'official_tp_best')
        for population in populations:
            prediction_types = ('official_tp', 'alternative') \
                if population == 'better_alternative' else ('official_tp',)
            categories_to_report = (('target_gt', 'target_gt'),
                                    ('other_same_class_gt', 'other_same_class_gt'),
                                    ('other_class_or_background',
                                     'other_class_or_background')) \
                if source is combined_scores else (
                    ('target_gt', 'target_gt'),
                    ('other_same_class_gt', 'other_same_class_gt'),
                    ('other_class_gt', 'other_class_gt'),
                    ('background', 'background'))
            for prediction_type in prediction_types:
                denominator = sum(len(source[population, prediction_type, key])
                                  for key, _ in categories_to_report)
                for key, label in categories_to_report:
                    values = source[population, prediction_type, key]
                    output.append({
                        'population': population,
                        'prediction_type': prediction_type,
                        'alignment_category': label,
                        'count': len(values), 'population_count': denominator,
                        'percent': 100*len(values)/denominator if denominator else 0,
                        'mean_confidence': mean(values),
                    })
        return output

    combined = summary_rows(combined_scores)
    detailed = summary_rows(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (('cases.csv', detailed_rows),
                              ('summary_three_categories.csv', combined),
                              ('summary_four_categories.csv', detailed)):
        with (args.output_dir/filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    payload = {
        'model': args.model_name, 'class_name': args.class_name,
        'threshold_m': 4.0,
        'background_iou_threshold': args.background_iou_threshold,
        'matched_gt_count': len(rows),
        'better_alternative_gt_count': sum(
            row['preferred_alt_better'] == '1' for row in rows),
        'official_tp_best_gt_count': sum(
            row['preferred_alt_better'] == '0' for row in rows),
        'three_category_summary': combined,
        'four_category_summary': detailed,
    }
    (args.output_dir/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
