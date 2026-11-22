#!/usr/bin/env python3
"""Measure whether each evaluated GT has a prediction pointing mainly to it."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from build_hypothesis_pair_gallery import CLASS_CATEGORIES
from localization_capacity import load_filtered_class
from pedestrian_image_alignment import camera_records, projected_rect, rect_iou


VISIBILITY_LABELS = {
    1: '0-40%', 2: '40-60%', 3: '60-80%', 4: '80-100%'}
SCORE_CUTOFFS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def annotation_box(record):
    return SimpleNamespace(translation=record['translation'], size=record['size'],
                           rotation=record['rotation'])


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cached-projections', type=Path,
                        help='Optional prior prediction-to-annotation assignments')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--class-name', default='pedestrian',
                        choices=tuple(CLASS_CATEGORIES))
    parser.add_argument('--projection-iou-threshold', type=float, default=0.01)
    args = parser.parse_args()
    if not args.results_json.is_file():
        parser.error('Missing input: {}'.format(args.results_json))
    if args.cached_projections is not None and \
            not args.cached_projections.is_file():
        parser.error('Missing cache: {}'.format(args.cached_projections))

    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                    verbose=False)
    gt_by_sample, predictions = load_filtered_class(
        args.results_json, args.data_root, args.class_name)
    target_categories = CLASS_CATEGORIES[args.class_name]

    category_cache = {}
    annotation_cache = {}
    camera_cache = {}
    projected_annotation_cache = {}

    def category_for_instance(instance_token):
        if instance_token not in category_cache:
            instance = nusc.get('instance', instance_token)
            category_cache[instance_token] = nusc.get(
                'category', instance['category_token'])['name']
        return category_cache[instance_token]

    def annotations(token):
        if token not in annotation_cache:
            records = []
            for annotation_token in nusc.get('sample', token)['anns']:
                ann = nusc.get('sample_annotation', annotation_token)
                if ann['num_lidar_pts']+ann['num_radar_pts'] <= 0:
                    continue
                records.append({
                    'token': annotation_token,
                    'category': category_for_instance(ann['instance_token']),
                    'translation': ann['translation'], 'size': ann['size'],
                    'rotation': ann['rotation'],
                    'visibility_token': int(ann['visibility_token']),
                })
            annotation_cache[token] = records
        return annotation_cache[token]

    def cameras(token):
        if token not in camera_cache:
            camera_cache[token] = camera_records(nusc, token)
        return camera_cache[token]

    def projected_annotations(token, camera_name, camera):
        key = (token, camera_name)
        if key not in projected_annotation_cache:
            projected_annotation_cache[key] = [
                (ann, rect)
                for ann in annotations(token)
                for rect in [projected_rect(annotation_box(ann), camera)]
                if rect is not None]
        return projected_annotation_cache[key]

    # Map every officially evaluated GT box back to its exact nuScenes annotation.
    evaluated_gt = []
    evaluated_tokens = set()
    for sample_token, boxes in gt_by_sample.items():
        candidates = [ann for ann in annotations(sample_token)
                      if ann['category'] in target_categories]
        used = set()
        for gt_index, gt in enumerate(boxes):
            available = [ann for ann in candidates if ann['token'] not in used]
            if not available:
                raise ValueError('No raw annotation for {} GT {}'
                                 .format(sample_token, gt_index))
            ann = min(available, key=lambda item: sum(
                (item['translation'][axis]-gt.translation[axis])**2
                for axis in range(3)))
            error = max(abs(ann['translation'][axis]-gt.translation[axis])
                        for axis in range(3))
            if error > 1e-3:
                raise ValueError('GT-to-annotation mismatch {} GT {}: {} m'
                                 .format(sample_token, gt_index, error))
            used.add(ann['token'])
            evaluated_tokens.add(ann['token'])
            evaluated_gt.append({
                'sample_token': sample_token, 'gt_index': gt_index,
                'annotation_token': ann['token'],
                'visibility_token': ann['visibility_token'],
                'visibility_label': VISIBILITY_LABELS[ann['visibility_token']],
            })

    # Reuse projection assignments from the prior remaining-FP audit.
    assignment = {}
    cached_ids = set()
    if args.cached_projections is not None:
        with args.cached_projections.open(newline='') as handle:
            for row in csv.DictReader(handle):
                prediction_id = int(row['prediction_id'])
                if prediction_id in cached_ids:
                    raise ValueError('Duplicate cached prediction ID {}'.format(
                        prediction_id))
                cached_ids.add(prediction_id)
                assignment[prediction_id] = {
                    'annotation_token': row['max_overlap_annotation_token'],
                    'annotation_category': row['max_overlap_annotation_category'],
                    'camera': row['max_overlap_camera'],
                    'iou2d': float(row['max_projected_iou2d']),
                    'score': float(row['score']),
                    'source': 'cached_remaining_fp',
                }
    if cached_ids and max(cached_ids) >= len(predictions):
        raise ValueError('Cached prediction IDs do not match prediction list')

    newly_projected = []
    missing = [index for index in range(len(predictions))
               if index not in cached_ids]
    for position, prediction_id in enumerate(missing, 1):
        prediction = predictions[prediction_id]
        token = prediction.sample_token
        best = None
        for camera_name, camera in cameras(token):
            prediction_rect = projected_rect(prediction, camera)
            if prediction_rect is None:
                continue
            for ann, ann_rect in projected_annotations(
                    token, camera_name, camera):
                candidate = (rect_iou(prediction_rect, ann_rect),
                             ann['token'], camera_name, ann)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        record = {
            'annotation_token': best[1] if best else '',
            'annotation_category': best[3]['category'] if best else '',
            'camera': best[2] if best else '',
            'iou2d': best[0] if best else 0.0,
            'score': prediction.detection_score,
            'source': 'new_projection',
        }
        assignment[prediction_id] = record
        newly_projected.append(dict({'prediction_id': prediction_id}, **record))
        if position % 2000 == 0:
            print('Projected {:,}/{:,} uncached predictions'.format(
                position, len(missing)), flush=True)

    pointing = defaultdict(list)
    for prediction_id, record in assignment.items():
        if record['iou2d'] > args.projection_iou_threshold and \
                record['annotation_token'] in evaluated_tokens and \
                record['annotation_category'] in target_categories:
            pointing[record['annotation_token']].append((
                prediction_id, record['score'], record['iou2d']))

    for row in evaluated_gt:
        candidates = pointing[row['annotation_token']]
        row['pointing_prediction_count'] = len(candidates)
        row['max_pointing_score'] = max(
            (candidate[1] for candidate in candidates), default='')
        row['max_pointing_iou2d'] = max(
            (candidate[2] for candidate in candidates), default='')
        row['highest_score_pointing_prediction_id'] = max(
            candidates, key=lambda item: item[1])[0] if candidates else ''

    summary_rows = []
    for visibility in range(1, 5):
        group = [row for row in evaluated_gt
                 if row['visibility_token'] == visibility]
        for cutoff in SCORE_CUTOFFS:
            covered = [row for row in group
                       if row['max_pointing_score'] != '' and
                       row['max_pointing_score'] >= cutoff]
            summary_rows.append({
                'visibility_token': visibility,
                'visibility_label': VISIBILITY_LABELS[visibility],
                'minimum_prediction_score': cutoff,
                'gt_count': len(group),
                'gt_with_pointing_prediction': len(covered),
                'gt_without_pointing_prediction': len(group)-len(covered),
                'coverage_percent':
                    100*len(covered)/len(group) if group else 0,
            })
    for cutoff in SCORE_CUTOFFS:
        covered = [row for row in evaluated_gt
                   if row['max_pointing_score'] != '' and
                   row['max_pointing_score'] >= cutoff]
        summary_rows.append({
            'visibility_token': 'all', 'visibility_label': 'all',
            'minimum_prediction_score': cutoff,
            'gt_count': len(evaluated_gt),
            'gt_with_pointing_prediction': len(covered),
            'gt_without_pointing_prediction': len(evaluated_gt)-len(covered),
            'coverage_percent': 100*len(covered)/len(evaluated_gt),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/'per_gt.csv', evaluated_gt)
    write_csv(args.output_dir/'visibility_summary.csv', summary_rows)
    if newly_projected:
        write_csv(args.output_dir/'newly_projected_predictions.csv',
                  newly_projected)
    payload = {
        'class': args.class_name,
        'projection_iou_threshold': args.projection_iou_threshold,
        'definition': ('A prediction points mainly to a GT when its maximum '
                       'projected IoU across all cameras and annotations belongs '
                       'to that evaluated GT and exceeds the IoU threshold.'),
        'prediction_count': len(predictions),
        'cached_prediction_count': len(cached_ids),
        'newly_projected_prediction_count': len(missing),
        'evaluated_gt_count': len(evaluated_gt),
        'score_cutoffs': list(SCORE_CUTOFFS),
        'summary': summary_rows,
    }
    (args.output_dir/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
