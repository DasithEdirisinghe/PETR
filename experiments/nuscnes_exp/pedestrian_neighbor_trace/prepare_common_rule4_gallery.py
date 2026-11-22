#!/usr/bin/env python3
"""Prepare synchronized common Rule-4 hypothesis-pair cases."""

import argparse
import csv
import json
import random
from pathlib import Path

from build_hypothesis_pair_gallery import build_cases, load_metadata
from localization_capacity import load_filtered_class
from pedestrian_image_alignment import measurements, projected_rect
from prepare_common_multimodel_gallery import MODEL_SPECS
from visualize import iter_json_array


def read_rows(path):
    with path.open(newline='') as handle:
        return {(row['sample_token'], int(row['gt_index'])): row
                for row in csv.DictReader(handle)}


def read_projection_assignments(paths):
    output = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                prediction_id = int(row['prediction_id'])
                output[prediction_id] = {
                    'category': row.get(
                        'annotation_category',
                        row.get('max_overlap_annotation_category', '')),
                    'iou2d': float(row.get(
                        'iou2d', row.get('max_projected_iou2d', 0))),
                }
    return output


def display_box(prediction, camera):
    from build_hypothesis_pair_gallery import corners, projected_segments
    box = {'translation': list(prediction.translation),
           'size': list(prediction.size),
           'rotation': list(prediction.rotation)}
    box['segments'] = projected_segments(box, camera)
    box['bev_corners'] = [point[:2] for point in corners(box)[:4]]
    return box


def gallery_row(row, has_alternative):
    output = dict(row)
    output['selected_box_json'] = row['official_tp_box_json']
    output['best_3d_box_json'] = (row['alternative_box_json'] if has_alternative
                                  else row['official_tp_box_json'])
    output['visibility_percent'] = {
        '1': '0-40%', '2': '40-60%', '3': '60-80%', '4': '80-100%'
    }[row['visibility_token']]
    for metric in ('score', 'center_m', 'iou3d', 'projected_iou2d',
                   'radial_error_m', 'transverse_error_m'):
        output['selected_'+metric] = row['official_tp_'+metric]
        output['best_3d_'+metric] = (row['alternative_'+metric]
                                     if has_alternative else
                                     row['official_tp_'+metric])
    if has_alternative:
        output['delta_iou3d'] = str(float(row['alternative_iou3d'])-
                                    float(row['official_tp_iou3d']))
        output['delta_projected_iou2d'] = str(
            float(row['alternative_projected_iou2d'])-
            float(row['official_tp_projected_iou2d']))
    else:
        output['delta_iou3d'] = '0'
        output['delta_projected_iou2d'] = '0'
    return output


def main():
    base = Path(__file__).resolve().parent
    outputs = base/'outputs'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-name', required=True,
                        choices=('pedestrian', 'car', 'bicycle'))
    parser.add_argument('--analysis-root', type=Path, default=outputs /
                        'multimodel_rule4_analysis_4m_margin_0p05')
    parser.add_argument('--data-root', type=Path, default=base.parents[3] /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-per-category', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--projection-iou-threshold', type=float, default=0.01)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = outputs/'visualizations'/(
            'hypothesis_pairs_rule4_common_'+args.class_name)

    paths = {slug: args.analysis_root/slug/args.class_name/'all_gt.csv'
             for slug, _ in MODEL_SPECS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error('Missing Rule-4 files:\n'+'\n'.join(missing))
    rows_by_model = {slug: read_rows(path) for slug, path in paths.items()}
    common = set.intersection(*[set(rows) for rows in rows_by_model.values()])
    available = {
        'alternative_exists': sorted(
            key for key in common if all(
                rows[key]['official_match'] == '1' and
                rows[key]['rule4_alternative_exists'] == '1'
                for rows in rows_by_model.values())),
        'official_tp_best': sorted(
            key for key in common if all(
                rows[key]['official_match'] == '1' and
                rows[key]['rule4_alternative_exists'] == '0'
                for rows in rows_by_model.values())),
    }
    rng = random.Random(args.seed)
    selected = {}
    for category, keys in available.items():
        count = min(args.max_per_category, len(keys))
        selected[category] = sorted(rng.sample(keys, count)) \
            if count < len(keys) else keys
    ordered = [(category, key) for category in
               ('alternative_exists', 'official_tp_best')
               for key in selected[category]]
    all_rows = [gallery_row(rows_by_model[slug][key], category ==
                            'alternative_exists')
                for slug, _ in MODEL_SPECS for category, key in ordered]
    metadata = load_metadata(args.data_root, all_rows, args.class_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = {key[0] for _, key in ordered}
    scene_by_sample = {
        record['token']: record['scene_token']
        for record in iter_json_array(args.data_root/'v1.0-trainval/sample.json')
        if record['token'] in tokens}
    cases_by_model = {}
    for slug, _ in MODEL_SPECS:
        converted = [gallery_row(rows_by_model[slug][key], category ==
                                 'alternative_exists')
                     for category, key in ordered]
        cases_by_model[slug] = build_cases(
            converted, metadata, args.data_root, args.output_dir)

    # Add maximum-confidence target-pointing predictions and high-confidence
    # non-pedestrian/background FPs for the already selected pedestrian cases.
    if args.class_name == 'pedestrian':
        petr_dir = base.parents[2]
        workspace = base.parents[3]
        result_files = {
            'petr_r50': petr_dir/'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json',
            'streampetr_r50': workspace/'StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json',
            'repdetr3d_eva02_l': workspace/'StreamPETR/experiments/eva02_gallery/outputs/inference/formatted/pts_bbox/results_nusc.json',
            'petr_eva02_l_epoch16': petr_dir/'results/petr_eva02_l_p4_800x320_nuscenes_global_batch_8/epoch_16_formatted/pts_bbox/results_nusc.json',
        }
        selected_tokens = {key[0] for _, key in ordered}
        for slug, _ in MODEL_SPECS:
            coverage_rows = read_rows(
                outputs/'gt_pointing_coverage'/slug/'pedestrian/per_gt.csv')
            projection_dir = outputs/'gt_pointing_coverage'/slug/'pedestrian'
            projection_paths = [projection_dir/'newly_projected_predictions.csv']
            fp_cache = outputs/'remaining_fp_projection_audit_4m'/slug/'pedestrian/remaining_fp_records.csv'
            if fp_cache.is_file():
                projection_paths.append(fp_cache)
            assignments = read_projection_assignments(projection_paths)
            gt_by_sample, predictions = load_filtered_class(
                result_files[slug], args.data_root, 'pedestrian')
            official_tp_ids = {
                int(row['official_tp_prediction_id'])
                for row in rows_by_model[slug].values()
                if row['official_match'] == '1'}
            frame_tp_scores = {}
            for prediction_id in official_tp_ids:
                prediction = predictions[prediction_id]
                frame_tp_scores.setdefault(prediction.sample_token, []).append(
                    prediction.detection_score)
            last_tp_score = {token: min(scores)
                             for token, scores in frame_tp_scores.items()}
            frame_high_fps = {token: [] for token in selected_tokens}
            for prediction_id, prediction in enumerate(predictions):
                token = prediction.sample_token
                if token not in frame_high_fps or prediction_id in official_tp_ids or \
                        token not in last_tp_score or \
                        prediction.detection_score <= last_tp_score[token]:
                    continue
                assignment = assignments.get(prediction_id, {})
                points_elsewhere = assignment.get('iou2d', 0) <= \
                    args.projection_iou_threshold or not assignment.get(
                        'category', '').startswith('human.pedestrian.')
                if points_elsewhere:
                    frame_high_fps[token].append(prediction_id)
            for index, (_, key) in enumerate(ordered):
                token, gi = key
                case = cases_by_model[slug][index]
                camera = dict(metadata[1][token, case['camera']])
                # build_hypothesis_pair_gallery keeps the image dimensions in
                # camera['record']; pedestrian_image_alignment expects them at
                # the top level. Preserve both representations because the
                # drawing helpers use camera['record'].
                camera['width'] = camera['record']['width']
                camera['height'] = camera['record']['height']
                gt = gt_by_sample[token][gi]
                gt_rect = projected_rect(gt, camera)
                coverage = coverage_rows[key]
                case['target_pointing_prediction_count'] = int(
                    coverage['pointing_prediction_count'])
                case['target_pointing_covered'] = (
                    case['target_pointing_prediction_count'] > 0)
                maximum_id = coverage['highest_score_pointing_prediction_id']
                if maximum_id != '':
                    maximum = predictions[int(maximum_id)]
                    case['boxes']['maximum'] = display_box(maximum, camera)
                    case['maximum'] = measurements(maximum, gt, camera, gt_rect)
                    case['maximum']['prediction_id'] = int(maximum_id)
                    case['maximum']['status'] = (
                        'TP' if int(maximum_id) in official_tp_ids else
                        'Rule-4 FP' if int(maximum_id) == int(
                            rows_by_model[slug][key]['alternative_prediction_id'] or -1)
                        else 'FP')
                else:
                    case['boxes']['maximum'] = None
                    case['maximum'] = None
                case['last_tp_score_in_frame'] = last_tp_score.get(token)
                case['high_conf_other_fps'] = []
                for prediction_id in frame_high_fps[token]:
                    prediction = predictions[prediction_id]
                    assignment = assignments.get(prediction_id, {})
                    case['high_conf_other_fps'].append({
                        'prediction_id': prediction_id,
                        'score': prediction.detection_score,
                        'projection_category': assignment.get('category', ''),
                        'projected_iou2d': assignment.get('iou2d', 0),
                        'box': display_box(prediction, camera),
                    })
    cases = []
    for index, (category, key) in enumerate(ordered):
        model_cases = {}
        for slug, _ in MODEL_SPECS:
            source = rows_by_model[slug][key]
            case = cases_by_model[slug][index]
            case['correspondence'] = {
                'official_tp': source['official_tp_alignment_category'],
                'alternative': source['alternative_alignment_category']
                if category == 'alternative_exists' else ''}
            model_cases[slug] = case
        cases.append({'category': category, 'sample_token': key[0],
                      'scene_token': scene_by_sample[key[0]],
                      'gt_index': key[1], 'models': model_cases})
    payload = {
        'rule_version': 4, 'class_name': args.class_name,
        'models': [{'slug': slug, 'name': name} for slug, name in MODEL_SPECS],
        'available_counts': {key: len(value) for key, value in available.items()},
        'selected_counts': {key: len(value) for key, value in selected.items()},
        'max_per_category': args.max_per_category, 'seed': args.seed,
        'cases': cases}
    (args.output_dir/'comparison_data.js').write_text(
        'const comparisonData='+json.dumps(payload, separators=(',', ':'))+';\n')
    (args.output_dir/'summary.json').write_text(json.dumps({
        key: payload[key] for key in ('rule_version', 'class_name',
                                      'available_counts', 'selected_counts',
                                      'max_per_category', 'seed')}, indent=2)+'\n')
    print('Rule 4 available:', json.dumps(payload['available_counts']))
    print('Rule 4 selected:', json.dumps(payload['selected_counts']))
    print('Wrote', len(cases), 'common cases to', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
