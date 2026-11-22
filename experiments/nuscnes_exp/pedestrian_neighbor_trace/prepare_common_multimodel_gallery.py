#!/usr/bin/env python3
"""Prepare synchronized hypothesis cases shared by all four detector variants."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from build_hypothesis_pair_gallery import build_cases, load_metadata
from visualize import iter_json_array


MODEL_SPECS = (
    ('petr_r50', 'PETR R50'),
    ('streampetr_r50', 'StreamPETR R50'),
    ('repdetr3d_eva02_l', 'RepDETR3D EVA02-L'),
    ('petr_eva02_l_epoch16', 'PETR EVA02-L epoch 16'),
)


def read_rows(path):
    with path.open(newline='') as handle:
        return {(row['sample_token'], int(row['gt_index'])): row
                for row in csv.DictReader(handle)}


def read_correspondence(path):
    output = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['sample_token'], int(row['gt_index']))
            record = output.setdefault(key, {})
            record[row['prediction_type']] = row['combined_category']
    return output


def main():
    base = Path(__file__).resolve().parent
    outputs = base/'outputs'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-name', required=True,
                        choices=('pedestrian', 'car', 'bicycle'))
    parser.add_argument('--analysis-root', type=Path,
                        default=outputs/'multimodel_hypothesis_analysis_4m')
    parser.add_argument('--correspondence-root', type=Path,
                        default=outputs/'projection_correspondence_audit_4m')
    parser.add_argument('--data-root', type=Path,
                        default=base.parents[3]/'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-per-category', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = outputs/'visualizations'/(
            'hypothesis_pairs_common_'+args.class_name)

    paths = {
        'petr_r50': args.analysis_root/'petr_r50'/args.class_name/'hypothesis/all_gt.csv',
        'streampetr_r50': args.analysis_root/'streampetr_r50'/args.class_name/'hypothesis/all_gt.csv',
        'repdetr3d_eva02_l': args.analysis_root/'repdetr3d_eva02_l'/args.class_name/'hypothesis/all_gt.csv',
        'petr_eva02_l_epoch16': args.analysis_root/'petr_eva02_l_epoch16'/args.class_name/'hypothesis/all_gt.csv',
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error('Missing analysis files:\n'+'\n'.join(missing))
    rows_by_model = {slug: read_rows(path) for slug, path in paths.items()}
    correspondence_paths = {
        slug: args.correspondence_root/slug/args.class_name/'cases.csv'
        for slug, _ in MODEL_SPECS}
    missing_correspondence = [str(path) for path in correspondence_paths.values()
                              if not path.is_file()]
    if missing_correspondence:
        parser.error('Missing correspondence audits:\n' +
                     '\n'.join(missing_correspondence))
    correspondence_by_model = {
        slug: read_correspondence(path)
        for slug, path in correspondence_paths.items()}
    common = set.intersection(*[set(rows) for rows in rows_by_model.values()])

    def eligible(key, alternative):
        values = [rows[key] for rows in rows_by_model.values()]
        return all(row['official_match'] == '1' and
                   row.get('preferred_alt_status') == 'unmatched' and
                   (row.get('preferred_alt_better') == '1') == alternative
                   for row in values) if alternative else all(
                       row['official_match'] == '1' and
                       row.get('preferred_alt_better') == '0'
                       for row in values)

    available = {
        'alternative_exists': sorted(key for key in common if eligible(key, True)),
        'official_tp_best': sorted(key for key in common if eligible(key, False)),
    }
    rng = random.Random(args.seed)
    selected = {}
    for category, keys in available.items():
        count = min(args.max_per_category, len(keys))
        selected[category] = sorted(rng.sample(keys, count)) if count < len(keys) else keys
    ordered = [(category, key) for category in
               ('alternative_exists', 'official_tp_best')
               for key in selected[category]]

    # Load raw GT/camera metadata once per class, then project all four models.
    all_selected_rows = []
    for slug, _ in MODEL_SPECS:
        all_selected_rows.extend(rows_by_model[slug][key] for _, key in ordered)
    metadata = load_metadata(args.data_root, all_selected_rows, args.class_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_tokens = {key[0] for _, key in ordered}
    scene_by_sample = {
        record['token']: record['scene_token']
        for record in iter_json_array(args.data_root/'v1.0-trainval/sample.json')
        if record['token'] in selected_tokens
    }

    cases_by_model = {}
    for slug, _ in MODEL_SPECS:
        model_rows = []
        for category, key in ordered:
            row = dict(rows_by_model[slug][key])
            if category == 'alternative_exists':
                row['best_3d_prediction_id'] = row['preferred_alt_prediction_id']
                row['best_3d_status'] = row['preferred_alt_status']
                row['best_3d_box_json'] = row['preferred_alt_box_json']
                for metric in ('score', 'center_m', 'iou3d',
                               'projected_iou2d', 'radial_error_m',
                               'transverse_error_m'):
                    row['best_3d_'+metric] = row['preferred_alt_'+metric]
                row['delta_iou3d'] = str(float(row['preferred_alt_iou3d'])-
                                         float(row['selected_iou3d']))
                row['delta_projected_iou2d'] = str(
                    float(row['preferred_alt_projected_iou2d'])-
                    float(row['selected_projected_iou2d']))
            model_rows.append(row)
        cases_by_model[slug] = build_cases(
            model_rows, metadata, args.data_root, args.output_dir)

    cases = []
    for index, (category, key) in enumerate(ordered):
        model_cases = {}
        for slug, _ in MODEL_SPECS:
            model_case = cases_by_model[slug][index]
            model_case['correspondence'] = correspondence_by_model[slug][key]
            model_cases[slug] = model_case
        cases.append({
            'category': category,
            'sample_token': key[0], 'scene_token': scene_by_sample[key[0]],
            'gt_index': key[1],
            'models': model_cases,
        })
    payload = {
        'class_name': args.class_name,
        'models': [{'slug': slug, 'name': name} for slug, name in MODEL_SPECS],
        'available_counts': {key: len(value) for key, value in available.items()},
        'selected_counts': {key: len(value) for key, value in selected.items()},
        'max_per_category': args.max_per_category,
        'seed': args.seed,
        'cases': cases,
    }
    (args.output_dir/'comparison_data.js').write_text(
        'const comparisonData='+json.dumps(payload, separators=(',', ':'))+';\n')
    (args.output_dir/'summary.json').write_text(json.dumps({
        key: payload[key] for key in
        ('class_name', 'available_counts', 'selected_counts',
         'max_per_category', 'seed')}, indent=2)+'\n')
    print(json.dumps(payload['available_counts'], indent=2))
    print(json.dumps(payload['selected_counts'], indent=2))
    print('Wrote', len(cases), 'common cases to', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
