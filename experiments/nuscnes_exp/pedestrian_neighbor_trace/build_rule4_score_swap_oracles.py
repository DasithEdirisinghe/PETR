#!/usr/bin/env python3
"""Build Case-1, Case-2, and combined Rule-4 score-swap submissions."""

import argparse
import copy
import csv
import json
import os
from collections import Counter
from pathlib import Path


CLASSES = (
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')
VARIANTS = ('case1', 'case2', 'combined')
ARRAY_FIELDS = ('translation', 'size', 'rotation', 'velocity')
EPS = 1e-7
BUILD_VERSION = 3


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def matches_saved(raw, saved):
    if raw['sample_token'] != saved['sample_token'] or \
            raw['detection_name'] != saved['detection_name'] or \
            raw.get('attribute_name', '') != saved.get('attribute_name', ''):
        return False
    if abs(float(raw['detection_score'])-float(saved['detection_score'])) > EPS:
        return False
    for field in ARRAY_FIELDS:
        if len(raw[field]) != len(saved[field]) or any(
                abs(float(a)-float(b)) > EPS
                for a, b in zip(raw[field], saved[field])):
            return False
    return True


def locate_prediction(submission, saved):
    candidates = []
    for index, raw in enumerate(submission['results'][saved['sample_token']]):
        if matches_saved(raw, saved):
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError('Expected one exact submitted prediction for {} {} score {}; '
                         'found {}'.format(saved['sample_token'],
                                           saved['detection_name'],
                                           saved['detection_score'], candidates))
    return saved['sample_token'], candidates[0]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_submission_atomic(path, submission):
    """Stream JSON, fsync, parse-check it, then atomically publish it."""
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as handle:
        json.dump(submission, handle, separators=(',', ':'))
        handle.flush()
        os.fsync(handle.fileno())
    with temporary.open() as handle:
        validated = json.load(handle)
    if len(validated.get('results', {})) != len(submission.get('results', {})):
        raise ValueError('Written submission failed sample-count validation: {}'
                         .format(temporary))
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--source-results', type=Path, required=True)
    parser.add_argument('--rule4-model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.source_results.is_file():
        parser.error('Missing source submission: {}'.format(args.source_results))

    source = json.loads(args.source_results.read_text())
    pairs = []
    excluded = []
    for class_name in CLASSES:
        path = args.rule4_model_dir/class_name/'rule4_cases.csv'
        if not path.is_file():
            parser.error('Missing Rule-4 cases: {}'.format(path))
        rows = read_rows(path)
        alternative_counts = Counter(row['alternative_prediction_id'] for row in rows)
        repeated = {prediction_id for prediction_id, count in
                    alternative_counts.items() if count > 1}
        for row in rows:
            if row['alternative_prediction_id'] in repeated:
                excluded.append({
                    'class': class_name, 'sample_token': row['sample_token'],
                    'gt_index': row['gt_index'],
                    'alternative_prediction_id': row['alternative_prediction_id'],
                    'reason': 'alternative selected for multiple GTs'})
                continue
            alignment = row['official_tp_alignment_category']
            if alignment == 'target_gt':
                case_type = 'case1'
            elif alignment == 'other_same_class_gt':
                case_type = 'case2'
            else:
                continue
            official_saved = json.loads(row['official_tp_box_json'])
            alternative_saved = json.loads(row['alternative_box_json'])
            official_key = locate_prediction(source, official_saved)
            alternative_key = locate_prediction(source, alternative_saved)
            pairs.append({
                'model': args.model_name, 'class': class_name,
                'case_type': case_type, 'sample_token': row['sample_token'],
                'gt_index': int(row['gt_index']),
                'official_prediction_id': int(row['official_tp_prediction_id']),
                'alternative_prediction_id': int(row['alternative_prediction_id']),
                'official_record_index': official_key[1],
                'alternative_record_index': alternative_key[1],
                'official_score': float(row['official_tp_score']),
                'alternative_score': float(row['alternative_score']),
                'official_center_m': float(row['official_tp_center_m']),
                'alternative_center_m': float(row['alternative_center_m']),
                'official_iou3d': float(row['official_tp_iou3d']),
                'alternative_iou3d': float(row['alternative_iou3d']),
                'official_iou2d': float(row['official_tp_projected_iou2d']),
                'alternative_iou2d': float(row['alternative_projected_iou2d']),
                'alternative_target_margin_iou2d':
                    float(row['alternative_target_iou2d'])-
                    float(row['alternative_max_competitor_iou2d']),
            })

    official_keys = [(p['sample_token'], p['official_record_index']) for p in pairs]
    alternative_keys = [(p['sample_token'], p['alternative_record_index']) for p in pairs]
    if len(official_keys) != len(set(official_keys)):
        raise ValueError('An official TP appears in multiple retained score-swap pairs')
    if len(alternative_keys) != len(set(alternative_keys)):
        raise ValueError('An alternative appears in multiple retained score-swap pairs')
    if set(official_keys) & set(alternative_keys):
        raise ValueError('A retained prediction is both an official TP and an alternative')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.output_dir/'baseline'
    baseline_dir.mkdir(exist_ok=True)
    write_submission_atomic(baseline_dir/'results_nusc.json', source)

    summary = {
        'build_version': BUILD_VERSION,
        'model': args.model_name,
        'classes': list(CLASSES),
        'source_results': str(args.source_results.resolve()),
        'rule4_model_dir': str(args.rule4_model_dir.resolve()),
        'conflict_policy': ('Exclude every pair row whose alternative prediction '
                            'was selected for more than one GT.'),
        'excluded_conflict_rows': len(excluded),
        'retained_pairs': len(pairs),
        'counts': {},
    }
    for variant in VARIANTS:
        selected = [pair for pair in pairs
                    if variant == 'combined' or pair['case_type'] == variant]
        submission = copy.deepcopy(source)
        changed = set()
        for pair in selected:
            token = pair['sample_token']
            official = submission['results'][token][pair['official_record_index']]
            alternative = submission['results'][token][pair['alternative_record_index']]
            if abs(official['detection_score']-pair['official_score']) > EPS or \
                    abs(alternative['detection_score']-pair['alternative_score']) > EPS:
                raise ValueError('Source score changed before applying pair {}:{}'
                                 .format(token, pair['gt_index']))
            official['detection_score'], alternative['detection_score'] = \
                alternative['detection_score'], official['detection_score']
            changed.add((token, pair['official_record_index']))
            changed.add((token, pair['alternative_record_index']))
        if len(changed) != 2*len(selected):
            raise ValueError('A prediction was modified more than once in {}'.format(variant))
        directory = args.output_dir/variant
        directory.mkdir(exist_ok=True)
        write_submission_atomic(directory/'results_nusc.json', submission)
        manifest = []
        for pair in selected:
            record = dict(pair)
            record['variant'] = variant
            record['official_new_score'] = pair['alternative_score']
            record['alternative_new_score'] = pair['official_score']
            manifest.append(record)
        write_csv(directory/'swaps.csv', manifest)
        summary['counts'][variant] = {
            'total': len(selected),
            'by_class': {class_name: sum(pair['class'] == class_name
                                         for pair in selected)
                         for class_name in CLASSES},
        }
        print('{}: {:,} swaps'.format(variant, len(selected)), flush=True)
    write_csv(args.output_dir/'excluded_conflicts.csv', excluded)
    (args.output_dir/'build_summary.json').write_text(
        json.dumps(summary, indent=2)+'\n')
    print('Excluded conflict rows:', len(excluded), flush=True)
    print(args.output_dir/'build_summary.json', flush=True)


if __name__ == '__main__':
    main()
