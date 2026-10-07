#!/usr/bin/env python3
"""Official-validated car AP@4m and ranked FP audit on R1-f *test* frames.

Uses existing final prediction JSON and PCCR evaluator tables. No inference,
checkpoint load, or post-hoc prediction correction is performed.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from audit_car_ap_ranking import (  # noqa: E402
    FIELDS, read_car_predictions, read_other_gt, replay, save_pr_svg,
    summarize)
from audit_tide3d_car_full import (  # noqa: E402
    RANGES, car_false_positives, load_json, match_class, rotate_inverse)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'data/pccr/R1-f/v1.0-test'
RESULTS = ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr'
DEFAULT_OUTPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_test/car_ap_fp_audit'
MODELS = ('R1-f', 'R1')
DISTANCES = ('0.5', '1.0', '2.0', '4.0')
FP_FIELDS = ('model', 'rank', 'sample_token', 'prediction_index', 'score',
             'recall_at_rank', 'before_25pct_recall', 'category',
             'nearest_car_gt_distance_m', 'nearest_car_gt_index',
             'nearest_other_gt_distance_m', 'nearest_other_gt_index',
             'nearest_other_gt_class')


def paths(model):
    formatted = (RESULTS / model / 'cross_rig/evaluations/R1-f/formatted')
    return formatted / 'results_pccr.json', formatted / 'metrics_summary.json'


def evaluator_gt(tokens):
    """Recreate evaluator-eligible GT using the same PCCR rules as val audits."""
    tokens = set(tokens)
    poses = {row['token']: row for row in load_json(TABLES / 'ego_pose.json')}
    pose_by_sample = {}
    for row in load_json(TABLES / 'sample_data.json'):
        token = row['sample_token']
        if token not in tokens or not row['is_key_frame']:
            continue
        pose_token = row['ego_pose_token']
        if token in pose_by_sample and pose_by_sample[token] != pose_token:
            raise ValueError('Unsynchronized keyframe poses: {}'.format(token))
        pose_by_sample[token] = pose_token
    if set(pose_by_sample) != tokens:
        raise ValueError('Keyframe ego poses do not cover prediction samples')
    categories = {row['token']: row['name']
                  for row in load_json(TABLES / 'category.json')}
    instance_names = {row['token']: categories[row['category_token']]
                      for row in load_json(TABLES / 'instance.json')}
    gt = {token: [] for token in tokens}
    for row in load_json(TABLES / 'sample_annotation.json'):
        token = row['sample_token']
        if token not in tokens:
            continue
        name = instance_names[row['instance_token']]
        if name not in RANGES or row['num_lidar_pts']+row['num_radar_pts'] == 0:
            continue
        pose = poses[pose_by_sample[token]]
        relative = [row['translation'][i]-pose['translation'][i] for i in range(3)]
        ego = rotate_inverse(pose['rotation'], relative)
        if math.hypot(ego[0], ego[1]) >= RANGES[name]:
            continue
        gt[token].append({'class': name,
                          'point': tuple(float(x) for x in row['translation'][:2]),
                          'annotation_token': row['token']})
    return gt


def car_gt_for_replay(gt):
    cars = defaultdict(list)
    for token, entries in gt.items():
        for index, entry in enumerate(entries):
            if entry['class'] == 'car':
                cars[token].append((index, entry['point']))
    return cars


def fp_checkpoint(fp_rows, rank_record, recall):
    checkpoint = rank_record['at_recall'][str(recall)]
    if checkpoint is None:
        return {'reached': False, 'fp': None, 'categories': {}}
    selected = [row for row in fp_rows if row['rank'] <= checkpoint['rank']]
    if len(selected) != checkpoint['fp']:
        raise ValueError('FP checkpoint differs from official AP replay')
    return {'reached': True, 'rank': checkpoint['rank'],
            'tp': checkpoint['tp'], 'fp': checkpoint['fp'],
            'precision': checkpoint['precision'],
            'score_cutoff': checkpoint['score'],
            'categories': dict(Counter(row['category'] for row in selected))}


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--score-threshold', type=float, default=0.60,
                        help='High-confidence FP cohort; AP replay uses all scores.')
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        raise ValueError('Score threshold must be in [0, 1]')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_sets, metric_sets = {}, {}
    for model in MODELS:
        result_path, metric_path = paths(model)
        if not result_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError('Missing saved {} R1-f test evaluation'.format(model))
        prediction_sets[model] = load_json(result_path)['results']
        metric_sets[model] = load_json(metric_path)
    tokens = set(prediction_sets[MODELS[0]])
    if tokens != set(prediction_sets[MODELS[1]]):
        raise ValueError('Model test predictions cover different frame tokens')
    gt = evaluator_gt(tokens)
    cars = car_gt_for_replay(gt)
    others = read_other_gt(TABLES, tokens)
    summary = {'rig': 'R1-f', 'split': 'test', 'threshold_m': 4.0,
               'score_threshold_for_high_confidence_fp': args.score_threshold,
               'gt_cars': sum(len(items) for items in cars.values()),
               'models': {},
               'note': 'AP and TP/FP use all final car predictions at the official '
                       '4 m threshold. GT proximity categories are descriptive, '
                       'not confirmed object identities or causes.'}
    all_ranked = []
    for model in MODELS:
        result_path, _ = paths(model)
        ranked = read_car_predictions(result_path, model)
        rows, gt_count, tp_scores = replay(model, ranked, cars, others, 4.0)
        if gt_count != summary['gt_cars']:
            raise ValueError('GT denominator changed between models')
        official_car = metric_sets[model]['label_aps']['car']
        rank_record = summarize(rows, gt_count, tp_scores, official_car['4.0'])
        official_class_mean = sum(official_car[key] for key in DISTANCES)/4.0
        car_matches = match_class(prediction_sets[model], gt, 'car')
        replay_tp = {(row['sample_token'], row['prediction_index'])
                     for row in rows if row['is_tp']}
        if set(car_matches) != replay_tp:
            raise ValueError('Rank replay and TIDE-style car TP identities differ')
        fp_rows = car_false_positives(
            model, prediction_sets[model], gt,
            {'car': car_matches}, near=8.0)
        if len(fp_rows) != rank_record['false_positives']:
            raise ValueError('FP ledger count differs from AP replay')
        high = [row for row in fp_rows if row['score'] >= args.score_threshold]
        high_ranked = [row for row in rows if row['car_score'] >= args.score_threshold]
        high_tp = sum(row['is_tp'] for row in high_ranked)
        if len(high_ranked)-high_tp != len(high):
            raise ValueError('High-score FP subset differs from rank ledger')
        high_summary = {
            'predictions': len(high_ranked), 'tp': high_tp, 'fp': len(high),
            'precision': high_tp/len(high_ranked) if high_ranked else None,
            'recall': high_tp/gt_count,
            'categories': dict(Counter(row['category'] for row in high))}
        rank_record.update({
            'official_car_ap_by_distance': official_car,
            'official_car_mean_ap_over_distances': official_class_mean,
            'official_overall_map': metric_sets[model]['mean_ap'],
            'fp_before_25pct_recall': fp_checkpoint(fp_rows, rank_record, .25),
            'fp_before_50pct_recall': fp_checkpoint(fp_rows, rank_record, .50),
            'fp_score_at_least_threshold': high_summary})
        summary['models'][model] = rank_record
        all_ranked.extend(rows)
        prefix = model.replace('-', '')
        write_csv(output / (prefix+'_high_score_car_fps.csv'), FP_FIELDS, high)
        before25 = [row for row in fp_rows if rank_record[
            'fp_before_25pct_recall']['reached'] and
            row['rank'] <= rank_record['fp_before_25pct_recall']['rank']]
        write_csv(output / (prefix+'_fp_before_25pct_recall.csv'),
                  FP_FIELDS, before25)
    write_csv(output / 'ranked_car_predictions.csv', FIELDS, all_ranked)
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    pr_svg = output / 'precision_recall.svg'
    save_pr_svg(pr_svg, summary)
    pr_svg.write_text(pr_svg.read_text().replace(
        'Car precision–recall on R1-f frames',
        'Car precision–recall on R1-f test frames'))
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-w', '1200', '-h', '620', str(pr_svg),
                        '-o', str(output / 'precision_recall.png')], check=True)
    print('Saved', output / 'summary.json')
    print('Saved', output / 'ranked_car_predictions.csv')
    print('Saved', output / 'precision_recall.svg')


if __name__ == '__main__':
    main()
