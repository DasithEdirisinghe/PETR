#!/usr/bin/env python3
"""Explain GT coverage and prediction scatter in saved crowded groups."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


GT_FIELDS = ('sample_token', 'threshold_m', 'gt_index', 'annotation_token',
             'instance_token', 'category', 'official_prediction_id',
             'official_center_distance_m', 'official_bev_iou',
             'nearest_prediction_id', 'nearest_prediction_distance_m',
             'nearest_prediction_score', 'nearest_prediction_official_gt_index',
             'best_iou_prediction_id', 'best_iou_prediction_score',
             'best_iou_prediction_official_gt_index', 'best_iou',
             'predictions_within_threshold',
             'predictions_within_1m', 'predictions_within_2m',
             'predictions_within_4m', 'close_predictions_claimed_elsewhere',
             'close_predictions_false_positive',
             'local_center_reassignment_covers_gt')
PRED_FIELDS = ('sample_token', 'threshold_m', 'prediction_id', 'score',
               'local_score_rank', 'global_class_rank', 'official_status',
               'official_gt_index', 'nearest_group_gt_index',
               'nearest_group_gt_distance_m', 'highest_iou_group_gt_index',
               'highest_group_bev_iou', 'closer_group_gt_than_assignment',
               'better_overlap_group_gt_than_assignment')


def interpret(input_path, output_dir, low_iou=0.1, overlap_margin=0.1):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(Counter)
    pred_counts = defaultdict(Counter)
    with input_path.open() as source, \
            (output_dir / 'gt_diagnostics.csv').open('w', newline='') as gt_file, \
            (output_dir / 'prediction_diagnostics.csv').open('w', newline='') as pred_file:
        gt_writer = csv.DictWriter(gt_file, fieldnames=GT_FIELDS)
        pred_writer = csv.DictWriter(pred_file, fieldnames=PRED_FIELDS)
        gt_writer.writeheader()
        pred_writer.writeheader()
        for line in source:
            group = json.loads(line)
            token = group['sample_token']
            predictions = {p['prediction_index']: p for p in group['predictions']}
            gt_ids = {g['gt_index'] for g in group['ground_truth']}
            pairs = {(p['prediction_index'], p['gt_index']): p
                     for p in group['pair_matrix']}
            for threshold_text, outcome in group['thresholds'].items():
                threshold = float(threshold_text)
                official = {int(gi): int(pred_id.rsplit(':', 1)[1])
                            for gi, pred_id in outcome['official_gt_to_prediction'].items()}
                pred_official = {int(pi): gi for pi, gi in
                                 outcome['official_prediction_to_gt'].items()}
                counterfactual_gts = {p['gt_index'] for p in
                                      outcome['center_counterfactual']}
                counts[threshold]['groups'] += 1
                for gt in group['ground_truth']:
                    gi = gt['gt_index']
                    edges = [pairs[pi, gi] for pi in predictions]
                    nearest = min(edges, key=lambda e: e['center_distance_m'],
                                  default=None)
                    best_iou = max(edges, key=lambda e: e['bev_iou'], default=None)
                    close = [e for e in edges if e['center_distance_m'] < threshold]
                    owner = official.get(gi)
                    owner_edge = pairs.get((owner, gi)) if owner is not None else None
                    claimed_elsewhere = sum(pred_official[e['prediction_index']] != gi
                                            and pred_official[e['prediction_index']] is not None
                                            for e in close)
                    nearby_fp = sum(pred_official[e['prediction_index']] is None
                                    for e in close)
                    if owner is None:
                        if close:
                            category = 'missed_gt_close_prediction_claimed_elsewhere'
                        else:
                            category = 'missed_gt_no_prediction_within_threshold'
                    elif owner_edge['bev_iou'] < low_iou:
                        if best_iou and best_iou['bev_iou'] >= low_iou and \
                                best_iou['bev_iou'] > owner_edge['bev_iou'] + overlap_margin:
                            category = 'matched_gt_better_overlapping_prediction_elsewhere'
                        else:
                            category = 'matched_gt_low_overlap'
                    else:
                        category = 'matched_gt_overlap'
                    counts[threshold][category] += 1
                    if owner is None and gi in counterfactual_gts:
                        counts[threshold]['missed_gt_local_reassignment_covers'] += 1
                    gt_writer.writerow({
                        'sample_token': token, 'threshold_m': threshold, 'gt_index': gi,
                        'annotation_token': gt['annotation_token'],
                        'instance_token': gt['instance_token'], 'category': category,
                        'official_prediction_id': token + ':' + str(owner)
                        if owner is not None else '',
                        'official_center_distance_m': owner_edge['center_distance_m']
                        if owner_edge else '',
                        'official_bev_iou': owner_edge['bev_iou'] if owner_edge else '',
                        'nearest_prediction_id': token + ':' + str(nearest['prediction_index'])
                        if nearest else '',
                        'nearest_prediction_distance_m': nearest['center_distance_m']
                        if nearest else '',
                        'nearest_prediction_score': predictions[nearest['prediction_index']]['score']
                        if nearest else '',
                        'nearest_prediction_official_gt_index':
                        pred_official[nearest['prediction_index']]
                        if nearest and pred_official[nearest['prediction_index']] is not None else '',
                        'best_iou_prediction_id': token + ':' + str(best_iou['prediction_index'])
                        if best_iou else '',
                        'best_iou_prediction_score': predictions[best_iou['prediction_index']]['score']
                        if best_iou else '',
                        'best_iou_prediction_official_gt_index':
                        pred_official[best_iou['prediction_index']]
                        if best_iou and pred_official[best_iou['prediction_index']] is not None else '',
                        'best_iou': best_iou['bev_iou'] if best_iou else '',
                        'predictions_within_threshold': len(close),
                        'predictions_within_1m': sum(e['center_distance_m'] < 1 for e in edges),
                        'predictions_within_2m': sum(e['center_distance_m'] < 2 for e in edges),
                        'predictions_within_4m': sum(e['center_distance_m'] < 4 for e in edges),
                        'close_predictions_claimed_elsewhere': claimed_elsewhere,
                        'close_predictions_false_positive': nearby_fp,
                        'local_center_reassignment_covers_gt': int(gi in counterfactual_gts)})
                    if owner is None and close and nearby_fp:
                        raise AssertionError('An unmatched GT has a close unclaimed prediction')
                for pi, prediction in predictions.items():
                    edges = [pairs[pi, gi] for gi in gt_ids]
                    nearest = min(edges, key=lambda e: e['center_distance_m'])
                    best_iou = max(edges, key=lambda e: e['bev_iou'])
                    assigned_gt = pred_official[pi]
                    if assigned_gt is None:
                        status = 'FP'
                    elif assigned_gt in gt_ids:
                        status = 'TP_in_group'
                    else:
                        status = 'TP_outside_group'
                    assigned_edge = pairs.get((pi, assigned_gt))
                    closer = int(assigned_edge is not None and
                                 nearest['gt_index'] != assigned_gt and
                                 nearest['center_distance_m'] <
                                 assigned_edge['center_distance_m'])
                    better_overlap = int(assigned_edge is not None and
                                         best_iou['gt_index'] != assigned_gt and
                                         best_iou['bev_iou'] >= low_iou and
                                         best_iou['bev_iou'] >
                                         assigned_edge['bev_iou'] + overlap_margin)
                    pred_counts[threshold][status] += 1
                    if status == 'FP':
                        for radius in (0.5, 1.0, 2.0, 4.0):
                            if nearest['center_distance_m'] < radius:
                                pred_counts[threshold]['FP_within_{}m_of_group_GT'.format(radius)] += 1
                    pred_counts[threshold]['closer_group_gt_than_assignment'] += closer
                    pred_counts[threshold]['better_overlap_group_gt_than_assignment'] += better_overlap
                    pred_writer.writerow({
                        'sample_token': token, 'threshold_m': threshold,
                        'prediction_id': prediction['prediction_id'],
                        'score': prediction['score'],
                        'local_score_rank': prediction['local_score_rank'],
                        'global_class_rank': prediction['global_class_rank'],
                        'official_status': status, 'official_gt_index': assigned_gt
                        if assigned_gt is not None else '',
                        'nearest_group_gt_index': nearest['gt_index'],
                        'nearest_group_gt_distance_m': nearest['center_distance_m'],
                        'highest_iou_group_gt_index': best_iou['gt_index'],
                        'highest_group_bev_iou': best_iou['bev_iou'],
                        'closer_group_gt_than_assignment': closer,
                        'better_overlap_group_gt_than_assignment': better_overlap})
    summary = {'low_overlap_bev_iou_threshold': low_iou,
               'better_overlap_margin': overlap_margin,
               'gt_counts_by_threshold': {str(k): dict(v) for k, v in counts.items()},
               'prediction_counts_by_threshold': {str(k): dict(v)
                                                  for k, v in pred_counts.items()}}
    (output_dir / 'interpretation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).parent / 'outputs/groups.jsonl')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'outputs')
    parser.add_argument('--low-bev-iou', type=float, default=0.1)
    parser.add_argument('--better-overlap-margin', type=float, default=0.1)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error('Missing group matrix file: {}'.format(args.input))
    if not 0 <= args.low_bev_iou <= 1 or not 0 <= args.better_overlap_margin <= 1:
        parser.error('IoU thresholds must be in [0, 1]')
    summary = interpret(args.input, args.output_dir, args.low_bev_iou,
                        args.better_overlap_margin)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
