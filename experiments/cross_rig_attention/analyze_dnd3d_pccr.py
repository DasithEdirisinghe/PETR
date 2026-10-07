#!/usr/bin/env python3
"""DnD-style paired 3D detection audit using official PCCR AP@4m matching.

Runs offline on saved final PETR outputs. Candidate error labels are geometric
diagnostics, never replacements for official TP/FP matching or causal claims.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from audit_tide3d_car_full import (  # noqa: E402
    INPUT, TABLES, RANGES, THRESHOLD, all_matches, car_false_positives,
    car_gt_csv, class_ap, eligible_gt, load_json, point, predictions,
    score_order)

MODELS = ('R1-f', 'R1')
GT_FIELDS = (
    'sample_token', 'annotation_token', 'gt_list_index', 'gt_class', 'gt_x',
    'gt_y', 'transition', 'native_status', 'native_prediction_index',
    'native_prediction_class', 'native_score', 'native_distance_m',
    'native_candidate_count', 'native_candidate_kinds', 'native_blocked_tp_count', 'cross_status',
    'cross_prediction_index', 'cross_prediction_class', 'cross_score',
    'cross_distance_m', 'cross_candidate_count', 'cross_candidate_kinds',
    'cross_blocked_tp_count')


def candidate_kind(gt_class, pred_class, distance, near):
    if distance >= near:
        return None
    if distance < THRESHOLD:
        return 'Competition' if gt_class == pred_class else 'Cls'
    return 'Loc' if gt_class == pred_class else 'Both'


def diagnose_model(gt, preds, matches, near):
    """Pair TP identities exactly; assign free candidate boxes only for diagnosis.

    Candidate assignment is global across classes and one-to-one. The closest
    candidate wins, then higher score; this is NOT the official AP matcher.
    """
    matched_gt = {}
    taken = set()
    for class_matches in matches.values():
        for (token, pred_index), (gt_index, distance, score) in class_matches.items():
            gt_key = (token, gt_index)
            if gt_key in matched_gt:
                raise ValueError('GT matched in multiple classes: {}'.format(gt_key))
            matched_gt[gt_key] = (pred_index, distance, score)
            taken.add((token, pred_index))
    edges = []
    possible = defaultdict(list)
    competition = set()
    blocked_tp = Counter()
    for token, entries in gt.items():
        for gi, truth in enumerate(entries):
            key = (token, gi)
            if key in matched_gt:
                continue
            for pi, box in enumerate(preds[token]):
                distance = math.dist(truth['point'], point(box))
                kind = candidate_kind(truth['class'], box['detection_name'],
                                      distance, near)
                if kind is None:
                    continue
                if (token, pi) in taken:
                    competition.add(key)
                    blocked_tp[key] += 1
                    continue
                if kind == 'Competition':
                    # A free same-class box within 4 m would contradict the
                    # official greedy match; keep it visible if encountered.
                    competition.add(key)
                    continue
                possible[key].append((pi, kind, distance))
                edges.append((distance, -float(box['detection_score']),
                              token, gi, pi, kind))
    chosen = {}
    used_candidate = set()
    for distance, _, token, gi, pi, kind in sorted(edges):
        key, pred_key = (token, gi), (token, pi)
        if key not in chosen and pred_key not in used_candidate:
            chosen[key] = (pi, kind, distance)
            used_candidate.add(pred_key)
    records = {}
    candidate_owner = {}
    for token, entries in gt.items():
        for gi, truth in enumerate(entries):
            key = (token, gi)
            if key in matched_gt:
                pi, distance, score = matched_gt[key]
                status = 'Matched'
            elif key in chosen:
                pi, status, distance = chosen[key]
                score = float(preds[token][pi]['detection_score'])
                candidate_owner[(token, pi)] = truth['annotation_token']
            else:
                pi, score, distance = None, None, None
                status = 'Competition' if key in competition or possible[key] else 'Miss'
            box = preds[token][pi] if pi is not None else None
            records[key] = {
                'status': status, 'prediction_index': pi,
                'prediction_class': box['detection_name'] if box else None,
                'score': score, 'distance_m': distance,
                'candidate_count': len(possible[key]),
                'candidate_kinds': sorted({entry[1] for entry in possible[key]}),
                'blocked_tp_count': blocked_tp[key]}
    return records, candidate_owner


def transition(native, cross):
    a, b = native['status'] == 'Matched', cross['status'] == 'Matched'
    return ('both_tp' if a and b else 'native_only' if a else
            'cross_only' if b else 'neither')


def checkpoint(preds, matches, gt_count, target):
    required = math.ceil(target*gt_count)
    tp = 0
    for rank, (score, token, index, _) in enumerate(score_order(preds, 'car'), 1):
        tp += (token, index) in matches['car']
        if tp >= required:
            return {'rank': rank, 'tp': tp, 'fp': rank-tp,
                    'precision': tp/rank, 'recall': tp/gt_count,
                    'score_cutoff': score}
    return None


def early_fp_rows(model, preds, gt, matches, near, owners):
    rows = car_false_positives(model, preds, gt, matches, near)
    gt_count = sum(entry['class'] == 'car' for entries in gt.values()
                   for entry in entries)
    cutoffs = {str(target): checkpoint(preds, matches, gt_count, target)
               for target in (0.25, 0.50)}
    for row in rows:
        row['candidate_for_missed_gt'] = owners.get(
            (row['sample_token'], row['prediction_index']), '')
    early = [row for row in rows if cutoffs['0.5'] is not None and
             row['rank'] <= cutoffs['0.5']['rank']]
    for target, result in cutoffs.items():
        if result is None:
            continue
        prefix = [row for row in rows if row['rank'] <= result['rank']]
        if len(prefix) != result['fp']:
            raise ValueError('FP rank replay mismatch for {} at {}'.format(model, target))
        result['fp_proximity_categories'] = dict(Counter(row['category'] for row in prefix))
        result['fp_linked_to_missed_gt_candidate'] = sum(
            bool(row['candidate_for_missed_gt']) for row in prefix)
    return early, cutoffs


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(input_dir, table_dir, output_dir, near):
    if near <= THRESHOLD:
        raise ValueError('Diagnostic search radius must exceed official 4 m')
    cars = car_gt_csv(input_dir / 'analysis/car_per_gt.csv')
    preds = {model: predictions(input_dir / model / 'formatted/results_pccr.json')
             for model in MODELS}
    if set(preds['R1-f']) != set(preds['R1']):
        raise ValueError('Models must be evaluated on the exact same frames')
    gt = eligible_gt(table_dir, cars, preds['R1-f'])
    metrics = {model: load_json(input_dir / model /
               'formatted/metrics_summary.json') for model in MODELS}
    matches, states, owners = {}, {}, {}
    ap_validation = {}
    for model in MODELS:
        matches[model] = all_matches(preds[model], gt)
        errors = {}
        for name in RANGES:
            actual = class_ap(preds[model], gt, name)
            official = metrics[model]['label_aps'][name][str(THRESHOLD)]
            errors[name] = actual-official
            if abs(errors[name]) > 1e-4:
                raise ValueError('{} {} AP@4m replay differs: {} vs {}'.format(
                    model, name, actual, official))
        ap_validation[model] = {'max_abs_error': max(map(abs, errors.values())),
                                'car_ap': metrics[model]['label_aps']['car']['4.0']}
        states[model], owners[model] = diagnose_model(
            gt, preds[model], matches[model], near)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = []
    per_class = defaultdict(Counter)
    native_only_errors = defaultdict(Counter)
    cross_only_errors = defaultdict(Counter)
    shared_error_matrix = defaultdict(Counter)
    for token, entries in gt.items():
        for gi, truth in enumerate(entries):
            key = (token, gi)
            native, cross = states['R1-f'][key], states['R1'][key]
            group = transition(native, cross)
            class_name = truth['class']
            per_class[class_name][group] += 1
            if group == 'native_only':
                native_only_errors[class_name][cross['status']] += 1
            elif group == 'cross_only':
                cross_only_errors[class_name][native['status']] += 1
            elif group == 'neither':
                shared_error_matrix[class_name][
                    native['status']+'__'+cross['status']] += 1
            row = {'sample_token': token, 'annotation_token': truth['annotation_token'],
                   'gt_list_index': gi, 'gt_class': class_name,
                   'gt_x': truth['point'][0], 'gt_y': truth['point'][1],
                   'transition': group}
            for prefix, item in (('native', native), ('cross', cross)):
                row.update({prefix+'_status': item['status'],
                            prefix+'_prediction_index': item['prediction_index'],
                            prefix+'_prediction_class': item['prediction_class'],
                            prefix+'_score': item['score'],
                            prefix+'_distance_m': item['distance_m'],
                            prefix+'_candidate_count': item['candidate_count'],
                            prefix+'_candidate_kinds': '|'.join(item['candidate_kinds']),
                            prefix+'_blocked_tp_count': item['blocked_tp_count']})
            gt_rows.append(row)
    gt_rows.sort(key=lambda row: (row['sample_token'], row['gt_list_index']))
    write_csv(output_dir / 'paired_gt.csv', GT_FIELDS, gt_rows)
    fp_summary = {}
    for model in MODELS:
        early, cutoffs = early_fp_rows(model, preds[model], gt,
                                      matches[model], near, owners[model])
        if early:
            write_csv(output_dir / ('{}_car_fp_before_50pct.csv'.format(
                model.replace('-', ''))), list(early[0]), early)
        fp_summary[model] = cutoffs
    summary = {
        'rig': 'R1-f', 'official_match_threshold_m': THRESHOLD,
        'diagnostic_candidate_radius_m': near,
        'models': {'native': 'R1-f', 'cross': 'R1'},
        'ap_validation': ap_validation,
        'gt_counts_by_class_and_transition': {
            name: dict(counts) for name, counts in per_class.items()},
        'native_only_cross_error_candidates': {
            name: dict(counts) for name, counts in native_only_errors.items()},
        'cross_only_native_error_candidates': {
            name: dict(counts) for name, counts in cross_only_errors.items()},
        'neither_error_pair_matrix': {
            name: dict(counts) for name, counts in shared_error_matrix.items()},
        'car_ranked_fp': fp_summary,
        'status_definition': {
            'Matched': 'Official class-specific, score-ranked BEV-center TP <4 m.',
            'Cls': 'Unmatched GT; selected free different-class box <4 m.',
            'Loc': 'Unmatched GT; selected free same-class box 4 m to search radius.',
            'Both': 'Unmatched GT; selected free different-class box 4 m to search radius.',
            'Miss': 'No free final-output candidate within search radius.',
            'Competition': 'No free candidate assigned; nearby box is a TP for another GT or assigned to another diagnostic GT.'},
        'caveat': 'Cls/Loc/Both are one-to-one geometric candidate labels, not '
                  'proven object identity or causal errors. Ranked FPs are a '
                  'separate prediction-centred ledger; AP impacts are not additive.'}
    car = summary['gt_counts_by_class_and_transition'].get('car', {})
    if input_dir.resolve() == INPUT.resolve() and sum(car.values()) != 3591:
        raise ValueError('Expected 3591 evaluator-valid GT cars')
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    from plot_dnd3d_pccr import plot  # noqa: E402
    plot(output_dir / 'summary.json', output_dir / 'dnd3d_car_summary.svg')
    print('Saved', output_dir / 'summary.json')
    print('Saved', output_dir / 'paired_gt.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--table-dir', type=Path, default=TABLES)
    parser.add_argument('--near-radius', type=float, default=8.0)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / 'dnd3d_car_4m'
    run(args.input_dir, args.table_dir, output_dir, args.near_radius)


if __name__ == '__main__':
    main()
