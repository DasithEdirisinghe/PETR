#!/usr/bin/env python3
"""Paired car GT/FP TIDE-style audit on the same R1-f validation frames.

Only saved, final PETR outputs are analyzed. Oracle changes exist in memory
solely to estimate diagnostic AP upper bounds; official results stay intact.
"""

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_val'
TABLES = ROOT / 'data/pccr/R1-f/v1.0-trainval'
RANGES = {'car': 50, 'truck': 50, 'bus': 50, 'motorcycle': 40,
          'bicycle': 40, 'adult': 40, 'child': 40,
          'traffic_light': 30, 'traffic_sign': 30}
THRESHOLD = 4.0
DEFAULT_NEAR = 8.0


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def rotate_inverse(q, v):
    w, x, y, z = (float(a) for a in q)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, -x/norm, -y/norm, -z/norm
    # Quaternion-derived rotation matrix, applied to the relative vector.
    return ((1-2*(y*y+z*z))*v[0]+2*(x*y-w*z)*v[1]+2*(x*z+w*y)*v[2],
            2*(x*y+w*z)*v[0]+(1-2*(x*x+z*z))*v[1]+2*(y*z-w*x)*v[2],
            2*(x*z-w*y)*v[0]+2*(y*z+w*x)*v[1]+(1-2*(x*x+y*y))*v[2])


def car_gt_csv(path):
    cars = defaultdict(dict)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            cars[row['sample_token']][int(row['gt_index'])] = (
                float(row['gt_x']), float(row['gt_y']))
    return cars


def eligible_gt(tables, cars, tokens):
    tokens = set(tokens)
    poses = {row['token']: row for row in load_json(tables / 'ego_pose.json')}
    pose_by_sample = {}
    for row in load_json(tables / 'sample_data.json'):
        token = row['sample_token']
        if token not in tokens or not row['is_key_frame']:
            continue
        pose_token = row['ego_pose_token']
        if token in pose_by_sample and pose_by_sample[token] != pose_token:
            raise ValueError('Unsynchronized keyframe poses: {}'.format(token))
        pose_by_sample[token] = pose_token
    if set(pose_by_sample) != tokens:
        raise ValueError('Missing ego pose for some evaluated frames')

    categories = {row['token']: row['name']
                  for row in load_json(tables / 'category.json')}
    instance_names = {row['token']: categories[row['category_token']]
                      for row in load_json(tables / 'instance.json')}
    gt = {token: [] for token in tokens}
    for row in load_json(tables / 'sample_annotation.json'):
        token = row['sample_token']
        if token not in tokens:
            continue
        name = instance_names[row['instance_token']]
        if name not in RANGES or row['num_lidar_pts'] + row['num_radar_pts'] == 0:
            continue
        pose = poses[pose_by_sample[token]]
        relative = [row['translation'][i]-pose['translation'][i] for i in range(3)]
        ego = rotate_inverse(pose['rotation'], relative)
        if math.hypot(ego[0], ego[1]) >= RANGES[name]:
            continue
        gt[token].append({'class': name,
                          'point': tuple(float(x) for x in row['translation'][:2]),
                          'annotation_token': row['token']})

    count = 0
    for token, entries in gt.items():
        found = Counter(tuple(round(value, 3) for value in entry['point'])
                        for entry in entries if entry['class'] == 'car')
        expected = Counter(tuple(round(value, 3) for value in center)
                           for center in cars.get(token, {}).values())
        if found != expected:
            raise ValueError('Car GT set mismatch for {}: computed={} csv={}'.format(
                token, found, expected))
        count += sum(found.values())
    if count != 3591:
        raise ValueError('Unexpected car GT count: {}'.format(count))
    return gt


def predictions(path):
    return load_json(path)['results']


def point(box):
    return box['translation'][:2]


def score_order(preds, class_name):
    selected = [(float(box['detection_score']), token, index, box)
                for token, boxes in preds.items()
                for index, box in enumerate(boxes)
                if box['detection_name'] == class_name]
    selected.sort(key=lambda entry: -entry[0])
    return selected


def match_class(preds, gt, class_name):
    used = set()
    matches = {}
    for score, token, index, box in score_order(preds, class_name):
        nearest = min(((math.dist(point(box), entry['point']), gt_index)
                       for gt_index, entry in enumerate(gt[token])
                       if entry['class'] == class_name and
                       (token, gt_index) not in used), default=None)
        if nearest is not None and nearest[0] < THRESHOLD:
            used.add((token, nearest[1]))
            matches[(token, index)] = (nearest[1], nearest[0], score)
    return matches


def class_ap(preds, gt, class_name='car'):
    total_gt = sum(entry['class'] == class_name for entries in gt.values()
                   for entry in entries)
    if total_gt == 0:
        return 0.0
    used = set()
    recalls, precisions = [], []
    tp = 0
    for rank, (score, token, index, box) in enumerate(score_order(preds, class_name), 1):
        nearest = min(((math.dist(point(box), entry['point']), gt_index)
                       for gt_index, entry in enumerate(gt[token])
                       if entry['class'] == class_name and
                       (token, gt_index) not in used), default=None)
        if nearest is not None and nearest[0] < THRESHOLD:
            used.add((token, nearest[1]))
            tp += 1
        recalls.append(tp / total_gt)
        precisions.append(tp / rank)
    if not recalls:
        return 0.0
    values = []
    for grid in range(11, 101):
        target = grid/100
        index = bisect.bisect_right(recalls, target)
        if index == 0:
            precision = precisions[0]
        elif index == len(recalls):
            precision = precisions[-1] if target <= recalls[-1] else 0.0
        else:
            left, right = recalls[index-1], recalls[index]
            weight = (target-left)/(right-left)
            precision = precisions[index-1] + weight*(precisions[index]-precisions[index-1])
        values.append(max(precision-.1, 0))
    return sum(values)/90/.9


def all_matches(preds, gt):
    return {name: match_class(preds, gt, name) for name in RANGES}


def candidate_edges(token, gt_index, gt_entry, boxes, taken, near):
    edges = []
    for index, box in enumerate(boxes):
        if (token, index) in taken:
            continue
        dist = math.dist(gt_entry['point'], point(box))
        if dist >= near:
            continue
        label = box['detection_name']
        if label == 'car' and dist < THRESHOLD:
            kind = 'matching_competition'
        elif label != 'car' and dist < THRESHOLD:
            kind = 'class_only_candidate'
        elif label == 'car':
            kind = 'localization_candidate'
        else:
            kind = 'class_and_localization_candidate'
        edges.append((dist, -float(box['detection_score']), index, kind))
    return sorted(edges)


def assign_unmatched(gt, preds, matches, all_model_matches, near):
    car_match_gt = {(token, entry[0]): (index, entry[1], entry[2])
                    for (token, index), entry in matches.items()}
    taken = {key for class_matches in all_model_matches.values()
             for key in class_matches}
    car_tp_by_token = defaultdict(list)
    for (token, index), (matched_index, _, _) in all_model_matches['car'].items():
        car_tp_by_token[token].append((index, matched_index))
    pending, diagnostics, competition = [], {}, {}
    for token, entries in gt.items():
        for gt_index, entry in enumerate(entries):
            if entry['class'] != 'car' or (token, gt_index) in car_match_gt:
                continue
            edges = candidate_edges(token, gt_index, entry, preds[token], taken, near)
            diagnostics[(token, gt_index)] = edges
            competition[(token, gt_index)] = sum(
                matched_index != gt_index and
                math.dist(point(preds[token][index]), entry['point']) < THRESHOLD
                for index, matched_index in car_tp_by_token[token])
            for dist, neg_score, index, kind in edges:
                pending.append((dist, neg_score, token, gt_index, index, kind))
    # One prediction can suggest at most one missed GT; nearest distance first.
    chosen, used_predictions = {}, set()
    for dist, neg_score, token, gt_index, index, kind in sorted(pending):
        key = (token, gt_index)
        pred_key = (token, index)
        if key not in chosen and pred_key not in used_predictions:
            chosen[key] = (index, dist, kind)
            used_predictions.add(pred_key)
    return car_match_gt, diagnostics, chosen, competition


def clone_predictions(preds):
    return {token: [dict(box) for box in boxes] for token, boxes in preds.items()}


def oracle(preds, gt, chosen, kind):
    changed = clone_predictions(preds)
    n = 0
    for (token, gt_index), (index, dist, actual_kind) in chosen.items():
        if actual_kind != kind:
            continue
        box = changed[token][index]
        if kind in ('class_only_candidate', 'class_and_localization_candidate'):
            box['detection_name'] = 'car'
        if kind in ('localization_candidate', 'class_and_localization_candidate'):
            box['translation'] = list(box['translation'])
            box['translation'][:2] = gt[token][gt_index]['point']
        n += 1
    return changed, n


def report_model(model, preds, gt, official_ap, near):
    baseline_ap = class_ap(preds, gt)
    if abs(baseline_ap-official_ap) > 1e-4:
        raise ValueError('{} baseline car AP mismatch: {} vs {}'.format(
            model, baseline_ap, official_ap))
    matches = all_matches(preds, gt)
    car_matches, diagnostics, chosen, competition = assign_unmatched(
        gt, preds, matches['car'], matches, near)
    # The number of car matches must agree with the earlier score-ranked audit.
    expected = 3440 if model == 'R1-f' else 2630
    if len(car_matches) != expected:
        raise ValueError('{} car TP count mismatch: {}'.format(model, len(car_matches)))
    oracle_results = {}
    for kind in ('class_only_candidate', 'localization_candidate',
                 'class_and_localization_candidate'):
        fixed, n = oracle(preds, gt, chosen, kind)
        fixed_ap = class_ap(fixed, gt)
        oracle_results[kind] = {'changed_predictions': n, 'oracle_ap': fixed_ap,
                                'delta_ap': fixed_ap-baseline_ap}
    categories = Counter(value[2] for value in chosen.values())
    for key in diagnostics:
        if key not in chosen:
            categories['matching_competition' if competition[key] else
                       'no_free_final_prediction_within_near_radius'] += 1
    return matches, car_matches, diagnostics, chosen, competition, {
        'official_car_ap_4m': official_ap, 'replayed_car_ap_4m': baseline_ap,
        'matched_car_gt': len(car_matches),
        'unmatched_car_gt': len(diagnostics),
        'unmatched_chosen_categories': dict(categories),
        'unmatched_with_competing_car_tp_within_4m': sum(
            count > 0 for count in competition.values()),
        'oracles': oracle_results,
    }


def car_false_positives(model, preds, gt, matches, near):
    """Official 4 m car FP identities plus all-class GT proximity evidence."""
    matched = matches['car']
    total_cars = sum(entry['class'] == 'car' for entries in gt.values()
                     for entry in entries)
    tp = 0
    rows = []
    for rank, (score, token, index, box) in enumerate(score_order(preds, 'car'), 1):
        if (token, index) in matched:
            tp += 1
            continue
        nearest_car = min(((math.dist(point(box), entry['point']), gt_index)
                           for gt_index, entry in enumerate(gt[token])
                           if entry['class'] == 'car'), default=None)
        nearest_other = min(((math.dist(point(box), entry['point']), gt_index,
                              entry['class'])
                             for gt_index, entry in enumerate(gt[token])
                             if entry['class'] != 'car'), default=None)
        car_d = nearest_car[0] if nearest_car else float('inf')
        other_d = nearest_other[0] if nearest_other else float('inf')
        if car_d < THRESHOLD:
            category = 'duplicate_or_competing_car'
        elif other_d < THRESHOLD:
            category = 'car_label_on_other_class_gt_candidate'
        elif min(car_d, other_d) < near:
            category = ('car_label_near_car_outside_4m' if car_d <= other_d
                        else 'car_label_near_other_outside_4m')
        else:
            category = 'no_evaluator_gt_within_near_radius'
        rows.append({
            'model': model, 'rank': rank, 'sample_token': token,
            'prediction_index': index, 'score': score,
            'recall_at_rank': tp/total_cars,
            'before_25pct_recall': int(tp/total_cars < .25),
            'category': category,
            'nearest_car_gt_distance_m': car_d if nearest_car else '',
            'nearest_car_gt_index': nearest_car[1] if nearest_car else '',
            'nearest_other_gt_distance_m': other_d if nearest_other else '',
            'nearest_other_gt_index': nearest_other[1] if nearest_other else '',
            'nearest_other_gt_class': nearest_other[2] if nearest_other else '',
        })
    return rows


def inverse_class_oracle(preds, fp_rows, early_only):
    changed = clone_predictions(preds)
    n = 0
    for row in fp_rows:
        if row['category'] != 'car_label_on_other_class_gt_candidate':
            continue
        if early_only and not row['before_25pct_recall']:
            continue
        changed[row['sample_token']][row['prediction_index']]['detection_name'] = (
            row['nearest_other_gt_class'])
        n += 1
    return changed, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--table-dir', type=Path, default=TABLES)
    parser.add_argument('--near-radius', type=float, default=DEFAULT_NEAR)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.near_radius <= THRESHOLD:
        raise ValueError('Diagnostic near radius must exceed official 4 m threshold')
    output = args.output_dir or args.input_dir / 'car_ap_rank_audit/tide3d_car_full'
    cars = car_gt_csv(args.input_dir / 'analysis/car_per_gt.csv')
    results = {model: predictions(args.input_dir / model /
               'formatted/results_pccr.json') for model in ('R1-f', 'R1')}
    if set(results['R1-f']) != set(results['R1']):
        raise ValueError('Native and cross-rig result frames differ')
    gt = eligible_gt(args.table_dir, cars, results['R1-f'])
    metrics = {model: load_json(args.input_dir / model /
               'formatted/metrics_summary.json') for model in results}
    output.mkdir(parents=True, exist_ok=True)
    summary = {'rig': 'R1-f', 'gt_cars': sum(map(len, cars.values())),
               'official_threshold_m': THRESHOLD,
               'diagnostic_near_radius_m': args.near_radius,
               'note': 'Unmatched-GT categories are proximity candidates, not '
                       'proven causal explanations. Oracle APs are independent '
                       'post-hoc tests on final outputs, not model results.',
               'models': {}}
    details = {}
    for model, preds in results.items():
        official = metrics[model]['label_aps']['car']['4.0']
        matches, car_matches, edges, chosen, competition, record = report_model(
            model, preds, gt, official, args.near_radius)
        class_errors = {}
        for name in RANGES:
            replayed = class_ap(preds, gt, name)
            expected = metrics[model]['label_aps'][name]['4.0']
            class_errors[name] = replayed - expected
            if abs(class_errors[name]) > 1e-4:
                raise ValueError('{} {} AP@4 mismatch: {} vs {}'.format(
                    model, name, replayed, expected))
        record['all_class_ap_4m_max_abs_replay_error'] = max(
            abs(error) for error in class_errors.values())
        fp_rows = car_false_positives(model, preds, gt, matches,
                                      args.near_radius)
        early = [row for row in fp_rows if row['before_25pct_recall']]
        record['car_fp_before_25pct_recall'] = len(early)
        record['car_fp_early_categories'] = dict(Counter(
            row['category'] for row in early))
        record['car_fp_early_other_gt_classes'] = dict(Counter(
            row['nearest_other_gt_class'] for row in early
            if row['category'] == 'car_label_on_other_class_gt_candidate'))
        record['inverse_class_oracles'] = {}
        for label, early_only in (('early_other_gt', True),
                                  ('all_other_gt', False)):
            fixed, n = inverse_class_oracle(preds, fp_rows, early_only)
            fixed_ap = class_ap(fixed, gt)
            record['inverse_class_oracles'][label] = {
                'changed_predictions': n, 'oracle_ap': fixed_ap,
                'delta_ap': fixed_ap-record['official_car_ap_4m']}
        with (output / '{}_car_fp_before_25pct.csv'.format(
                model.replace('-', ''))).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fp_rows[0]))
            writer.writeheader()
            writer.writerows(early)
        summary['models'][model] = record
        details[model] = (preds, matches, car_matches, edges, chosen, competition)

    native = details['R1-f'][2]
    cross = details['R1'][2]
    transitions = Counter()
    fields = ('sample_token', 'gt_index', 'gt_list_index', 'gt_x', 'gt_y', 'native_tp',
              'cross_tp', 'transition', 'model', 'diagnostic_category',
              'candidate_count', 'candidate_prediction_index',
              'candidate_class', 'candidate_score', 'candidate_distance_m',
              'candidate_is_nearest_gt_of_any_class',
              'competing_car_tp_within_4m_count')
    transition_categories = defaultdict(Counter)
    candidate_nearest_counts = defaultdict(Counter)
    with (output / 'paired_car_gt.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for token, entries in gt.items():
            for gt_index, entry in enumerate(entries):
                if entry['class'] != 'car':
                    continue
                ledger_indices = [index for index, center in cars[token].items()
                                  if math.dist(center, entry['point']) < .001]
                if len(ledger_indices) != 1:
                    raise ValueError('Cannot map GT car to original ledger ID')
                key = (token, gt_index)
                native_tp = key in native
                cross_tp = key in cross
                transition = ('both_tp' if native_tp and cross_tp else
                              'native_only' if native_tp else
                              'cross_only' if cross_tp else 'neither')
                transitions[transition] += 1
                for model in ('R1-f', 'R1'):
                    preds, matches, car_matches, edges, chosen, competition = details[model]
                    candidate = chosen.get(key)
                    if key in car_matches:
                        category = 'true_positive'
                    elif candidate:
                        category = candidate[2]
                    else:
                        category = ('matching_competition' if competition.get(key)
                                    else 'no_free_final_prediction_within_near_radius')
                    transition_categories[(transition, model)][category] += 1
                    pred_box = preds[token][candidate[0]] if candidate else None
                    nearest_class = ''
                    if candidate:
                        nearest_gt = min((math.dist(point(pred_box), other['point']),
                                          index, other['class'])
                                         for index, other in enumerate(entries))
                        nearest_class = int(nearest_gt[1] == gt_index)
                        candidate_nearest_counts[model][category + '__is_nearest'] += nearest_class
                        candidate_nearest_counts[model][category + '__total'] += 1
                    writer.writerow({
                        'sample_token': token, 'gt_index': ledger_indices[0],
                        'gt_list_index': gt_index,
                        'gt_x': entry['point'][0], 'gt_y': entry['point'][1],
                        'native_tp': int(native_tp), 'cross_tp': int(cross_tp),
                        'transition': transition, 'model': model,
                        'diagnostic_category': category,
                        'candidate_count': len(edges.get(key, ())),
                        'candidate_prediction_index': candidate[0] if candidate else '',
                        'candidate_class': pred_box['detection_name'] if pred_box else '',
                        'candidate_score': pred_box['detection_score'] if pred_box else '',
                        'candidate_distance_m': candidate[1] if candidate else '',
                        'candidate_is_nearest_gt_of_any_class': nearest_class,
                        'competing_car_tp_within_4m_count': competition.get(key, 0),
                    })
    summary['paired_gt_transitions'] = dict(transitions)
    summary['paired_gt_transition_categories'] = {
        '{}__{}'.format(transition, model): dict(counts)
        for (transition, model), counts in transition_categories.items()}
    summary['candidate_nearest_gt_checks'] = {
        model: dict(counts) for model, counts in candidate_nearest_counts.items()}
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print('Saved', output / 'summary.json')
    print('Saved', output / 'paired_car_gt.csv')


if __name__ == '__main__':
    main()
