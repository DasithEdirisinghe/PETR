#!/usr/bin/env python3
"""Paired R1-f car AP@4m TIDE-style audit and final-layer attention summary.

Reads saved, standard PETR predictions and previously captured attention only.
GT is used after inference. Oracle predictions are in-memory diagnostics and are
never presented as ordinary model results.
"""

import argparse
import bisect
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from audit_tide3d_car_full import (INPUT, TABLES, THRESHOLD,
                                   all_matches, car_gt_csv, class_ap,
                                   clone_predictions, eligible_gt, load_json,
                                   point, predictions, score_order)

TYPES = ('Cls', 'Loc', 'Both', 'Dupe', 'Bkgd', 'Missed')


def distance(a, b):
    return math.dist(a, b)


def associations(token, pred, entries, radius):
    return sorted((distance(point(pred), gt['point']), i, gt['class'])
                  for i, gt in enumerate(entries)
                  if distance(point(pred), gt['point']) < radius)


def classify_fp(token, index, box, gt, car_matches, all_model_matches, radius):
    """Return a conservative TIDE-inspired class and possible missed-car link."""
    edges = associations(token, box, gt[token], radius)
    close = [edge for edge in edges if edge[0] < THRESHOLD]
    if len(close) > 1:
        return 'Ambiguous', None, 'multiple_gt_within_4m'
    if close:
        _, gi, label = close[0]
        if label == 'car':
            if (token, gi) in car_matches:
                return 'Dupe', None, 'already_matched_car_gt'
            return 'Ambiguous', None, 'unmatched_car_within_4m'
        if any(match[0] == gi for (tk, _), match in
               all_model_matches[label].items() if tk == token):
            return 'Ambiguous', None, 'other_class_gt_already_detected'
        return 'Cls', None, 'car_label_near_{}_gt'.format(label)
    if not edges:
        return 'Bkgd', None, 'no_eligible_gt_within_radius'
    nearest = edges[0]
    if len(edges) > 1 and edges[1][0] - nearest[0] < 1.0:
        return 'Ambiguous', None, 'competing_gt_in_near_band'
    _, gi, label = nearest
    if label != 'car':
        return 'Both', None, 'car_label_4m_to_radius_from_{}_gt'.format(label)
    if (token, gi) in car_matches:
        return 'Ambiguous', None, 'near_already_detected_car'
    return 'Loc', (token, gi), 'car_label_4m_to_radius_from_car_gt'


def assign_missing_cars(model, gt, preds, all_model_matches, car_matches,
                        fp_rows, radius):
    """Associate at most one unused prediction to each missed car GT."""
    occupied = {(token, index) for matches in all_model_matches.values()
                for token, index in matches}
    linked = {(row['sample_token'], row['linked_gt_index']) for row in fp_rows
              if row['type'] == 'Loc' and row['linked_gt_index'] is not None}
    taken = {(row['sample_token'], row['prediction_index']) for row in fp_rows
             if row['type'] == 'Loc'}
    car_fp_near = defaultdict(list)
    for row in fp_rows:
        if row['type'] == 'Ambiguous':
            for d, gi, label in associations(row['sample_token'],
                    preds[row['sample_token']][row['prediction_index']],
                    gt[row['sample_token']], radius):
                if label == 'car':
                    car_fp_near[(row['sample_token'], gi)].append(row)
    candidates = []
    for token, entries in gt.items():
        for gi, entry in enumerate(entries):
            if entry['class'] != 'car' or (token, gi) in car_matches or (token, gi) in linked:
                continue
            for index, box in enumerate(preds[token]):
                if box['detection_name'] == 'car' or (token, index) in occupied:
                    continue
                d = distance(point(box), entry['point'])
                if d < radius:
                    candidates.append((d, -float(box['detection_score']),
                                       token, gi, index))
    selected = {}
    for d, neg_score, token, gi, index in sorted(candidates):
        if (token, gi) not in selected and (token, index) not in taken:
            selected[(token, gi)] = (index, d)
            taken.add((token, index))
    rows = []
    for token, entries in gt.items():
        for gi, entry in enumerate(entries):
            if entry['class'] != 'car' or (token, gi) in car_matches:
                continue
            key = (token, gi)
            if key in linked:
                kind, index, d, reason = 'Loc', None, None, 'linked_car_fp'
            elif key in selected:
                index, d = selected[key]
                nearby = associations(token, preds[token][index], entries, radius)
                if len(nearby) > 1 and nearby[1][0] - nearby[0][0] < 1.0:
                    kind, reason = 'Ambiguous', 'noncar_prediction_near_multiple_gt'
                elif nearby[0][1] != gi:
                    kind, reason = 'Ambiguous', 'noncar_prediction_closer_to_other_gt'
                else:
                    kind = 'Cls' if d < THRESHOLD else 'Both'
                    reason = 'unmatched_noncar_prediction_near_car_gt'
            elif car_fp_near.get(key):
                kind, index, d, reason = 'Ambiguous', None, None, 'ambiguous_car_fp_near_gt'
            else:
                kind, index, d, reason = 'Missed', None, None, 'no_free_prediction_within_radius'
            rows.append({'model': model, 'sample_token': token, 'gt_index': gi,
                         'type': kind, 'candidate_prediction_index': index,
                         'distance_m': d, 'reason': reason})
    return rows


def apply_oracle(preds, gt, fp_rows, fn_rows, kind):
    changed = clone_predictions(preds)
    deleted = defaultdict(set)
    changed_count = 0
    for row in fp_rows:
        if row['type'] != kind:
            continue
        token, index = row['sample_token'], row['prediction_index']
        if kind == 'Loc':
            gi = row['linked_gt_index']
            if gi is None:
                continue
            changed[token][index]['translation'] = list(changed[token][index]['translation'])
            changed[token][index]['translation'][:2] = gt[token][gi]['point']
        else:
            deleted[token].add(index)
        changed_count += 1
    for row in fn_rows:
        if row['type'] != kind or row['candidate_prediction_index'] is None:
            continue
        token, index = row['sample_token'], row['candidate_prediction_index']
        box = changed[token][index]
        box['detection_name'] = 'car'
        if kind == 'Both':
            box['translation'] = list(box['translation'])
            box['translation'][:2] = gt[token][row['gt_index']]['point']
        changed_count += 1
    if kind == 'Missed':
        # TIDE's Miss oracle reduces the GT-positive count; it does not append
        # low-ranked perfect detections (which may leave AP unchanged).
        for row in fn_rows:
            if row['type'] == 'Missed':
                changed_count += 1
    for token, indexes in deleted.items():
        changed[token] = [box for i, box in enumerate(changed[token])
                          if i not in indexes]
    return changed, changed_count


def ap_with_reduced_gt_count(preds, gt, n_removed):
    """TIDE-style Miss oracle: same ranked TP/FP decisions, smaller GT count."""
    remaining = sum(item['class'] == 'car' for entries in gt.values()
                    for item in entries) - n_removed
    if remaining <= 0:
        raise ValueError('Miss oracle removed all car GT')
    used = set()
    recall, precision = [], []
    tp = 0
    for rank, (_, token, _, box) in enumerate(score_order(preds, 'car'), 1):
        nearest = min(((distance(point(box), entry['point']), gi)
                       for gi, entry in enumerate(gt[token])
                       if entry['class'] == 'car' and (token, gi) not in used),
                      default=None)
        if nearest and nearest[0] < THRESHOLD:
            used.add((token, nearest[1]))
            tp += 1
        recall.append(tp/remaining)
        precision.append(tp/rank)
    values = []
    for grid in range(11, 101):
        target = grid/100
        index = bisect.bisect_right(recall, target)
        if index == 0:
            p = precision[0]
        elif index == len(recall):
            p = precision[-1] if target <= recall[-1] else 0.0
        else:
            left, right = recall[index-1], recall[index]
            weight = (target-left)/(right-left)
            p = precision[index-1] + weight*(precision[index]-precision[index-1])
        values.append(max(p-.1, 0))
    return sum(values)/90/.9


def audit_model(model, preds, gt, official, radius):
    baseline = class_ap(preds, gt, 'car')
    if abs(baseline - official) > 1e-4:
        raise ValueError('{} AP replay mismatch: {} vs {}'.format(model, baseline, official))
    matches = all_matches(preds, gt)
    car_matches = {(token, gi) for (token, _), (gi, _, _) in matches['car'].items()}
    fp_rows = []
    tp = 0
    total_gt = sum(item['class'] == 'car' for entries in gt.values() for item in entries)
    for rank, (score, token, index, box) in enumerate(score_order(preds, 'car'), 1):
        if (token, index) in matches['car']:
            tp += 1
            continue
        kind, linked, reason = classify_fp(token, index, box, gt,
                                          car_matches, matches, radius)
        fp_rows.append({'model': model, 'rank': rank, 'sample_token': token,
                        'prediction_index': index, 'score': score,
                        'recall_at_rank': tp/total_gt,
                        'before_25pct_recall': int(tp/total_gt < .25),
                        'type': kind,
                        'linked_gt_index': linked[1] if linked else None,
                        'reason': reason})
    # A single missed GT cannot be "repaired" by several localization FPs.
    loc_owner = {}
    for row in fp_rows:
        if row['type'] != 'Loc':
            continue
        key = (row['sample_token'], row['linked_gt_index'])
        if key in loc_owner:
            row['type'] = 'Ambiguous'
            row['reason'] = 'second_car_fp_for_same_missed_gt'
            row['linked_gt_index'] = None
        else:
            loc_owner[key] = row['prediction_index']
    fn_rows = assign_missing_cars(model, gt, preds, matches, car_matches,
                                  fp_rows, radius)
    oracle_results = {}
    for kind in TYPES:
        modified, n = apply_oracle(preds, gt, fp_rows, fn_rows, kind)
        value = (ap_with_reduced_gt_count(preds, gt, n) if kind == 'Missed'
                 else class_ap(modified, gt, 'car'))
        oracle_results[kind] = {'changed_items': n, 'ap': value,
                                'delta_ap': value-baseline}
    return fp_rows, fn_rows, {
        'official_car_ap_4m': official, 'replayed_car_ap_4m': baseline,
        'car_tp': len(car_matches), 'car_fp': len(fp_rows), 'car_fn': len(fn_rows),
        'fp_types_all': dict(Counter(row['type'] for row in fp_rows)),
        'fp_types_before_25pct': dict(Counter(row['type'] for row in fp_rows
                                               if row['before_25pct_recall'])),
        'fn_types': dict(Counter(row['type'] for row in fn_rows)),
        'fp_before_25pct': sum(row['before_25pct_recall'] for row in fp_rows),
        'isolated_oracles': oracle_results}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    return (ordered[n//2] if n % 2 else
            (ordered[n//2-1] + ordered[n//2])/2) if n else None


def attention_summary(input_dir, tables, output):
    source = input_dir / 'query_scorecard'
    query = read_csv(source / 'per_car_query.csv')
    paired = read_csv(source / 'paired_attention.csv')
    by_key = defaultdict(dict)
    for row in query:
        by_key[(row['sample_token'], row['gt_index'])][row['model']] = row
    scene = {row['token']: row['scene_token']
             for row in load_json(tables / 'sample.json')}
    records = []
    for row in paired:
        if int(row['layer']) != 6:
            continue
        key = (row['sample_token'], row['gt_index'])
        pair = by_key[key]
        if set(pair) != {'r1', 'r1f'}:
            continue
        a, b = pair['r1']['attention_auc_l6'], pair['r1f']['attention_auc_l6']
        if not a or not b:
            continue
        records.append({'sample_token': key[0], 'scene_token': scene[key[0]],
                        'gt_index': key[1], 'r1_auc': float(a),
                        'r1f_auc': float(b), 'delta_auc': float(b)-float(a),
                        'r1_gt_mass': float(row['r1_car_attention_mass']),
                        'r1f_gt_mass': float(row['r1f_car_attention_mass']),
                        'full_overlap': float(row['full_spatial_overlap']),
                        'joint_gt_mass': float(row['joint_car_attention_mass']),
                        'within_gt_overlap': (float(row['within_car_spatial_overlap'])
                           if row['within_car_spatial_overlap'] else None)})
    if len(records) != 3591:
        raise ValueError('Expected 3,591 paired final-layer GT cars, got {}'.format(len(records)))
    write_csv(output / 'paired_attention_l6.csv', records)
    groups = defaultdict(list)
    for row in records:
        groups[row['scene_token']].append(row['delta_auc'])
    rng = random.Random(17)
    keys = sorted(groups)
    bootstrap = []
    for _ in range(2000):
        sample = [rng.choice(keys) for _ in keys]
        values = [v for key in sample for v in groups[key]]
        bootstrap.append(sum(values)/len(values))
    bootstrap.sort()
    return {'n_gt_cars': len(records), 'n_scenes': len(groups),
            'layer': 6, 'cameras': 'pooled_all',
            'median_r1_auc': median([r['r1_auc'] for r in records]),
            'median_r1f_auc': median([r['r1f_auc'] for r in records]),
            'mean_paired_delta_auc_r1f_minus_r1': sum(r['delta_auc'] for r in records)/len(records),
            'median_paired_delta_auc_r1f_minus_r1': median([r['delta_auc'] for r in records]),
            'scene_bootstrap_mean_delta_95pct_ci': [bootstrap[49], bootstrap[1949]],
            'fraction_gt_r1f_auc_higher': sum(r['delta_auc'] > 0 for r in records)/len(records),
            'median_r1_gt_mass': median([r['r1_gt_mass'] for r in records]),
            'median_r1f_gt_mass': median([r['r1f_gt_mass'] for r in records]),
            'median_full_spatial_overlap': median([r['full_overlap'] for r in records]),
            'median_joint_gt_mass': median([r['joint_gt_mass'] for r in records]),
            'median_within_gt_overlap': median([r['within_gt_overlap'] for r in records
                                                 if r['within_gt_overlap'] is not None])}


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write: {}'.format(path))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_svg(summary, output):
    """Dependency-free, compact comparison plot (SVG)."""
    models = summary['models']
    colors = {'Cls':'#7551a8', 'Loc':'#df9141', 'Both':'#b85e79',
              'Dupe':'#3d80a2', 'Bkgd':'#718e57', 'Missed':'#9b6670',
              'Ambiguous':'#87919c'}
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="660" viewBox="0 0 1050 660">',
             '<rect width="1050" height="660" fill="#fff"/>',
             '<text x="45" y="52" font-family="Arial" font-size="27" font-weight="bold">R1-f frames: car AP@4m failure audit</text>',
             '<text x="45" y="80" font-family="Arial" font-size="15" fill="#526071">Six TIDE-inspired types; ambiguous cases retained separately</text>']
    for j, model in enumerate(('R1-f', 'R1')):
        rec = models[model]
        y = 135+j*225
        lines.append('<text x="45" y="{}" font-family="Arial" font-size="21" font-weight="bold">{}: AP {:.2f}% · early FP {:,}</text>'.format(
            y, model, rec['official_car_ap_4m']*100, rec['fp_before_25pct']))
        counts = rec['fp_types_before_25pct']
        total = max(sum(counts.values()), 1)
        x = 45
        for kind in TYPES[:-1]+('Ambiguous',):
            width = 930*counts.get(kind, 0)/total
            if width:
                lines.append('<rect x="{:.1f}" y="{}" width="{:.1f}" height="38" fill="{}"/>'.format(
                    x, y+18, width, colors[kind]))
                x += width
        legend = '  ·  '.join('{} {}'.format(kind, counts.get(kind, 0))
                             for kind in TYPES[:-1]+('Ambiguous',))
        lines.append('<text x="45" y="{}" font-family="Arial" font-size="14" fill="#334">{}</text>'.format(y+79, legend))
        gains = rec['isolated_oracles']
        oracle = '  ·  '.join('{} {:+.2f}pp'.format(kind, gains[kind]['delta_ap']*100)
                              for kind in TYPES)
        lines.append('<text x="45" y="{}" font-family="Arial" font-size="13" fill="#526071">Isolated AP oracle: {}</text>'.format(y+105, oracle))
        lines.append('<text x="45" y="{}" font-family="Arial" font-size="13" fill="#526071">GT-car FN types: {}</text>'.format(
            y+129, ' · '.join('{} {}'.format(k, rec['fn_types'].get(k, 0)) for k in TYPES+('Ambiguous',))))
    attention = summary['attention']
    lines.append('<text x="45" y="590" font-family="Arial" font-size="18" font-weight="bold">Last-layer GT-ray attention AUROC, all cameras</text>')
    lines.append('<text x="45" y="620" font-family="Arial" font-size="16">R1 median {:.3f} · R1-f median {:.3f} · paired mean difference {:+.3f} (95% scene CI {:.3f} to {:.3f})</text>'.format(
        attention['median_r1_auc'], attention['median_r1f_auc'],
        attention['mean_paired_delta_auc_r1f_minus_r1'],
        *attention['scene_bootstrap_mean_delta_95pct_ci']))
    lines.append('</svg>')
    (output / 'bounded_diagnosis.svg').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--table-dir', type=Path, default=TABLES)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--near-radius', type=float, default=8.0)
    args = parser.parse_args()
    if args.near_radius <= THRESHOLD:
        parser.error('--near-radius must exceed 4m')
    output = args.output_dir or args.input_dir / 'bounded_tide3d_attention'
    output.mkdir(parents=True, exist_ok=True)
    results = {model: predictions(args.input_dir / model / 'formatted/results_pccr.json')
               for model in ('R1-f', 'R1')}
    if set(results['R1-f']) != set(results['R1']):
        raise ValueError('Model sample tokens do not match')
    cars = car_gt_csv(args.input_dir / 'analysis/car_per_gt.csv')
    gt = eligible_gt(args.table_dir, cars, results['R1-f'])
    summary = {'metric': 'car AP@4m', 'near_radius_m': args.near_radius,
               'models': {}, 'caveats': [
                   'TIDE-inspired center-distance adaptation, not official TIDE IoU.',
                   'Proximity categories are output-level diagnostics, not proven network causes.',
                   'Bkgd means no evaluator-eligible GT within radius, not verified empty image.',
                   'Ambiguous predictions are not forced into the six types.',
                   'Oracle APs are independent post-hoc modifications; they do not add.']}
    for model in ('R1-f', 'R1'):
        metrics = load_json(args.input_dir / model / 'formatted/metrics_summary.json')
        official = metrics['label_aps']['car']['4.0']
        fp, fn, rec = audit_model(model, results[model], gt, official,
                                  args.near_radius)
        write_csv(output / '{}_car_fp.csv'.format(model.replace('-', '')),
                  fp)
        write_csv(output / '{}_car_fn.csv'.format(model.replace('-', '')),
                  fn)
        summary['models'][model] = rec
    summary['attention'] = attention_summary(args.input_dir, args.table_dir, output)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_svg(summary, output)
    from plot_bounded_r1f_attention import render as render_attention
    from plot_bounded_r1f_attention_mass import render as render_attention_mass
    render_attention(args.input_dir, output)
    render_attention_mass(args.input_dir, output)
    print('Saved {}'.format(output))


if __name__ == '__main__':
    main()
