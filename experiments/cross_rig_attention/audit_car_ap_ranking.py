#!/usr/bin/env python3
"""Replay PCCR car AP and explain the confidence-ranked false positives."""

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_val'
FIELDS = ('model', 'rank', 'sample_token', 'prediction_index', 'car_score',
          'pred_x', 'pred_y', 'is_tp', 'fp_reason', 'matched_gt_index',
          'matched_distance_m', 'nearest_car_gt_distance_m',
          'nearest_car_gt_index',
          'nearest_other_gt_distance_m', 'nearest_other_gt_class',
          'cumulative_tp', 'cumulative_fp', 'precision', 'recall')
CHECKPOINT_RECALL = (0.10, 0.25, 0.50, 0.70)
SCORE_CUTOFFS = (0.50, 0.35, 0.10, 0.05, 0.01, 0.0)


def read_car_gt(path):
    cars = defaultdict(list)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            cars[row['sample_token']].append((int(row['gt_index']),
                    (float(row['gt_x']), float(row['gt_y']))))
    return cars


def read_other_gt(root, tokens):
    """Other-class GT is a diagnostic proximity check, not AP matching."""
    with (root / 'category.json').open() as handle:
        category = {row['token']: row['name'] for row in json.load(handle)}
    with (root / 'instance.json').open() as handle:
        instances = {row['token']: category[row['category_token']]
                     for row in json.load(handle)}
    others = defaultdict(list)
    with (root / 'sample_annotation.json').open() as handle:
        annotations = json.load(handle)
    for row in annotations:
        if row['sample_token'] not in tokens or row['num_lidar_pts'] + row['num_radar_pts'] == 0:
            continue
        name = instances[row['instance_token']]
        if name != 'car':
            others[row['sample_token']].append(
                ((float(row['translation'][0]), float(row['translation'][1])), name))
    return others


def read_car_predictions(path, model):
    with path.open() as handle:
        results = json.load(handle)['results']
    predictions = []
    for token, boxes in results.items():
        for index, box in enumerate(boxes):
            if box['detection_name'] == 'car':
                predictions.append((float(box['detection_score']), token, index,
                                    (float(box['translation'][0]),
                                     float(box['translation'][1]))))
    # Stable sort reproduces the evaluator's descending-score order.
    predictions.sort(key=lambda item: item[0], reverse=True)
    return predictions


def nearest(point, entries, excluded=None):
    best = None
    for index, (gt_index, center) in enumerate(entries):
        if excluded is not None and gt_index in excluded:
            continue
        distance = math.dist(point, center)
        if best is None or distance < best[0]:
            best = (distance, gt_index)
    return best


def nearest_other(point, entries):
    if not entries:
        return None
    distance, name = min(((math.dist(point, center), name)
                          for center, name in entries), key=lambda item: item[0])
    return distance, name


def replay(model, predictions, cars, others, threshold):
    total_gt = sum(len(entries) for entries in cars.values())
    used = defaultdict(set)
    rows = []
    tp_scores = []
    for rank, (score, token, pred_index, point) in enumerate(predictions, 1):
        all_nearest = nearest(point, cars.get(token, ()))
        available = nearest(point, cars.get(token, ()), used[token])
        matched = available is not None and available[0] < threshold
        other = nearest_other(point, others.get(token, ()))
        if matched:
            used[token].add(available[1])
            tp_scores.append(score)
            reason = ''
        elif all_nearest is not None and all_nearest[0] < threshold:
            reason = 'duplicate_or_competing_car'
        elif other is not None and other[0] < threshold:
            reason = 'near_other_class_gt'
        else:
            reason = 'no_gt_within_threshold'
        tp = len(tp_scores)
        fp = rank - tp
        rows.append({
            'model': model, 'rank': rank, 'sample_token': token,
            'prediction_index': pred_index, 'car_score': score,
            'pred_x': point[0], 'pred_y': point[1],
            'is_tp': int(matched), 'fp_reason': reason,
            'matched_gt_index': available[1] if matched else '',
            'matched_distance_m': available[0] if matched else '',
            'nearest_car_gt_distance_m': all_nearest[0] if all_nearest else '',
            'nearest_car_gt_index': all_nearest[1] if all_nearest else '',
            'nearest_other_gt_distance_m': other[0] if other else '',
            'nearest_other_gt_class': other[1] if other else '',
            'cumulative_tp': tp, 'cumulative_fp': fp,
            'precision': tp / rank, 'recall': tp / total_gt,
        })
    return rows, total_gt, tp_scores


def interpolated_precision(rows, recall):
    """Equivalent to np.interp(recall, cumulative_recall, precision, right=0)."""
    if not rows:
        return 0.0
    recalls = [row['recall'] for row in rows]
    index = bisect.bisect_right(recalls, recall)
    if index == 0:
        return rows[0]['precision']
    if index == len(rows):
        return rows[-1]['precision'] if recall <= recalls[-1] else 0.0
    left, right = rows[index-1], rows[index]
    if right['recall'] == left['recall']:
        return right['precision']
    weight = (recall-left['recall'])/(right['recall']-left['recall'])
    return left['precision'] + weight*(right['precision']-left['precision'])


def summarize(rows, total_gt, tp_scores, official_ap):
    precision_grid = [interpolated_precision(rows, i/100) for i in range(101)]
    replay_ap = sum(max(value-.1, 0) for value in precision_grid[11:])/90/.9
    if abs(replay_ap-official_ap) > 1e-4:
        raise ValueError('AP replay differs from official: {} vs {}'.format(
            replay_ap, official_ap))
    checkpoints = {}
    for recall in CHECKPOINT_RECALL:
        entry = next((row for row in rows if row['recall'] >= recall), None)
        if entry is None:
            checkpoints[str(recall)] = None
            continue
        prefix = rows[:entry['rank']]
        far_car_bins = Counter()
        for row in prefix:
            if row['fp_reason'] != 'no_gt_within_threshold':
                continue
            distance = row['nearest_car_gt_distance_m']
            if distance == '' or distance >= 20:
                far_car_bins['at_least_20m_or_no_car_gt'] += 1
            elif distance >= 8:
                far_car_bins['8_to_20m'] += 1
            else:
                far_car_bins['threshold_to_8m'] += 1
        checkpoints[str(recall)] = {
            'rank': entry['rank'], 'tp': entry['cumulative_tp'],
            'fp': entry['cumulative_fp'], 'precision': entry['precision'],
            'score': entry['car_score'],
            'fp_reasons': dict(Counter(row['fp_reason'] for row in prefix
                                       if not row['is_tp'])),
            'no_near_gt_nearest_car_distance_bins': dict(far_car_bins),
        }
    cutoffs = {}
    for cutoff in SCORE_CUTOFFS:
        prefix = [row for row in rows if row['car_score'] >= cutoff]
        tp = sum(row['is_tp'] for row in prefix)
        cutoffs[str(cutoff)] = {'predictions': len(prefix), 'tp': tp,
                               'fp': len(prefix)-tp,
                               'precision': tp/len(prefix) if prefix else 0,
                               'recall': tp/total_gt}
    return {'gt_cars': total_gt, 'car_predictions': len(rows),
            'true_positives': len(tp_scores),
            'false_positives': len(rows)-len(tp_scores),
            'maximum_recall': len(tp_scores)/total_gt,
            'median_true_positive_score': statistics.median(tp_scores),
            'official_car_ap': official_ap, 'replayed_car_ap': replay_ap,
            'precision_at_recall_grid': precision_grid,
            'fp_reasons_all': dict(Counter(row['fp_reason'] for row in rows
                                           if not row['is_tp'])),
            'at_recall': checkpoints, 'at_score': cutoffs}


def save_pr_svg(path, summary):
    x0, x1, y0, y1 = 95, 765, 530, 90
    rig = summary['rig']
    native = rig
    cross = 'R1-f' if rig == 'R1' else 'R1'
    models = ((native, '#168176'), (cross, '#c9574c'))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" '
             'viewBox="0 0 1200 620">',
             '<rect width="1200" height="620" fill="#f4f7fa"/>',
             '<text x="55" y="46" font-family="Arial" font-size="27" '
             'font-weight="bold" fill="#183047">Car precision–recall on {} frames · {} m</text>'.format(
                 rig, summary['threshold_m']),
             '<rect x="55" y="65" width="1100" height="510" rx="14" fill="white"/>']
    for step in range(6):
        value = step/5
        x = x0 + (x1-x0)*value
        y = y0 - (y0-y1)*value
        parts.append('<path d="M{:.1f} {}V{} M{} {:.1f}H{}" stroke="#e2e8ee"/>'.format(
            x, y0, y1, x0, y, x1))
        parts.append('<text x="{:.1f}" y="555" text-anchor="middle" '
                     'font-family="Arial" font-size="14" fill="#53687b">{:.0f}%</text>'.format(x, value*100))
        parts.append('<text x="78" y="{:.1f}" text-anchor="end" '
                     'font-family="Arial" font-size="14" fill="#53687b">{:.0f}%</text>'.format(y+5, value*100))
    parts.append('<text x="430" y="590" text-anchor="middle" font-family="Arial" '
                 'font-size="17" fill="#183047">Recall</text>')
    parts.append('<text x="25" y="320" transform="rotate(-90 25 320)" '
                 'text-anchor="middle" font-family="Arial" font-size="17" '
                 'fill="#183047">Precision</text>')
    for recall in (0.25, 0.50):
        x = x0 + (x1-x0)*recall
        parts.append('<path d="M{:.1f} {}V{}" stroke="#94a6b6" '
                     'stroke-dasharray="5 5"/>'.format(x, y0, y1))
    for model, color in models:
        record = summary['models'][model]
        points = ' '.join('{:.1f},{:.1f}'.format(
            x0+(x1-x0)*i/100, y0-(y0-y1)*value)
            for i, value in enumerate(record['precision_at_recall_grid']))
        parts.append('<polyline points="{}" fill="none" stroke="{}" '
                     'stroke-width="3"/>'.format(points, color))
        for recall in (0.25, 0.50):
            checkpoint = record['at_recall'][str(recall)]
            if checkpoint is None:
                continue
            x = x0 + (x1-x0)*checkpoint['tp']/record['gt_cars']
            y = y0 - (y0-y1)*checkpoint['precision']
            parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="6" '
                         'fill="{}" stroke="white" stroke-width="2"/>'.format(
                             x, y, color))
    for index, (model, color) in enumerate(models):
        record = summary['models'][model]
        y = 135+index*205
        parts.append('<rect x="815" y="{}" width="18" height="18" fill="{}"/>'.format(y-15, color))
        parts.append('<text x="845" y="{}" font-family="Arial" font-size="21" '
                     'font-weight="bold" fill="#183047">{}</text>'.format(y, model))
        parts.append('<text x="815" y="{}" font-family="Arial" font-size="18" '
                     'fill="#53687b">AP: {:.2f}%</text>'.format(y+37, 100*record['official_car_ap']))
        for offset, recall in ((75, 0.25), (108, 0.50)):
            checkpoint = record['at_recall'][str(recall)]
            label = ('{}%: {:,} TP · {:,} FP · {:.1f}% precision'.format(
                int(recall*100), checkpoint['tp'], checkpoint['fp'],
                100*checkpoint['precision']) if checkpoint is not None else
                '{}%: not reached'.format(int(recall*100)))
            parts.append('<text x="815" y="{}" font-family="Arial" font-size="16" '
                         'fill="#53687b">{}</text>'.format(y+offset, label))
        parts.append('<text x="815" y="{}" font-family="Arial" font-size="15" '
                     'fill="#53687b">Max recall: {:.1f}%</text>'.format(
                         y+141, 100*record['maximum_recall']))
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def save_25_inspection(output, rows, cars, native_matches, summary):
    cutoff_rank = summary['at_recall']['0.25']['rank']
    selected = [row for row in rows if row['rank'] <= cutoff_rank and
                row['fp_reason'] == 'no_gt_within_threshold']
    fieldnames = list(FIELDS) + ['nearest_car_native_tp', 'gt_cars_in_frame']
    with (output / 'r1_unassociated_before_25pct_recall.csv').open(
            'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            record = dict(row)
            record['nearest_car_native_tp'] = int(
                (row['sample_token'], row['nearest_car_gt_index']) in native_matches)
            record['gt_cars_in_frame'] = len(cars.get(row['sample_token'], ()))
            writer.writerow(record)
    scores = [row['car_score'] for row in selected]
    summary['unassociated_before_25pct_recall'] = {
        'n': len(selected),
        'score_median': statistics.median(scores) if scores else None,
        'score_min': min(scores) if scores else None,
        'score_max': max(scores) if scores else None,
        'in_frames_without_gt_cars': sum(
            not cars.get(row['sample_token']) for row in selected),
        'nearest_car_native_true_positive': sum(
            (row['sample_token'], row['nearest_car_gt_index']) in native_matches
            for row in selected),
        'nearest_car_distance_bins': summary['at_recall']['0.25'][
            'no_near_gt_nearest_car_distance_bins'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--rig', choices=('R1', 'R1-f'), default='R1-f')
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.threshold not in (0.5, 1.0, 2.0, 4.0):
        raise ValueError('Use an official AP center-distance threshold')
    output = args.output_dir or args.input_dir / 'car_ap_rank_audit'
    cars = read_car_gt(args.input_dir / 'analysis/car_per_gt.csv')
    others = read_other_gt(ROOT / 'data/pccr' / args.rig / 'v1.0-trainval',
                           set(cars))
    summary = {'rig': args.rig, 'threshold_m': args.threshold,
               'fp_reason_note': 'Other-class GT proximity is diagnostic; '
                                 'car AP matching is replayed exactly.',
               'models': {}}
    output.mkdir(parents=True, exist_ok=True)
    native_matches = set()
    with (output / 'ranked_car_predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for model in ('R1-f', 'R1'):
            predictions = read_car_predictions(
                args.input_dir / model / 'formatted/results_pccr.json', model)
            rows, total_gt, tp_scores = replay(
                model, predictions, cars, others, args.threshold)
            with (args.input_dir / model / 'formatted/metrics_summary.json').open() as metric_file:
                official = json.load(metric_file)['label_aps']['car'][str(args.threshold)]
            summary['models'][model] = summarize(rows, total_gt, tp_scores, official)
            if args.rig == 'R1-f' and args.threshold == 4.0:
                if model == 'R1-f':
                    native_matches = {(row['sample_token'], row['matched_gt_index'])
                                      for row in rows if row['is_tp']}
                elif model == 'R1':
                    save_25_inspection(output, rows, cars, native_matches,
                                       summary['models'][model])
            writer.writerows(rows)
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    if args.threshold == 4.0:
        save_pr_svg(output / 'precision_recall.svg', summary)
    print('Saved', output / 'ranked_car_predictions.csv')
    print('Saved', output / 'summary.json')


if __name__ == '__main__':
    main()
