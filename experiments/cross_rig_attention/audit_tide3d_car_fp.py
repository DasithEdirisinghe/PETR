#!/usr/bin/env python3
"""TIDE-inspired, evaluator-faithful car FP audit on R1-f frames.

The category names describe BEV-center proximity, not proven model intent.
Suppression oracles are diagnostic counterfactuals on saved final detections;
they never modify PETR inference or the official metrics files.
"""

import argparse
import bisect
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_val'
CATEGORIES = (
    'duplicate_or_competing_car',
    'near_other_gt_under_4m',
    'near_car_4_to_8m',
    'near_other_gt_4_to_8m',
    'near_car_8_to_20m',
    'near_other_gt_8_to_20m',
    'no_considered_gt_within_20m',
)
EXTRA_FIELDS = ('proximity_category', 'nearest_any_gt_distance_m',
                'nearest_any_gt_class', 'native_matched_car_within_8m')


def distance(value):
    return float(value) if value != '' else float('inf')


def category(row):
    if row['fp_reason'] == 'duplicate_or_competing_car':
        return CATEGORIES[0]
    car = distance(row['nearest_car_gt_distance_m'])
    other = distance(row['nearest_other_gt_distance_m'])
    if other < 4:
        return CATEGORIES[1]
    if car < 8 or other < 8:
        return CATEGORIES[2] if car <= other else CATEGORIES[3]
    if car < 20 or other < 20:
        return CATEGORIES[4] if car <= other else CATEGORIES[5]
    return CATEGORIES[6]


def ap_after_suppressing(rows, total_gt, suppress):
    """Same AP interpolation as official replay, with only selected FPs removed."""
    recalls = []
    precisions = []
    tp = 0
    for row in rows:
        if not row['is_tp'] and suppress(row):
            continue
        tp += row['is_tp']
        rank = len(recalls) + 1
        recalls.append(tp / total_gt)
        precisions.append(tp / rank)
    grid = []
    for i in range(11, 101):
        target = i / 100
        index = bisect.bisect_right(recalls, target)
        if index == 0:
            value = precisions[0]
        elif index == len(recalls):
            value = precisions[-1] if target <= recalls[-1] else 0.0
        elif recalls[index] == recalls[index - 1]:
            value = precisions[index]
        else:
            weight = (target - recalls[index - 1]) / (
                recalls[index] - recalls[index - 1])
            value = precisions[index - 1] + weight * (
                precisions[index] - precisions[index - 1])
        grid.append(value)
    return sum(max(value - .1, 0) for value in grid) / 90 / .9


def as_number(row):
    row['rank'] = int(row['rank'])
    row['is_tp'] = int(row['is_tp'])
    row['recall'] = float(row['recall'])
    row['precision'] = float(row['precision'])
    row['car_score'] = float(row['car_score'])
    return row


def read_ledger(path):
    rows = {'R1-f': [], 'R1': []}
    with path.open(newline='') as handle:
        for record in csv.DictReader(handle):
            row = as_number(record)
            rows[row['model']].append(row)
    return rows


def analyze(model, rows, total_gt, official_ap, output, native_matches):
    baseline = ap_after_suppressing(rows, total_gt, lambda _: False)
    if abs(baseline - official_ap) > 1e-4:
        raise ValueError('{} AP mismatch: replay={} official={}'.format(
            model, baseline, official_ap))
    quarter = next(row for row in rows if row['recall'] >= .25)
    quarter_rank = quarter['rank']
    early = [row for row in rows[:quarter_rank] if not row['is_tp']]
    fps = [row for row in rows if not row['is_tp']]
    for row in fps:
        row['proximity_category'] = category(row)
        car = distance(row['nearest_car_gt_distance_m'])
        other = distance(row['nearest_other_gt_distance_m'])
        row['nearest_any_gt_distance_m'] = min(car, other)
        row['nearest_any_gt_class'] = (
            'car' if car <= other else row['nearest_other_gt_class'])
        row['native_matched_car_within_8m'] = int(
            car < 8 and (row['sample_token'], row['nearest_car_gt_index'])
            in native_matches)

    fields = list(rows[0]) + list(EXTRA_FIELDS)
    with (output / '{}_early_fp_25pct.csv'.format(model.replace('-', ''))).open(
            'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(early)

    categories = {}
    for name in CATEGORIES:
        early_n = sum(row['proximity_category'] == name for row in early)
        all_n = sum(row['proximity_category'] == name for row in fps)
        early_fixed = ap_after_suppressing(
            rows, total_gt,
            lambda row, target=name: row['rank'] <= quarter_rank and
            row.get('proximity_category') == target)
        all_fixed = ap_after_suppressing(
            rows, total_gt,
            lambda row, target=name: row.get('proximity_category') == target)
        categories[name] = {
            'early_fp_count': early_n,
            'all_fp_count': all_n,
            'native_matched_car_within_8m_early_count': sum(
                row['proximity_category'] == name and
                row['native_matched_car_within_8m'] for row in early),
            'early_suppression_delta_ap': early_fixed - baseline,
            'all_suppression_delta_ap': all_fixed - baseline,
        }
    if sum(value['early_fp_count'] for value in categories.values()) != len(early):
        raise AssertionError('FP categories do not partition the early FPs')
    all_fp_suppressed_ap = ap_after_suppressing(
        rows, total_gt, lambda row: not row['is_tp'])
    return {
        'official_car_ap_4m': official_ap,
        'replayed_car_ap_4m': baseline,
        'gt_cars': total_gt,
        'rank_at_25pct_recall': quarter_rank,
        'tp_at_25pct_recall': int(quarter['cumulative_tp']),
        'fp_at_25pct_recall': len(early),
        'precision_at_25pct_recall': float(quarter['precision']),
        'score_at_25pct_recall': float(quarter['car_score']),
        'categories': categories,
        'all_early_fp_suppression_delta_ap': ap_after_suppressing(
            rows, total_gt,
            lambda row: row['rank'] <= quarter_rank and not row['is_tp']) - baseline,
        'all_fp_suppressed_ap': all_fp_suppressed_ap,
        'all_fp_suppression_delta_ap': all_fp_suppressed_ap - baseline,
    }


def save_svg(path, summary):
    width = 1280
    labels = {
        'duplicate_or_competing_car': 'Duplicate / competing car',
        'near_other_gt_under_4m': 'Near other-class GT, under 4 m',
        'near_car_4_to_8m': 'Nearest GT car, 4–8 m',
        'near_other_gt_4_to_8m': 'Nearest other GT, 4–8 m',
        'near_car_8_to_20m': 'Nearest GT car, 8–20 m',
        'near_other_gt_8_to_20m': 'Nearest other GT, 8–20 m',
        'no_considered_gt_within_20m': 'No considered GT within 20 m',
    }
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="620" '
             'viewBox="0 0 {} 620">'.format(width, width),
             '<rect width="1280" height="620" fill="#f5f8fb"/>',
             '<text x="35" y="49" font-family="Arial" font-size="28" '
             'font-weight="bold" fill="#173047">High-ranked car false positives on R1-f frames</text>',
             '<text x="35" y="78" font-family="Arial" font-size="17" fill="#53687b">'
             'Before each model reaches 25% recall · PETR car AP at 4 m · proximity categories are diagnostic</text>']
    scale = 600 / max(1, max(
        summary['models']['R1']['categories'][name]['early_fp_count']
        for name in CATEGORIES))
    for index, name in enumerate(CATEGORIES):
        y = 125 + index * 65
        parts.append('<text x="35" y="{}" font-family="Arial" font-size="18" '
                     'fill="#173047">{}</text>'.format(y + 13, labels[name]))
        for model, offset, color in (('R1-f', 0, '#168176'), ('R1', 23, '#c9574c')):
            count = summary['models'][model]['categories'][name]['early_fp_count']
            bar = count * scale
            parts.append('<rect x="350" y="{}" width="{:.1f}" height="18" '
                         'fill="{}"/>'.format(y + offset - 9, bar, color))
            parts.append('<text x="{:.1f}" y="{}" font-family="Arial" '
                         'font-size="16" fill="#173047">{}</text>'.format(
                             360 + bar, y + offset + 5, count))
    parts.append('<rect x="36" y="584" width="17" height="17" fill="#168176"/>'
                 '<text x="61" y="599" font-family="Arial" font-size="16">R1-f native</text>'
                 '<rect x="250" y="584" width="17" height="17" fill="#c9574c"/>'
                 '<text x="275" y="599" font-family="Arial" font-size="16">R1 cross-rig</text>'
                 '</svg>')
    path.write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    audit = args.input_dir / 'car_ap_rank_audit'
    with (audit / 'summary.json').open() as handle:
        rank_summary = json.load(handle)
    if rank_summary['rig'] != 'R1-f' or rank_summary['threshold_m'] != 4.0:
        raise ValueError('Requires the R1-f, 4 m rank audit')
    rows = read_ledger(audit / 'ranked_car_predictions.csv')
    native_matches = {(row['sample_token'], row['matched_gt_index'])
                      for row in rows['R1-f'] if row['is_tp']}
    output = args.output_dir or audit / 'tide3d_car_fp'
    output.mkdir(parents=True, exist_ok=True)
    summary = {'rig': 'R1-f', 'threshold_m': 4.0, 'recall_checkpoint': .25,
               'method': 'TIDE-inspired FP proximity categories and independent '
                         'suppression oracles; not the original 2D TIDE toolbox',
               'oracle_note': 'Each delta AP is computed independently from the '
                              'unchanged baseline. Suppression is post-hoc, not '
                              'a model result; gains are not additive.',
               'models': {}}
    for model in ('R1-f', 'R1'):
        rank_record = rank_summary['models'][model]
        summary['models'][model] = analyze(
            model, rows[model], rank_record['gt_cars'],
            rank_record['official_car_ap'], output, native_matches)
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    save_svg(output / 'early_fp_categories.svg', summary)
    print('Saved', output / 'summary.json')
    print('Saved', output / 'early_fp_categories.svg')


if __name__ == '__main__':
    main()
