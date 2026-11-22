#!/usr/bin/env python3
"""Audit PETR's 4 m to 0.5 m AP gap using saved, evaluator-filtered data.

No detector inference or nuScenes Python package is required. Official AP is
read from the saved nuScenes evaluator output; matching is replayed and checked
against the existing GT-level evaluator replay before diagnostics are written.
"""

import argparse
import csv
import gc
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
VISIBILITY = {'1': '0-40%', '2': '40-60%', '3': '60-80%', '4': '80-100%'}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ap(metrics, name, threshold):
    return 100.0 * metrics['label_aps'][name][str(float(threshold))]


def load_gt(records, annotations_path):
    """Keep only evaluator-eligible annotations, then release the large table."""
    needed = {row['annotation_token'] for row in records}
    print('Loading annotation centers...', flush=True)
    with annotations_path.open() as handle:
        annotations = json.load(handle)
    centers = {ann['token']: ann['translation'][:2]
               for ann in annotations if ann['token'] in needed}
    del annotations
    gc.collect()
    if len(centers) != len(needed):
        raise ValueError('Missing {} eligible annotation centers'.format(
            len(needed) - len(centers)))
    by_key, by_sample_class = {}, defaultdict(list)
    for row in records:
        row['xy'] = centers[row['annotation_token']]
        key = row['annotation_token']
        by_key[key] = row
        by_sample_class[row['sample_token'], row['class']].append(row)
    return by_key, by_sample_class


def load_predictions(path, classes):
    print('Loading evaluator-filtered predictions...', flush=True)
    with path.open() as handle:
        data = json.load(handle)
    by_class, by_sample_class = defaultdict(list), defaultdict(list)
    for sample, boxes in data['results'].items():
        for index, box in enumerate(boxes):
            name = box['detection_name']
            if name not in classes:
                continue
            pred = {'id': '{}:{}'.format(sample, index), 'sample': sample,
                    'class': name, 'xy': box['translation'][:2],
                    'score': float(box['detection_score'])}
            by_class[name].append(pred)
            by_sample_class[sample, name].append(pred)
    del data
    gc.collect()
    for name in classes:
        # Stable tie handling; the supplied scores are normally distinct.
        by_class[name].sort(key=lambda p: p['score'], reverse=True)
        for rank, pred in enumerate(by_class[name], 1):
            pred['rank'] = rank
            pred['rank_pct'] = 100.0 * rank / len(by_class[name])
    return by_class, by_sample_class


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match(by_class, by_sample_class, threshold):
    """Confidence order; nearest currently unmatched same-class GT in sample."""
    pred_to_gt, gt_to_pred, fp_reason = {}, {}, {}
    for name, predictions in by_class.items():
        taken = set()
        for pred in predictions:
            gts = by_sample_class.get((pred['sample'], name), ())
            available = (gt for gt in gts if gt['annotation_token'] not in taken)
            nearest = min(available, key=lambda gt: distance(pred['xy'], gt['xy']),
                          default=None)
            if nearest is not None and distance(pred['xy'], nearest['xy']) < threshold:
                key = nearest['annotation_token']
                taken.add(key)
                pred_to_gt[pred['id']] = key
                gt_to_pred[key] = pred
            else:
                nearby = [gt for gt in gts if distance(pred['xy'], gt['xy']) < threshold]
                fp_reason[pred['id']] = ('duplicate_or_competition' if nearby
                                          else 'no_gt_within_threshold')
    return pred_to_gt, gt_to_pred, fp_reason


def validate_match(gt_records, matches):
    for threshold, (_, gt_to_pred, _) in matches.items():
        field = 'status_at_{}m'.format(threshold)
        expected = {row['annotation_token'] for row in gt_records
                    if row[field] in ('both', 'petr_only')}
        actual = set(gt_to_pred)
        if actual != expected:
            raise ValueError('Match replay differs from prior evaluator replay '
                             'at {} m: {} missing, {} extra GT. Stop before '
                             'interpreting outputs.'.format(threshold,
                                 len(expected - actual), len(actual - expected)))
        print('Validated <{} m: {:,} GT matches'.format(threshold, len(actual)),
              flush=True)


def nearby_context(gt, pred, by_sample_class):
    others = [other for other in by_sample_class[gt['sample_token'], gt['class']]
              if other['annotation_token'] != gt['annotation_token']]
    other_dist = min((distance(pred['xy'], other['xy']) for other in others),
                     default=None)
    target_dist = distance(pred['xy'], gt['xy'])
    return other_dist, int(other_dist is not None and other_dist < target_dist)


def radial_context(gt, pred, ego_xy):
    gx, gy = gt['xy']
    ex, ey = ego_xy
    radius = math.hypot(gx - ex, gy - ey)
    if radius == 0:
        return None, None
    dx, dy = pred['xy'][0] - gx, pred['xy'][1] - gy
    radial = (dx * (gx - ex) + dy * (gy - ey)) / radius
    tangent = (-dx * (gy - ey) + dy * (gx - ex)) / radius
    return radial, tangent


def read_ego(root, needed_samples):
    """Use LIDAR_TOP pose, as in the existing oracle, for radial error."""
    with (root / 'sample_data.json').open() as handle:
        sample_data = json.load(handle)
    poses = {s['sample_token']: s['ego_pose_token'] for s in sample_data
             if s['sample_token'] in needed_samples and s['is_key_frame']
             and '/LIDAR_TOP/' in s['filename']}
    del sample_data
    with (root / 'ego_pose.json').open() as handle:
        ego_poses = json.load(handle)
    needed_poses = set(poses.values())
    xy = {p['token']: p['translation'][:2] for p in ego_poses
          if p['token'] in needed_poses}
    del ego_poses
    result = {sample: xy[token] for sample, token in poses.items() if token in xy}
    if set(result) != needed_samples:
        raise ValueError('Missing LIDAR_TOP ego pose for {} samples'.format(
            len(needed_samples - set(result))))
    return result


def summarize(records, matches, by_sample_class, ego, metrics, oracle):
    pred4, gt4, fp4 = matches[4.0]
    pred05, gt05, fp05 = matches[0.5]
    details, groups = [], defaultdict(list)
    all_scopes = lambda row: ((row['class'], 'all'),
                              (row['class'], row['distance_bin']))
    for gt in records:
        p4 = gt4.get(gt['annotation_token'])
        p05 = gt05.get(gt['annotation_token'])
        if p4 is None:
            transition = 'unmatched_at_4m'
        elif p05 is not None:
            transition = ('same_prediction_at_0.5m' if p05['id'] == p4['id']
                          else 'different_prediction_at_0.5m')
        else:
            transition = 'lost_gt_match_at_0.5m'
        other_dist, other_closer = (nearby_context(gt, p4, by_sample_class)
                                    if p4 is not None else (None, None))
        radial, tangent = (radial_context(gt, p4, ego[gt['sample_token']])
                           if p4 is not None and gt['sample_token'] in ego
                           else (None, None))
        if p4 is None:
            pred4_status05 = ''
        elif p4['id'] in pred05:
            pred4_status05 = ('matched_same_gt' if p05 is not None and
                              p05['id'] == p4['id'] else 'matched_other_gt')
        else:
            pred4_status05 = fp05[p4['id']]
        row = {'class': gt['class'], 'distance_bin': gt['distance_bin'],
               'sample_token': gt['sample_token'],
               'annotation_token': gt['annotation_token'],
               'visibility_token': gt['visibility_token'],
               'visibility_band': gt['visibility_band'],
               'gt_range_m': gt['distance_m'], 'transition': transition,
               'pred4_id': p4['id'] if p4 else '',
               'pred4_score': p4['score'] if p4 else '',
               'pred4_rank_percent': p4['rank_pct'] if p4 else '',
               'pred4_error_m': distance(p4['xy'], gt['xy']) if p4 else '',
               'pred4_radial_error_m': radial if radial is not None else '',
               'pred4_tangential_error_m': tangent if tangent is not None else '',
               'nearest_other_gt_to_pred_m': other_dist if other_dist is not None else '',
               'other_gt_closer': other_closer if other_closer is not None else '',
               'pred4_status_at_0.5m': pred4_status05,
               'gt05_pred_id': p05['id'] if p05 else ''}
        details.append(row)
        for scope in all_scopes(row):
            groups[scope].append(row)
    summary = []
    for (name, bin_name), rows in sorted(groups.items()):
        n4 = [r for r in rows if r['pred4_id']]
        lost = [r for r in rows if r['transition'] == 'lost_gt_match_at_0.5m']
        imprecise = [r for r in n4 if r['pred4_error_m'] >= 0.5]
        def count(rows, predicate):
            return sum(bool(predicate(r)) for r in rows)
        def mean(rows, field):
            values = [abs(float(r[field])) for r in rows if r[field] != '']
            return sum(values) / len(values) if values else ''
        summary.append({
            'class': name, 'distance_bin': bin_name, 'eligible_gt': len(rows),
            'matched_gt_4m': len(n4),
            'matched_gt_0.5m': count(rows, lambda r: r['gt05_pred_id']),
            'lost_gt_match_4_to_0.5': len(lost),
            'imprecise_original_4m_pairs': len(imprecise),
            'lost_gt_80_100_visible': count(lost, lambda r: r['visibility_token'] == '4'),
            'lost_gt_0_40_visible': count(lost, lambda r: r['visibility_token'] == '1'),
            'lost_gt_other_closer': count(lost, lambda r: r['other_gt_closer'] == 1),
            'lost_gt_pred4_duplicate_at_0.5': count(lost, lambda r:
                r['pred4_status_at_0.5m'] == 'duplicate_or_competition'),
            'lost_gt_pred4_matched_other_at_0.5': count(lost, lambda r:
                r['pred4_status_at_0.5m'] == 'matched_other_gt'),
            'lost_mean_abs_radial_m': mean(lost, 'pred4_radial_error_m'),
            'lost_mean_abs_tangential_m': mean(lost, 'pred4_tangential_error_m'),
            'lost_mean_score': (sum(float(r['pred4_score']) for r in lost) / len(lost)
                                if lost else ''),
            'lost_mean_rank_percent': (sum(float(r['pred4_rank_percent']) for r in lost) /
                                       len(lost) if lost else '')})
    for row in summary:
        if row['distance_bin'] != 'all':
            continue
        name = row['class']
        row['official_ap_0.5_percent'] = ap(metrics, name, 0.5)
        row['official_ap_4_percent'] = ap(metrics, name, 4.0)
        row['official_ap_gap_pp'] = row['official_ap_4_percent'] - row['official_ap_0.5_percent']
        for mode in ('radial', 'full_center'):
            row[mode + '_oracle_ap_0.5_percent'] = ap(oracle[mode], name, 0.5)
            row[mode + '_oracle_gain_pp'] = (row[mode + '_oracle_ap_0.5_percent'] -
                                             row['official_ap_0.5_percent'])
    return details, summary


def prediction_summary(by_class, matches):
    p4, _, fp4 = matches[4.0]
    p05, _, fp05 = matches[0.5]
    out = []
    for name, preds in sorted(by_class.items()):
        labels = Counter()
        scores = defaultdict(list)
        for p in preds:
            pid = p['id']
            if pid in p4:
                status = ('tp_both_thresholds' if pid in p05 else
                          'tp4_fp05_' + fp05[pid])
            elif pid in p05:
                status = 'fp4_tp05_after_rematching'
            else:
                status = 'fp_both_' + fp4[pid]
            labels[status] += 1
            scores[status].append(p['score'])
        for status, n in sorted(labels.items()):
            out.append({'class': name, 'prediction_transition': status,
                        'count': n, 'mean_score': sum(scores[status]) / n,
                        'fraction_of_class_predictions_percent': 100 * n / len(preds)})
    return out


def transition_factors(details, prediction_rows):
    """Describe the two mutually exclusive outcomes of original 4 m TPs."""
    counts = defaultdict(Counter)
    totals = Counter()
    for row in details:
        if not row['pred4_id']:
            continue
        name = row['class']
        outcome = ('tp_at_0.5m' if row['pred4_status_at_0.5m'].startswith('matched_')
                   else 'fp_at_0.5m')
        totals[name, outcome] += 1
        radial = float(row['pred4_radial_error_m'])
        tangent = float(row['pred4_tangential_error_m'])
        other = row['nearest_other_gt_to_pred_m']
        rank = float(row['pred4_rank_percent'])
        error = float(row['pred4_error_m'])
        factors = {
            'visibility': row['visibility_band'],
            'gt_range': row['distance_bin'],
            'gt_range_30m_plus': 'yes' if float(row['gt_range_m']) >= 30 else 'no',
            'other_same_class_gt_within_4m_of_prediction':
                'yes' if other != '' and float(other) < 4 else 'no',
            'different_gt_closer_to_prediction':
                'yes' if row['other_gt_closer'] in (1, '1') else 'no',
            'abs_radial_exceeds_abs_tangential':
                'yes' if abs(radial) > abs(tangent) else 'no',
            'signed_radial_error': 'farther_than_gt' if radial > 0 else
                                   'nearer_than_gt' if radial < 0 else 'zero',
            'confidence_rank_within_class':
                'top_10_percent' if rank <= 10 else
                '10_to_50_percent' if rank <= 50 else 'bottom_50_percent',
            'original_center_error_band':
                '0-0.5m' if error < 0.5 else '0.5-1m' if error < 1 else
                '1-2m' if error < 2 else '2-4m',
        }
        for factor, category in factors.items():
            counts[name, outcome, factor][category] += 1
    prediction_counts = defaultdict(Counter)
    for row in prediction_rows:
        prediction_counts[row['class']][row['prediction_transition']] = int(row['count'])
    for name in {key[0] for key in totals}:
        c = prediction_counts[name]
        if totals[name, 'tp_at_0.5m'] != c['tp_both_thresholds'] or \
                totals[name, 'fp_at_0.5m'] != sum(
                    value for key, value in c.items() if key.startswith('tp4_fp05_')):
            raise ValueError('4 m TP factor rows differ from prediction transitions '
                             'for {}'.format(name))
    rows = []
    for (name, outcome, factor), categories in sorted(counts.items()):
        total = totals[name, outcome]
        if sum(categories.values()) != total:
            raise ValueError('Factor categories do not partition {}'.format(
                (name, outcome, factor)))
        for category, n in sorted(categories.items()):
            rows.append({'class': name, 'outcome': outcome, 'factor': factor,
                         'category': category, 'count': n,
                         'outcome_total': total,
                         'percent_of_outcome': 100.0 * n / total})
    return rows


def factor_outcome_rates(factor_rows):
    grouped = defaultdict(Counter)
    for row in factor_rows:
        grouped[row['class'], row['factor'], row['category']][row['outcome']] = int(
            row['count'])
    output = []
    for (name, factor, category), counts in sorted(grouped.items()):
        retained = counts['tp_at_0.5m']
        lost = counts['fp_at_0.5m']
        total = retained + lost
        output.append({'class': name, 'factor': factor, 'category': category,
                       'tp4_count': total, 'tp4_retained_count': retained,
                       'tp4_lost_count': lost,
                       'loss_rate_within_category_percent': 100 * lost / total})
    return output


def report(path, summary, prediction_rows, factor_rows=()):
    rows = [r for r in summary if r['distance_bin'] == 'all']
    lines = ['# PETR AP threshold gap audit', '',
             'Official nuScenes AP is read from saved evaluator metrics. The '
             'prediction and GT transition counts are from a replay of the '
             'evaluator-filtered predictions, validated against the existing '
             'independent GT match replay at all four thresholds. No inference '
             'was run.', '',
             '| Class | AP@0.5 | AP@4 | Gap | 4 m GT matches | Lost GT at 0.5 | '
             'Radial oracle AP@0.5 | Full-center oracle AP@0.5 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {} | {:.1f}% | {:.1f}% | {:.1f} pp | {:,} | {:,} | '
                     '{:.1f}% | {:.1f}% |'.format(
                         r['class'], float(r['official_ap_0.5_percent']),
                         float(r['official_ap_4_percent']),
                         float(r['official_ap_gap_pp']), int(r['matched_gt_4m']),
                         int(r['lost_gt_match_4_to_0.5']),
                         float(r['radial_oracle_ap_0.5_percent']),
                         float(r['full_center_oracle_ap_0.5_percent'])))
    lines += ['', '## Characteristics of GT matches lost at 0.5 m', '',
              'Percentages in this table use lost GT matches as the denominator. '
              'The radial and tangential columns are mean absolute error on '
              'the original 4 m matched prediction.', '',
              '| Class | Lost / 4 m matched | 0–40% visible | Other GT closer | '
              'Abs radial | Abs tangential |',
              '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lost = int(r['lost_gt_match_4_to_0.5'])
        matched = int(r['matched_gt_4m'])
        lines.append('| {} | {:.1f}% | {:.1f}% | {:.1f}% | {:.2f} m | {:.2f} m |'.format(
            r['class'], 100 * lost / matched if matched else 0,
            100 * int(r['lost_gt_0_40_visible']) / lost if lost else 0,
            100 * int(r['lost_gt_other_closer']) / lost if lost else 0,
            float(r['lost_mean_abs_radial_m']),
            float(r['lost_mean_abs_tangential_m'])))
    range_bins = ('0-10m', '10-20m', '20-30m', '30-40m', '40-50m')
    by_range = {(r['class'], r['distance_bin']): r for r in summary}
    lines += ['', '## Where 4 m GT matches lose their 0.5 m match', '',
              'Each cell is lost GT / GT matched at 4 m, with the lost count '
              'in parentheses. A dash means no eligible GT in that range.', '',
              '| Class | ' + ' | '.join(range_bins) + ' |',
              '|---|' + '|'.join('---:' for _ in range_bins) + '|']
    for r in rows:
        cells = []
        for bin_name in range_bins:
            value = by_range.get((r['class'], bin_name))
            if value is None or not int(value['matched_gt_4m']):
                cells.append('—')
            else:
                lost = int(value['lost_gt_match_4_to_0.5'])
                matched = int(value['matched_gt_4m'])
                cells.append('{:.1f}% ({:,})'.format(100 * lost / matched, lost))
        lines.append('| {} | {} |'.format(r['class'], ' | '.join(cells)))
    lines += ['', '## Prediction transitions', '',
              'These counts include all evaluator-filtered predictions, not '
              'only predictions matched at 4 m. A 4 m TP that fails at 0.5 m '
              'usually has no eligible GT within 0.5 m; a small subset is '
              'blocked by an earlier, higher-confidence match.', '',
              'Retention percentages use original 4 m TPs as denominator. '
              'Competition percentage uses 4 m TP → 0.5 m FP as denominator.', '',
              '| Class | FP at 4 m | TP at 4 m | TP4 → TP0.5 | TP4 → FP0.5 | '
              'Competition among lost |',
              '|---|---:|---:|---:|---:|---:|']
    prediction_counts = defaultdict(Counter)
    for r in prediction_rows:
        prediction_counts[r['class']][r['prediction_transition']] = int(r['count'])
    for r in rows:
        c = prediction_counts[r['class']]
        fp4 = sum(v for key, v in c.items() if key.startswith('fp_both_') or
                  key == 'fp4_tp05_after_rematching')
        lost = sum(v for key, v in c.items() if key.startswith('tp4_fp05_'))
        retained = c['tp_both_thresholds']
        tp4 = retained + lost
        competition = c['tp4_fp05_duplicate_or_competition']
        lines.append('| {} | {:,} | {:,} | {:,} ({:.1f}%) | '
                     '{:,} ({:.1f}%) | {:,} ({:.1f}%) |'.format(
                         r['class'], fp4, tp4, retained,
                         100 * retained / tp4 if tp4 else 0, lost,
                         100 * lost / tp4 if tp4 else 0, competition,
                         100 * competition / lost if lost else 0))
    if factor_rows:
        lookup = {(r['class'], r['outcome'], r['factor'], r['category']): r
                  for r in factor_rows}
        selected = (
            ('visibility', '0-40%', '0–40% visible'),
            ('gt_range_30m_plus', 'yes', 'GT ≥30 m'),
            ('different_gt_closer_to_prediction', 'yes', 'Other GT closer'),
            ('abs_radial_exceeds_abs_tangential', 'yes', 'Radial dominant'),
            ('confidence_rank_within_class', 'top_10_percent', 'Top 10% score rank'),
        )
        lines += ['', '## Candidate indicators in retained and lost 4 m TPs', '',
                  'Each percentage uses its own outcome group as denominator: '
                  'TP4 → TP0.5 or TP4 → FP0.5. Indicators overlap and are '
                  'descriptive, so their percentages must not be added. '
                  'Indicators describe the original 4 m assigned GT and '
                  'prediction. A retained prediction can match a different '
                  'GT at 0.5 m. Confidence rank is among all evaluator-filtered '
                  'predictions of that class. Full category counts are in '
                  '`tp_transition_factors.csv`; within-category loss rates '
                  'are in `tp_factor_outcome_rates.csv`.', '']
        for factor, category, label in selected:
            lines += ['| Class | {}: retained | {}: lost |'.format(label, label),
                      '|---|---:|---:|']
            for r in rows:
                name = r['class']
                values = []
                for outcome in ('tp_at_0.5m', 'fp_at_0.5m'):
                    item = lookup.get((name, outcome, factor, category))
                    n = int(item['count']) if item else 0
                    total = int(item['outcome_total']) if item else (
                        prediction_counts[name]['tp_both_thresholds'] if
                        outcome == 'tp_at_0.5m' else sum(
                            value for key, value in prediction_counts[name].items()
                            if key.startswith('tp4_fp05_')))
                    values.append('{:,} ({:.1f}%)'.format(
                        n, 100 * n / total if total else 0))
                lines.append('| {} | {} | {} |'.format(name, *values))
            lines.append('')
    lines += ['', '## Reading the outputs', '',
              '`class_range_summary.csv` contains class × GT-range counts, '
              'visibility, nearby-GT, confidence, and radial/tangential diagnostics. '
              '`gt_transitions.csv` contains each eligible GT. '
              '`prediction_transitions.csv` counts every evaluator-filtered '
              'prediction, including FPs and duplicate/competition flags. '
              '`tp_transition_factors.csv` has all category counts for both '
              '4 m TP outcomes.', '',
              'A 4 m TP can become an FP at 0.5 m while its GT becomes unmatched; '
              'these are two views of the same event, not additive causes. '
              'The duplicate flag means another eligible GT lies within the '
              'threshold but was already taken in confidence order; it does not '
              'prove object identity confusion. Visibility and nearby-GT rates '
              'are associations, not causal explanations. Official AP gaps '
              'cannot be apportioned by adding GT counts or oracle gains.', '',
              'Distance bins are GT-defined, so FP predictions without a matched '
              'GT are summarized by class rather than forced into a GT bin. '
              'A separate bin-restricted AP would define a different evaluation '
              'population and is not presented as a contribution to global AP. '
              'Saved PETR query features are not linked to every evaluated '
              'prediction in these artifacts, so query-level claims are outside '
              'this run.', '']
    path.write_text('\n'.join(lines))


def main():
    here = Path(__file__).resolve().parent
    exp = here.parent
    workspace = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=workspace / 'StreamPETR/data/nuscenes')
    parser.add_argument('--paired-gt', type=Path,
                        default=exp / 'paired_visibility_comparison/outputs/paired_gt.csv')
    parser.add_argument('--oracle-dir', type=Path,
                        default=exp / 'localization_oracle/outputs/baseline')
    parser.add_argument('--output-dir', type=Path, default=here / 'outputs')
    args = parser.parse_args()
    table = args.data_root / 'v1.0-trainval'
    prediction_path = args.oracle_dir / 'baseline/results_nusc.json'
    metric_path = args.oracle_dir / 'baseline/metrics_summary.json'
    paths = [args.paired_gt, prediction_path, metric_path,
             args.oracle_dir / 'radial/metrics_summary.json',
             args.oracle_dir / 'full_center/metrics_summary.json',
             table / 'sample_annotation.json', table / 'sample_data.json',
             table / 'ego_pose.json']
    for path in paths:
        if not path.is_file():
            parser.error('Missing input: {}'.format(path))
    with metric_path.open() as handle:
        metrics = json.load(handle)
    oracle = {}
    for mode in ('radial', 'full_center'):
        with (args.oracle_dir / mode / 'metrics_summary.json').open() as handle:
            oracle[mode] = json.load(handle)
    records = read_csv(args.paired_gt)
    classes = set(metrics['label_aps'])
    records = [r for r in records if r['class'] in classes]
    if not records:
        raise ValueError('No eligible GT records')
    _, gt_by_sample_class = load_gt(records, table / 'sample_annotation.json')
    by_class, _ = load_predictions(prediction_path, classes)
    matches = {}
    for threshold in THRESHOLDS:
        print('Matching at <{} m...'.format(threshold), flush=True)
        matches[threshold] = match(by_class, gt_by_sample_class, threshold)
    validate_match(records, matches)
    print('Loading ego poses...', flush=True)
    ego = read_ego(table, {r['sample_token'] for r in records})
    details, summary = summarize(records, matches, gt_by_sample_class,
                                 ego, metrics, oracle)
    predictions = prediction_summary(by_class, matches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'gt_transitions.csv', details, list(details[0]))
    write_csv(args.output_dir / 'class_range_summary.csv', summary,
              list(dict.fromkeys(k for r in summary for k in r)))
    write_csv(args.output_dir / 'prediction_transitions.csv', predictions,
              list(predictions[0]))
    factors = transition_factors(details, predictions)
    write_csv(args.output_dir / 'tp_transition_factors.csv', factors,
              list(factors[0]))
    rates = factor_outcome_rates(factors)
    write_csv(args.output_dir / 'tp_factor_outcome_rates.csv', rates,
              list(rates[0]))
    report(args.output_dir / 'RESULTS.md', summary, predictions, factors)
    print('Wrote {}'.format(args.output_dir / 'RESULTS.md'), flush=True)


if __name__ == '__main__':
    main()
