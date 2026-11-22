#!/usr/bin/env python3
"""Count official nuScenes GT matches by ego range and distance threshold."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
RANGE_LABELS = ('0–10 m', '10–20 m', '20–30 m', '30–40 m', '40–50 m', '>50 m')
BAND_KEYS = ('error_lt_0.5m', 'error_0.5_to_1m',
             'error_1_to_2m', 'error_2_to_4m', 'no_match_lt_4m')


def range_bin(distance):
    if distance < 0:
        raise ValueError('Negative GT ego range')
    if distance < 10:
        return RANGE_LABELS[0]
    if distance < 20:
        return RANGE_LABELS[1]
    if distance < 30:
        return RANGE_LABELS[2]
    if distance < 40:
        return RANGE_LABELS[3]
    if distance <= 50:
        return RANGE_LABELS[4]
    return RANGE_LABELS[5]


def match_gt(evaluator):
    """Replay the devkit's score order at every threshold, sample by sample."""
    matched = {threshold: set() for threshold in THRESHOLDS}
    # Matching is independent between samples and classes. Working one sample
    # at a time lets us compute each prediction-to-GT distance only once.
    for token in evaluator.gt_boxes.sample_tokens:
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)
        for gt_index, gt in enumerate(evaluator.gt_boxes[token]):
            gt_by_class[gt.detection_name].append((gt_index, gt))
        for pred_index, pred in enumerate(evaluator.pred_boxes[token]):
            pred_by_class[pred.detection_name].append((pred_index, pred))
        for class_name in evaluator.cfg.class_names:
            gt_candidates = gt_by_class[class_name]
            if not gt_candidates:
                continue
            ranked = sorted(pred_by_class[class_name],
                            key=lambda item: (item[1].detection_score, item[0]),
                            reverse=True)
            candidates = []
            for _, pred in ranked:
                candidates.append(sorted(
                    (evaluator.cfg.dist_fcn_callable(gt, pred), gt_index)
                    for gt_index, gt in gt_candidates))
            for threshold in THRESHOLDS:
                taken = set()
                for distances in candidates:
                    for distance, gt_index in distances:
                        if gt_index not in taken:
                            if distance < threshold:
                                taken.add(gt_index)
                                matched[threshold].add((token, gt_index))
                            break
    return matched


def gt_records(evaluator, nusc):
    """Use the evaluator-filtered GT boxes and the ego pose of each keyframe."""
    origins = {}
    records = {}
    for token in evaluator.gt_boxes.sample_tokens:
        sample = nusc.get('sample', token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        origins[token] = nusc.get(
            'ego_pose', sample_data['ego_pose_token'])['translation'][:2]
        ego_x, ego_y = origins[token]
        for gt_index, gt in enumerate(evaluator.gt_boxes[token]):
            distance = math.hypot(gt.translation[0] - ego_x,
                                  gt.translation[1] - ego_y)
            records[(token, gt_index)] = {
                'class': gt.detection_name,
                'gt_range_m': distance,
                'range_bin': range_bin(distance),
            }
    return records


def summarize(records, matches, class_name=None):
    keys_by_range = defaultdict(set)
    for key, record in records.items():
        if class_name is None or record['class'] == class_name:
            keys_by_range[record['range_bin']].add(key)
    rows = []
    for label in RANGE_LABELS:
        gt_keys = keys_by_range[label]
        counts = {threshold: len(gt_keys & matches[threshold])
                  for threshold in THRESHOLDS}
        total = len(gt_keys)
        row = {
            'gt_range': label,
            'total_gt': total,
            'matched_lt_4m': counts[4.0],
            'no_match_lt_4m': total - counts[4.0],
            'error_lt_0.5m': counts[0.5],
            'error_0.5_to_1m': counts[1.0] - counts[0.5],
            'error_1_to_2m': counts[2.0] - counts[1.0],
            'error_2_to_4m': counts[4.0] - counts[2.0],
        }
        for threshold in THRESHOLDS:
            row['recall_lt_{}m_percent'.format(threshold)] = (
                100 * counts[threshold] / total if total else None)
        if sum(row[key] for key in BAND_KEYS) != total:
            raise AssertionError('GT categories do not partition {}'.format(label))
        rows.append(row)
    return rows


def with_total(rows):
    fields = ('total_gt', 'matched_lt_4m', 'no_match_lt_4m') + BAND_KEYS[:-1]
    total = {'gt_range': 'Total'}
    for field in fields:
        total[field] = sum(row[field] for row in rows)
    cumulative_fields = {
        0.5: ('error_lt_0.5m',),
        1.0: ('error_lt_0.5m', 'error_0.5_to_1m'),
        2.0: ('error_lt_0.5m', 'error_0.5_to_1m', 'error_1_to_2m'),
        4.0: BAND_KEYS[:-1],
    }
    for threshold in THRESHOLDS:
        matched = sum(total[field] for field in cumulative_fields[threshold])
        total['recall_lt_{}m_percent'.format(threshold)] = (
            100 * matched / total['total_gt'] if total['total_gt'] else None)
    return rows + [total]


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows):
    lines = [
        '| GT ego range | Total GT | Matched <4 m | No match <4 m | '
        'Error <0.5 m | Error 0.5–1 m | Error 1–2 m | Error 2–4 m |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        values = [row['gt_range']] + [
            '{:,}'.format(row[key]) for key in
            ('total_gt', 'matched_lt_4m', 'no_match_lt_4m',
             'error_lt_0.5m', 'error_0.5_to_1m',
             'error_1_to_2m', 'error_2_to_4m')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines.extend((
        '',
        '| GT ego range | GT recall <0.5 m | GT recall <1 m | '
        'GT recall <2 m | GT recall <4 m |',
        '|---|---:|---:|---:|---:|',
    ))
    for row in rows:
        values = [row['gt_range']]
        for threshold in THRESHOLDS:
            value = row['recall_lt_{}m_percent'.format(threshold)]
            values.append('—' if value is None else '{:.1f}%'.format(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--results-json', type=Path, default=Path(
        'experiments/detection_error_analysis/outputs/petr_car_ap_aligned/'
        'formatted/pts_bbox/results_nusc.json'))
    parser.add_argument('--output-dir', type=Path, default=experiment_dir /
                        'outputs/baseline')
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error('nuScenes data root does not exist: {}'.format(args.data_root))
    if not args.results_json.is_file():
        parser.error('Baseline predictions not found: {}'.format(args.results_json))

    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('Loading nuScenes metadata and filtered evaluation boxes...', flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                   verbose=False)
    evaluator = NuScenesEval(
        nusc, config=config_factory('detection_cvpr_2019'),
        result_path=str(args.results_json), eval_set='val',
        output_dir=str(args.output_dir), verbose=False)
    records = gt_records(evaluator, nusc)
    print('Evaluator-eligible GT: {:,}'.format(len(records)), flush=True)
    print('Replaying matching at all thresholds...', flush=True)
    matches = match_gt(evaluator)
    for threshold in THRESHOLDS:
        print('Matched GT at <{} m: {:,}'.format(
            threshold, len(matches[threshold])), flush=True)
    for lower, upper in zip(THRESHOLDS, THRESHOLDS[1:]):
        if not matches[lower] <= matches[upper]:
            raise AssertionError('Matched GT sets are not nested')
    if not matches[4.0] <= set(records):
        raise AssertionError('Matched GT missing from evaluator-filtered GT')
    previous_summary = experiment_dir.parent / (
        'center_error_decomposition/outputs/baseline/center_error_summary.json')
    if previous_summary.is_file():
        with previous_summary.open() as handle:
            previous = json.load(handle)
        if (Path(previous['source']).resolve() == args.results_json.resolve()
                and len(matches[4.0]) != previous['matches']):
            raise AssertionError(
                '4 m matched GT count {} differs from previous center-error '
                'count {}'.format(len(matches[4.0]), previous['matches']))

    overall = with_total(summarize(records, matches))
    by_class = []
    for class_name in evaluator.cfg.class_names:
        by_class.extend({'class': class_name, **row}
                        for row in with_total(summarize(records, matches, class_name)))
    write_csv(args.output_dir / 'gt_coverage_by_range.csv', overall)
    write_csv(args.output_dir / 'gt_coverage_by_class.csv', by_class)
    report = markdown(overall)
    (args.output_dir / 'gt_coverage_by_range.md').write_text(
        '# Baseline PETR GT detection coverage by ego range\n\n' + report + '\n\n'
        'GT and predictions use the official nuScenes evaluation filters. '
        'The four localization bands plus the no-match group partition all '
        'GT in each range bin. Recall percentages use all eligible GT in that '
        'range bin as denominator. Matching is replayed independently at each '
        'threshold.\n')
    with (args.output_dir / 'gt_coverage_by_range.json').open('w') as handle:
        json.dump({'source': str(args.results_json),
                   'gt_denominator': 'official evaluator-filtered validation GT',
                   'total_gt': len(records),
                   'matched_gt_by_threshold': {
                       str(t): len(matches[t]) for t in THRESHOLDS},
                   'rows': overall}, handle, indent=2)
    print(report)
    print('\nTotal GT:', len(records))
    print('Saved to:', args.output_dir)


if __name__ == '__main__':
    main()
