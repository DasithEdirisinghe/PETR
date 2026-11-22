#!/usr/bin/env python3
"""Compare PETR and StreamPETR on the same eligible nuScenes validation GT."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
RANGE_EDGES = (10, 20, 30, 40, 50)
VISIBILITY = {'1': '0-40%', '2': '40-60%', '3': '60-80%', '4': '80-100%'}


def range_bin(distance):
    for upper in RANGE_EDGES:
        if distance < upper:
            return '{}-{}m'.format(upper - 10, upper)
    return '50m+'


def official_matches(evaluator, threshold):
    """Replay score-ordered, nearest-unmatched-GT devkit matching."""
    import numpy as np
    matched = {}
    for name in evaluator.cfg.class_names:
        predictions = [(token, box) for token in evaluator.pred_boxes.sample_tokens
                       for box in evaluator.pred_boxes[token]
                       if box.detection_name == name]
        scores = np.asarray([box.detection_score for _, box in predictions])
        taken = set()
        for rank in np.argsort(scores)[::-1]:
            token, pred = predictions[int(rank)]
            nearest, distance = None, float('inf')
            for index, gt in enumerate(evaluator.gt_boxes[token]):
                if gt.detection_name != name or (token, index) in taken:
                    continue
                candidate = evaluator.cfg.dist_fcn_callable(gt, pred)
                if candidate < distance:
                    nearest, distance = index, candidate
            if nearest is not None and distance < threshold:
                key = (token, nearest)
                taken.add(key)
                matched[key] = (distance, pred.detection_score)
    return matched


def gt_records(nusc, evaluator):
    from nuscenes.eval.detection.utils import category_to_detection_name
    records = {}
    for token in evaluator.gt_boxes.sample_tokens:
        sample = nusc.get('sample', token)
        annotations = defaultdict(list)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            name = category_to_detection_name(ann['category_name'])
            annotations[(name, tuple(ann['translation']))].append(ann)
        for index, gt in enumerate(evaluator.gt_boxes[token]):
            matches = annotations[gt.detection_name, tuple(gt.translation)]
            if len(matches) != 1:
                raise ValueError('Cannot uniquely identify GT annotation {} {}'.format(token, index))
            ann = matches[0]
            visibility = ann['visibility_token']
            if visibility not in VISIBILITY:
                raise ValueError('Unexpected visibility token {}'.format(visibility))
            distance = math.hypot(*gt.ego_translation[:2])
            records[token, index] = {
                'sample_token': token,
                'annotation_token': ann['token'],
                'instance_token': ann['instance_token'],
                'class': gt.detection_name,
                'distance_m': distance,
                'distance_bin': range_bin(distance),
                'visibility_token': visibility,
                'visibility_band': VISIBILITY[visibility],
            }
    return records


def validate_same_gt(first, second):
    if set(first.gt_boxes.sample_tokens) != set(second.gt_boxes.sample_tokens):
        raise ValueError('Models have different validation sample sets')
    for token in first.gt_boxes.sample_tokens:
        a, b = first.gt_boxes[token], second.gt_boxes[token]
        if len(a) != len(b) or any(
                x.detection_name != y.detection_name or
                tuple(x.translation) != tuple(y.translation)
                for x, y in zip(a, b)):
            raise ValueError('Evaluator-filtered GT differs at sample {}'.format(token))


def status(key, matches, threshold):
    petr = key in matches['PETR'][threshold]
    stream = key in matches['StreamPETR'][threshold]
    if petr and stream:
        return 'both'
    if petr:
        return 'petr_only'
    if stream:
        return 'stream_only'
    return 'neither'


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    here = Path(__file__).resolve().parent
    # PETR and StreamPETR are siblings under dev_ws on both local and cluster mounts.
    workspace = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=workspace / 'StreamPETR/data/nuscenes')
    parser.add_argument('--petr-results', type=Path,
                        default=here.parent / 'localization_oracle/outputs/baseline/'
                        'baseline/results_nusc.json')
    parser.add_argument('--stream-results', type=Path,
                        default=workspace / 'StreamPETR/test/'
                        'stream_petr_r50_noflash_428q_nui_60e/'
                        'Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json')
    parser.add_argument('--output-dir', type=Path, default=here / 'outputs')
    args = parser.parse_args()
    for path in (args.petr_results, args.stream_results):
        if not path.is_file():
            parser.error('Missing prediction JSON: {}'.format(path))
    if not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Missing nuScenes metadata under {}'.format(args.data_root))

    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    config = config_factory('detection_cvpr_2019')
    evaluators = {}
    matches = {}
    for name, path in (('PETR', args.petr_results),
                       ('StreamPETR', args.stream_results)):
        print('Loading {} predictions...'.format(name), flush=True)
        evaluator = NuScenesEval(nusc, config, str(path), 'val',
                                str(args.output_dir / ('devkit_' + name.lower())),
                                verbose=False)
        evaluators[name] = evaluator
        matches[name] = {}
        for threshold in THRESHOLDS:
            print('Matching {} at <{} m...'.format(name, threshold), flush=True)
            matches[name][threshold] = official_matches(evaluator, threshold)
        for lower, upper in zip(THRESHOLDS, THRESHOLDS[1:]):
            if not matches[name][lower].keys() <= matches[name][upper].keys():
                raise AssertionError('{} GT match sets are not nested'.format(name))

    validate_same_gt(evaluators['PETR'], evaluators['StreamPETR'])
    records = gt_records(nusc, evaluators['PETR'])
    if any(key not in records for model in matches.values()
           for threshold_matches in model.values() for key in threshold_matches):
        raise AssertionError('Matched GT missing from common eligible GT set')

    detail = []
    counters = defaultdict(Counter)
    for key, record in records.items():
        row = dict(record)
        for name, prefix in (('PETR', 'petr'), ('StreamPETR', 'stream')):
            pair = matches[name][4.0].get(key)
            row[prefix + '_error_at_4m_m'] = pair[0] if pair else ''
            row[prefix + '_score_at_4m'] = pair[1] if pair else ''
        for threshold in THRESHOLDS:
            row['status_at_{}m'.format(threshold)] = status(key, matches, threshold)
        detail.append(row)
        scopes = (
            ('all', 'all', 'all'),
            (row['class'], 'all', 'all'),
            ('all', row['distance_bin'], 'all'),
            ('all', 'all', row['visibility_band']),
            (row['class'], row['distance_bin'], 'all'),
            (row['class'], 'all', row['visibility_band']),
            ('all', row['distance_bin'], row['visibility_band']),
            (row['class'], row['distance_bin'], row['visibility_band']),
        )
        for scope in scopes:
            counters[scope]['total_gt'] += 1
            for threshold in THRESHOLDS:
                counters[scope][(threshold, row['status_at_{}m'.format(threshold)])] += 1
            if row['status_at_4.0m'] == 'both':
                diff = row['stream_error_at_4m_m'] - row['petr_error_at_4m_m']
                counters[scope]['both_error_delta_sum_m'] += diff
                counters[scope]['both_stream_better'] += diff < 0
                counters[scope]['both_petr_better'] += diff > 0

    detail.sort(key=lambda row: (row['class'], row['sample_token'], row['annotation_token']))
    write_csv(args.output_dir / 'paired_gt.csv', detail)
    summary = []
    for (name, distance, visibility), counts in sorted(counters.items()):
        total = counts['total_gt']
        both = counts[(4.0, 'both')]
        for threshold in THRESHOLDS:
            row = {'class': name, 'distance_bin': distance,
                   'visibility_band': visibility, 'threshold_m': threshold,
                   'total_gt': total}
            for category in ('neither', 'petr_only', 'stream_only', 'both'):
                row[category] = counts[(threshold, category)]
            row['petr_recall_percent'] = 100 * (row['petr_only'] + row['both']) / total
            row['stream_recall_percent'] = 100 * (row['stream_only'] + row['both']) / total
            row['recall_gain_pp'] = row['stream_recall_percent'] - row['petr_recall_percent']
            row['mean_stream_minus_petr_error_m_on_both_4m'] = (
                counts['both_error_delta_sum_m'] / both if both else '')
            row['stream_better_on_both_4m_count'] = counts['both_stream_better']
            row['petr_better_on_both_4m_count'] = counts['both_petr_better']
            summary.append(row)
    write_csv(args.output_dir / 'summary.csv', summary)
    overall = next(row for row in summary if row['class'] == 'all' and
                   row['distance_bin'] == 'all' and row['visibility_band'] == 'all' and
                   row['threshold_m'] == 4.0)
    lines = ['# Paired PETR–StreamPETR GT comparison', '',
             'Uses the same official evaluator-filtered nuScenes validation GT '
             'and saved model predictions. No inference was run. Matching is '
             'replayed independently at each threshold in descending score order.', '',
             'These are GT recall and center-error comparisons, **not AP**. '
             'AP also depends on false positives and ranking. Visibility is the '
             'nuScenes six-camera panorama annotation band.', '',
             'Total eligible GT: {:,}. At <4 m: PETR {:,}, StreamPETR {:,}, '
             'both {:,}, PETR only {:,}, StreamPETR only {:,}.'.format(
                 overall['total_gt'], overall['petr_only'] + overall['both'],
                 overall['stream_only'] + overall['both'], overall['both'],
                 overall['petr_only'], overall['stream_only']), '',
             '| Visibility | GT | PETR <4 m | StreamPETR <4 m | Gain (pp) | '
             'Stream only | PETR only |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for band in VISIBILITY.values():
        row = next(x for x in summary if x['class'] == 'all' and
                   x['distance_bin'] == 'all' and x['visibility_band'] == band and
                   x['threshold_m'] == 4.0)
        lines.append('| {} | {:,} | {:.1f}% | {:.1f}% | {:+.1f} | {:,} | {:,} |'.format(
            band, row['total_gt'], row['petr_recall_percent'],
            row['stream_recall_percent'], row['recall_gain_pp'],
            row['stream_only'], row['petr_only']))
    lines += ['', 'See `summary.csv` for class × distance × visibility at all '
              'thresholds and `paired_gt.csv` for each GT and the two fixed '
              '<4 m center errors. Differences within the “both” cohort are '
              'descriptive; cohort membership changes at tighter thresholds.', '']
    (args.output_dir / 'RESULTS.md').write_text('\n'.join(lines))
    print('Wrote {}'.format(args.output_dir / 'RESULTS.md'), flush=True)


if __name__ == '__main__':
    main()
