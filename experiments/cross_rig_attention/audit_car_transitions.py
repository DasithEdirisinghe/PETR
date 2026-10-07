#!/usr/bin/env python3
"""Audit native-correct GT cars under cross-rig PETR inference outputs.

Reads the ordinary formatted detections and official metric summaries. It does
not rerun inference or alter predictions. Per-car categories are diagnostic;
official AP is read from metrics_summary.json.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compare_r1_r1f_cars import gt_frames


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/cross_rig_attention/output'
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
CATEGORIES = ('still_correct', 'localization', 'classification', 'both',
              'competing_outputs', 'no_output_within_4m')


def read_predictions(path):
    with path.open() as handle:
        payload = json.load(handle)['results']
    return {token: [(np.asarray(box['translation'][:2], dtype=float),
                     box['detection_name'], float(box['detection_score']))
                    for box in boxes] for token, boxes in payload.items()}


def car_matches(centers, predictions, threshold):
    """Mirror the evaluator's confidence-ordered, nearest-unmatched car rule."""
    matched = {}
    taken = set()
    for pred_index in sorted(range(len(predictions)),
                             key=lambda index: -predictions[index][2]):
        point, label, _ = predictions[pred_index]
        if label != 'car':
            continue
        available = [i for i in range(len(centers)) if i not in taken]
        if not available:
            break
        distances = np.linalg.norm(centers[available] - point, axis=1)
        choice = int(np.argmin(distances))
        if distances[choice] < threshold:
            gt_index = available[choice]
            matched[gt_index] = pred_index
            taken.add(gt_index)
    return matched


def owned_candidates(centers, predictions):
    """For diagnosis only, give each final output to its nearest GT car."""
    owned = [[] for _ in centers]
    if len(centers) == 0:
        return owned
    for pred_index, (point, label, score) in enumerate(predictions):
        distances = np.linalg.norm(centers - point, axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance < 4.0:
            owned[index].append((distance, label, score, pred_index))
    return owned


def best(candidates):
    return min(candidates, key=lambda item: (item[0], -item[2])) if candidates else None


def outcome(candidates, threshold):
    cars = [item for item in candidates if item[1] == 'car']
    others = [item for item in candidates if item[1] != 'car']
    car = best(cars)
    other = best(others)
    if car and other:
        category = 'competing_outputs'
    elif car:
        category = 'localization' if car[0] >= threshold else 'competing_outputs'
    elif other:
        category = 'classification' if other[0] < threshold else 'both'
    else:
        category = 'no_output_within_4m'
    return category, car, other


def official_ap(path):
    with path.open() as handle:
        values = json.load(handle)['label_aps']['car']
    return {str(t): float(values[str(t)]) for t in THRESHOLDS}


def run_rig(rig):
    native = rig
    cross = 'R1-f' if rig == 'R1' else 'R1'
    stem = 'r1_val' if rig == 'R1' else 'r1f_val'
    source = EXPERIMENT / stem
    frames = gt_frames(ROOT / 'data/pccr' / rig / (rig + '_infos_val.pkl'))
    paths = {name: source / name / 'formatted' for name in (native, cross)}
    predictions = {name: read_predictions(paths[name] / 'results_pccr.json')
                   for name in (native, cross)}
    for name, records in predictions.items():
        if set(records) != set(frames):
            raise ValueError('{} {}: result tokens differ from validation GT'.format(rig, name))
    ap = {name: official_ap(paths[name] / 'metrics_summary.json')
          for name in (native, cross)}
    rows = []
    counts = {}
    for threshold in THRESHOLDS:
        summary = Counter()
        for token, frame in frames.items():
            centers = frame['centers']
            native_preds = predictions[native][token]
            cross_preds = predictions[cross][token]
            native_hits = car_matches(centers, native_preds, threshold)
            cross_hits = car_matches(centers, cross_preds, threshold)
            owned = owned_candidates(centers, cross_preds)
            summary['all_gt_cars'] += len(centers)
            summary['native_correct'] += len(native_hits)
            summary['cross_correct_all_gt'] += len(cross_hits)
            for index, native_pred_index in native_hits.items():
                if index in cross_hits:
                    category = 'still_correct'
                    car = other = None
                else:
                    category, car, other = outcome(owned[index], threshold)
                summary[category] += 1
                native_pred = native_preds[native_pred_index]
                rows.append({
                    'rig': rig, 'sample_token': token,
                    'gt_index': int(frame['gt_indices'][index]),
                    'threshold_m': threshold, 'native_model': native,
                    'cross_model': cross, 'outcome': category,
                    'native_car_score': native_pred[2],
                    'native_center_error_m': float(np.linalg.norm(
                        native_pred[0] - centers[index])),
                    'nearest_cross_car_distance_m': car[0] if car else '',
                    'nearest_cross_car_score': car[2] if car else '',
                    'nearest_cross_other_distance_m': other[0] if other else '',
                    'nearest_cross_other_class': other[1] if other else '',
                    'nearest_cross_other_score': other[2] if other else '',
                    'cross_owned_outputs_within_4m': len(owned[index]),
                })
        counts[str(threshold)] = {
            'all_gt_cars': summary['all_gt_cars'],
            'native_correct': summary['native_correct'],
            'cross_correct_all_gt': summary['cross_correct_all_gt'],
            'native_correct_transitions': {name: summary[name] for name in CATEGORIES},
            'native_car_ap': ap[native][str(threshold)],
            'cross_car_ap': ap[cross][str(threshold)],
        }
    return rows, {'native_model': native, 'cross_model': cross,
                  'frames': len(frames), 'by_threshold': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=EXPERIMENT / 'car_transition_audit')
    args = parser.parse_args()
    all_rows = []
    summary = {'method': 'standard saved PETR outputs; evaluator-style car matches; '
                         'nearest-GT ownership only for diagnostic failure labels',
               'radius_m': 4.0, 'rigs': {}}
    for rig in ('R1', 'R1-f'):
        rows, rig_summary = run_rig(rig)
        all_rows.extend(rows)
        summary['rigs'][rig] = rig_summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'per_car_transitions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print('Saved', args.output_dir / 'per_car_transitions.csv')
    print('Saved', args.output_dir / 'summary.json')


if __name__ == '__main__':
    main()
