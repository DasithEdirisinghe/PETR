#!/usr/bin/env python3
"""Run fixed-assignment radial, tangential, and full-center nuScenes oracles."""

import argparse
import csv
import json
import math
from pathlib import Path


MODES = ('baseline', 'radial', 'tangential', 'full_center')
RESULT_FIELDS = ('sample_token', 'translation', 'size', 'rotation', 'velocity',
                 'detection_name', 'detection_score', 'attribute_name')


def match_at_four_m(evaluator):
    """Replay official score order and greedy same-class 4 m matching."""
    matches = []
    for name in evaluator.cfg.class_names:
        predictions = [box for box in evaluator.pred_boxes.all
                       if box.detection_name == name]
        ranked = [i for _, i in sorted(
            (box.detection_score, i) for i, box in enumerate(predictions))][::-1]
        taken = set()
        for i in ranked:
            pred = predictions[i]
            token = pred.sample_token
            nearest, distance = None, float('inf')
            for j, gt in enumerate(evaluator.gt_boxes[token]):
                if gt.detection_name != name or (token, j) in taken:
                    continue
                d = evaluator.cfg.dist_fcn_callable(gt, pred)
                if d < distance:
                    nearest, distance = j, d
            if nearest is not None and distance < 4.0:
                taken.add((token, nearest))
                matches.append((pred, evaluator.gt_boxes[token][nearest], distance))
    return matches


def corrected_xy(pred_xy, gt_xy, ego_xy):
    dx, dy = pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]
    rx, ry = gt_xy[0] - ego_xy[0], gt_xy[1] - ego_xy[1]
    radius = math.hypot(rx, ry)
    if radius < 1e-8:
        raise ValueError('GT center lies at ego origin; radial direction undefined')
    ux, uy = rx / radius, ry / radius
    radial = dx * ux + dy * uy
    radial_xy = (pred_xy[0] - radial * ux, pred_xy[1] - radial * uy)
    tangential_xy = (gt_xy[0] + radial * ux, gt_xy[1] + radial * uy)
    return {'radial': radial_xy, 'tangential': tangential_xy,
            'full_center': (gt_xy[0], gt_xy[1])}


def write_results(path, boxes, meta, corrections, mode):
    results = {}
    for token in boxes.sample_tokens:
        rows = []
        for box in boxes[token]:
            serialized = box.serialize()
            row = {key: serialized[key] for key in RESULT_FIELDS}
            if mode != 'baseline' and id(box) in corrections:
                row['translation'] = list(row['translation'])
                row['translation'][:2] = corrections[id(box)][mode]
            rows.append(row)
        results[token] = rows
    with path.open('w') as handle:
        json.dump({'meta': meta, 'results': results}, handle)


def ap_row(mode, metrics, classes):
    row = {'oracle': mode}
    for threshold in (0.5, 1.0, 2.0, 4.0):
        key = str(threshold)
        row['AP@{}m'.format(key)] = 100 * sum(
            metrics['label_aps'][name][key] for name in classes) / len(classes)
    row['mAP'] = 100 * metrics['mean_ap']
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--results-json', type=Path, default=Path(
        'experiments/detection_error_analysis/outputs/petr_car_ap_aligned/'
        'formatted/pts_bbox/results_nusc.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(
        'experiments/nuscnes_exp/localization_oracle/outputs/baseline'))
    args = parser.parse_args()
    if not args.data_root.exists() or not args.results_json.is_file():
        parser.error('Check --data-root and --results-json; both must exist')

    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    config = config_factory('detection_cvpr_2019')
    source = NuScenesEval(nusc, config=config, result_path=str(args.results_json),
                         eval_set='val', output_dir=str(args.output_dir / 'source'),
                         verbose=False)
    matches = match_at_four_m(source)
    ego = {}
    corrections = {}
    for pred, gt, distance in matches:
        token = pred.sample_token
        if token not in ego:
            sample = nusc.get('sample', token)
            lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego[token] = nusc.get('ego_pose', lidar['ego_pose_token'])['translation'][:2]
        corrections[id(pred)] = corrected_xy(pred.translation, gt.translation, ego[token])
        if abs(math.hypot(pred.translation[0] - gt.translation[0],
                          pred.translation[1] - gt.translation[1]) - distance) > 1e-5:
            raise ValueError('Matching distance differs from BEV center distance')

    rows = []
    for mode in MODES:
        directory = args.output_dir / mode
        directory.mkdir(parents=True, exist_ok=True)
        submission = directory / 'results_nusc.json'
        write_results(submission, source.pred_boxes, source.meta, corrections, mode)
        evaluator = NuScenesEval(nusc, config=config, result_path=str(submission),
                                 eval_set='val', output_dir=str(directory), verbose=False)
        evaluator.main(render_curves=False)
        with (directory / 'metrics_summary.json').open() as handle:
            metrics = json.load(handle)
        if mode == 'baseline':
            original_metrics = args.results_json.with_name('metrics_summary.json')
            if original_metrics.is_file():
                with original_metrics.open() as handle:
                    original = json.load(handle)
                if abs(metrics['mean_ap'] - original['mean_ap']) > 1e-8:
                    raise ValueError('Re-evaluated baseline differs from source metrics; '
                                     'inspect the submission before interpreting oracles')
        rows.append(ap_row(mode, metrics, config.class_names))
        print(mode, rows[-1], flush=True)

    with (args.output_dir / 'ap_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / 'summary.md').open('w') as handle:
        handle.write('# PETR localization oracles\n\n')
        handle.write('Fixed original 4 m greedy matches: {:,}. Only global BEV XY '
                     'centers of matched predictions were changed. Scores, classes, '
                     'sizes, heights, and rotations were preserved.\n\n'.format(len(matches)))
        handle.write('| Oracle | AP@0.5m | AP@1m | AP@2m | AP@4m | mAP | ΔmAP |\n')
        handle.write('|---|---:|---:|---:|---:|---:|---:|\n')
        baseline = rows[0]['mAP']
        for row in rows:
            handle.write('| {} | {} | {:+.2f} pp |\n'.format(
                row['oracle'], ' | '.join('{:.2f}%'.format(row[k]) for k in
                    ('AP@0.5m', 'AP@1.0m', 'AP@2.0m', 'AP@4.0m', 'mAP')),
                row['mAP'] - baseline))
        handle.write('\nOfficial evaluator rematches each corrected submission. '
                     'Unmatched predictions are unchanged. This is an oracle for '
                     'the original 4 m matched subset, not a claim about an '
                     'achievable learned correction.\n')
    print('Fixed 4 m matches:', len(matches))
    print('Wrote:', args.output_dir / 'summary.md')


if __name__ == '__main__':
    main()
