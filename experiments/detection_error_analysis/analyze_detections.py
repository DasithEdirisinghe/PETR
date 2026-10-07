#!/usr/bin/env python3
"""Explain missed GT objects using the official nuScenes AP input and matching.

Inference is unchanged: PETR submits its top-scored boxes without an extra
confidence cutoff. The nuScenes devkit formats, filters, and evaluates those
boxes. We replay its class-specific, score-ordered distance matching to label
true positives, false positives, and missed GT. Explanations for misses are
post-hoc spatial evidence, not an additive or causal decomposition of AP.
"""

import argparse
import csv
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from nuscenes import NuScenes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap
from nuscenes.eval.detection.evaluate import NuScenesEval


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GT_FIELDS = (
    'sample_token', 'gt_index', 'target_class', 'distance_threshold_m',
    'status', 'evidence', 'matched_prediction_score',
    'nearest_same_class_m', 'nearest_other_class_m',
)
EVIDENCE = (
    'matched', 'same_class_near_but_unmatched', 'mixed_evidence',
    'possible_class_confusion', 'possible_localization_error',
    'no_nearby_prediction',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--target-class', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--ann-file')
    parser.add_argument('--nearby-distance', type=float, default=8.0,
                        help='Radius for post-hoc evidence around a missed GT.')
    parser.add_argument('--gpu-id', type=int, default=0)
    return parser.parse_args()


def import_plugin(cfg):
    if cfg.get('plugin', False):
        module = os.path.dirname(cfg.get('plugin_dir', '')).replace('/', '.')
        importlib.import_module(module)


def run_inference(args):
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    if args.data_root is not None:
        cfg.data.test.data_root = args.data_root
        if 'data_root' in cfg:
            cfg.data_root = args.data_root
    if args.ann_file is not None:
        cfg.data.test.ann_file = args.ann_file
        if 'test_ann_file' in cfg:
            cfg.test_ann_file = args.ann_file
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    if args.target_class not in model.CLASSES:
        raise ValueError('Unknown target class {!r}; choices are {}'.format(
            args.target_class, list(model.CLASSES)))
    torch.cuda.set_device(args.gpu_id)
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    model.eval()

    outputs = []
    progress = mmcv.ProgressBar(len(dataset))
    for data in loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        outputs.extend(result)
        for _ in result:
            progress.update()
    if len(outputs) != len(dataset):
        raise RuntimeError('Expected {} predictions, got {}'.format(
            len(dataset), len(outputs)))
    return dataset, outputs


def official_evaluation(dataset, outputs, output_dir):
    """Use the dataset formatter and nuScenes devkit's exact filtered boxes."""
    formatted_prefix = output_dir / 'formatted'
    result_files, temporary = dataset.format_results(
        outputs, jsonfile_prefix=str(formatted_prefix))
    if temporary is not None:
        raise RuntimeError('Expected persistent formatted results')
    result_path = result_files['pts_bbox'] if isinstance(
        result_files, dict) else result_files
    result_path = Path(result_path)
    nusc = NuScenes(
        version=dataset.version, dataroot=dataset.data_root, verbose=False)
    eval_set = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}[
        dataset.version]
    evaluator = NuScenesEval(
        nusc, config=dataset.eval_detection_configs,
        result_path=str(result_path), eval_set=eval_set,
        output_dir=str(result_path.parent), verbose=False)
    evaluator.main(render_curves=False)
    metrics_path = result_path.parent / 'metrics_summary.json'
    metrics = mmcv.load(str(metrics_path))
    return evaluator, metrics, metrics_path


def replay_match(evaluator, target_class, distance_threshold):
    """Replay the devkit's greedy, class-specific, score-ranked matching."""
    gt_by_token = evaluator.gt_boxes.boxes
    predictions = [
        box for box in evaluator.pred_boxes.all
        if box.detection_name == target_class
    ]
    # Match the devkit's tie-breaker as well as its descending score order.
    ranked_indices = [index for _, index in sorted(
        (box.detection_score, index)
        for index, box in enumerate(predictions))][::-1]

    taken = set()
    matched = {}
    false_positives = 0
    for pred_index in ranked_indices:
        prediction = predictions[pred_index]
        token = prediction.sample_token
        nearest_index, nearest_distance = None, float('inf')
        for gt_index, gt in enumerate(gt_by_token[token]):
            if gt.detection_name != target_class or (token, gt_index) in taken:
                continue
            distance = evaluator.cfg.dist_fcn_callable(gt, prediction)
            if distance < nearest_distance:
                nearest_index, nearest_distance = gt_index, distance
        if nearest_index is not None and nearest_distance < distance_threshold:
            key = (token, nearest_index)
            taken.add(key)
            matched[key] = (prediction, nearest_distance)
        else:
            false_positives += 1
    return matched, false_positives


def nearest_distances(gt, predictions, target_class):
    same = other = float('inf')
    for prediction in predictions:
        distance = center_distance(gt, prediction)
        if prediction.detection_name == target_class:
            same = min(same, distance)
        else:
            other = min(other, distance)
    return same, other


def gt_rows(evaluator, target_class, threshold, nearby_distance, matched):
    rows = []
    for token in evaluator.gt_boxes.sample_tokens:
        predictions = evaluator.pred_boxes.boxes[token]
        for gt_index, gt in enumerate(evaluator.gt_boxes.boxes[token]):
            if gt.detection_name != target_class:
                continue
            key = (token, gt_index)
            same, other = nearest_distances(gt, predictions, target_class)
            if key in matched:
                prediction, _ = matched[key]
                status, evidence = 'TP', 'matched'
                score = float(prediction.detection_score)
            else:
                status, score = 'FN', ''
                if same < threshold:
                    evidence = 'same_class_near_but_unmatched'
                elif other < threshold and same <= nearby_distance:
                    evidence = 'mixed_evidence'
                elif other < threshold:
                    evidence = 'possible_class_confusion'
                elif same <= nearby_distance:
                    evidence = 'possible_localization_error'
                else:
                    evidence = 'no_nearby_prediction'
            rows.append({
                'sample_token': token,
                'gt_index': gt_index,
                'target_class': target_class,
                'distance_threshold_m': threshold,
                'status': status,
                'evidence': evidence,
                'matched_prediction_score': score,
                'nearest_same_class_m': '' if same == float('inf') else same,
                'nearest_other_class_m': '' if other == float('inf') else other,
            })
    return rows


def main():
    args = parse_args()
    if args.nearby_distance <= 0:
        raise ValueError('--nearby-distance must be positive')
    os.chdir(str(REPO_ROOT))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, outputs = run_inference(args)
    evaluator, metrics, metrics_path = official_evaluation(
        dataset, outputs, output_dir)
    thresholds = dataset.eval_detection_configs.dist_ths
    if args.nearby_distance < max(thresholds):
        raise ValueError('--nearby-distance must be at least {}'.format(
            max(thresholds)))

    all_rows = []
    summary = {
        'target_class': args.target_class,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'score_threshold': None,
        'prediction_source': 'PETR final top-k boxes, formatted and filtered by nuScenes',
        'official_metrics_summary': str(metrics_path),
        'official_nds': metrics['nd_score'],
        'official_map': metrics['mean_ap'],
        'nearby_distance_m': args.nearby_distance,
        'distance_thresholds': {},
    }
    print('\n{}: official nuScenes AP and score-ranked matching'.format(
        args.target_class))
    print('No additional confidence threshold; PETR top-k is unchanged.')
    print('{:<8} {:>9} {:>8} {:>8} {:>8} {:>12}'.format(
        'Dist (m)', 'AP', 'TP', 'FN', 'FP', 'Replay check'))
    for threshold in thresholds:
        # Calling the same devkit functions used by NuScenesEval verifies that
        # the metric and filtered prediction/GT inputs are aligned.
        metric_data = accumulate(
            evaluator.gt_boxes, evaluator.pred_boxes, args.target_class,
            evaluator.cfg.dist_fcn_callable, threshold)
        replay_ap = calc_ap(
            metric_data, dataset.eval_detection_configs.min_recall,
            dataset.eval_detection_configs.min_precision)
        official_ap = float(metrics['label_aps'][args.target_class][
            str(threshold)])
        ap_agrees = abs(replay_ap - official_ap) <= 1e-6
        if not ap_agrees:
            raise AssertionError(
                'Official AP mismatch at {} m: replay={} summary={}'.format(
                    threshold, replay_ap, official_ap))
        matched, false_positives = replay_match(
            evaluator, args.target_class, threshold)
        rows = gt_rows(
            evaluator, args.target_class, threshold,
            args.nearby_distance, matched)
        all_rows.extend(rows)
        counts = Counter(row['evidence'] for row in rows)
        fn_count = sum(row['status'] == 'FN' for row in rows)
        summary['distance_thresholds'][str(threshold)] = {
            'official_ap': official_ap,
            'replay_ap': replay_ap,
            'replay_agrees_with_official_ap': ap_agrees,
            'true_positives': len(matched),
            'false_negatives': fn_count,
            'false_positives': false_positives,
            'miss_evidence_counts': {
                name: counts[name] for name in EVIDENCE if name != 'matched'},
        }
        print('{:<8g} {:>9.4f} {:>8} {:>8} {:>8} {:>12}'.format(
            threshold, official_ap, len(matched), fn_count,
            false_positives, 'OK'))
        print('  Miss evidence: {}'.format(', '.join(
            '{}={}'.format(name, counts[name])
            for name in EVIDENCE if name != 'matched')))

    rows_path = output_dir / 'gt_match_analysis.csv'
    with rows_path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=GT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = output_dir / 'ap_aligned_summary.json'
    with summary_path.open('w') as handle:
        json.dump(summary, handle, indent=2)
    print('Official metrics:', metrics_path)
    print('Per-GT matches:', rows_path)
    print('Summary:', summary_path)
    print('Miss-evidence categories are post-hoc clues, not AP-loss fractions.')


if __name__ == '__main__':
    main()
