#!/usr/bin/env python3
"""Run the official nuScenes detection evaluator on train-split predictions."""

import argparse
import json
from pathlib import Path


def normalize_train_submission(nusc, cfg, source_path, output_dir):
    """Complete a train submission with empty entries for omitted samples.

    PETR info files can omit samples that do not satisfy their preprocessing
    policy.  nuScenes evaluation nevertheless requires one result-list entry
    for every official split token.  Extra tokens are never accepted because
    they indicate that inference used the wrong split.
    """
    from nuscenes.eval.common.loaders import load_gt, load_prediction
    from nuscenes.eval.detection.data_classes import DetectionBox

    predictions, _ = load_prediction(
        str(source_path), cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    ground_truth = load_gt(nusc, 'train', DetectionBox, verbose=False)
    predicted = set(predictions.sample_tokens)
    expected = set(ground_truth.sample_tokens)
    missing = sorted(expected-predicted)
    extra = sorted(predicted-expected)
    report = {
        'eval_set': 'train',
        'expected_sample_count': len(expected),
        'prediction_sample_count': len(predicted),
        'missing_sample_count': len(missing),
        'extra_sample_count': len(extra),
        'missing_sample_examples': missing[:20],
        'extra_sample_examples': extra[:20],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'sample_token_audit.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print('Train token audit: expected {:,}, predicted {:,}, missing {:,}, extra {:,}'
          .format(len(expected), len(predicted), len(missing), len(extra)),
          flush=True)
    if extra:
        raise ValueError(
            'Prediction JSON contains {} tokens outside the official train split. '
            'The inference configuration used the wrong annotations. Examples: {}'
            .format(len(extra), ', '.join(extra[:5])))
    if not missing:
        return source_path

    submission = json.loads(source_path.read_text())
    for token in missing:
        submission['results'][token] = []
    normalized = output_dir/'results_nusc_train_complete.json'
    normalized.write_text(json.dumps(submission))
    print('Added empty predictions for {:,} omitted train samples: {}'.format(
        len(missing), normalized), flush=True)
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not args.results_json.is_file():
        parser.error('Missing train predictions: {}'.format(args.results_json))
    if not (args.data_root/'v1.0-trainval').is_dir():
        parser.error('Invalid nuScenes data root: {}'.format(args.data_root))

    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                    verbose=False)
    cfg = config_factory('detection_cvpr_2019')
    evaluation_results = normalize_train_submission(
        nusc, cfg, args.results_json, args.output_dir)
    (args.output_dir/'evaluation_results_path.txt').write_text(
        str(evaluation_results.resolve())+'\n')

    metrics_path = args.output_dir/'metrics_summary.json'
    if args.force or not metrics_path.is_file():
        evaluator = NuScenesEval(
            nusc, config=cfg, result_path=str(evaluation_results), eval_set='train',
            output_dir=str(args.output_dir), verbose=False)
        evaluator.main(render_curves=False)

    metrics = json.loads(metrics_path.read_text())
    print('nuScenes train diagnostic: mAP {:.4f}, NDS {:.4f}'.format(
        metrics['mean_ap'], metrics['nd_score']))
    print('Evaluated submission: {}'.format(evaluation_results))
    print(metrics_path)


if __name__ == '__main__':
    main()

