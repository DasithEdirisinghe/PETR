#!/usr/bin/env python3
"""Compare realized baseline TP IoU with direct 3D-IoU-oracle TP IoU."""
import argparse
import csv
import statistics
from pathlib import Path

from all_class_reranking_oracle import class_boxes
from analyze import overlaps
from localization_capacity import THRESHOLDS
from reranking_oracle import greedy_match_records, iou3d_oracle_order


def model_arg(value):
    name, sep, path = value.partition('=')
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError('Expected NAME=/path/to/results_nusc.json')
    return name, Path(path)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.mean(values) if values else float('nan')


def median(values):
    return statistics.median(values) if values else float('nan')


def main():
    base = Path(__file__).resolve().parent
    workspace = base.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=model_arg, action='append', required=True,
                        help='NAME=/path/to/results_nusc.json; repeat')
    parser.add_argument('--data-root', type=Path,
                        default=workspace/'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path,
                        default=base/'outputs/tp_iou3d_comparison')
    args = parser.parse_args()
    if len({name for name, _ in args.model}) != len(args.model):
        parser.error('Model names must be unique')
    for _, path in args.model:
        if not path.is_file():
            parser.error('Missing model predictions: '+str(path))
    if not (args.data_root/'v1.0-trainval').is_dir():
        parser.error('Missing nuScenes metadata: '+str(args.data_root))

    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    cfg = config_factory('detection_cvpr_2019')
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in args.model:
        evaluator = NuScenesEval(nusc, config=cfg, result_path=str(path),
                                 eval_set='val', output_dir=str(args.output_dir/name/'source_eval'),
                                 verbose=False)
        for class_name in cfg.class_names:
            gt, _, predictions = class_boxes(evaluator, class_name)
            # Reuse filtered boxes once per class for all four thresholds.
            for threshold in THRESHOLDS:
                rows.append(compare_boxes(gt, predictions, name, class_name, threshold))
            print(name, class_name, 'done', flush=True)
    write_csv(args.output_dir/'summary.csv', rows)
    print('Saved', args.output_dir/'summary.csv')


def compare_boxes(gt, predictions, model, class_name, threshold):
    baseline = greedy_match_records(gt, predictions, range(len(predictions)), threshold)
    _, selected = iou3d_oracle_order(gt, predictions, threshold)
    oracle = {(predictions[pi].sample_token, match[0]): (pi, match[1])
              for pi, match in selected.items()}
    if len(oracle) != len(selected):
        raise AssertionError('Oracle reused a GT')
    if len({pi for pi, _ in oracle.values()}) != len(oracle):
        raise AssertionError('Oracle reused a prediction')
    common = baseline.keys() & oracle.keys()
    baseline_iou = {key: overlaps(predictions[match[0]], gt[key[0]][key[1]])[1]
                    for key, match in baseline.items()}
    oracle_iou = {key: overlaps(predictions[match[0]], gt[key[0]][key[1]])[1]
                  for key, match in oracle.items()}
    paired_delta = [oracle_iou[key] - baseline_iou[key] for key in common]
    tp_box_iou = [overlaps(predictions[baseline[key][0]],
                           predictions[oracle[key][0]])[1] for key in common]
    row = dict(model=model, class_name=class_name, threshold_m=threshold,
               gt_count=sum(len(boxes) for boxes in gt.values()),
               baseline_tp_count=len(baseline), oracle_tp_count=len(oracle),
               common_gt_count=len(common),
               baseline_mean_iou3d=mean(list(baseline_iou.values())),
               oracle_mean_iou3d=mean(list(oracle_iou.values())),
               baseline_median_iou3d=median(list(baseline_iou.values())),
               oracle_median_iou3d=median(list(oracle_iou.values())),
               common_baseline_mean_iou3d=mean([baseline_iou[k] for k in common]),
               common_oracle_mean_iou3d=mean([oracle_iou[k] for k in common]),
               baseline_oracle_tp_box_mean_iou3d=mean(tp_box_iou),
               baseline_oracle_tp_box_median_iou3d=median(tp_box_iou),
               same_prediction_count=sum(baseline[k][0] == oracle[k][0]
                                         for k in common),
               paired_mean_delta_iou3d=mean(paired_delta),
               paired_median_delta_iou3d=median(paired_delta),
               paired_improved_count=sum(x > 1e-9 for x in paired_delta),
               paired_worsened_count=sum(x < -1e-9 for x in paired_delta),
               baseline_zero_iou_count=sum(x <= 1e-9 for x in baseline_iou.values()),
               oracle_zero_iou_count=sum(x <= 1e-9 for x in oracle_iou.values()))
    return row


if __name__ == '__main__':
    main()
