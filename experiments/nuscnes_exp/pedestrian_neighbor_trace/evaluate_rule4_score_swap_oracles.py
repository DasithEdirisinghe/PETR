#!/usr/bin/env python3
"""Run official nuScenes evaluation for Rule-4 score-swap submissions."""

import argparse
import csv
import json
from pathlib import Path


VARIANTS = ('baseline', 'case1', 'case2', 'combined')
THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')


def class_mean_ap(metrics, class_name):
    return sum(metrics['label_aps'][class_name][threshold]
               for threshold in THRESHOLDS)/len(THRESHOLDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                    verbose=False)
    config = config_factory('detection_cvpr_2019')
    metrics_by_variant = {}
    build_summary = json.loads((args.model_dir/'build_summary.json').read_text())
    analyzed_classes = tuple(build_summary.get(
        'classes', ('pedestrian', 'car', 'bicycle')))
    for variant in VARIANTS:
        directory = args.model_dir/variant
        result_path = directory/'results_nusc.json'
        metrics_path = directory/'metrics_summary.json'
        if not result_path.is_file():
            parser.error('Missing submission: {}'.format(result_path))
        if args.force or not metrics_path.is_file():
            evaluator = NuScenesEval(
                nusc, config=config, result_path=str(result_path), eval_set='val',
                output_dir=str(directory), verbose=False)
            evaluator.main(render_curves=False)
        metrics_by_variant[variant] = json.loads(metrics_path.read_text())
        print('{} {} mAP {:.6f} NDS {:.6f}'.format(
            args.model_name, variant,
            metrics_by_variant[variant]['mean_ap'],
            metrics_by_variant[variant]['nd_score']), flush=True)

    baseline = metrics_by_variant['baseline']
    detail_rows = []
    class_rows = []
    overall_rows = []
    for variant in VARIANTS:
        metrics = metrics_by_variant[variant]
        for class_name in config.class_names:
            for threshold in THRESHOLDS:
                value = metrics['label_aps'][class_name][threshold]
                base = baseline['label_aps'][class_name][threshold]
                detail_rows.append({
                    'model': args.model_name, 'variant': variant,
                    'class': class_name, 'threshold_m': threshold,
                    'ap': value, 'baseline_ap': base, 'delta_ap': value-base,
                })
            value = class_mean_ap(metrics, class_name)
            base = class_mean_ap(baseline, class_name)
            class_rows.append({
                'model': args.model_name, 'variant': variant,
                'class': class_name, 'mean_ap': value,
                'baseline_mean_ap': base, 'delta_mean_ap': value-base,
            })
        analyzed_map = sum(class_mean_ap(metrics, name)
                           for name in analyzed_classes)/len(analyzed_classes)
        baseline_analyzed_map = sum(class_mean_ap(baseline, name)
                                    for name in analyzed_classes)/len(analyzed_classes)
        swaps = 0 if variant == 'baseline' else \
            build_summary['counts'][variant]['total']
        overall_rows.append({
            'model': args.model_name, 'variant': variant, 'swaps': swaps,
            'analyzed_class_count': len(analyzed_classes),
            'analyzed_classes_map': analyzed_map,
            'baseline_analyzed_classes_map': baseline_analyzed_map,
            'delta_analyzed_classes_map': analyzed_map-baseline_analyzed_map,
            'official_mean_ap': metrics['mean_ap'],
            'baseline_official_mean_ap': baseline['mean_ap'],
            'delta_official_mean_ap': metrics['mean_ap']-baseline['mean_ap'],
            'nds': metrics['nd_score'],
            'baseline_nds': baseline['nd_score'],
            'delta_nds': metrics['nd_score']-baseline['nd_score'],
        })

    for filename, rows in (('threshold_ap.csv', detail_rows),
                           ('class_mean_ap.csv', class_rows),
                           ('overall.csv', overall_rows)):
        with (args.model_dir/filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.model_dir/'evaluation_summary.json').write_text(json.dumps({
        'model': args.model_name, 'threshold_ap': detail_rows,
        'class_mean_ap': class_rows, 'overall': overall_rows,
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
