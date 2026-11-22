#!/usr/bin/env python3
"""Count precise TPs displaced by a higher-score box at the 2 m threshold.

Run with repeated --model NAME=/path/to/results_nusc.json arguments. Inputs
must be nuScenes validation result files. Outputs are per-model/class CSV and
three SVG heatmaps, each with separate 0.5->2 m and 1->2 m columns.
"""
import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape

from all_class_reranking_oracle import class_boxes
from reranking_oracle import greedy_match_records

PAIRS = ((0.5, 2.0), (1.0, 2.0))


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


def analyze(evaluator, model, class_name):
    gt, _, predictions = class_boxes(evaluator, class_name)
    gt_count = sum(len(boxes) for boxes in gt.values())
    matches = {t: greedy_match_records(gt, predictions, range(len(predictions)), t)
               for t in (0.5, 1.0, 2.0)}
    total = len(predictions)
    rows, cases = [], []
    loose = matches[2.0]
    loose_tp_ids = {p[0] for p in loose.values()}
    for tight_threshold, loose_threshold in PAIRS:
        tight = matches[tight_threshold]
        selected = []
        for key, (precise_id, precise_error) in tight.items():
            claimant = loose.get(key)
            if claimant is None:
                continue
            coarse_id, coarse_error = claimant
            # The tight TP must itself be a loose FP; a different GT match
            # at 2 m does not satisfy this experiment's definition.
            if (coarse_id == precise_id or precise_id in loose_tp_ids or
                    not predictions[coarse_id].detection_score >
                    predictions[precise_id].detection_score):
                continue
            token, gi = key
            selected.append(key)
            cases.append(dict(model=model, class_name=class_name,
                              tight_threshold_m=tight_threshold,
                              loose_threshold_m=loose_threshold,
                              sample_token=token, gt_index=gi,
                              precise_prediction_index=precise_id,
                              claimant_prediction_index=coarse_id,
                              precise_score=predictions[precise_id].detection_score,
                              claimant_score=predictions[coarse_id].detection_score,
                              precise_center_error_m=precise_error,
                              claimant_center_error_m=coarse_error))
        n = len(selected)
        tight_tp = len(tight)
        loose_fp = total - len(loose)
        rows.append(dict(model=model, class_name=class_name,
                         tight_threshold_m=tight_threshold,
                         loose_threshold_m=loose_threshold,
                         switch_count=n, tight_tp_count=tight_tp,
                         loose_fp_count=loose_fp, gt_count=gt_count,
                         pct_of_tight_tp=100*n/tight_tp if tight_tp else 0,
                         pct_of_loose_fp=100*n/loose_fp if loose_fp else 0,
                         pct_of_gt=100*n/gt_count if gt_count else 0))
    return rows, cases


def plot(path, rows, models, classes, field, title, suffix):
    values = {(r['model'], r['class_name'], float(r['tight_threshold_m'])): r
              for r in rows}
    cell_w, cell_h, left, top = 170, 36, 175, 120
    width = left + cell_w * len(models) * 2 + 25
    height = top + cell_h * len(classes) + 50
    vmax = max((float(r[field]) for r in rows), default=0) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:21px;font-weight:bold}.label{font-size:13px}.small{font-size:11px;fill:#475569}</style>',
             f'<text x="22" y="32" class="title">{escape(title)}</text>',
             '<text x="22" y="56" class="small">Same GT; tight TP becomes a 2 m FP; higher-score box claims that GT at 2 m.</text>']
    for mi, model in enumerate(models):
        x = left + mi*2*cell_w
        parts.append(f'<text x="{x+cell_w}" y="81" text-anchor="middle" class="label">{escape(model)}</text>')
        for pi, (tight, _) in enumerate(PAIRS):
            parts.append(f'<text x="{x+(pi+.5)*cell_w}" y="104" text-anchor="middle" class="small">{tight:g} → 2 m</text>')
    for ci, cls in enumerate(classes):
        y = top + ci*cell_h
        parts.append(f'<text x="15" y="{y+23}" class="label">{escape(cls)}</text>')
        for mi, model in enumerate(models):
            for pi, (tight, _) in enumerate(PAIRS):
                r = values[(model, cls, tight)]
                value = float(r[field]); count = r['switch_count']
                strength = value/vmax
                # Light blue scale keeps black annotations legible.
                color = f'rgb({int(245-130*strength)},{int(249-95*strength)},{int(255-30*strength)})'
                x = left+(2*mi+pi)*cell_w
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color}"/>')
                if suffix == 'count':
                    label = str(count)
                else:
                    denominator = (r['tight_tp_count'] if field == 'pct_of_tight_tp'
                                   else r['loose_fp_count'] if field == 'pct_of_loose_fp'
                                   else r['gt_count'])
                    label = f'{value:.1f}% ({count}/{denominator})'
                parts.append(f'<text x="{x+cell_w/2}" y="{y+23}" text-anchor="middle" class="label">{label}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts)+'\n')


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', type=model_arg, required=True,
                        help='NAME=/path/to/results_nusc.json; repeat for each model')
    parser.add_argument('--data-root', type=Path,
                        default=root.parent/'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path,
                        default=base/'outputs/threshold_switch_duplicates')
    args = parser.parse_args()
    if len({name for name, _ in args.model}) != len(args.model):
        parser.error('Model names must be unique')
    for _, path in args.model:
        if not path.is_file():
            parser.error('Missing results: '+str(path))
    if not (args.data_root/'v1.0-trainval').is_dir():
        parser.error('Missing nuScenes trainval metadata: '+str(args.data_root))

    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    cfg = config_factory('detection_cvpr_2019')
    classes = list(cfg.class_names)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, cases = [], []
    for model, path in args.model:
        evaluator = NuScenesEval(nusc, config=cfg, result_path=str(path),
                                 eval_set='val', output_dir=str(args.output_dir/model/'source_eval'),
                                 verbose=False)
        for cls in classes:
            r, c = analyze(evaluator, model, cls)
            rows.extend(r); cases.extend(c)
    write_csv(args.output_dir/'summary.csv', rows)
    if cases:
        write_csv(args.output_dir/'cases.csv', cases)
    for field, title, suffix in (
            ('switch_count', 'Threshold-switch duplicate count', 'count'),
            ('pct_of_tight_tp', 'Switches as share of tight-threshold TPs', 'percent'),
            ('pct_of_loose_fp', 'Switches as share of 2 m FPs', 'percent'),
            ('pct_of_gt', 'GTs with a threshold-switch TP', 'percent')):
        plot(args.output_dir/(field+'.svg'), rows, [m for m, _ in args.model],
             classes, field, title, suffix)
    print('Wrote', args.output_dir/'summary.csv', 'and three SVG plots')


if __name__ == '__main__':
    main()
