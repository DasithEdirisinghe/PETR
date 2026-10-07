#!/usr/bin/env python3
"""Trace high-score R1 car FPs on R1-f val/test frames to PETR attention.

Read-only diagnostic on R1-f validation frames. Official AP decisions and
standard PETR inference are not changed. Region assignment is geometric and
does not establish which visible object caused a prediction.
"""

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/cross_rig_attention'))

from analyze_pair import feature_rays  # noqa: E402
from attention_region_metrics import REGIONS, region_labels, score_regions  # noqa: E402
from audit_common_tp_attention import decode_query_ids  # noqa: E402
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, prepare_one, relocate_dataset_paths)
from score_r1f_car_queries import (  # noqa: E402
    CHECKPOINTS, CONFIG, CaptureCrossAttention, attention_selected,
    build_petr_checkpoint)

MODEL, SHORT = 'R1', 'r1'
METRICS = ('cell_count', 'cell_fraction', 'attention_mass', 'attention_auroc')
FIELDS = ('sample_token', 'model', 'fp_proximity_category', 'rank',
          'prediction_index', 'query_index', 'car_score',
          'query_score_delta', 'query_center_delta_m', 'valid_cell_count',
          'multi_gt_ray_fraction',
          'camera_attention_mass_json') + tuple(
              '{}_{}'.format(region, metric) for region in REGIONS
              for metric in METRICS)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--test-set', choices=('r1f_val', 'r1f_test'),
                        default='r1f_val')
    parser.add_argument('--score-threshold', type=float, default=0.60)
    parser.add_argument('--box-expansion', type=float, default=1.0,
                        help='Metres added to every GT box half-size.')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Smoke test on first N selected frames.')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def load_cohort(path, threshold):
    """Select only official R1 car FPs, preserving final-output IDs."""
    selected = defaultdict(list)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if (row['model'] != MODEL or row['is_tp'] != '0' or
                    float(row['car_score']) < threshold):
                continue
            selected[row['sample_token']].append({
                'prediction_index': int(row['prediction_index']),
                'score': float(row['car_score']),
                'rank': int(row['rank'])})
    for token, entries in selected.items():
        indices = [entry['prediction_index'] for entry in entries]
        if len(indices) != len(set(indices)):
            raise ValueError('Repeated saved prediction in {}'.format(token))
    return selected


def proximity_categories(path):
    result = {}
    if not path.exists():
        return result
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            result[(row['model'], row['sample_token'],
                    int(row['prediction_index']))] = row['category']
    return result


def annotated_boxes(info):
    """Use all annotated LiDAR boxes, including those outside AP eligibility.

    PCCR info boxes use bottom-centre z; the ray test expects gravity-centre z.
    Unknown annotated classes remain in `other_object`.
    """
    boxes = np.asarray(info['gt_boxes'], dtype=np.float64).copy()
    names = np.asarray(info['gt_names'])
    if len(boxes) != len(names):
        raise ValueError('Raw GT box/name count mismatch')
    if len(boxes):
        boxes[:, 2] += boxes[:, 5]*0.5
    return boxes, names


def trace_frame(model, dataset, index, entries, saved_boxes, device,
                expansion, categories):
    selected_boxes = []
    for entry in entries:
        box = saved_boxes[entry['prediction_index']]
        if box['detection_name'] != 'car' or abs(
                box['detection_score']-entry['score']) > 1e-4:
            raise ValueError('Rank ledger differs from saved final car prediction')
        selected_boxes.append(box)
    data = prepare_one(dataset, index,
                       device.index if device.type == 'cuda' else device)
    images, metas = first_augmentation(data)
    head = model.pts_bbox_head
    if head.bbox_coder.__class__.__name__ != 'NMSFreeCoder':
        raise ValueError('Expected standard PETR NMSFreeCoder')
    with torch.no_grad():
        features = model.extract_feat(img=images, img_metas=metas)
        with CaptureCrossAttention(head) as capture:
            output = head(features, metas)
        queries, query_audit = decode_query_ids(
            head, model.CLASSES, output, selected_boxes,
            dataset.data_infos[index], metas, dataset)
        attention, valid = attention_selected(
            head.transformer.decoder.layers[-1].attentions[1],
            capture.calls[6], queries)
    info = dataset.data_infos[index]
    layout = features[0].shape
    source = {'key_layout': {'num_cameras': int(layout[1]),
                             'height': int(layout[-2]), 'width': int(layout[-1])},
              'lidar2img': metas[0]['lidar2img'],
              'pad_shape': metas[0]['pad_shape']}
    origins, directions, camera_ids = feature_rays(source)
    boxes, names = annotated_boxes(info)
    labels, multi_gt = region_labels(origins, directions, boxes, names, expansion)
    if len(labels) != len(valid) or not valid.any():
        raise ValueError('Feature-ray/attention token layout mismatch')
    rows = []
    for position, entry in enumerate(entries):
        weights = attention[position]
        scores = score_regions(weights, valid, labels)
        total = float(weights[valid].sum())
        camera_mass = [float(weights[valid & (camera_ids == camera)].sum()/total)
                       for camera in range(int(layout[1]))]
        row = {'sample_token': info['token'], 'model': MODEL,
               'fp_proximity_category': categories.get(
                   (MODEL, info['token'], entry['prediction_index']), ''),
               'rank': entry['rank'],
               'prediction_index': entry['prediction_index'],
               'query_index': int(queries[position]),
               'car_score': entry['score'],
               'query_score_delta': query_audit[position]['score_delta'],
               'query_center_delta_m': query_audit[position]['center_delta_m'],
               'valid_cell_count': int(valid.sum()),
               'multi_gt_ray_fraction': float((multi_gt & valid).sum()/valid.sum()),
               'camera_attention_mass_json': json.dumps(camera_mass)}
        for region in REGIONS:
            for metric in METRICS:
                row['{}_{}'.format(region, metric)] = scores[region][metric]
        rows.append(row)
    return rows


def median_present(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None]
    return statistics.median(values) if values else None


def group_summary(rows):
    result = {'n': len(rows), 'median_car_score': median_present(rows, 'car_score'),
              'fp_proximity_categories': dict(Counter(
                  row['fp_proximity_category'] for row in rows
                  if row['fp_proximity_category']))}
    result['regions'] = {}
    for region in REGIONS:
        values = {'n_auroc_defined': sum(
            row[region+'_attention_auroc'] is not None for row in rows)}
        for metric in METRICS:
            if metric != 'cell_count':
                values['median_{}'.format(metric)] = median_present(
                    rows, '{}_{}'.format(region, metric))
        result['regions'][region] = values
    return result


def summarize(rows, full_cohort, frames, options):
    fp_categories = {}
    for category in sorted({row['fp_proximity_category'] for row in rows
                            if row['fp_proximity_category']}):
        fp_categories[category] = group_summary([
            row for row in rows if row['fp_proximity_category'] == category])
    expected = sum(len(entries) for entries in full_cohort.values())
    return {
        'cohort': 'Official R1 car FPs on R1-f {} frames'.format(
            'validation' if options.test_set == 'r1f_val' else 'test'),
        'test_set': options.test_set,
        'score_threshold': options.score_threshold,
        'box_expansion_m': options.box_expansion,
        'processed_frames': len(frames), 'full_cohort_fp_count': expected,
        'traced_fp_count': len(rows),
        'is_complete': options.max_frames is None,
        'region_definition': 'Nearest positive intersection of feature-cell '
            'camera ray with any expanded annotated 3D GT box: car, other '
            'annotated object, or no annotated GT. Overlapping boxes take '
            'nearest entry depth; this is not a visibility oracle.',
        'attention_definition': 'PETR decoder layer 6 cross-attention, mean over '
            'heads. AUROC ranks region-cell weights against all other valid '
            'cells; 0.5=no ranking preference, 1=all region cells higher. '
            'Attention mass and cell fraction are interpretation checks. '
            'Undefined AUROC is null '
            'when the region or its complement has no valid cells.',
        'caveat': 'GT-box rays are not pixel masks; no annotated GT does not '
            'guarantee empty background. Attention is descriptive, not causal.',
        'all_r1_fps': group_summary(rows),
        'r1_fp_proximity_groups': fp_categories}


def plot(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ('Car GT', 'Other GT', 'No annotated GT')
    colours = ('#d48a21', '#357cb3', '#728192')
    group = summary['all_r1_fps']
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for axis, metric, ylabel in zip(
            axes, ('median_attention_auroc', 'median_attention_mass'),
            ('Attention AUROC (0.5 = no preference)',
             'Fraction of total attention mass')):
        values = [group['regions'][region][metric] for region in REGIONS]
        axis.bar(np.arange(3),
                 [np.nan if value is None else value for value in values],
                 color=colours)
        axis.set_xticks(np.arange(3))
        axis.set_xticklabels(names, rotation=18, ha='right')
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.2)
        axis.set_ylim(0, 1)
        if metric == 'median_attention_auroc':
            axis.axhline(.5, color='#555', linestyle='--', linewidth=1)
    figure.suptitle('R1 car FPs on R1-f frames: last-layer attention '
                    '(score >= {:.2f}, n={}{}).'.format(
                        summary['score_threshold'],
                        summary['traced_fp_count'],
                        '' if summary['is_complete'] else '; partial smoke test'))
    figure.tight_layout()
    figure.savefig(str(destination.with_suffix('.png')), dpi=180)
    figure.savefig(str(destination.with_suffix('.svg')))
    plt.close(figure)


def plot_fp_categories(summary, destination):
    """Show whether the five R1 FP proximity groups attend differently."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    categories = summary['r1_fp_proximity_groups']
    if not categories:
        return
    aliases = {
        'no_evaluator_gt_within_near_radius': 'No GT <8 m',
        'car_label_near_car_outside_4m': 'Near car 4–8 m',
        'car_label_near_other_outside_4m': 'Near other 4–8 m',
        'car_label_on_other_class_gt_candidate': 'Other GT <4 m',
        'duplicate_or_competing_car': 'Competing car'}
    ordered = sorted(categories, key=lambda name: (-categories[name]['n'], name))
    region_names = ('Car GT rays', 'Other GT rays', 'No annotated GT rays')
    colours = ('#d48a21', '#357cb3', '#728192')
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    x = np.arange(len(ordered))
    for axis, metric, ylabel in zip(
            axes, ('median_attention_auroc', 'median_attention_mass'),
            ('Attention AUROC (0.5 = no preference)',
             'Fraction of total attention mass')):
        for j, (region, title, colour) in enumerate(zip(REGIONS, region_names,
                                                        colours)):
            values = [categories[name]['regions'][region][metric]
                      for name in ordered]
            axis.bar(x+(j-1)*.25,
                     [np.nan if value is None else value for value in values],
                     width=.24, label=title, color=colour)
        axis.set_xticks(x)
        axis.set_xticklabels(['{}\n(n={})'.format(aliases.get(name, name),
                                              categories[name]['n'])
                              for name in ordered], fontsize=9)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.grid(axis='y', alpha=.2)
        if metric == 'median_attention_auroc':
            axis.axhline(.5, color='#555', linestyle='--', linewidth=1)
    axes[0].legend(loc='upper left', fontsize=8)
    figure.suptitle('R1 high-confidence car FPs by existing GT-proximity category')
    figure.tight_layout()
    figure.savefig(str(destination.with_suffix('.png')), dpi=180)
    figure.savefig(str(destination.with_suffix('.svg')))
    plt.close(figure)


def main():
    options = arguments()
    if not (0 <= options.score_threshold <= 1) or options.box_expansion < 0 or (
            options.max_frames is not None and options.max_frames < 1):
        raise ValueError('Invalid score threshold, box expansion or frame limit')
    base = ROOT / 'experiments/cross_rig_attention/output' / options.test_set
    split = 'val' if options.test_set == 'r1f_val' else 'test'
    rank_dir = 'car_ap_rank_audit' if split == 'val' else 'car_ap_fp_audit'
    rank = base / rank_dir / 'ranked_car_predictions.csv'
    if split == 'val':
        saved_path = base / MODEL / 'formatted/results_pccr.json'
        category_path = base / 'dnd3d_car_4m/R1_car_fp_before_50pct.csv'
    else:
        saved_path = (ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr' /
                      MODEL / 'cross_rig/evaluations/R1-f/formatted/results_pccr.json')
        category_path = base / rank_dir / 'R1_high_score_car_fps.csv'
    if not rank.is_file() or not saved_path.is_file():
        raise FileNotFoundError('Missing official FP rank ledger or saved final '
                                'predictions: {}, {}'.format(rank, saved_path))
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/R1-f/'
    definition.ann_file = 'data/pccr/R1-f/R1-f_infos_{}.pkl'.format(split)
    definition.test_mode = True
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    with rank.with_name('summary.json').open() as handle:
        rank_summary = json.load(handle)
    if rank_summary['rig'] != 'R1-f' or rank_summary['threshold_m'] != 4.0:
        raise ValueError('Expected official R1-f car AP@4m rank ledger')
    ap = rank_summary['models'][MODEL]
    if abs(ap['replayed_car_ap']-ap['official_car_ap']) > 1e-8:
        raise ValueError('Rank ledger AP differs from official evaluator')
    cohort = load_cohort(rank, options.score_threshold)
    token_to_index = {info['token']: index
                      for index, info in enumerate(dataset.data_infos)}
    tokens = sorted(cohort)
    if not tokens or not set(tokens) <= set(token_to_index):
        raise ValueError('Selected AP ledger frames absent from dataset')
    tokens = tokens[:options.max_frames]
    categories = proximity_categories(category_path)
    output = (options.output_dir or base / 'r1_fp_attention_regions').resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(options.device)
    checkpoint_stat = CHECKPOINTS[SHORT].stat()
    ledger_stat = rank.stat()
    signature = {'schema': 2, 'model': MODEL,
                 'checkpoint_size': checkpoint_stat.st_size,
                 'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
                 'rank_size': ledger_stat.st_size,
                 'rank_mtime_ns': ledger_stat.st_mtime_ns,
                 'score_threshold': options.score_threshold,
                 'box_expansion_m': options.box_expansion}
    if split == 'test':
        signature['test_set'] = options.test_set
    model = build_petr_checkpoint(cfg, dataset, SHORT, device)
    with saved_path.open() as handle:
        saved = json.load(handle)['results']
    for count, token in enumerate(tokens, 1):
        case_path = output / 'cases' / SHORT / (token+'.json')
        if case_path.exists() and not options.force:
            with case_path.open() as handle:
                case = json.load(handle)
            if case['signature'] != signature or len(case['rows']) != len(
                    cohort[token]):
                raise ValueError('Cached case settings/cohort differ: {}'.format(
                    case_path))
            continue
        rows = trace_frame(model, dataset, token_to_index[token], cohort[token],
                           saved[token], device, options.box_expansion,
                           categories)
        case_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = case_path.with_suffix('.partial.json')
        with temporary.open('w') as handle:
            json.dump({'signature': signature, 'rows': rows}, handle)
        os.replace(str(temporary), str(case_path))
        if count % 50 == 0:
            print('{}: {}/{} frames'.format(MODEL, count, len(tokens)), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    rows = []
    for token in tokens:
        with (output / 'cases' / SHORT / (token+'.json')).open() as handle:
            rows.extend(json.load(handle)['rows'])
    expected_count = sum(len(cohort[token]) for token in tokens)
    if len(rows) != expected_count:
        raise ValueError('Traced output count mismatch: {} vs {}'.format(
            len(rows), expected_count))
    with (output / 'per_prediction.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, cohort, tokens, options)
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    plot(summary, output / 'r1_fp_attention_regions')
    plot_fp_categories(summary, output / 'r1_fp_category_attention_regions')
    print('Saved', output / 'per_prediction.csv')
    print('Saved', output / 'summary.json')
    print('Saved', output / 'r1_fp_attention_regions.png')
    if summary['r1_fp_proximity_groups']:
        print('Saved', output / 'r1_fp_category_attention_regions.png')


if __name__ == '__main__':
    main()
