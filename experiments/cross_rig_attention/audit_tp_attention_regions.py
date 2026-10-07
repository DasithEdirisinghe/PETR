#!/usr/bin/env python3
"""Trace official car TPs to last-layer PETR attention on annotated GT regions.

Select checkpoint, evaluation rig/split, and minimum saved car score via CLI.
The AP ledger defines TPs; unmodified PETR forward passes recover their query
attention. No inference correction or alternative GT matching is applied.
"""

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
from collections import defaultdict
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

MODEL_LABELS = {'r1': 'R1', 'r1f': 'R1-f'}
TEST_SETS = {
    'r1_val': ('R1', 'val'), 'r1f_val': ('R1-f', 'val'),
    'r1_test': ('R1', 'test'), 'r1f_test': ('R1-f', 'test')}
METRICS = ('cell_count', 'cell_fraction', 'attention_mass', 'attention_auroc')
FIELDS = ('sample_token', 'model', 'test_set', 'rank', 'prediction_index',
          'matched_gt_index', 'matched_distance_m', 'query_index', 'car_score',
          'query_score_delta', 'query_center_delta_m', 'valid_cell_count',
          'multi_gt_ray_fraction', 'camera_attention_mass_json') + tuple(
              '{}_{}'.format(region, metric) for region in REGIONS
              for metric in METRICS)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=sorted(MODEL_LABELS),
                        help='Checkpoint: r1 or r1f.')
    parser.add_argument('--test-set', required=True, choices=sorted(TEST_SETS),
                        help='Evaluation frames/rig. A labelled AP ledger is required.')
    parser.add_argument('--score-threshold', type=float, default=0.0,
                        help='Minimum saved car sigmoid score; default all TPs.')
    parser.add_argument('--box-expansion', type=float, default=1.0,
                        help='Metres added to each GT box half-size.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Smoke test on first N selected frames.')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def load_tp_cohort(path, model_label, threshold):
    selected = defaultdict(list)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if (row['model'] != model_label or row['is_tp'] != '1' or
                    float(row['car_score']) < threshold):
                continue
            selected[row['sample_token']].append({
                'prediction_index': int(row['prediction_index']),
                'rank': int(row['rank']),
                'score': float(row['car_score']),
                'matched_gt_index': int(row['matched_gt_index']),
                'matched_distance_m': float(row['matched_distance_m'])})
    for token, entries in selected.items():
        indices = [item['prediction_index'] for item in entries]
        if len(indices) != len(set(indices)):
            raise ValueError('Duplicate TP prediction in {}'.format(token))
    return selected


def annotated_boxes(info):
    """All info annotations, including objects excluded from AP eligibility."""
    boxes = np.asarray(info['gt_boxes'], dtype=np.float64).copy()
    names = np.asarray(info['gt_names'])
    if len(boxes) != len(names):
        raise ValueError('Raw GT box/name count mismatch')
    if len(boxes):
        boxes[:, 2] += boxes[:, 5]*0.5  # bottom z to gravity centre
    return boxes, names


def trace_frame(model, dataset, index, entries, saved_boxes, device,
                expansion, model_label, test_set):
    selected_boxes = []
    for entry in entries:
        box = saved_boxes[entry['prediction_index']]
        if (box['detection_name'] != 'car' or
                abs(box['detection_score']-entry['score']) > 1e-4):
            raise ValueError('AP rank ledger differs from saved final prediction')
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
        region_scores = score_regions(weights, valid, labels)
        total = float(weights[valid].sum())
        camera_mass = [float(weights[valid & (camera_ids == camera)].sum()/total)
                       for camera in range(int(layout[1]))]
        row = {'sample_token': info['token'], 'model': model_label,
               'test_set': test_set, 'rank': entry['rank'],
               'prediction_index': entry['prediction_index'],
               'matched_gt_index': entry['matched_gt_index'],
               'matched_distance_m': entry['matched_distance_m'],
               'query_index': int(queries[position]),
               'car_score': entry['score'],
               'query_score_delta': query_audit[position]['score_delta'],
               'query_center_delta_m': query_audit[position]['center_delta_m'],
               'valid_cell_count': int(valid.sum()),
               'multi_gt_ray_fraction': float((multi_gt & valid).sum()/valid.sum()),
               'camera_attention_mass_json': json.dumps(camera_mass)}
        for region in REGIONS:
            for metric in METRICS:
                row['{}_{}'.format(region, metric)] = region_scores[region][metric]
        rows.append(row)
    return rows


def median_defined(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None]
    return statistics.median(values) if values else None


def summarize(rows, cohort, tokens, options, official_ap):
    regions = {}
    for region in REGIONS:
        regions[region] = {
            'n_auroc_defined': sum(row[region+'_attention_auroc'] is not None
                                   for row in rows),
            'median_attention_auroc': median_defined(
                rows, region+'_attention_auroc'),
            'median_attention_mass': median_defined(
                rows, region+'_attention_mass'),
            'median_cell_fraction': median_defined(
                rows, region+'_cell_fraction')}
    return {
        'cohort': 'Official car TP@4m final predictions',
        'model': MODEL_LABELS[options.model], 'test_set': options.test_set,
        'official_car_ap_4m': official_ap,
        'score_threshold': options.score_threshold,
        'box_expansion_m': options.box_expansion,
        'full_selected_tp_count': sum(len(items) for items in cohort.values()),
        'traced_tp_count': len(rows), 'processed_frames': len(tokens),
        'is_complete': options.max_frames is None,
        'region_definition': 'Each valid image-feature-cell ray belongs to its '
            'nearest expanded annotated GT 3D box: car, other object, or no '
            'annotated GT. Multiple intersections use nearest entry depth.',
        'attention_definition': 'Last decoder layer, mean over heads, all '
            'cameras. One-vs-rest AUROC ranks region weights against the '
            'complement: 0.5=no preference, 1=all region cells higher. '
            'Undefined when either side has zero cells.',
        'caveat': 'No annotated GT is not proof of empty background; box rays '
            'are not pixel segmentations. Attention is descriptive, not causal.',
        'regions': regions}


def plot(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ('Car GT', 'Other GT', 'No annotated GT')
    colours = ('#d48a21', '#357cb3', '#728192')
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for axis, key, ylabel in zip(
            axes, ('median_attention_auroc', 'median_attention_mass'),
            ('Attention AUROC (0.5 = no preference)',
             'Fraction of total attention mass')):
        values = [summary['regions'][region][key] for region in REGIONS]
        axis.bar(np.arange(3),
                 [np.nan if value is None else value for value in values],
                 color=colours)
        axis.set_xticks(np.arange(3))
        axis.set_xticklabels(labels, rotation=18, ha='right')
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.grid(axis='y', alpha=.2)
        if key == 'median_attention_auroc':
            axis.axhline(.5, color='#555', linestyle='--', linewidth=1)
    figure.suptitle('{} TPs on {} (score >= {:.2f}; n={}{}).'.format(
        summary['model'], summary['test_set'], summary['score_threshold'],
        summary['traced_tp_count'],
        '' if summary['is_complete'] else '; partial smoke test'))
    figure.tight_layout()
    figure.savefig(str(destination.with_suffix('.png')), dpi=180)
    figure.savefig(str(destination.with_suffix('.svg')))
    plt.close(figure)


def main():
    options = arguments()
    if not (0 <= options.score_threshold <= 1) or options.box_expansion < 0 or (
            options.max_frames is not None and options.max_frames < 1):
        raise ValueError('Invalid score threshold, box expansion or frame limit')
    rig_name, split = TEST_SETS[options.test_set]
    base = ROOT / 'experiments/cross_rig_attention/output' / options.test_set
    rank_dir = 'car_ap_rank_audit' if split == 'val' else 'car_ap_fp_audit'
    rank = base / rank_dir / 'ranked_car_predictions.csv'
    rank_summary_path = rank.with_name('summary.json')
    if split == 'val':
        saved_path = (base / MODEL_LABELS[options.model] /
                      'formatted/results_pccr.json')
    else:
        saved_path = (ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr' /
                      MODEL_LABELS[options.model] / 'cross_rig/evaluations' /
                      rig_name / 'formatted/results_pccr.json')
    if (not rank.is_file() or not rank_summary_path.is_file() or
            not saved_path.is_file()):
        raise FileNotFoundError(
            'Selected test set needs official car AP@4m rank ledger, its '
            'summary, and saved final predictions: {}, {}, {}. No replacement '
            'matching is used.'.format(rank, rank_summary_path, saved_path))
    output = (options.output_dir or
              base / 'tp_attention_regions' / options.model).resolve()
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/{}/'.format(rig_name)
    definition.ann_file = 'data/pccr/{}/{}_infos_{}.pkl'.format(
        rig_name, rig_name, split)
    definition.test_mode = True
    if not (ROOT / definition.ann_file).is_file():
        raise FileNotFoundError('Selected dataset info is absent: {}'.format(
            definition.ann_file))
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    with rank_summary_path.open() as handle:
        rank_summary = json.load(handle)
    if rank_summary['rig'] != rig_name or rank_summary['threshold_m'] != 4.0:
        raise ValueError('Expected official {} car AP@4m rank ledger'.format(
            rig_name))
    ap = rank_summary['models'][MODEL_LABELS[options.model]]
    if abs(ap['replayed_car_ap']-ap['official_car_ap']) > 1e-8:
        raise ValueError('Rank-ledger replay differs from official AP')
    cohort = load_tp_cohort(rank, MODEL_LABELS[options.model],
                            options.score_threshold)
    token_to_index = {info['token']: index
                      for index, info in enumerate(dataset.data_infos)}
    tokens = sorted(cohort)
    if not tokens or not set(tokens) <= set(token_to_index):
        raise ValueError('Selected TPs absent from dataset frames')
    tokens = tokens[:options.max_frames]
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(options.device)
    checkpoint_stat = CHECKPOINTS[options.model].stat()
    rank_stat = rank.stat()
    signature = {'schema': 1, 'model': options.model,
                 'test_set': options.test_set,
                 'checkpoint_size': checkpoint_stat.st_size,
                 'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
                 'rank_size': rank_stat.st_size,
                 'rank_mtime_ns': rank_stat.st_mtime_ns,
                 'score_threshold': options.score_threshold,
                 'box_expansion_m': options.box_expansion}
    model = build_petr_checkpoint(cfg, dataset, options.model, device)
    with saved_path.open() as handle:
        saved = json.load(handle)['results']
    for count, token in enumerate(tokens, 1):
        case_path = output / 'cases' / (token+'.json')
        if case_path.exists() and not options.force:
            with case_path.open() as handle:
                case = json.load(handle)
            if case['signature'] != signature or len(case['rows']) != len(
                    cohort[token]):
                raise ValueError('Cached case has different settings/cohort: {}'.format(
                    case_path))
            continue
        rows = trace_frame(model, dataset, token_to_index[token], cohort[token],
                           saved[token], device, options.box_expansion,
                           MODEL_LABELS[options.model], options.test_set)
        case_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = case_path.with_suffix('.partial.json')
        with temporary.open('w') as handle:
            json.dump({'signature': signature, 'rows': rows}, handle)
        os.replace(str(temporary), str(case_path))
        if count % 50 == 0:
            print('{} on {}: {}/{} frames'.format(
                options.model, options.test_set, count, len(tokens)), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    rows = []
    for token in tokens:
        with (output / 'cases' / (token+'.json')).open() as handle:
            rows.extend(json.load(handle)['rows'])
    expected = sum(len(cohort[token]) for token in tokens)
    if len(rows) != expected:
        raise ValueError('Traced TP count mismatch: {} vs {}'.format(
            len(rows), expected))
    with (output / 'per_prediction.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, cohort, tokens, options, ap['official_car_ap'])
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    plot(summary, output / 'tp_attention_regions')
    print('Saved', output / 'per_prediction.csv')
    print('Saved', output / 'summary.json')
    print('Saved', output / 'tp_attention_regions.png')


if __name__ == '__main__':
    main()
