#!/usr/bin/env python3
"""Four-way PETR decoder-state × regression-head compatibility experiment."""

import argparse
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, gt_targets, prepare_one, relocate_dataset_paths)

CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
CHECKPOINTS = {
    'r1': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth',
    'r1f': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth',
}


def signature(model):
    stat = CHECKPOINTS[model].stat()
    return [str(CHECKPOINTS[model]), stat.st_size,
            getattr(stat, 'st_mtime_ns', int(stat.st_mtime*1e9))]


def make_dataset(cfg, rig, split):
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/{}/'.format(rig)
    definition.ann_file = 'data/pccr/{0}/{0}_infos_{1}.pkl'.format(rig, split)
    definition.test_mode = True
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    return dataset


def make_model(cfg, dataset, key, device):
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(CHECKPOINTS[key]), map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.to(device).eval()
    return model


def recompute_boxes(states, branches, refs, pc_range):
    """Exact vanilla PETR post-decoder regression using host references."""
    reference = inverse_sigmoid(refs)
    layers = []
    for layer, state in enumerate(torch.nan_to_num(states)):
        delta = branches[layer](state).clone()
        xy = (delta[..., 0:2] + reference[..., 0:2]).sigmoid()
        z = (delta[..., 4:5] + reference[..., 2:3]).sigmoid()
        delta[..., 0:2] = xy
        delta[..., 4:5] = z
        layers.append(delta)
    boxes = torch.stack(layers)
    bounds = boxes.new_tensor(pc_range)
    boxes[..., 0:1] = boxes[..., 0:1]*(bounds[3]-bounds[0])+bounds[0]
    boxes[..., 1:2] = boxes[..., 1:2]*(bounds[4]-bounds[1])+bounds[1]
    boxes[..., 4:5] = boxes[..., 4:5]*(bounds[5]-bounds[2])+bounds[2]
    return boxes


def assigned_car_queries(head, boxes, scores, gt_boxes, gt_labels, car_id):
    assignment = head.assigner.assign(boxes[-1, 0], scores[-1, 0],
                                      gt_boxes, gt_labels, None)
    result = {}
    for query in torch.nonzero(assignment.gt_inds > 0).flatten().tolist():
        gt = int(assignment.gt_inds[query].item())-1
        if int(gt_labels[gt]) == car_id:
            result[gt] = query
    return result


def decoded_result(head, boxes, scores, metas):
    box3d, confidence, labels = head.get_bboxes(
        {'all_cls_scores': scores, 'all_bbox_preds': boxes}, metas)[0]
    return {'boxes_3d': box3d.to('cpu'),
            'scores_3d': confidence.detach().cpu(),
            'labels_3d': labels.detach().cpu()}


def one_frame(host, donor_branches, dataset, index, device):
    head = host.pts_bbox_head
    data = prepare_one(dataset, index, device.index if device.type == 'cuda'
                       else device)
    images, metas = first_augmentation(data)
    trace = {}

    def capture(_module, _input, output):
        trace['states'] = output[0]

    handle = head.transformer.register_forward_hook(capture)
    try:
        with torch.no_grad():
            features = host.extract_feat(img=images, img_metas=metas)
            native = head(features, metas)
            states = trace['states']
            refs = head.reference_points.weight.unsqueeze(0)
            replay = recompute_boxes(states, head.reg_branches,
                                     refs, head.pc_range)
            difference = float((replay-native['all_bbox_preds']).abs().max())
            if difference > 1e-4:
                raise AssertionError('Native regression replay differs by {}'.format(
                    difference))
            swapped = recompute_boxes(states, donor_branches,
                                      refs, head.pc_range)
            scores = native['all_cls_scores']
            gt_boxes, gt_labels, _ = gt_targets(dataset, index, device,
                                                head.pc_range)
            car_id = list(host.CLASSES).index('car')
            if gt_labels.numel():
                native_queries = assigned_car_queries(
                    head, replay, scores, gt_boxes, gt_labels, car_id)
                swapped_queries = assigned_car_queries(
                    head, swapped, scores, gt_boxes, gt_labels, car_id)
            else:
                native_queries, swapped_queries = {}, {}
            rows = []
            car_indices = torch.nonzero(gt_labels == car_id).flatten().tolist()
            car_scores = scores[-1, 0, :, car_id].sigmoid()
            top_classes = scores[-1, 0].sigmoid().argmax(-1)
            for gt in car_indices:
                center = gt_boxes[gt, :2]
                native_query = native_queries.get(gt)
                swapped_query = swapped_queries.get(gt)
                row = {'gt_index': gt, 'native_query': native_query,
                       'swapped_query': swapped_query,
                       'query_changed': (native_query is not None and
                                         swapped_query is not None and
                                         native_query != swapped_query)}
                for label, box_set, query in [
                        ('native', replay, native_query),
                        ('swapped_reassigned', swapped, swapped_query),
                        ('swapped_fixed_query', swapped, native_query)]:
                    if query is None:
                        row[label+'_error_m'] = None
                        row[label+'_score'] = None
                        continue
                    errors = torch.norm(
                        box_set[:, 0, query, :2]-center[None], dim=-1)
                    row[label+'_layer_errors_m'] = errors.detach().cpu().tolist()
                    row[label+'_error_m'] = float(errors[-1])
                    row[label+'_score'] = float(car_scores[query])
                    row[label+'_class_correct'] = int(top_classes[query]) == car_id
                rows.append(row)
            results = {
                'native': decoded_result(head, replay, scores, metas),
                'swapped': decoded_result(head, swapped, scores, metas),
            }
    finally:
        handle.remove()
    return {'cars': rows, 'results': results,
            'native_replay_max_abs_error': difference}


def stats(rows):
    both = [row for row in rows if row['native_error_m'] is not None and
            row['swapped_fixed_query_error_m'] is not None]
    reassigned = [row for row in rows if row['native_error_m'] is not None and
                  row['swapped_reassigned_error_m'] is not None]
    if not both:
        return {'n_gt_cars': len(rows), 'n_both': 0}
    delta = np.asarray([r['swapped_fixed_query_error_m']-r['native_error_m']
                        for r in both])
    result = {
        'n_gt_cars': len(rows), 'n_both': len(both),
        'mean_native_error_m': float(np.mean([r['native_error_m'] for r in both])),
        'mean_swapped_fixed_error_m': float(np.mean([
            r['swapped_fixed_query_error_m'] for r in both])),
        'mean_swap_minus_native_m': float(delta.mean()),
        'median_swap_minus_native_m': float(np.median(delta)),
        'fraction_cars_improved_by_swap': float((delta < 0).mean()),
        'fraction_query_reassigned': float(np.mean([
            r['query_changed'] for r in reassigned])) if reassigned else None,
        'mean_swapped_reassigned_error_m': float(np.mean([
            r['swapped_reassigned_error_m'] for r in reassigned]
        )) if reassigned else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--rigs', nargs='+', choices=['R1', 'R1-f'],
                        default=['R1-f', 'R1'])
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--evaluate', action='store_true',
                        help='Run official dataset metrics; requires full split.')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError('--max-frames must be positive')
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    device = torch.device(args.device)
    root = args.output_dir or ROOT / 'experiments/cross_rig_attention/output' / (
        'regression_head_swap_{}'.format(args.split))
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for rig in args.rigs:
        dataset = make_dataset(cfg, rig, args.split)
        limit = min(len(dataset), args.max_frames) if args.max_frames else len(dataset)
        if args.evaluate and limit != len(dataset):
            raise ValueError('--evaluate requires the full split (omit --max-frames)')
        summary[rig] = {}
        for host_key in ('r1', 'r1f'):
            donor_key = 'r1f' if host_key == 'r1' else 'r1'
            frame_dir = root / rig.replace('-', '') / host_key
            frame_dir.mkdir(parents=True, exist_ok=True)
            expected = {'host': signature(host_key), 'donor': signature(donor_key)}
            tokens = [dataset.data_infos[i]['token'] for i in range(limit)]
            remaining = []
            for index, token in enumerate(tokens):
                path = frame_dir / (token+'.pt')
                if not path.is_file():
                    remaining.append((index, token))
                    continue
                cached = torch.load(str(path), map_location='cpu')
                if cached['signatures'] != expected:
                    raise ValueError('Checkpoint changed since cached result: {}'.format(path))
            print(rig, host_key, 'remaining frames:', len(remaining), flush=True)
            if remaining:
                host = make_model(cfg, dataset, host_key, device)
                donor = make_model(cfg, dataset, donor_key, torch.device('cpu'))
                donor_branches = copy.deepcopy(donor.pts_bbox_head.reg_branches)
                donor_branches.to(device).eval()
                del donor
                for progress, (index, token) in enumerate(remaining, 1):
                    result = one_frame(host, donor_branches, dataset, index, device)
                    result['signatures'] = expected
                    result['token'] = token
                    path = frame_dir / (token+'.pt')
                    temporary = frame_dir / (token+'.partial.pt')
                    torch.save(result, str(temporary))
                    os.replace(str(temporary), str(path))
                    if progress % 25 == 0 or progress == len(remaining):
                        print(rig, host_key, progress, '/', len(remaining),
                              'new frames', flush=True)
                del host, donor_branches
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            records = [torch.load(str(frame_dir / (token+'.pt')),
                                  map_location='cpu') for token in tokens]
            cars = [row for record in records for row in record['cars']]
            summary[rig][host_key] = stats(cars)
            summary[rig][host_key]['processed_frames'] = len(records)
            if args.evaluate:
                evaluations = {}
                for condition in ('native', 'swapped'):
                    predictions = [record['results'][condition] for record in records]
                    evaluations[condition] = dataset.evaluate(predictions,
                                                               metric='bbox')
                summary[rig][host_key]['official_metrics'] = evaluations
            print(rig, host_key, summary[rig][host_key], flush=True)
        with (root / ('summary_{}.json'.format(rig.replace('-', '')))).open('w') as handle:
            json.dump(summary[rig], handle, indent=2)
    with (root / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    fig, axes = plt.subplots(1, len(args.rigs), figsize=(6*len(args.rigs), 5))
    axes = np.atleast_1d(axes)
    for axis, rig in zip(axes, args.rigs):
        values = [summary[rig][key].get('mean_swap_minus_native_m', np.nan)
                  for key in ('r1', 'r1f')]
        axis.bar(['R1 states', 'R1-f states'], values,
                 color=['#d14d40', '#209b83'])
        axis.axhline(0, color='#64748b', linewidth=1)
        axis.set_title('{} input'.format(rig))
        axis.set_ylabel('Mean swap − native center error (m)')
        axis.grid(axis='y', alpha=.2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.suptitle('Regression-head swap at fixed host query (negative helps)',
                 x=.04, ha='left', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(root / 'regression_head_swap.png', dpi=220)
    fig.savefig(root / 'regression_head_swap.svg')
    plt.close(fig)
    if args.evaluate and 'R1-f' in summary:
        plotter = (ROOT / 'experiments/cross_rig_attention/'
                   'plot_regression_head_swap_ap.py')
        svg = root / 'regression_head_swap_r1f_car_ap.svg'
        subprocess.run([sys.executable, str(plotter), '--summary',
                        str(root / 'summary.json'), '--output', str(svg)],
                       cwd=str(ROOT), check=True)
        converter = shutil.which('rsvg-convert')
        if converter:
            subprocess.run([converter, '-o', str(svg.with_suffix('.png')),
                            str(svg)], check=True)
    print('Saved:', root)


if __name__ == '__main__':
    main()
