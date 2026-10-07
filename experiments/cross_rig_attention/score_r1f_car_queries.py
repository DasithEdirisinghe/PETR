#!/usr/bin/env python3
"""Full-validation PETR query-level car attention/class/center scorecard."""

import argparse
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model as build_detector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/cross_rig_attention'))
from analyze_pair import feature_rays, rays_intersect_box  # noqa: E402
from compare_r1_r1f_cars import gt_frames  # noqa: E402
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, gt_targets, prepare_one, relocate_dataset_paths)

CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
CHECKPOINTS = {
    'r1': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth',
    'r1f': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth',
}
FIELDS = ('sample_token', 'gt_index', 'model', 'query_index', 'car_score',
          'top_class', 'top_score', 'car_top_class', 'car_score_pass',
          'center_error_m', 'center_pass_1m', 'center_pass_4m',
          'attention_auc_l1', 'attention_auc_l6',
          'attention_lift_l1', 'attention_lift_l6', 'car_ray_count')
CAMERA_FIELDS = ('sample_token', 'gt_index', 'model', 'query_index',
                 'camera_index', 'camera_name', 'gt_ray_visible',
                 'car_ray_count', 'attention_auc_l1', 'attention_auc_l6',
                 'attention_lift_l1', 'attention_lift_l6',
                 'camera_attention_mass_l1', 'camera_attention_mass_l6',
                 'car_ray_attention_mass_l1', 'car_ray_attention_mass_l6')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-threshold', type=float, default=.35)
    parser.add_argument('--box-expansion', type=float, default=.5)
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Smoke test with a small number of frames.')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


class CaptureCrossAttention:
    """Record the actual cross-attention call; never modify its result."""

    def __init__(self, head):
        self.head = head
        self.calls = {}
        self.originals = []

    def __enter__(self):
        for index in (0, 5):
            module = self.head.transformer.decoder.layers[index].attentions[1]
            original = module.forward

            def captured(*args, _index=index, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                def get(name, position):
                    return kwargs.get(name, args[position]
                                      if len(args) > position else None)
                self.calls[_index+1] = {
                    'query': get('query', 0), 'key': get('key', 1),
                    'query_pos': get('query_pos', 4),
                    'key_pos': get('key_pos', 5),
                    'attn_mask': get('attn_mask', 6),
                    'padding_mask': get('key_padding_mask', 7),
                }
                return result
            module.forward = captured
            self.originals.append((module, original))
        return self

    def __exit__(self, _type, _value, _traceback):
        for module, original in self.originals:
            module.forward = original


def attention_selected(module, call, query_indices):
    """Mean-over-head attention [selected query, camera*y*x]."""
    if module.batch_first or getattr(module, 'urope', False):
        raise ValueError('This experiment supports vanilla PETR only')
    q, k = call['query'], call['key']
    if k is None:
        k = q
    qpos, kpos = call['query_pos'], call['key_pos']
    dims, heads = module.embed_dims, module.num_heads
    depth = dims // heads
    weight, bias = module.attn.in_proj_weight, module.attn.in_proj_bias
    q = F.linear(q[query_indices]+qpos[query_indices], weight[:dims],
                 None if bias is None else bias[:dims])
    k = F.linear(k+kpos, weight[dims:2*dims],
                 None if bias is None else bias[dims:2*dims])
    q = q[:, 0].view(len(query_indices), heads, depth).transpose(0, 1)
    k = k[:, 0].view(-1, heads, depth).transpose(0, 1)
    logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(depth)
    mask = call['attn_mask']
    if mask is not None:
        if mask.dim() == 2:
            mask = mask[query_indices][None]
        else:
            mask = mask.view(-1, heads, mask.shape[-2], mask.shape[-1])[
                0, :, query_indices]
        if mask.dtype in (torch.bool, torch.uint8):
            logits = logits.masked_fill(mask.bool(), float('-inf'))
        else:
            logits = logits + mask
    padding = call['padding_mask']
    if padding is not None:
        logits = logits.masked_fill(padding[0][None, None].bool(), float('-inf'))
    attention = F.softmax(logits.float(), dim=-1).mean(dim=0)
    return attention.detach().cpu().numpy(), torch.isfinite(
        logits).all(dim=0).all(dim=0).detach().cpu().numpy()


def ray_scores(attention, valid, car_mask, candidate_mask=None):
    if candidate_mask is None:
        candidate_mask = np.ones(len(valid), dtype=np.bool_)
    valid = valid & candidate_mask
    car = car_mask & valid
    if not car.any():
        return None, None, 0
    background = (~car_mask) & valid
    if not background.any():
        return None, None, int(car.sum())
    positive = attention[car]
    negative = attention[background]
    # Exact pairwise AUROC is feasible at PETR's feature resolution; cap the
    # negative comparison set deterministically for crowded/high-resolution rigs.
    if len(negative) > 2000:
        rng = np.random.RandomState(17)
        negative = negative[rng.choice(len(negative), 2000, replace=False)]
    auc = ((positive[:, None] > negative).mean() +
           .5*(positive[:, None] == negative).mean())
    attention_total = attention[valid].sum()
    lift = (attention[car].sum()/max(float(attention_total), 1e-12) /
            (car.sum()/float(valid.sum())))
    return float(auc), float(lift), int(car.sum())


def build_petr_checkpoint(cfg, dataset, name, device):
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(CHECKPOINTS[name]),
                                 map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    return model.to(device).eval()


def run_frame(model, dataset, index, frame, name, device, options):
    data = prepare_one(dataset, index,
                       device.index if device.type == 'cuda' else device)
    images, metas = first_augmentation(data)
    head = model.pts_bbox_head
    with torch.no_grad():
        features = model.extract_feat(img=images, img_metas=metas)
        gt_boxes, gt_labels, _ = gt_targets(dataset, index, device, head.pc_range)
        usable = [(i, int(value)) for i, value in enumerate(frame['tracer_indices'])
                  if value is not None and int(value) < len(gt_boxes) and
                  int(gt_labels[int(value)]) == list(model.CLASSES).index('car')]
        for local, gt_index in usable:
            actual = gt_boxes[gt_index, :2].detach().float().cpu().numpy()
            if np.linalg.norm(actual-frame['lidar_centers'][local]) > .2:
                raise ValueError('GT annotation/tracer index mismatch in {}'.format(
                    dataset.data_infos[index]['token']))
        if not usable:
            return [], [], None, len(frame['tracer_indices'])
        with CaptureCrossAttention(head) as capture:
            output = head(features, metas)
        assignment = head.assigner.assign(output['all_bbox_preds'][-1, 0],
                                          output['all_cls_scores'][-1, 0],
                                          gt_boxes, gt_labels, None)
        pairs = []
        for local, gt_index in usable:
            matches = torch.nonzero(assignment.gt_inds == gt_index+1).flatten()
            if matches.numel() == 1:
                pairs.append((local, gt_index, int(matches.item())))
        if not pairs:
            return [], [], None, len(frame['tracer_indices'])
        layout = features[0].shape
        source = {'key_layout': {'num_cameras': layout[1],
                                 'height': layout[-2], 'width': layout[-1]},
                  'lidar2img': metas[0]['lidar2img'],
                  'pad_shape': metas[0]['pad_shape']}
        origins, directions, camera_ids = feature_rays(source)
        camera_names = list(dataset.data_infos[index]['cams'])
        if len(camera_names) != layout[1]:
            raise ValueError('Camera metadata order/count mismatch')
        query_indices = [pair[2] for pair in pairs]
        attentions = {}
        for layer in (1, 6):
            module = head.transformer.decoder.layers[layer-1].attentions[1]
            attentions[layer] = attention_selected(
                module, capture.calls[layer], query_indices)
        logits = output['all_cls_scores'][-1, 0].float()
        centers = output['all_bbox_preds'][-1, 0, :, :2].float()
        car_id = list(model.CLASSES).index('car')
        rows, camera_rows = [], []
        car_masks = []
        for position, (local, gt_index, query) in enumerate(pairs):
            gt_box = gt_boxes[gt_index].detach().float().cpu().numpy()
            ray_mask = rays_intersect_box(origins, directions, gt_box,
                                          options.box_expansion)
            car_masks.append(ray_mask)
            probabilities = logits[query].sigmoid()
            top = int(torch.argmax(probabilities))
            error = float(torch.norm(centers[query]-gt_boxes[gt_index, :2]))
            row = {'sample_token': dataset.data_infos[index]['token'],
                   'gt_index': int(frame['gt_indices'][local]),
                   'model': name, 'query_index': query,
                   'car_score': float(probabilities[car_id]),
                   'top_class': model.CLASSES[top],
                   'top_score': float(probabilities[top]),
                   'car_top_class': int(top == car_id),
                   'car_score_pass': int(float(probabilities[car_id]) >=
                                         options.score_threshold),
                   'center_error_m': error,
                   'center_pass_1m': int(error < 1.0),
                   'center_pass_4m': int(error < 4.0)}
            for layer in (1, 6):
                attention, valid = attentions[layer]
                auc, lift, count = ray_scores(attention[position], valid,
                                              ray_mask)
                row['attention_auc_l{}'.format(layer)] = auc
                row['attention_lift_l{}'.format(layer)] = lift
                row['car_ray_count'] = count
            rows.append(row)
            for camera, camera_name in enumerate(camera_names):
                on_camera = camera_ids == camera
                camera_row = {
                    'sample_token': row['sample_token'],
                    'gt_index': row['gt_index'], 'model': name,
                    'query_index': query, 'camera_index': camera,
                    'camera_name': camera_name,
                    'gt_ray_visible': int((ray_mask & on_camera).any()),
                    'car_ray_count': int((ray_mask & on_camera).sum())}
                for layer in (1, 6):
                    attention, valid = attentions[layer]
                    values = attention[position]
                    auc, lift, _ = ray_scores(values, valid, ray_mask,
                                               on_camera)
                    camera_row['attention_auc_l{}'.format(layer)] = auc
                    camera_row['attention_lift_l{}'.format(layer)] = lift
                    camera_row['camera_attention_mass_l{}'.format(layer)] = float(
                        values[on_camera & valid].sum())
                    camera_row['car_ray_attention_mass_l{}'.format(layer)] = float(
                        values[on_camera & valid & ray_mask].sum())
                camera_rows.append(camera_row)
        map_payload = {
            'gt_indices': np.asarray([row['gt_index'] for row in rows],
                                     dtype=np.int32),
            'car_masks': np.asarray(car_masks, dtype=np.bool_),
            'camera_ids': camera_ids.astype(np.int8),
            'attention_l1': attentions[1][0].astype(np.float16),
            'attention_l6': attentions[6][0].astype(np.float16),
            'valid_l1': attentions[1][1].astype(np.bool_),
            'valid_l6': attentions[6][1].astype(np.bool_),
        }
        return rows, camera_rows, map_payload, len(frame['tracer_indices'])-len(rows)


def main():
    options = arguments()
    if not 0 <= options.score_threshold <= 1 or options.box_expansion < 0:
        raise ValueError('Invalid score threshold or box expansion')
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/R1-f/'
    definition.ann_file = 'data/pccr/R1-f/R1-f_infos_{}.pkl'.format(
        options.split)
    definition.test_mode = True
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    frames = gt_frames(ROOT / definition.ann_file)
    token_to_index = {item['token']: i for i, item in
                      enumerate(dataset.data_infos)}
    if set(frames) != set(token_to_index):
        raise ValueError('GT and PETR dataset sample tokens differ')
    output = options.output_dir or ROOT / (
        'experiments/cross_rig_attention/output/r1f_{}/query_scorecard'.format(
            options.split))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(options.device)
    selected = list(frames.items())[:options.max_frames]
    for name in ('r1', 'r1f'):
        checkpoint = CHECKPOINTS[name]
        stat = checkpoint.stat()
        signature = {'checkpoint': str(checkpoint), 'size': stat.st_size,
                     'mtime_ns': getattr(stat, 'st_mtime_ns',
                                         int(stat.st_mtime*1e9)),
                     'split': options.split,
                     'score_threshold': options.score_threshold,
                     'box_expansion': options.box_expansion,
                     'schema': 3}
        model = build_petr_checkpoint(cfg, dataset, name, device)
        for number, (token, frame) in enumerate(selected, 1):
            target = output / 'cases' / name / (token+'.json')
            map_target = output / 'attention_maps' / name / (token+'.npz')
            if target.exists() and not options.force:
                with target.open() as handle:
                    old = json.load(handle)
                if old.get('signature') != signature:
                    raise ValueError('Cached query case has different settings: {}'.format(
                        target))
                if old['rows'] and not map_target.exists():
                    raise ValueError('Cached case lacks its attention maps: {}'.format(
                        map_target))
                continue
            rows, camera_rows, map_payload, unmatched = run_frame(
                model, dataset, token_to_index[token], frame, name,
                device, options)
            if map_payload is not None:
                map_target.parent.mkdir(parents=True, exist_ok=True)
                temporary_map = map_target.with_suffix('.partial.npz')
                with temporary_map.open('wb') as handle:
                    np.savez_compressed(handle, **map_payload)
                os.replace(str(temporary_map), str(map_target))
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix('.partial.json')
            with temp.open('w') as handle:
                json.dump({'rows': rows, 'camera_rows': camera_rows,
                           'unmatched_gt': unmatched,
                           'signature': signature}, handle)
            os.replace(str(temp), str(target))
            if number % 50 == 0:
                print('{}: {}/{} frames'.format(name, number, len(selected)),
                      flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    rows, camera_rows = [], []
    skipped = {}
    for name in ('r1', 'r1f'):
        skipped[name] = 0
        for token, _ in selected:
            with (output / 'cases' / name / (token+'.json')).open() as handle:
                case = json.load(handle)
            rows.extend(case['rows'])
            camera_rows.extend(case['camera_rows'])
            skipped[name] += case['unmatched_gt']
    with (output / 'per_car_query.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'per_car_camera_attention.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMERA_FIELDS)
        writer.writeheader()
        writer.writerows(camera_rows)
    with (output / 'run_metadata.json').open('w') as handle:
        json.dump({'processed_frames': len(selected),
                   'gt_cars': sum(len(frame['gt_indices']) for _, frame in selected),
                   'unmatched_gt_by_model': skipped,
                   'cases_by_model': {name: sum(row['model'] == name for row in rows)
                                      for name in ('r1', 'r1f')}}, handle, indent=2)
    print('Saved query-level scorecard data:', output / 'per_car_query.csv')
    from plot_r1f_car_query_scorecard import main as plot_main  # noqa: E402
    sys.argv = [sys.argv[0], '--input-dir', str(output),
                '--score-threshold', str(options.score_threshold)]
    plot_main()
    from compare_r1f_query_attention_maps import main as pair_main  # noqa: E402
    sys.argv = [sys.argv[0], '--input-dir', str(output)]
    pair_main()


if __name__ == '__main__':
    main()
