#!/usr/bin/env python3
"""Frozen-model, same-R1-f-frame PETR cross-attention pathway diagnosis."""

import argparse
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
from mmdet3d.models import build_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/cross_rig_attention'))
from analyze_pair import feature_rays, rays_intersect_box  # noqa: E402
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, gt_targets, position_components, prepare_one,
    relocate_dataset_paths)

CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
CHECKPOINTS = {
    'r1': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth',
    'r1f': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth',
}
MODES = ('value_car', 'value_control', 'key_x', 'key_g3d', 'key_gmv')


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--selection', type=Path, default=None,
                        help='R1-f trace_candidates.json or another list of R1-f GT cars.')
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--layers', nargs='+', type=int, default=list(range(1, 7)))
    parser.add_argument('--strengths', nargs='+', type=float, default=[0.5])
    parser.add_argument('--control-repeats', type=int, default=3)
    parser.add_argument('--box-expansion', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def checkpoint_signature(key):
    path = CHECKPOINTS[key]
    stat = path.stat()
    return [str(path), stat.st_size,
            getattr(stat, 'st_mtime_ns', int(stat.st_mtime*1e9))]


def make_model(cfg, dataset, key, device):
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(CHECKPOINTS[key]), map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.to(device).eval()
    return model


def make_masks(metas, feature_shape, gt_box, expansion):
    cameras, _, height, width = feature_shape
    source = {'key_layout': {'num_cameras': cameras, 'height': height,
                             'width': width},
              'lidar2img': metas[0]['lidar2img'],
              'pad_shape': metas[0]['pad_shape']}
    origins, directions, cam_ids = feature_rays(source)
    mask = rays_intersect_box(origins, directions, gt_box, expansion)
    return mask, cam_ids


def control_mask(car_mask, valid_mask, cam_ids, seed):
    rng = np.random.RandomState(seed)
    result = np.zeros(len(car_mask), dtype=np.bool_)
    for camera in np.unique(cam_ids[car_mask & valid_mask]):
        needed = int(((cam_ids == camera) & car_mask & valid_mask).sum())
        pool = np.flatnonzero((cam_ids == camera) & ~car_mask & valid_mask)
        if len(pool) < needed:
            raise ValueError('Not enough same-camera control tokens')
        result[rng.choice(pool, needed, replace=False)] = True
    return result


def score(outputs, query, gt_box, car_id, reference):
    boxes = outputs['all_bbox_preds'][:, 0, query].detach().float().cpu().numpy()
    gt_xy = np.asarray(gt_box[:2], dtype=np.float64)
    reference_xy = np.asarray(reference[:2], dtype=np.float64)
    direction = gt_xy-reference_xy
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    centers = boxes[:, :2]
    errors = np.linalg.norm(centers-gt_xy[None], axis=1)
    radial = (centers-gt_xy[None]) @ direction
    confidences = outputs['all_cls_scores'][:, 0, query, car_id].sigmoid()
    return {'layer_center_error_m': errors.tolist(),
            'layer_radial_ref_to_gt_m': radial.tolist(),
            'layer_car_confidence': confidences.detach().float().cpu().tolist(),
            'layer_center_xy': centers.tolist()}


class CrossAttentionInstrument:
    """Trace or perturb one query of one PETR cross-attention layer."""

    def __init__(self, head, query, car_mask, cam_ids, g3d, gmv, layer=None,
                 mode=None, strength=0.0, control_seed=0):
        self.head = head
        self.query = query
        self.car_mask = car_mask
        self.cam_ids = cam_ids
        self.g3d = g3d
        self.gmv = gmv
        self.target_layer = layer
        self.mode = mode
        self.strength = strength
        self.control_seed = control_seed
        self.records = []
        self.originals = []

    def __enter__(self):
        for index, layer in enumerate(self.head.transformer.decoder.layers):
            module = layer.attentions[1]
            original = module.forward

            def instrumented(*call_args, _index=index, _module=module,
                             _original=original, **call_kwargs):
                result = _original(*call_args, **call_kwargs)
                if self.target_layer is not None and _index != self.target_layer:
                    return result
                return self.analyze(_index, _module, result,
                                    call_args, call_kwargs)

            module.forward = instrumented
            self.originals.append((module, original))
        return self

    def __exit__(self, _type, _value, _traceback):
        for module, original in self.originals:
            module.forward = original

    def analyze(self, layer, module, result, call_args, kwargs):
        def arg(name, index, default=None):
            return kwargs.get(name, call_args[index]
                              if len(call_args) > index else default)
        query = arg('query', 0)
        key = arg('key', 1, query)
        value = arg('value', 2, key)
        identity = arg('identity', 3, query)
        qpos = arg('query_pos', 4)
        kpos = arg('key_pos', 5)
        attn_mask = arg('attn_mask', 6)
        padding_mask = arg('key_padding_mask', 7)
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query is None or key is None or value is None:
            raise ValueError('Expected PETR cross-attention Q/K/V')
        if module.batch_first or module.urope or query.shape[1] != 1:
            raise ValueError('This diagnostic supports vanilla PETR batch size 1 only')
        if kpos is None or float((kpos-(self.g3d+self.gmv)).abs().max()) > 1e-4:
            raise AssertionError('Key PE does not decompose into G3D + GMV')
        dims, heads = module.embed_dims, module.num_heads
        depth = dims // heads
        weight, bias = module.attn.in_proj_weight, module.attn.in_proj_bias
        def project(tensor, start, end, with_bias=True):
            component_bias = None if bias is None or not with_bias else bias[start:end]
            return F.linear(tensor, weight[start:end], component_bias)
        q = project(query+(qpos if qpos is not None else 0), 0, dims)
        k = project(key+(kpos if kpos is not None else 0), dims, 2*dims)
        v = project(value, 2*dims, 3*dims)
        qh = q[self.query, 0].view(heads, depth)
        kh = k[:, 0].view(-1, heads, depth).transpose(0, 1)
        vh = v[:, 0].view(-1, heads, depth).transpose(0, 1)
        logits = torch.einsum('hd,htd->ht', qh, kh)/math.sqrt(depth)
        if attn_mask is not None:
            selected = attn_mask[self.query] if attn_mask.dim() == 2 else (
                attn_mask.view(-1, heads, attn_mask.shape[-2],
                               attn_mask.shape[-1])[0, :, self.query])
            if selected.dtype in (torch.bool, torch.uint8):
                logits = logits.masked_fill(selected.bool(), float('-inf'))
            else:
                logits = logits + selected
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask[0][None].bool(), float('-inf'))
        attention = F.softmax(logits.float(), dim=-1).to(v.dtype)
        valid = torch.isfinite(logits).all(dim=0).detach().cpu().numpy()
        car = self.car_mask & valid
        if not car.any():
            raise ValueError('GT car has no valid intersecting image rays')
        car_tensor = torch.as_tensor(car, device=v.device)
        raw = torch.einsum('ht,htd->hd', attention, vh).reshape(dims)
        output_projection = module.attn.out_proj
        projected = F.linear(raw, output_projection.weight,
                             output_projection.bias)
        actual = result[self.query, 0]-identity[self.query, 0]
        replay_error = float((projected-actual).abs().max())
        if replay_error > 1e-4:
            raise AssertionError('Cross-attention replay mismatch at L{}: {}'.format(
                layer+1, replay_error))
        car_raw = torch.einsum('ht,htd->hd', attention[:, car_tensor],
                               vh[:, car_tensor]).reshape(dims)
        car_update = F.linear(car_raw, output_projection.weight, None)
        total_update = F.linear(raw, output_projection.weight, None)
        record = {
            'layer': layer+1,
            'car_token_count': int(car.sum()),
            'valid_token_count': int(valid.sum()),
            'car_attention_mass': float(attention[:, car_tensor].sum(-1).mean()),
            'car_attention_enrichment': float(
                attention[:, car_tensor].sum(-1).mean() /
                (int(car.sum())/int(valid.sum()))),
            'car_value_update_norm': float(car_update.norm()),
            'total_value_update_norm': float(total_update.norm()),
            'car_to_total_update_norm_ratio': float(
                car_update.norm()/total_update.norm().clamp_min(1e-8)),
            'attention_replay_max_abs_error': replay_error,
        }
        if self.target_layer is None:
            self.records.append(record)
            return result
        new_raw = raw
        if self.mode in ('value_car', 'value_control'):
            selected = car if self.mode == 'value_car' else control_mask(
                car, valid, self.cam_ids, self.control_seed)
            replacement = vh.clone()
            selected_t = torch.as_tensor(selected, device=v.device)
            for camera in np.unique(self.cam_ids[selected]):
                where = (self.cam_ids == camera) & selected
                background = (self.cam_ids == camera) & ~car & valid & ~where
                if not background.any():
                    raise ValueError('No background value reference for camera')
                background_t = torch.as_tensor(background, device=v.device)
                mean = vh[:, background_t].mean(dim=1, keepdim=True)
                where_t = torch.as_tensor(where, device=v.device)
                replacement[:, where_t] = (
                    (1-self.strength)*vh[:, where_t]+self.strength*mean)
            new_raw = torch.einsum('ht,htd->hd', attention,
                                   replacement).reshape(dims)
            record['intervened_token_count'] = int(selected.sum())
            record['intervened_attention_mass'] = float(
                attention[:, selected_t].sum(-1).mean())
        elif self.mode in ('key_x', 'key_g3d', 'key_gmv'):
            if self.mode == 'key_x':
                component = project(key, dims, 2*dims)
            else:
                pe = self.g3d if self.mode == 'key_g3d' else self.gmv
                component = project(pe, dims, 2*dims, with_bias=False)
            component_h = component[:, 0].view(-1, heads, depth).transpose(0, 1)
            term = torch.einsum('hd,htd->ht', qh, component_h)/math.sqrt(depth)
            rerouted = F.softmax((logits-self.strength*term).float(),
                                 dim=-1).to(v.dtype)
            new_raw = torch.einsum('ht,htd->hd', rerouted, vh).reshape(dims)
            record['new_car_attention_mass'] = float(
                rerouted[:, car_tensor].sum(-1).mean())
        else:
            raise ValueError('Unknown intervention: {}'.format(self.mode))
        delta = F.linear(new_raw-raw, output_projection.weight, None)
        updated = result.clone()
        updated[self.query, 0] = updated[self.query, 0]+delta
        record.update({'mode': self.mode, 'strength': self.strength,
                       'attention_output_delta_norm': float(delta.norm())})
        self.records.append(record)
        return updated


class StageStateCapture:
    """Record the selected query around self-attention, cross-attention and FFN."""

    def __init__(self, head, query):
        self.head = head
        self.query = query
        self.handles = []
        self.records = [dict(layer=layer+1) for layer in
                        range(len(head.transformer.decoder.layers))]

    def __enter__(self):
        for layer_index, layer in enumerate(self.head.transformer.decoder.layers):
            for name, module in [('self_attention', layer.attentions[0]),
                                 ('cross_attention', layer.attentions[1]),
                                 ('ffn', layer.ffns[0])]:
                def capture(_module, inputs, output, _layer=layer_index,
                            _name=name):
                    before = inputs[0][self.query, 0].detach().float()
                    after = output[self.query, 0].detach().float()
                    self.records[_layer][_name] = {
                        'before_norm': float(before.norm()),
                        'after_norm': float(after.norm()),
                        'update_norm': float((after-before).norm()),
                        'before': before.cpu().half().tolist(),
                        'after': after.cpu().half().tolist(),
                    }
                self.handles.append(module.register_forward_hook(capture))
            for norm_index, module in enumerate(layer.norms):
                def capture_norm(_module, _inputs, output, _layer=layer_index,
                                 _norm=norm_index):
                    vector = output[self.query, 0].detach().float()
                    self.records[_layer]['post_norm_{}'.format(_norm+1)] = {
                        'norm': float(vector.norm()),
                        'state': vector.cpu().half().tolist(),
                    }
                self.handles.append(module.register_forward_hook(capture_norm))
        return self

    def __exit__(self, _type, _value, _traceback):
        for handle in self.handles:
            handle.remove()


def flatten_pe(tensor):
    batch, cameras, channels, height, width = tensor.shape
    if batch != 1:
        raise ValueError('Expected batch size one')
    return tensor.permute(1, 3, 4, 0, 2).reshape(
        cameras*height*width, batch, channels)


def process_car(model, dataset, sample_index, selection, device, options):
    head = model.pts_bbox_head
    data = prepare_one(dataset, sample_index,
                       device.index if device.type == 'cuda' else device)
    images, metas = first_augmentation(data)
    with torch.no_grad():
        features = model.extract_feat(img=images, img_metas=metas)
        gt_boxes, gt_labels, _ = gt_targets(dataset, sample_index,
                                            device, head.pc_range)
        gt_index = int(selection['tracer_gt_index'])
        if gt_index >= len(gt_boxes) or int(gt_labels[gt_index]) != list(
                model.CLASSES).index('car'):
            raise ValueError('Selection GT car index does not match filtered PETR GT')
        gt_box = gt_boxes[gt_index].detach().float().cpu().numpy()
        if np.linalg.norm(gt_box[:2]-np.asarray([
                selection['gt_lidar_x'], selection['gt_lidar_y']])) > .02:
            raise ValueError('Selected car center does not match this frame')
        baseline = head(features, metas)
        assignment = head.assigner.assign(
            baseline['all_bbox_preds'][-1, 0],
            baseline['all_cls_scores'][-1, 0], gt_boxes, gt_labels, None)
        candidates = torch.nonzero(assignment.gt_inds == gt_index+1).flatten()
        if candidates.numel() != 1:
            raise ValueError('Expected exactly one assigned query for GT car')
        query = int(candidates.item())
        reference = head.reference_points.weight[query].detach()
        bounds = reference.new_tensor(head.pc_range)
        reference = (reference*(bounds[3:6]-bounds[:3])+bounds[:3])
        reference = reference.float().cpu().numpy()
        car_id = list(model.CLASSES).index('car')
        mask, cam_ids = make_masks(metas, features[0].shape[1:],
                                   gt_box, options.box_expansion)
        g3d, gmv = position_components(head, features, metas)
        g3d, gmv = flatten_pe(g3d), flatten_pe(gmv)
        with StageStateCapture(head, query) as stages:
            with CrossAttentionInstrument(head, query, mask, cam_ids,
                                          g3d, gmv) as trace:
                replay = head(features, metas)
        difference = float(max(
            (baseline['all_bbox_preds']-replay['all_bbox_preds']).abs().max(),
            (baseline['all_cls_scores']-replay['all_cls_scores']).abs().max()))
        if difference > 1e-5 or len(trace.records) != 6:
            raise AssertionError('Native traced forward changed model output: {}'.format(
                difference))
        base_score = score(baseline, query, gt_box, car_id, reference)
        interventions = []
        for layer in options.layers:
            for mode in MODES:
                repetitions = (range(options.control_repeats)
                               if mode == 'value_control' else range(1))
                for repeat in repetitions:
                    for strength in options.strengths:
                        seed = (int(selection['tracer_gt_index'])+101*sample_index+
                                1009*layer+7919*repeat)
                        with CrossAttentionInstrument(
                                head, query, mask, cam_ids, g3d, gmv,
                                layer=layer-1, mode=mode,
                                strength=strength, control_seed=seed) as probe:
                            altered = head(features, metas)
                        after = score(altered, query, gt_box, car_id, reference)
                        event = probe.records[0]
                        event.update({
                            'control_repeat': repeat,
                            'final_center_error_m': after['layer_center_error_m'][-1],
                            'delta_final_error_m': (
                                after['layer_center_error_m'][-1]-
                                base_score['layer_center_error_m'][-1]),
                            'final_car_confidence': after['layer_car_confidence'][-1],
                            'delta_final_confidence': (
                                after['layer_car_confidence'][-1]-
                                base_score['layer_car_confidence'][-1]),
                            'final_center_xy': after['layer_center_xy'][-1],
                        })
                        if strength == 0 and abs(event['delta_final_error_m']) > 1e-5:
                            raise AssertionError('Zero-strength intervention changed box')
                        interventions.append(event)
        return {
            'sample_token': selection['sample_token'],
            'gt_index': gt_index, 'query_index': query,
            'gt_center_xy': gt_box[:2].tolist(),
            'reference_xy': reference[:2].tolist(),
            'reference_error_m': float(np.linalg.norm(reference[:2]-gt_box[:2])),
            'baseline': base_score,
            'observed_layers': trace.records,
            'query_state_stages': stages.records,
            'interventions': interventions,
            'native_replay_max_abs_error': difference,
        }


def main():
    options = args()
    if (options.max_examples < 1 or options.control_repeats < 1 or
            options.box_expansion < 0 or not options.strengths or
            any(x < 0 or x > 1 for x in options.strengths) or
            any(layer < 1 or layer > 6 for layer in options.layers)):
        raise ValueError('Invalid examples, layer, strength, repeat or box expansion')
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
    token_to_index = {info['token']: index for index, info in
                      enumerate(dataset.data_infos)}
    selection_path = options.selection or ROOT / (
        'experiments/cross_rig_attention/output/r1f_{}/analysis/'
        'trace_candidates.json'.format(options.split))
    with selection_path.open() as handle:
        selected = json.load(handle)[:options.max_examples]
    if not selected:
        raise ValueError('Selection is empty')
    output = options.output_dir or ROOT / (
        'experiments/cross_rig_attention/output/r1f_{}/'
        'inference_pathway'.format(options.split))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(options.device)
    config = {'schema_version': 1,
              'selection': str(selection_path), 'max_examples': options.max_examples,
              'layers': options.layers, 'strengths': options.strengths,
              'control_repeats': options.control_repeats,
              'box_expansion': options.box_expansion}
    for model_key in ('r1', 'r1f'):
        model = None
        for index, candidate in enumerate(selected):
            token = candidate['sample_token']
            if token not in token_to_index:
                raise ValueError('Selected token absent from dataset: {}'.format(token))
            destination = output / 'cases' / model_key / (
                '{}_gt{:03d}.json'.format(token, int(candidate['tracer_gt_index'])))
            signature = {'checkpoint': checkpoint_signature(model_key),
                         'settings': config}
            if destination.is_file() and not options.force:
                with destination.open() as handle:
                    cached = json.load(handle)
                if cached.get('signature') != signature:
                    raise ValueError('Cached case uses different checkpoint/settings: {}'.format(
                        destination))
                print('Reusing', model_key, token, flush=True)
                continue
            if model is None:
                model = make_model(cfg, dataset, model_key, device)
            case = process_car(model, dataset, token_to_index[token],
                               candidate, device, options)
            case['signature'] = signature
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.partial.json')
            with temporary.open('w') as handle:
                json.dump(case, handle)
            os.replace(str(temporary), str(destination))
            print('Saved', model_key, index+1, '/', len(selected), token, flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    from plot_r1f_inference_pathway import make_report  # noqa: E402
    make_report(output, selected, options.layers, options.strengths)


if __name__ == '__main__':
    main()
