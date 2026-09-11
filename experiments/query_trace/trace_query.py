#!/usr/bin/env python
"""Trace clean and poorly localized PETR queries through all decoder layers."""

import argparse
import csv
import importlib
import json
import math
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox  # noqa: E402


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(
        REPO_ROOT / 'projects/configs/petr/'
        'petr_r50dcn_gridmask_p4_nuscenes_local.py'))
    parser.add_argument('--checkpoint', default=str(
        REPO_ROOT / 'ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth'))
    parser.add_argument('--selection-json', default=str(
        REPO_ROOT / 'experiments/keyframe_selection/outputs/selection.json'))
    parser.add_argument('--scene-name', default=None,
                        help='Explicit nuScenes scene, e.g. scene-0018.')
    parser.add_argument('--scene-frame-index', type=int, default=None,
                        help='Zero-based keyframe index within --scene-name.')
    parser.add_argument('--sample-index', type=int, default=None)
    parser.add_argument('--sample-token', default=None)
    parser.add_argument('--output-dir', default=str(
        REPO_ROOT / 'experiments/query_trace/outputs'))
    parser.add_argument('--score-threshold', type=float, default=0.35)
    parser.add_argument('--classes', nargs='+', default=None,
                        help='Restrict query selection to class names, e.g. car pedestrian.')
    parser.add_argument('--list-queries', action='store_true',
                        help='Write matched-query precheck tables and exit.')
    parser.add_argument('--query-indices', nargs='+', type=int, default=None,
                        help='Trace these exact final-layer query indices.')
    parser.add_argument('--clean-error', type=float, default=1.0)
    parser.add_argument('--poor-error', type=float, default=2.0)
    parser.add_argument('--max-poor-error', type=float, default=4.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--plot-bev-rays', action='store_true',
                        help='Plot attention-weighted camera rays in LiDAR BEV.')
    parser.add_argument('--bev-ray-layers', nargs='+', type=int, default=[6],
                        help='One-based decoder layers for BEV ray plots.')
    parser.add_argument('--bev-ray-top-k', type=int, default=200,
                        help='Number of highest-attention rays to draw per plot.')
    return parser.parse_args()


def import_plugin(cfg):
    if cfg.get('plugin', False):
        importlib.import_module(
            os.path.dirname(cfg.get('plugin_dir', '')).replace('/', '.'))


def relocate_dataset_paths(dataset, data_root):
    root = Path(data_root).resolve()

    def relocate(value):
        if not isinstance(value, str) or os.path.exists(value):
            return value
        normalized = value.replace('\\', '/')
        for directory in ('samples', 'sweeps', 'maps'):
            marker = '/' + directory + '/'
            if marker in normalized:
                candidate = root / directory / normalized.split(marker, 1)[1]
                if candidate.exists():
                    return str(candidate)
        return value

    for info in dataset.data_infos:
        if 'lidar_path' in info:
            info['lidar_path'] = relocate(info['lidar_path'])
        for camera in info.get('cams', {}).values():
            camera['data_path'] = relocate(camera.get('data_path'))


def resolve_sample(dataset, args):
    index, token = args.sample_index, args.sample_token
    if (args.scene_name is None) != (args.scene_frame_index is None):
        raise ValueError('--scene-name and --scene-frame-index must be used together')
    if args.scene_name is not None:
        if index is not None or token is not None:
            raise ValueError('Use either scene/frame or sample index/token, not both')
        if args.scene_frame_index < 0:
            raise ValueError('--scene-frame-index is zero-based and must be >= 0')
        table_root = Path(dataset.data_root) / 'v1.0-trainval'
        with open(str(table_root / 'sample.json'), 'r') as handle:
            samples = {row['token']: row for row in json.load(handle)}
        with open(str(table_root / 'scene.json'), 'r') as handle:
            scenes = {row['token']: row for row in json.load(handle)}
        scene_indices = []
        for candidate, info in enumerate(dataset.data_infos):
            sample = samples.get(info['token'])
            if sample is None:
                continue
            if scenes[sample['scene_token']]['name'] == args.scene_name:
                scene_indices.append(candidate)
        scene_indices.sort(key=lambda i: dataset.data_infos[i]['timestamp'])
        if args.scene_frame_index >= len(scene_indices):
            raise IndexError('{} has {} validation keyframes, requested {}'.format(
                args.scene_name, len(scene_indices), args.scene_frame_index))
        index = scene_indices[args.scene_frame_index]
        token = dataset.data_infos[index]['token']
    if index is None and token is None:
        with open(args.selection_json, 'r') as handle:
            selection = json.load(handle)['selection']
        index, token = selection.get('dataset_index'), selection.get('sample_token')
    if token is not None:
        matches = [i for i, info in enumerate(dataset.data_infos)
                   if info.get('token') == token]
        if not matches:
            raise ValueError('Sample token is absent from this dataset: ' + token)
        token_index = matches[0]
        if index is not None and token_index != index:
            print('Selection dataset index changed; using stable sample token.')
        index = token_index
    if index is None or not 0 <= index < len(dataset):
        raise IndexError('A valid sample index or token is required')
    return index, dataset.data_infos[index]['token']


def prepare_one(dataset, index, device):
    sample = dataset.prepare_test_data(index)
    if sample is None:
        raise RuntimeError('Dataset returned no test data for index {}'.format(index))
    return scatter(collate([sample], samples_per_gpu=1), [device])[0]


def first_augmentation(data):
    img, metas = data['img'], data['img_metas']
    if isinstance(img, (list, tuple)):
        img = img[0]
    if (isinstance(metas, (list, tuple)) and metas and
            isinstance(metas[0], (list, tuple))):
        metas = metas[0]
    return img, metas


def gt_targets(dataset, index, device, pc_range):
    annotation = dataset.get_ann_info(index)
    labels = torch.as_tensor(
        annotation['gt_labels_3d'], dtype=torch.long, device=device)
    boxes = torch.cat((annotation['gt_bboxes_3d'].gravity_center,
                       annotation['gt_bboxes_3d'].tensor[:, 3:]), dim=1).to(device)
    bounds = boxes.new_tensor(pc_range)
    valid = labels >= 0
    valid &= ((boxes[:, :3] >= bounds[:3]) &
              (boxes[:, :3] <= bounds[3:6])).all(dim=1)
    return boxes[valid], labels[valid], annotation


def query_candidates(head, outputs, gt_boxes, gt_labels, class_names,
                     score_threshold, allowed_label_ids=None):
    cls = outputs['all_cls_scores'][-1, 0]
    boxes = outputs['all_bbox_preds'][-1, 0]
    assignment = head.assigner.assign(boxes, cls, gt_boxes, gt_labels, None)
    probs = cls.sigmoid()
    candidates = []
    for query in torch.nonzero(assignment.gt_inds > 0).flatten():
        query_index = int(query.item())
        gt_index = int(assignment.gt_inds[query] - 1)
        gt_label = int(gt_labels[gt_index])
        if allowed_label_ids is not None and gt_label not in allowed_label_ids:
            continue
        predicted_label = int(probs[query].argmax())
        confidence = float(probs[query, gt_label])
        error = float(torch.norm(
            boxes[query][[0, 1]] - gt_boxes[gt_index, :2]))
        candidates.append(dict(
            query_index=query_index,
            gt_index=gt_index,
            gt_label=gt_label,
            gt_class=class_names[gt_label],
            predicted_label=predicted_label,
            predicted_class=class_names[predicted_label],
            class_correct=int(predicted_label == gt_label),
            confidence=confidence,
            above_score_threshold=int(confidence >= score_threshold),
            bev_error_m=error))
    candidates.sort(key=lambda row: row['query_index'])
    return candidates


def choose_queries(candidates, args):
    if args.query_indices:
        by_query = {row['query_index']: row for row in candidates}
        missing = [query for query in args.query_indices if query not in by_query]
        if missing:
            raise ValueError(
                'Requested queries are not final-layer Hungarian matches in '
                'the selected class set: {}'.format(missing))
        selected = [dict(by_query[query]) for query in args.query_indices]
    else:
        eligible = [row for row in candidates
                    if row['class_correct'] and row['above_score_threshold']]
        clean = [row for row in eligible
                 if row['bev_error_m'] <= args.clean_error]
        poor = [row for row in eligible
                if args.poor_error <= row['bev_error_m'] <= args.max_poor_error]
        if not clean:
            raise RuntimeError('No correctly classified clean query meets the threshold')
        if not poor:
            raise RuntimeError(
                'No correctly classified poorly localized query meets the '
                'threshold; run with --list-queries and select explicitly')
        clean.sort(key=lambda row: (-row['confidence'], row['bev_error_m']))
        poor.sort(key=lambda row: (-row['bev_error_m'], -row['confidence']))
        selected = [dict(clean[0]), dict(poor[0])]

    targets = {}
    for slot, row in enumerate(selected):
        name = 'query_{:04d}_{}'.format(row['query_index'], row['gt_class'])
        if name in targets:
            raise ValueError('Duplicate query index requested: {}'.format(
                row['query_index']))
        row['trace_slot'] = slot
        targets[name] = row
    return targets


def write_query_precheck(output_dir, candidates, sample_token, dataset_index,
                         score_threshold, selected_classes):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'query_candidates.csv'
    if candidates:
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
    with (output_dir / 'query_candidates.json').open('w') as handle:
        json.dump({
            'sample_token': sample_token,
            'dataset_index': dataset_index,
            'score_threshold': score_threshold,
            'selected_classes': selected_classes,
            'queries': candidates,
        }, handle, indent=2)
    return csv_path


def tensor_cpu(value, half=False):
    value = value.detach().cpu()
    return value.half() if half else value.float()


def position_components(head, features, img_metas):
    """Reconstruct PETR's separate learned-3D and multiview PE tensors."""
    if head.position_embedding_mode != 'petr' or not head.with_position:
        raise NotImplementedError(
            'Component decomposition currently requires vanilla PETR 3DPE')
    feature = features[0]
    batch, cameras = feature.shape[:2]
    input_h, input_w, _ = img_metas[0]['pad_shape'][0]
    masks = feature.new_ones((batch, cameras, input_h, input_w))
    for batch_index in range(batch):
        for camera in range(cameras):
            image_h, image_w, _ = img_metas[batch_index]['img_shape'][camera]
            masks[batch_index, camera, :image_h, :image_w] = 0
    projected_shape = head.input_proj(feature.flatten(0, 1)).shape[-2:]
    masks = F.interpolate(masks, size=projected_shape).to(torch.bool)
    learned_3d, _ = head.position_embeding(features, img_metas, masks)
    if head.with_multiview:
        multiview = head.positional_encoding(masks)
        multiview = head.adapt_pos3d(multiview.flatten(0, 1)).view(
            batch, cameras, head.embed_dims, *projected_shape)
    else:
        per_camera = []
        for camera in range(cameras):
            encoded = head.positional_encoding(masks[:, camera])
            per_camera.append(encoded.unsqueeze(1))
        multiview = torch.cat(per_camera, dim=1)
        multiview = head.adapt_pos3d(multiview.flatten(0, 1)).view(
            batch, cameras, head.embed_dims, *projected_shape)
    return learned_3d, multiview


def flatten_keys(tensor):
    batch, cameras, channels, height, width = tensor.shape
    return tensor.permute(1, 3, 4, 0, 2).reshape(
        cameras * height * width, batch, channels)


def js_divergence(probability, alternative):
    probability = probability.float().clamp_min(1e-12)
    alternative = alternative.float().clamp_min(1e-12)
    midpoint = 0.5 * (probability + alternative)
    return 0.5 * (
        (probability * (probability.log() - midpoint.log())).sum(-1) +
        (alternative * (alternative.log() - midpoint.log())).sum(-1))


def install_attention_trace(head, query_indices, records, image_3d_pe,
                            image_multiview_pe):
    """Monkey-patch only this model instance; restore methods after tracing."""
    originals = []
    for layer_index, layer in enumerate(head.transformer.decoder.layers):
        module = layer.attentions[1]
        if getattr(module, 'urope', False):
            raise NotImplementedError('This baseline tracer targets vanilla PETR attention')
        original = module.forward
        originals.append((module, original))

        def traced(this, *call_args, _layer=layer_index, _original=original,
                   **call_kwargs):
            query = call_kwargs.get('query', call_args[0] if call_args else None)
            key = call_kwargs.get('key', call_args[1] if len(call_args) > 1 else query)
            value = call_kwargs.get('value', call_args[2] if len(call_args) > 2 else key)
            identity = call_kwargs.get(
                'identity', call_args[3] if len(call_args) > 3 else query)
            if key is None:
                key = query
            if value is None:
                value = key
            if identity is None:
                identity = query
            query_pos = call_kwargs.get(
                'query_pos', call_args[4] if len(call_args) > 4 else None)
            key_pos = call_kwargs.get(
                'key_pos', call_args[5] if len(call_args) > 5 else None)
            attention_mask = call_kwargs.get(
                'attn_mask', call_args[6] if len(call_args) > 6 else None)
            padding_mask = call_kwargs.get(
                'key_padding_mask', call_args[7] if len(call_args) > 7 else None)
            result = _original(*call_args, **call_kwargs)

            q_input = query if query_pos is None else query + query_pos
            k_input = key if key_pos is None else key + key_pos
            position_split_error = float(
                (key_pos - image_3d_pe - image_multiview_pe).abs().max())
            dims, heads = this.embed_dims, this.num_heads
            head_dim = dims // heads
            weight, bias = this.attn.in_proj_weight, this.attn.in_proj_bias
            q = F.linear(q_input, weight[:dims], None if bias is None else bias[:dims])
            k = F.linear(k_input, weight[dims:2*dims],
                         None if bias is None else bias[dims:2*dims])
            v = F.linear(value, weight[2*dims:],
                         None if bias is None else bias[2*dims:])
            q_selected = q[query_indices, 0].view(len(query_indices), heads,
                                                  head_dim).transpose(0, 1)
            k_heads = k[:, 0].view(k.shape[0], heads, head_dim).transpose(0, 1)
            v_heads = v[:, 0].view(v.shape[0], heads, head_dim).transpose(0, 1)
            logits = torch.matmul(q_selected, k_heads.transpose(-2, -1))
            logits = logits / math.sqrt(head_dim)
            if attention_mask is not None:
                selected_mask = attention_mask
                if selected_mask.dim() == 2:
                    selected_mask = selected_mask[query_indices][None]
                elif selected_mask.dim() == 3:
                    selected_mask = selected_mask.view(
                        -1, heads, selected_mask.shape[-2],
                        selected_mask.shape[-1])
                    selected_mask = selected_mask[0, :, query_indices]
                if selected_mask.dtype in (torch.bool, torch.uint8):
                    logits = logits.masked_fill(selected_mask.bool(), float('-inf'))
                else:
                    logits = logits + selected_mask
            if padding_mask is not None:
                logits = logits.masked_fill(
                    padding_mask[0][None, None].bool(), float('-inf'))
            attention = F.softmax(logits.float(), dim=-1).to(q.dtype)

            q_content = F.linear(query, weight[:dims], None)
            q_position = F.linear(query_pos, weight[:dims], None)
            k_content = F.linear(key, weight[dims:2*dims], None)
            k_3d = F.linear(image_3d_pe, weight[dims:2*dims], None)
            k_multiview = F.linear(
                image_multiview_pe, weight[dims:2*dims], None)

            def selected_query_heads(projected):
                return projected[query_indices, 0].view(
                    len(query_indices), heads, head_dim).transpose(0, 1)

            def all_key_heads(projected):
                return projected[:, 0].view(
                    projected.shape[0], heads, head_dim).transpose(0, 1)

            query_parts = {
                'H': selected_query_heads(q_content),
                'E': selected_query_heads(q_position),
            }
            key_parts = {
                'X': all_key_heads(k_content),
                'G3D': all_key_heads(k_3d),
                'GMV': all_key_heads(k_multiview),
            }
            component_logits = {}
            component_impacts = {}
            for query_name, query_part in query_parts.items():
                for key_name, key_part in key_parts.items():
                    component_name = '{}-{}'.format(query_name, key_name)
                    component = torch.matmul(
                        query_part, key_part.transpose(-2, -1))
                    component = component / math.sqrt(head_dim)
                    component_logits[component_name] = component
                    without_component = logits - component
                    alternative = F.softmax(
                        without_component.float(), dim=-1).to(q.dtype)
                    # Mean JS over heads, independently for each traced query.
                    component_impacts[component_name] = js_divergence(
                        attention, alternative).mean(dim=0)
            impact_stack = torch.stack(list(component_impacts.values()), dim=0)
            impact_percent = 100.0 * impact_stack / impact_stack.sum(
                dim=0, keepdim=True).clamp_min(1e-12)
            replay = torch.matmul(attention, v_heads).transpose(0, 1).reshape(
                len(query_indices), dims)
            replay = this.attn.out_proj(replay) + identity[query_indices, 0]
            manual_difference = float(
                (replay - result[query_indices, 0]).abs().max())

            # Ask the installed PyTorch MHA for its native output and averaged
            # weights. This is the strict compatibility check on older PETR
            # environments; the explicit per-head calculation remains the
            # source of the head-resolved tensors saved below.
            native_out, native_attention = this.attn(
                query=q_input, key=k_input, value=value,
                attn_mask=attention_mask, key_padding_mask=padding_mask)
            native_replay = native_out[query_indices, 0] + identity[
                query_indices, 0]
            native_difference = float(
                (native_replay - result[query_indices, 0]).abs().max())
            native_selected = native_attention[0, query_indices]
            attention_difference = float(
                (attention.mean(dim=0) - native_selected).abs().max())
            records.append({
                'layer': _layer + 1,
                'h': tensor_cpu(query[query_indices, 0]),
                'query_pe': tensor_cpu(query_pos[query_indices, 0]),
                'projected_q': tensor_cpu(q_selected),
                'projected_k': tensor_cpu(k_heads, half=True),
                'logits': tensor_cpu(logits),
                'attention': tensor_cpu(attention),
                'component_logits': {
                    name: tensor_cpu(component)
                    for name, component in component_logits.items()},
                'component_impact_js': {
                    name: tensor_cpu(component_impacts[name])
                    for name in component_impacts},
                'component_impact_percent': {
                    name: tensor_cpu(impact_percent[position])
                    for position, name in enumerate(component_impacts)},
                'manual_replay_max_abs_error': manual_difference,
                'native_replay_max_abs_error': native_difference,
                'attention_weight_max_abs_error': attention_difference,
                'position_split_max_abs_error': position_split_error,
            })
            if _layer == 0:
                records[-1]['image_features'] = tensor_cpu(key[:, 0], half=True)
                records[-1]['image_position'] = tensor_cpu(key_pos[:, 0], half=True)
            return result

        module.forward = types.MethodType(traced, module)
    return originals


def restore_attention(originals):
    for module, original in originals:
        module.forward = original


def model_images(img, cfg):
    tensor = img.detach().float().cpu()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError('Query tracing requires batch size 1')
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError('Expected image shape [N,C,H,W] or [1,N,C,H,W], got {}'.format(
            tuple(tensor.shape)))
    tensor = tensor.numpy().transpose(0, 2, 3, 1)
    norm = cfg.get('img_norm_cfg', {})
    mean = np.asarray(norm.get('mean', [0, 0, 0]), dtype=np.float32)
    std = np.asarray(norm.get('std', [1, 1, 1]), dtype=np.float32)
    tensor = np.clip(tensor * std + mean, 0, 255).astype(np.uint8)
    if norm.get('to_rgb', False):
        tensor = tensor[..., ::-1]
    return tensor


def draw_box(image, corners, lidar2img, color=(70, 255, 70)):
    result = image.copy()
    xyz1 = np.concatenate((corners, np.ones((8, 1))), axis=1)
    projected = xyz1 @ np.asarray(lidar2img).T
    depth = projected[:, 2]
    uv = projected[:, :2] / np.maximum(depth[:, None], 1e-5)
    for start, end in EDGES:
        if depth[start] > 0.1 and depth[end] > 0.1:
            cv2.line(result, tuple(np.round(uv[start]).astype(int)),
                     tuple(np.round(uv[end]).astype(int)), color, 2,
                     cv2.LINE_AA)
    return result


def draw_projected_point(image, xyz, lidar2img, marker, color):
    """Project a LiDAR-frame point and draw it only inside this camera."""
    homogeneous = np.concatenate((np.asarray(xyz, dtype=np.float64), [1.0]))
    projected = np.asarray(lidar2img, dtype=np.float64) @ homogeneous
    if projected[2] <= 0.1:
        return False
    u, v = projected[0] / projected[2], projected[1] / projected[2]
    height, width = image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return False
    center = (int(round(u)), int(round(v)))
    if marker == 'circle':
        cv2.circle(image, center, 9, color, 3, cv2.LINE_AA)
    elif marker == 'cross':
        cv2.line(image, (center[0] - 9, center[1] - 9),
                 (center[0] + 9, center[1] + 9), color, 3, cv2.LINE_AA)
        cv2.line(image, (center[0] - 9, center[1] + 9),
                 (center[0] + 9, center[1] - 9), color, 3, cv2.LINE_AA)
    elif marker == 'diamond':
        points = np.asarray([(center[0], center[1] - 10),
                             (center[0] + 10, center[1]),
                             (center[0], center[1] + 10),
                             (center[0] - 10, center[1])], dtype=np.int32)
        cv2.polylines(image, [points], True, color, 3, cv2.LINE_AA)
    return True


def render_target(target_name, target_position, traces, outputs, gt_boxes,
                  reference_xyz, annotation, images, img_metas, feature_shape,
                  class_names, output_dir):
    query = target_position['query_index']
    num_cams, feat_h, feat_w = feature_shape
    camera_names = img_metas[0].get(
        'camera_names', ['CAM_{}'.format(i) for i in range(num_cams)])
    projections = img_metas[0]['lidar2img']
    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    layer_attention = []
    for trace in traces:
        attention = trace['attention'][:, target_position['trace_slot']].mean(0)
        layer_attention.append(
            attention.reshape(num_cams, feat_h, feat_w).numpy())
    shared_scale = max(
        float(np.percentile(np.stack(layer_attention), 99.5)), 1e-12)
    box_template = annotation['gt_bboxes_3d']
    gt_tensor = gt_boxes[target_position['gt_index']].detach().cpu().clone()
    gt_center = gt_tensor[:3].numpy().copy()
    reference_center = np.asarray(reference_xyz, dtype=np.float32)
    reference_gt_bev = float(np.linalg.norm(
        reference_center[:2] - gt_center[:2]))
    gt_tensor[2] -= gt_tensor[5] * 0.5
    if gt_tensor.numel() != box_template.box_dim:
        raise ValueError(
            'GT box dimension {} does not match dataset box dimension {}'.format(
                gt_tensor.numel(), box_template.box_dim))
    gt_corners = box_template.new_box(gt_tensor.unsqueeze(0)).corners[0].numpy()
    tile_width = 600
    tile_height = int(round(images[0].shape[0] * tile_width /
                            images[0].shape[1]))
    summary_width = 600

    def summary_panel(lines):
        panel = np.full((tile_height, summary_width, 3), (24, 31, 43),
                        dtype=np.uint8)
        for line_index, (text_value, color, scale) in enumerate(lines):
            cv2.putText(panel, text_value, (18, 31 + line_index * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                        cv2.LINE_AA)
        return panel

    gt_tiles = []
    for camera in range(num_cams):
        gt_view = draw_box(
            images[camera], gt_corners, projections[camera], color=(40, 40, 255))
        draw_projected_point(
            gt_view, gt_center, projections[camera], 'diamond', (255, 0, 255))
        cv2.putText(gt_view, '{} | GROUND TRUTH'.format(camera_names[camera]),
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        gt_tiles.append(cv2.resize(
            gt_view, (tile_width, tile_height), interpolation=cv2.INTER_AREA))
    gt_summary = summary_panel([
        ('GROUND TRUTH', (255, 255, 255), 0.68),
        ('class: {}'.format(target_position['gt_class']),
         (200, 210, 225), 0.58),
        ('R-GT: {:.2f} m'.format(reference_gt_bev), (255, 255, 0), 0.62),
    ])
    rows = [np.concatenate([gt_summary] + gt_tiles, axis=1)]
    for layer_index, trace in enumerate(traces):
        attention = layer_attention[layer_index]
        camera_mass = attention.reshape(num_cams, -1).sum(1)
        decoded = denormalize_bbox(
            outputs['all_bbox_preds'][layer_index, 0, query:query + 1], None)
        box_tensor = decoded.detach().cpu().clone()
        predicted_center = box_tensor[0, :3].numpy().copy()
        predicted_gt_bev = float(np.linalg.norm(
            predicted_center[:2] - gt_center[:2]))
        predicted_reference_bev = float(np.linalg.norm(
            predicted_center[:2] - reference_center[:2]))
        layer_confidence = float(outputs['all_cls_scores'][
            layer_index, 0, query, target_position['gt_label']].sigmoid())
        predicted_label = int(outputs['all_cls_scores'][
            layer_index, 0, query].sigmoid().argmax())
        predicted_class = class_names[predicted_label]
        box_tensor[:, 2] -= box_tensor[:, 5] * 0.5
        if box_tensor.shape[-1] != box_template.box_dim:
            raise ValueError(
                'Decoded PETR box dimension {} does not match dataset box '
                'dimension {}'.format(box_tensor.shape[-1],
                                      box_template.box_dim))
        corners = box_template.new_box(box_tensor).corners[0].numpy()
        tiles = []
        for camera in range(num_cams):
            base = images[camera]
            heat = cv2.resize(attention[camera], (base.shape[1], base.shape[0]))
            heat = np.clip(heat / shared_scale, 0, 1)
            color = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                      cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(base, 0.58, color, 0.42, 0)
            overlay = draw_box(overlay, corners, projections[camera])
            draw_projected_point(
                overlay, reference_center, projections[camera], 'circle',
                (255, 255, 0))
            draw_projected_point(
                overlay, predicted_center, projections[camera], 'cross',
                (0, 255, 255))
            draw_projected_point(
                overlay, gt_center, projections[camera], 'diamond',
                (255, 0, 255))
            label = '{} | L{} | mass={:.3f}'.format(
                camera_names[camera], layer_index + 1, camera_mass[camera])
            cv2.putText(overlay, label, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA)
            tiles.append(cv2.resize(
                overlay, (tile_width, tile_height),
                interpolation=cv2.INTER_AREA))
        impacts = trace['component_impact_percent']
        slot = target_position['trace_slot']
        impact_text = '  '.join('{}:{:.1f}%'.format(
            name, float(values[slot])) for name, values in impacts.items())
        first_half = '  '.join(impact_text.split('  ')[:3])
        second_half = '  '.join(impact_text.split('  ')[3:])
        layer_summary = summary_panel([
            ('DECODER LAYER {}'.format(layer_index + 1),
             (255, 255, 255), 0.68),
            ('GT: {}   Pred: {}'.format(
                target_position['gt_class'], predicted_class),
             (200, 210, 225), 0.56),
            ('confidence: {:.3f}'.format(layer_confidence),
             (200, 210, 225), 0.58),
            ('P-GT: {:.2f} m   P-R: {:.2f} m'.format(
                predicted_gt_bev, predicted_reference_bev),
             (200, 210, 225), 0.57),
            (first_half, (120, 205, 255), 0.48),
            (second_half, (120, 205, 255), 0.48),
        ])
        rows.append(np.concatenate([layer_summary] + tiles, axis=1))
    canvas = np.concatenate(rows, axis=0)
    title = ('{} query {} | GT: {} | Pred: {} | rows=GT,L1-L{} | conf {:.3f} | '
             'R-GT={:.2f}m').format(
                 target_name, query, target_position['gt_class'],
                 target_position['predicted_class'], len(traces),
                 target_position['confidence'], reference_gt_bev)
    header = np.zeros((142, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    legend_y = 64
    cv2.circle(header, (28, legend_y - 4), 8, (255, 255, 0), 3,
               cv2.LINE_AA)
    cv2.putText(header, 'Reference point', (45, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 220, 235), 1,
                cv2.LINE_AA)
    cross_x = 225
    cv2.line(header, (cross_x - 8, legend_y - 12),
             (cross_x + 8, legend_y + 4), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(header, (cross_x - 8, legend_y + 4),
             (cross_x + 8, legend_y - 12), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(header, 'Predicted center', (242, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 220, 235), 1,
                cv2.LINE_AA)
    diamond_x = 440
    diamond_y = legend_y - 4
    diamond = np.asarray([(diamond_x, diamond_y - 9),
                          (diamond_x + 9, diamond_y),
                          (diamond_x, diamond_y + 9),
                          (diamond_x - 9, diamond_y)], dtype=np.int32)
    cv2.polylines(header, [diamond], True, (255, 0, 255), 3,
                  cv2.LINE_AA)
    cv2.putText(header, 'GT center', (457, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 220, 235), 1,
                cv2.LINE_AA)
    cv2.putText(header, 'Distances: BEV metres', (600, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 220, 235), 1,
                cv2.LINE_AA)
    cv2.putText(
        header,
        'Impact terms: H=query state/content | E=query positional embedding | '
        'X=image appearance feature | G3D=image 3D positional encoding | '
        'GMV=multiview positional encoding',
        (18, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 200, 220), 1,
        cv2.LINE_AA)
    cv2.putText(
        header,
        'Impact % = relative change in attention over all valid image tokens '
        'when that term is removed (six terms sum to 100% per layer)',
        (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 200, 220), 1,
        cv2.LINE_AA)
    cv2.imwrite(str(target_dir / 'attention_all_layers.png'),
                np.concatenate((header, canvas), axis=0))


def serializable_targets(targets, outputs, gt_boxes, classes,
                         metric_references, traces):
    result = {}
    for name, target in targets.items():
        query, gt = target['query_index'], target['gt_index']
        row = dict(target)
        row['class_name'] = classes[target['gt_label']]
        row['gt_box'] = gt_boxes[gt].detach().cpu().tolist()
        row['layer_confidence'] = outputs['all_cls_scores'][:, 0, query,
            target['gt_label']].sigmoid().detach().cpu().tolist()
        encoded = outputs['all_bbox_preds'][:, 0, query]
        decoded = denormalize_bbox(encoded, None)
        reference = metric_references[target['trace_slot']]
        gt_center = gt_boxes[gt, :3]
        row['layer_box_head_encoding'] = encoded.detach().cpu().tolist()
        row['layer_box_decoded'] = decoded.detach().cpu().tolist()
        row['reference_point_metric'] = reference.detach().cpu().tolist()
        row['reference_to_gt_bev_m'] = float(torch.norm(
            reference[:2] - gt_center[:2]))
        row['layer_prediction_to_gt_bev_m'] = torch.norm(
            decoded[:, :2] - gt_center[None, :2], dim=1).detach().cpu().tolist()
        row['layer_prediction_to_reference_bev_m'] = torch.norm(
            decoded[:, :2] - reference[None, :2], dim=1).detach().cpu().tolist()
        row['layer_component_impact_percent'] = [
            {name: float(values[target['trace_slot']])
             for name, values in trace['component_impact_percent'].items()}
            for trace in traces]
        result[name] = row
    return result


def bev_box_corners(box):
    """Return four oriented corners for a decoded 3D box in LiDAR BEV."""
    x, y, width, length, yaw = [float(box[index])
                                for index in (0, 1, 3, 4, 6)]
    local = np.asarray([[-width / 2, -length / 2],
                        [width / 2, -length / 2],
                        [width / 2, length / 2],
                        [-width / 2, length / 2]])
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw)],
                           [math.sin(yaw), math.cos(yaw)]])
    return local @ rotation.T + np.asarray([x, y])


def draw_bev_box(axis, box, color, label):
    corners = bev_box_corners(box)
    closed = np.concatenate((corners, corners[:1]), axis=0)
    axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.2,
              label=label, zorder=7)


def render_bev_rays(target_name, target, traces, outputs, gt_boxes,
                    reference_xyz, img_metas, feature_shape, pc_range,
                    max_ray_depth, layers, top_k, class_names, output_dir):
    """Draw the most-attended image-token rays in the LiDAR BEV frame."""
    cameras, height, width = feature_shape
    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    projections = img_metas[0]['lidar2img']
    camera_names = img_metas[0].get(
        'camera_names', ['CAM_{}'.format(i) for i in range(cameras)])
    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    gt_box = gt_boxes[target['gt_index']].detach().cpu().numpy()
    reference = np.asarray(reference_xyz)

    for layer_number in layers:
        layer_index = layer_number - 1
        attention = traces[layer_index]['attention'][
            :, target['trace_slot']].mean(0).numpy()
        valid_indices = np.flatnonzero(attention > 0)
        selected_count = min(top_k, len(valid_indices))
        if selected_count == 0:
            continue
        selected_local = np.argpartition(
            attention[valid_indices], -selected_count)[-selected_count:]
        indices = valid_indices[selected_local]
        indices = indices[np.argsort(attention[indices])]
        scores = attention[indices]
        segments, ray_cameras, ray_scores = [], [], []
        camera_origins, camera_directions, camera_fov_edges = {}, {}, {}
        max_depth = float(max_ray_depth)
        # Use the ray through the image centre as the camera's forward
        # direction. This comes from the same calibrated lidar2img transform
        # used to construct the attention rays.
        for camera in range(cameras):
            inverse = np.linalg.inv(np.asarray(projections[camera]))
            origin_h = inverse @ np.asarray([0.0, 0.0, 0.0, 1.0])
            forward_h = inverse @ np.asarray(
                [0.5 * pad_w * max_depth, 0.5 * pad_h * max_depth,
                 max_depth, 1.0])
            origin = origin_h[:3] / max(abs(origin_h[3]), 1e-12)
            forward = forward_h[:3] / max(abs(forward_h[3]), 1e-12)
            direction = forward[:2] - origin[:2]
            direction_norm = np.linalg.norm(direction)
            if (np.isfinite(origin).all() and np.isfinite(direction).all()
                    and direction_norm > 1e-12):
                camera_origins[camera] = origin[:2]
                camera_directions[camera] = direction / direction_norm
                edge_points = []
                for edge_u in (0.0, float(pad_w)):
                    edge_h = inverse @ np.asarray(
                        [edge_u * max_depth, 0.5 * pad_h * max_depth,
                         max_depth, 1.0])
                    edge = edge_h[:3] / max(abs(edge_h[3]), 1e-12)
                    if np.isfinite(edge).all():
                        edge_points.append(edge[:2])
                if len(edge_points) == 2:
                    camera_fov_edges[camera] = edge_points
        for flat_index in indices:
            camera = int(flat_index // (height * width))
            spatial = int(flat_index % (height * width))
            row, column = divmod(spatial, width)
            u = column * float(pad_w) / width
            v = row * float(pad_h) / height
            inverse = np.linalg.inv(np.asarray(projections[camera]))
            origin_h = inverse @ np.asarray([0.0, 0.0, 0.0, 1.0])
            far_h = inverse @ np.asarray(
                [u * max_depth, v * max_depth, max_depth, 1.0])
            origin = origin_h[:3] / max(abs(origin_h[3]), 1e-12)
            far = far_h[:3] / max(abs(far_h[3]), 1e-12)
            if not (np.isfinite(origin).all() and np.isfinite(far).all()):
                continue
            segments.append([origin[:2], far[:2]])
            ray_cameras.append(camera)
            ray_scores.append(float(attention[flat_index]))

        scores = np.asarray(ray_scores, dtype=np.float64)
        if len(scores) == 0:
            continue

        figure, axis = plt.subplots(figsize=(12, 12), dpi=220)
        score_min = max(float(scores.min()), 1e-12)
        score_max = max(float(scores.max()), score_min)
        if score_max > score_min:
            normalization = LogNorm(vmin=score_min, vmax=score_max)
            log_scores = np.log(np.clip(scores, score_min, None))
            width_scale = ((log_scores - math.log(score_min)) /
                           (math.log(score_max) - math.log(score_min)))
            scale_label = 'Mean attention weight per image token (log scale)'
        else:
            normalization = Normalize(vmin=0.0, vmax=score_max + 1e-12)
            width_scale = np.ones_like(scores)
            scale_label = 'Mean attention weight per image token'
        collection = LineCollection(
            segments, cmap='turbo', norm=normalization,
            linewidths=0.6 + 3.4 * width_scale,
            alpha=0.72, zorder=3)
        collection.set_array(scores)
        axis.add_collection(collection)
        colorbar = figure.colorbar(collection, ax=axis, fraction=0.045,
                                   pad=0.03)
        colorbar.set_label(scale_label)
        camera_colors = plt.cm.tab10(np.linspace(0, 1, cameras))
        for camera, origin in camera_origins.items():
            for edge in camera_fov_edges.get(camera, []):
                axis.plot([origin[0], edge[0]], [origin[1], edge[1]],
                          color=camera_colors[camera], linestyle=(0, (2, 3)),
                          linewidth=1.25, alpha=0.75, zorder=2)
            direction = camera_directions[camera]
            angle = math.degrees(math.atan2(direction[1], direction[0]))
            marker_style = MarkerStyle('^')
            marker_path = marker_style.get_path().transformed(
                marker_style.get_transform()).transformed(
                    Affine2D().rotate_deg(angle - 90.0))
            axis.scatter(origin[0], origin[1], marker=marker_path, s=90,
                         color=camera_colors[camera], edgecolor='black',
                         linewidth=0.5, zorder=8)

        query = target['query_index']
        predicted_label = int(outputs['all_cls_scores'][
            layer_index, 0, query].sigmoid().argmax())
        predicted_class = class_names[predicted_label]
        decoded = denormalize_bbox(
            outputs['all_bbox_preds'][layer_index, 0, query:query + 1],
            None)[0].detach().cpu().numpy()
        draw_bev_box(axis, gt_box, '#d62728', 'GT box')
        draw_bev_box(axis, decoded, '#2ca02c', 'Predicted box')
        axis.scatter(reference[0], reference[1], facecolors='none',
                     edgecolors='#00a8c6', linewidths=2.5, s=130,
                     label='Query reference', zorder=9)
        axis.scatter(decoded[0], decoded[1], marker='x', color='#d4a900',
                     linewidths=2.5, s=110, label='Predicted center', zorder=9)
        axis.scatter(gt_box[0], gt_box[1], marker='D', color='#b51fad',
                     s=75, label='GT center', zorder=9)
        axis.set_xlim(float(pc_range[0]), float(pc_range[3]))
        axis.set_ylim(float(pc_range[1]), float(pc_range[4]))
        axis.set_aspect('equal', adjustable='box')
        axis.set_xlabel('LiDAR x (m)')
        axis.set_ylabel('LiDAR y (m)')
        axis.grid(alpha=0.2)
        axis.set_title(
            '{} query {} | GT: {} | Pred: {} | decoder L{} | top {} rays | '
            'retained mass {:.3f}'.format(
                target_name, query, target['gt_class'], predicted_class,
                layer_number, len(segments), float(scores.sum())))
        object_handles, object_labels = axis.get_legend_handles_labels()
        camera_handles = []
        for camera in range(cameras):
            direction = camera_directions.get(camera, np.asarray([0.0, 1.0]))
            angle = math.degrees(math.atan2(direction[1], direction[0]))
            marker_style = MarkerStyle('^')
            marker_path = marker_style.get_path().transformed(
                marker_style.get_transform()).transformed(
                    Affine2D().rotate_deg(angle - 90.0))
            camera_handles.append(Line2D(
                [0], [0], marker=marker_path, linestyle=(0, (2, 3)),
                linewidth=1.25, markersize=8,
                markerfacecolor=camera_colors[camera], markeredgecolor='black',
                color=camera_colors[camera],
                label='C{} = {}'.format(camera, camera_names[camera])))
        axis.legend(object_handles + camera_handles,
                    object_labels + [handle.get_label()
                                     for handle in camera_handles],
                    loc='upper right', fontsize=7, framealpha=0.92,
                    title='Objects and cameras')
        figure.tight_layout()
        figure.savefig(str(
            target_dir / 'bev_ray_attention_L{:02d}.png'.format(layer_number)),
            bbox_inches='tight')
        plt.close(figure)


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if args.bev_ray_top_k < 1:
        raise ValueError('--bev-ray-top-k must be positive')
    invalid_layers = [layer for layer in args.bev_ray_layers
                      if layer < 1 or layer > 6]
    if invalid_layers:
        raise ValueError('--bev-ray-layers must contain values from 1 to 6')
    os.chdir(str(REPO_ROOT))
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    relocate_dataset_paths(dataset, cfg.data.test.data_root)
    index, token = resolve_sample(dataset, args)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    class_names = list(model.CLASSES)
    allowed_label_ids = None
    if args.classes:
        unknown = sorted(set(args.classes) - set(class_names))
        if unknown:
            raise ValueError('Unknown classes: {}. Available classes: {}'.format(
                ', '.join(unknown), ', '.join(class_names)))
        allowed_label_ids = {class_names.index(name) for name in args.classes}
    device = torch.device(args.device)
    model = model.to(device).eval()
    scatter_device = device.index if device.type == 'cuda' else device
    data = prepare_one(dataset, index, scatter_device)
    img, img_metas = first_augmentation(data)
    img_metas[0]['camera_names'] = list(dataset.data_infos[index]['cams'])
    head = model.pts_bbox_head

    with torch.no_grad():
        features = model.extract_feat(img=img, img_metas=img_metas)
        baseline = head(features, img_metas)
        gt_boxes, gt_labels, annotation = gt_targets(
            dataset, index, device, head.pc_range)
        candidates = query_candidates(
            head, baseline, gt_boxes, gt_labels, class_names,
            args.score_threshold, allowed_label_ids)
        output_dir = Path(args.output_dir) / token
        precheck_path = write_query_precheck(
            output_dir, candidates, token, index, args.score_threshold,
            args.classes or class_names)
        if args.list_queries:
            print('Query precheck written to:', precheck_path)
            return
        targets = choose_queries(candidates, args)
        query_indices = [target['query_index']
                         for target in targets.values()]
        traces = []
        learned_3d_pe, multiview_pe = position_components(
            head, features, img_metas)
        learned_3d_pe = flatten_keys(learned_3d_pe)
        multiview_pe = flatten_keys(multiview_pe)
        originals = install_attention_trace(
            head, query_indices, traces, learned_3d_pe, multiview_pe)
        try:
            replay_outputs = head(features, img_metas)
        finally:
            restore_attention(originals)

    output_difference = float(max(
        (baseline['all_bbox_preds'] - replay_outputs['all_bbox_preds']).abs().max(),
        (baseline['all_cls_scores'] - replay_outputs['all_cls_scores']).abs().max()))
    max_manual_error = max(
        row['manual_replay_max_abs_error'] for row in traces)
    max_native_error = max(
        row['native_replay_max_abs_error'] for row in traces)
    max_weight_error = max(
        row['attention_weight_max_abs_error'] for row in traces)
    max_position_split_error = max(
        row['position_split_max_abs_error'] for row in traces)
    if (output_difference > 1e-5 or max_native_error > 1e-5 or
            max_position_split_error > 1e-5):
        raise AssertionError(
            'Attention validation failed: output={} native={} PE split={}'.format(
                output_difference, max_native_error,
                max_position_split_error))
    if max_weight_error > 1e-5:
        print('WARNING: explicit per-head versus native averaged attention '
              'differs by {:.6g}; retained in trace metadata. This is expected '
              'with the legacy PyTorch MHA implementation.'.format(
                  max_weight_error))
    if max_weight_error > 5e-2:
        raise AssertionError(
            'Per-head attention reconstruction has a gross mismatch: {}'.format(
                max_weight_error))

    batch, num_cams, _, feat_h, feat_w = features[0].shape
    assert batch == 1 and len(traces) == len(head.transformer.decoder.layers)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_references = head.reference_points.weight[query_indices]
    pc_range = selected_references.new_tensor(head.pc_range)
    metric_references = (selected_references *
                         (pc_range[3:6] - pc_range[:3]) + pc_range[:3])
    torch.save({
        'sample_token': token,
        'dataset_index': index,
        'scene_name': args.scene_name,
        'scene_frame_index': args.scene_frame_index,
        'query_indices': query_indices,
        'reference_points_normalized': tensor_cpu(
            selected_references),
        'reference_points_metric': tensor_cpu(metric_references),
        'traces': traces,
        'all_cls_scores': tensor_cpu(baseline['all_cls_scores']),
        'all_bbox_preds': tensor_cpu(baseline['all_bbox_preds']),
        'key_layout': dict(num_cameras=num_cams, height=feat_h, width=feat_w,
                           order='camera,y,x'),
    }, str(output_dir / 'trace_tensors.pt'))
    images = model_images(img, cfg)
    for name, target in targets.items():
        render_target(
            name, target, traces, baseline, gt_boxes,
            metric_references[target['trace_slot']].detach().cpu().numpy(),
            annotation, images, img_metas, (num_cams, feat_h, feat_w),
            class_names, output_dir)
        if args.plot_bev_rays:
            render_bev_rays(
                name, target, traces, baseline, gt_boxes,
                metric_references[target['trace_slot']].detach().cpu().numpy(),
                img_metas, (num_cams, feat_h, feat_w), head.pc_range,
                head.position_range[3], args.bev_ray_layers,
                args.bev_ray_top_k, class_names, output_dir)
    metadata = {
        'sample_token': token,
        'dataset_index': index,
        'score_threshold': args.score_threshold,
        'selected_classes': args.classes or class_names,
        'targets': serializable_targets(
            targets, baseline, gt_boxes, class_names, metric_references,
            traces),
        'feature_shape': [num_cams, feat_h, feat_w],
        'bev_ray_visualization': {
            'enabled': args.plot_bev_rays,
            'layers': args.bev_ray_layers,
            'top_k': args.bev_ray_top_k,
        },
        'manual_attention_replay_max_abs_error': max_manual_error,
        'native_attention_replay_max_abs_error': max_native_error,
        'attention_weight_max_abs_error': max_weight_error,
        'position_split_max_abs_error': max_position_split_error,
        'component_impact_definition': (
            'Per component: mean Jensen-Shannon divergence across attention '
            'heads between full attention and leave-one-component-out '
            'attention over all valid image tokens; normalized to 100% across '
            'the six components for each query and decoder layer.'),
        'second_forward_max_abs_error': output_difference,
    }
    with open(str(output_dir / 'trace_summary.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print('Trace written to:', output_dir)


if __name__ == '__main__':
    main()
