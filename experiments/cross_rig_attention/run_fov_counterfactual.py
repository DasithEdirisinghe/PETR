#!/usr/bin/env python3
"""Test calibration-consistent FoV normalization with a frozen PETR checkpoint.

Conditions: raw target image, image-space warp to source virtual cameras,
and approximate post-backbone feature warp to the same virtual cameras.
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_car_model_comparison import scene_frame_lookup
from experiments.query_trace.trace_query import (
    first_augmentation, import_plugin, prepare_one, relocate_dataset_paths)
from plot_car_bev_localization import quaternion_matrix


DEFAULT_CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-rig', default='R1')
    parser.add_argument('--target-rig', default='R1-f')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint', default=None,
                        help='Defaults to the checkpoint trained on --source-rig.')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--modes', nargs='+', default=[
        'raw', 'image_warp', 'feature_warp', 'source_view'],
        choices=['raw', 'image_warp', 'feature_warp', 'source_view'])
    parser.add_argument('--max-frames', type=int, default=0,
                        help='0 = full split with official AP; positive = smoke test, no AP.')
    parser.add_argument('--sample-tokens-file', default=None,
                        help='Optional JSON list or trace_candidates.json; no official AP.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--max-camera-shift-m', type=float, default=0.02)
    parser.add_argument('--max-camera-rotation-deg', type=float, default=0.1)
    parser.add_argument('--max-projection-residual', type=float, default=1e-3)
    parser.add_argument('--baseline-ap-tolerance', type=float, default=.002,
                        help='Maximum absolute difference from cached raw-target '
                             'car AP, when that cache exists.')
    return parser.parse_args()


def lidar_to_global_matrix(info):
    lidar_to_ego = np.eye(4)
    lidar_to_ego[:3, :3] = quaternion_matrix(info['lidar2ego_rotation'])
    lidar_to_ego[:3, 3] = info['lidar2ego_translation']
    ego_to_global = np.eye(4)
    ego_to_global[:3, :3] = quaternion_matrix(info['ego2global_rotation'])
    ego_to_global[:3, 3] = info['ego2global_translation']
    return ego_to_global @ lidar_to_ego


def camera_pose(info, name):
    cam = info['cams'][name]
    camera_to_lidar = np.eye(4)
    camera_to_lidar[:3, :3] = np.asarray(cam['sensor2lidar_rotation'])
    camera_to_lidar[:3, 3] = np.asarray(cam['sensor2lidar_translation'])
    return lidar_to_global_matrix(info) @ camera_to_lidar


def rotation_degrees(a, b):
    cosine = np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def camera_homographies(source_info, target_info, source_meta, target_meta,
                        options):
    names = list(target_info['cams'])
    if list(source_info['cams']) != names:
        raise ValueError('Camera names/order differ; no safe token alignment')
    source_lidar_to_global = lidar_to_global_matrix(source_info)
    target_lidar_to_global = lidar_to_global_matrix(target_info)
    homographies, diagnostics = [], []
    for index, name in enumerate(names):
        source_pose = camera_pose(source_info, name)
        target_pose = camera_pose(target_info, name)
        shift = float(np.linalg.norm(source_pose[:3, 3] - target_pose[:3, 3]))
        turn = rotation_degrees(source_pose[:3, :3], target_pose[:3, :3])
        if shift > options.max_camera_shift_m or turn > options.max_camera_rotation_deg:
            raise ValueError('{} pose differs: {:.4f} m / {:.4f} deg; FoV-only '
                             'homography is invalid'.format(name, shift, turn))
        source_p = (np.asarray(source_meta['lidar2img'][index], dtype=np.float64)[:3]
                    @ np.linalg.inv(source_lidar_to_global))
        target_p = (np.asarray(target_meta['lidar2img'][index], dtype=np.float64)[:3]
                    @ np.linalg.inv(target_lidar_to_global))
        homography = source_p[:, :3] @ np.linalg.inv(target_p[:, :3])
        reconstructed = homography @ target_p
        scale = float(np.sum(source_p * reconstructed) /
                      np.sum(reconstructed * reconstructed))
        residual = float(np.linalg.norm(source_p - scale * reconstructed) /
                         np.linalg.norm(source_p))
        if residual > options.max_projection_residual:
            raise ValueError('{} 3x4 projection is not related by a homography: '
                             'residual {:.6g}'.format(name, residual))
        homography /= homography[2, 2]
        homographies.append(homography)
        diagnostics.append({'camera': name, 'pose_shift_m': shift,
                            'pose_rotation_deg': turn,
                            'projection_residual': residual,
                            'source_focal_x': float(source_info['cams'][name]['cam_intrinsic'][0][0]),
                            'target_focal_x': float(target_info['cams'][name]['cam_intrinsic'][0][0])})
    return homographies, diagnostics


def virtual_meta(source_meta, target_meta, homographies):
    meta = copy.deepcopy(target_meta)
    meta['img_shape'] = copy.deepcopy(source_meta['img_shape'])
    meta['pad_shape'] = copy.deepcopy(source_meta['pad_shape'])
    meta['input_shape'] = tuple(source_meta['pad_shape'][0][:2])
    projected = []
    for index, homography in enumerate(homographies):
        transform = np.eye(4)
        transform[:3, :3] = homography
        projected.append(transform @ np.asarray(target_meta['lidar2img'][index]))
    meta['lidar2img'] = projected
    return meta


def warp_images(images, homographies, output_shape):
    height, width = output_shape
    warped, valid_fractions, valid_masks = [], [], []
    for index, homography in enumerate(homographies):
        array = images[0, index].detach().float().cpu().permute(1, 2, 0).numpy()
        result = cv2.warpPerspective(array, homography, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        valid = cv2.warpPerspective(
            np.ones(array.shape[:2], dtype=np.uint8), homography, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        valid_fractions.append(float(valid.mean()))
        valid_masks.append(torch.from_numpy(valid.astype(np.float32)))
        warped.append(torch.from_numpy(result).permute(2, 0, 1))
    return torch.stack(warped, 0).unsqueeze(0).to(
        device=images.device, dtype=images.dtype), valid_fractions, torch.stack(
            valid_masks, 0).unsqueeze(0).unsqueeze(2).to(images.device)


def image_to_feature_matrix(image_shape, feature_shape):
    image_h, image_w = image_shape
    feature_h, feature_w = feature_shape
    sx, sy = image_w / feature_w, image_h / feature_h
    return np.asarray([[sx, 0, (sx - 1) / 2],
                       [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)


def warp_features(features, homographies, source_image_shape,
                  target_image_shape, source_feature_shapes):
    output = []
    for level, source_feature_shape in zip(features, source_feature_shapes):
        _, cameras, _, target_h, target_w = level.shape
        source_h, source_w = source_feature_shape
        target_to_image = image_to_feature_matrix(
            target_image_shape, (target_h, target_w))
        source_to_image = image_to_feature_matrix(
            source_image_shape, (source_h, source_w))
        aligned = []
        for camera in range(cameras):
            feature_h = (np.linalg.inv(source_to_image) @ homographies[camera]
                         @ target_to_image)
            inverse = torch.as_tensor(np.linalg.inv(feature_h),
                                      device=level.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(
                torch.arange(source_h, device=level.device),
                torch.arange(source_w, device=level.device))
            points = torch.stack([xx, yy, torch.ones_like(xx)], -1).float()
            mapped = points @ inverse.T
            xy = mapped[..., :2] / mapped[..., 2:].clamp_min(1e-8)
            grid = torch.stack([2 * xy[..., 0] / max(target_w-1, 1)-1,
                                2 * xy[..., 1] / max(target_h-1, 1)-1], -1)
            aligned.append(F.grid_sample(level[:, camera].float(),
                                         grid.unsqueeze(0), mode='bilinear',
                                         padding_mode='zeros', align_corners=True))
        output.append(torch.stack(aligned, dim=1).to(level.dtype))
    return output


def cpu_result(result):
    box = result[0]['pts_bbox']
    return {'pts_bbox': {key: value.to('cpu') for key, value in box.items()}}


def selected_tokens(path):
    with open(path, 'r') as handle:
        values = json.load(handle)
    if isinstance(values, dict):
        values = values.get('selection', values)
    if not isinstance(values, list):
        raise ValueError('Sample selection must be a JSON list')
    return [row if isinstance(row, str) else row['sample_token'] for row in values]


def main():
    options = arguments()
    if options.source_rig == options.target_rig:
        raise ValueError('Source and target rigs must differ')
    if options.max_frames < 0 or options.save_every < 1:
        raise ValueError('--max-frames must be >=0 and --save-every >=1')
    output = Path(options.output_dir or
                  ROOT / 'experiments/cross_rig_attention/output/fov_counterfactual' /
                  '{}_to_{}_{}'.format(options.target_rig.lower().replace('-', ''),
                                       options.source_rig.lower().replace('-', ''),
                                       options.split))
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(options.config)
    import_plugin(cfg)
    cfg.model.pretrained = None

    def dataset_for(rig):
        definition = copy.deepcopy(cfg.data.test)
        definition.data_root = 'data/pccr/{}/'.format(rig)
        definition.ann_file = 'data/pccr/{0}/{0}_infos_{1}.pkl'.format(
            rig, options.split)
        definition.test_mode = True
        dataset = build_dataset(definition)
        relocate_dataset_paths(dataset, definition.data_root)
        return dataset

    source_dataset = dataset_for(options.source_rig)
    target_dataset = dataset_for(options.target_rig)
    source_by_token = {info['token']: index for index, info in
                       enumerate(source_dataset.data_infos)}
    source_lookup, source_scenes = scene_frame_lookup(options.source_rig,
                                                      options.split)
    target_lookup, _ = scene_frame_lookup(options.target_rig, options.split)
    del source_lookup
    all_tokens = [info['token'] for info in target_dataset.data_infos]
    if options.sample_tokens_file:
        requested = selected_tokens(options.sample_tokens_file)
        if len(requested) != len(set(requested)):
            raise ValueError('Duplicate sample tokens in selection')
        unknown = set(requested) - set(all_tokens)
        if unknown:
            raise ValueError('Unknown target sample tokens: {}'.format(sorted(unknown)[:5]))
        tokens = requested
    else:
        tokens = all_tokens
    if options.max_frames:
        tokens = tokens[:options.max_frames]
    target_by_token = {token: index for index, token in enumerate(all_tokens)}
    complete_split = len(tokens) == len(all_tokens) and set(tokens) == set(all_tokens)
    if complete_split and tokens != all_tokens:
        raise ValueError('Full evaluation requires dataset annotation order')

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = (options.checkpoint or str(
        ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr' /
        options.source_rig / 'latest.pth'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    device = torch.device(options.device)
    model.to(device).eval()
    scatter_device = device.index if device.type == 'cuda' else device

    for mode in options.modes:
        mode_dir = output / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        progress = mode_dir / 'progress.pkl'
        if progress.exists():
            with progress.open('rb') as handle:
                saved = pickle.load(handle)
            if saved['tokens'] != tokens[:len(saved['results'])]:
                raise ValueError('Cached progress token order differs: ' + str(progress))
            results, audits = saved['results'], saved['audits']
        else:
            results, audits = [], []
        for position in range(len(results), len(tokens)):
            token = tokens[position]
            scene, frame_index = target_lookup[token]
            if scene not in source_scenes or frame_index >= len(source_scenes[scene]):
                raise ValueError('No paired source frame for {} {}'.format(scene, frame_index))
            source_token = source_scenes[scene][frame_index]
            source_index = source_by_token[source_token]
            target_index = target_by_token[token]
            source_data = prepare_one(source_dataset, source_index, scatter_device)
            target_data = prepare_one(target_dataset, target_index, scatter_device)
            source_img, source_metas = first_augmentation(source_data)
            target_img, target_metas = first_augmentation(target_data)
            source_info = source_dataset.data_infos[source_index]
            target_info = target_dataset.data_infos[target_index]
            source_metas[0]['camera_names'] = list(source_info['cams'])
            target_metas[0]['camera_names'] = list(target_info['cams'])
            homographies, cameras = camera_homographies(
                source_info, target_info, source_metas[0], target_metas[0], options)
            source_shape = tuple(source_img.shape[-2:])
            target_shape = tuple(target_img.shape[-2:])
            meta = virtual_meta(source_metas[0], target_metas[0], homographies)
            with torch.no_grad():
                photometric_mae = None
                if mode == 'raw':
                    prediction = model.simple_test(target_metas,
                                                   target_img.clone())
                    coverage = None
                elif mode == 'image_warp':
                    warped, coverage, valid_mask = warp_images(
                        target_img, homographies, source_shape)
                    difference = (warped.float() - source_img.float()).abs()
                    photometric_mae = float((difference*valid_mask).sum() /
                                             (valid_mask.sum()*difference.shape[2]).clamp_min(1))
                    prediction = model.simple_test([meta], warped)
                elif mode == 'source_view':
                    # Paired R1 image upper bound, decoded in target LiDAR frame.
                    # It is never used to adapt the model or construct the warp.
                    prediction = model.simple_test([meta], source_img.clone())
                    coverage = None
                else:
                    target_features = model.extract_feat(
                        img=target_img.clone(), img_metas=target_metas)
                    source_features = model.extract_feat(
                        img=source_img.clone(), img_metas=source_metas)
                    feature_shapes = [level.shape[-2:] for level in source_features]
                    aligned = warp_features(target_features, homographies,
                                            source_shape, target_shape,
                                            feature_shapes)
                    prediction = [{'pts_bbox': value} for value in
                                  model.simple_test_pts(aligned, [meta])]
                    coverage = None
            results.append(cpu_result(prediction))
            audits.append({'target_token': token, 'source_token': source_token,
                           'scene': scene, 'frame_index': frame_index,
                           'cameras': cameras, 'image_overlap': coverage,
                           'overlap_normalized_image_mae': photometric_mae})
            if (position + 1) % options.save_every == 0 or position + 1 == len(tokens):
                with progress.open('wb') as handle:
                    pickle.dump({'tokens': tokens[:len(results)], 'results': results,
                                 'audits': audits}, handle)
                print(mode, position + 1, '/', len(tokens), flush=True)
        with (mode_dir / 'geometry_audit.json').open('w') as handle:
            json.dump(audits, handle, indent=2)
        if complete_split:
            metrics_path = mode_dir / 'formatted/metrics_summary.json'
            if not metrics_path.is_file():
                metrics = target_dataset.evaluate(
                    results, metric='bbox', jsonfile_prefix=str(mode_dir / 'formatted'))
                with (mode_dir / 'evaluation.json').open('w') as handle:
                    json.dump(metrics, handle, indent=2)
            with metrics_path.open('r') as handle:
                current_ap = json.load(handle)['label_aps']['car']
            print(mode, 'car AP:', current_ap, flush=True)
            if mode == 'raw':
                cached_rig = options.target_rig.lower().replace('-', '')
                saved_summary = (ROOT / 'experiments/cross_rig_attention/output' /
                                 '{}_{}'.format(cached_rig, options.split) /
                                 options.source_rig / 'formatted/metrics_summary.json')
                if saved_summary.is_file():
                    with saved_summary.open('r') as handle:
                        expected = json.load(handle)['label_aps']['car']
                    differences = {threshold: abs(current_ap[threshold]-expected[threshold])
                                   for threshold in expected}
                    if max(differences.values()) > options.baseline_ap_tolerance:
                        raise AssertionError('Raw run does not reproduce cached '
                                             'baseline car AP: {}'.format(differences))
        else:
            print(mode, 'smoke test: official AP skipped for partial split', flush=True)
    print('Saved FoV counterfactual experiment:', output)


if __name__ == '__main__':
    main()
