#!/usr/bin/env python
"""Select informative keyframes from any PETR-compatible dataset."""

import argparse
import csv
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rank PETR keyframes containing successes and failures.')
    parser.add_argument('--config', default=str(
        REPO_ROOT / 'projects/configs/petr/'
        'petr_r50dcn_gridmask_p4_nuscenes_local.py'))
    parser.add_argument('--checkpoint', default=str(
        REPO_ROOT / 'ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth'))
    parser.add_argument('--output-dir', default=str(
        REPO_ROOT / 'experiments/keyframe_selection/outputs'))
    parser.add_argument('--data-root', default=None,
                        help='Override cfg.data.test.data_root.')
    parser.add_argument('--ann-file', default=None,
                        help='Override cfg.data.test.ann_file.')
    parser.add_argument('--dataset', '--dataset-name', dest='dataset_name',
                        default=None,
                        help='Dataset/camera-rig label. If data/pccr/<name> '
                             'exists, its validation split is selected unless '
                             '--data-root/--ann-file override it.')
    parser.add_argument('--scene-name', default=None,
                        help='Restrict the scan, e.g. scene-0103.')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Samples to scan when --scene-name is omitted; 0=all.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-threshold', type=float, default=0.35)
    parser.add_argument('--classes', nargs='+', default=None,
                        help='Restrict ranking to class names, e.g. car pedestrian truck.')
    parser.add_argument('--clean-error', type=float, default=1.0,
                        help='Maximum BEV center error for a clean success.')
    parser.add_argument('--poor-error', type=float, default=2.0,
                        help='Minimum BEV center error for poor localization.')
    parser.add_argument('--max-match-distance', type=float, default=4.0,
                        help='Maximum class-aware BEV matching distance.')
    parser.add_argument('--top-k', type=int, default=20)
    return parser.parse_args()


def import_plugin(cfg):
    if cfg.get('plugin', False):
        module_path = os.path.dirname(cfg.get('plugin_dir', '')).replace('/', '.')
        importlib.import_module(module_path)


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


def load_scene_tables(dataset):
    """Load nuScenes-style scene tables when the dataset provides them."""
    versions = [getattr(dataset, 'version', None), 'v1.0-trainval']
    for version in versions:
        if not version:
            continue
        table_root = Path(dataset.data_root) / version
        sample_path = table_root / 'sample.json'
        scene_path = table_root / 'scene.json'
        if not sample_path.is_file() or not scene_path.is_file():
            continue
        with sample_path.open('r') as handle:
            samples = {row['token']: row for row in json.load(handle)}
        with scene_path.open('r') as handle:
            scenes = {row['token']: row for row in json.load(handle)}
        return samples, scenes
    return {}, {}


def scene_identity(info, samples, scenes, dataset_name):
    """Return a scene/log label without requiring nuScenes tables."""
    sample = samples.get(info.get('token'))
    if sample is not None:
        scene = scenes.get(sample.get('scene_token'))
        if scene is not None:
            return scene.get('name', scene.get('token'))
    for key in ('scene_name', 'scene_token', 'log_name', 'log_id',
                'log_token', 'sequence_name', 'sequence_id'):
        if info.get(key) is not None:
            return str(info[key])
    return dataset_name or 'dataset'


def candidate_indices(dataset, samples, scenes, scene_name, max_samples, seed,
                      dataset_name):
    scene_rows = {}
    for index, info in enumerate(dataset.data_infos):
        scene = scene_identity(info, samples, scenes, dataset_name)
        if scene_name is None or scene == scene_name:
            scene_rows.setdefault(scene, []).append(index)
    rows = []
    for scene, indices in scene_rows.items():
        indices.sort(key=lambda i: dataset.data_infos[i]['timestamp'])
        rows.extend((index, scene, frame_index)
                    for frame_index, index in enumerate(indices))
    if scene_name is not None and not rows:
        raise ValueError('Scene is not represented in validation infos: ' + scene_name)
    if scene_name is None:
        random.Random(seed).shuffle(rows)
        if max_samples > 0:
            rows = rows[:max_samples]
    else:
        rows.sort(key=lambda row: row[2])
    return rows


def prepare_one(dataset, index, device):
    sample = dataset.prepare_test_data(index)
    if sample is None:
        raise RuntimeError('Dataset returned no sample for index {}'.format(index))
    return scatter(collate([sample], samples_per_gpu=1), [device])[0]


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def class_aware_matches(gt_centers, gt_labels, pred_centers, pred_labels,
                        max_distance):
    """Greedily select the globally closest valid class-aware BEV pairs."""
    candidates = []
    for pred_index in range(len(pred_centers)):
        for gt_index in range(len(gt_centers)):
            if pred_labels[pred_index] != gt_labels[gt_index]:
                continue
            distance = float(np.linalg.norm(
                pred_centers[pred_index, :2] - gt_centers[gt_index, :2]))
            if distance <= max_distance:
                candidates.append((distance, pred_index, gt_index))
    matches, used_pred, used_gt = [], set(), set()
    for distance, pred_index, gt_index in sorted(candidates):
        if pred_index in used_pred or gt_index in used_gt:
            continue
        matches.append((pred_index, gt_index, distance))
        used_pred.add(pred_index)
        used_gt.add(gt_index)
    return matches, used_pred, used_gt


def camera_visibility(raw_info, gt_centers):
    visible = np.zeros(len(gt_centers), dtype=np.int64)
    # get_data_info() precedes the image-loading pipeline, so image dimensions
    # are not present yet. The images are already warm in the OS cache after
    # inference; reading their headers here keeps this independent of the
    # DataContainer representation used by different MMCV versions.
    shapes = []
    for path in raw_info.get('img_filename', []):
        image = mmcv.imread(path)
        if image is None:
            raise FileNotFoundError('Unable to read camera image: {}'.format(path))
        shapes.append(image.shape)
    for matrix, shape in zip(raw_info['lidar2img'], shapes):
        xyz1 = np.concatenate(
            (gt_centers[:, :3], np.ones((len(gt_centers), 1))), axis=1)
        projected = xyz1 @ np.asarray(matrix).T
        depth = projected[:, 2]
        uv = projected[:, :2] / np.maximum(depth[:, None], 1e-5)
        height, width = shape[:2]
        inside = ((depth > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < height))
        visible += inside.astype(np.int64)
    return visible


def analyze_frame(dataset, index, scene_name, scene_frame_index, prediction,
                  args, allowed_label_ids=None):
    annotation = dataset.get_ann_info(index)
    gt_labels = as_numpy(annotation['gt_labels_3d']).astype(np.int64)
    valid_gt = gt_labels >= 0
    if allowed_label_ids is not None:
        valid_gt &= np.isin(gt_labels, list(allowed_label_ids))
    gt_labels = gt_labels[valid_gt]
    gt_centers = as_numpy(annotation['gt_bboxes_3d'].gravity_center)[valid_gt]
    class_names = list(dataset.CLASSES)

    scores = as_numpy(prediction['scores_3d'])
    pred_all_labels = as_numpy(prediction['labels_3d']).astype(np.int64)
    keep = scores >= args.score_threshold
    if allowed_label_ids is not None:
        keep &= np.isin(pred_all_labels, list(allowed_label_ids))
    scores = scores[keep]
    pred_labels = pred_all_labels[keep]
    pred_centers = as_numpy(prediction['boxes_3d'].gravity_center)[keep]
    matches, used_pred, used_gt = class_aware_matches(
        gt_centers, gt_labels, pred_centers, pred_labels,
        args.max_match_distance)

    raw_info = dataset.get_data_info(index)
    visibility = camera_visibility(raw_info, gt_centers)
    gt_class_counts = {
        name: int((gt_labels == label).sum())
        for label, name in enumerate(class_names)
        if int((gt_labels == label).sum()) > 0}
    visible_gt_class_counts = {
        name: int(((gt_labels == label) & (visibility > 0)).sum())
        for label, name in enumerate(class_names)
        if int(((gt_labels == label) & (visibility > 0)).sum()) > 0}
    visible_objects = [
        {
            'gt_index': int(gt_index),
            'class': class_names[int(gt_labels[gt_index])],
            'distance_m': round(float(np.linalg.norm(
                gt_centers[gt_index, :2])), 3),
            'cameras_visible': int(visibility[gt_index]),
        }
        for gt_index in range(len(gt_centers)) if visibility[gt_index] > 0]
    clean = sum(distance <= args.clean_error for _, _, distance in matches)
    poor = sum(distance >= args.poor_error for _, _, distance in matches)
    multicam_gt = int((visibility >= 2).sum())
    false_positives = len(pred_centers) - len(used_pred)
    missed_gt = len(gt_centers) - len(used_gt)
    mean_score = float(scores.mean()) if len(scores) else 0.0
    satisfies = clean > 0 and poor > 0 and multicam_gt > 0
    rank_score = (100.0 * int(satisfies) + 4.0 * clean + 3.0 * poor +
                  2.0 * multicam_gt - 0.75 * false_positives -
                  0.5 * missed_gt + mean_score)
    token = dataset.data_infos[index]['token']
    return {
        'rank_score': rank_score,
        'satisfies_all': int(satisfies),
        'scene_name': scene_name,
        'scene_frame_index': scene_frame_index,
        'scene_frame_number': scene_frame_index + 1,
        'dataset_index': index,
        'sample_token': token,
        'timestamp': int(dataset.data_infos[index]['timestamp']),
        'num_gt': len(gt_centers),
        'gt_class_counts': json.dumps(gt_class_counts, sort_keys=True),
        'visible_gt_class_counts': json.dumps(
            visible_gt_class_counts, sort_keys=True),
        'visible_objects': json.dumps(visible_objects, sort_keys=True),
        'multicamera_gt': multicam_gt,
        'predictions_above_threshold': len(pred_centers),
        'matched_predictions': len(matches),
        'clean_successes': clean,
        'poor_localizations': poor,
        'false_positives': false_positives,
        'missed_gt': missed_gt,
        'mean_kept_confidence': mean_score,
        'best_center_error_m': min((m[2] for m in matches), default=None),
        'worst_matched_center_error_m': max((m[2] for m in matches), default=None),
        'max_gt_camera_count': int(visibility.max()) if len(visibility) else 0,
    }


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if args.poor_error <= args.clean_error:
        raise ValueError('--poor-error must be greater than --clean-error')
    os.chdir(str(REPO_ROOT))
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if args.dataset_name is not None:
        pccr_root = REPO_ROOT / 'data' / 'pccr' / args.dataset_name
        pccr_val = pccr_root / '{}_infos_val.pkl'.format(args.dataset_name)
        if args.data_root is None and pccr_root.is_dir():
            args.data_root = str(pccr_root) + '/'
        if args.ann_file is None and pccr_val.is_file():
            args.ann_file = str(pccr_val)
    if args.data_root is not None:
        cfg.data.test.data_root = args.data_root
        if 'data_root' in cfg:
            cfg.data_root = args.data_root
    if args.ann_file is not None:
        cfg.data.test.ann_file = args.ann_file
        if 'test_ann_file' in cfg:
            cfg.test_ann_file = args.ann_file
    if args.dataset_name is not None and 'dataset_name' in cfg:
        cfg.dataset_name = args.dataset_name
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    relocate_dataset_paths(dataset, cfg.data.test.data_root)
    samples, scenes = load_scene_tables(dataset)
    candidates = candidate_indices(
        dataset, samples, scenes, args.scene_name, args.max_samples, args.seed,
        args.dataset_name)

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

    records = []
    with torch.no_grad():
        for position, (index, scene_name, scene_frame_index) in enumerate(
                candidates, 1):
            data = prepare_one(dataset, index, scatter_device)
            output = model(return_loss=False, rescale=True, **data)[0]
            prediction = output.get('pts_bbox', output)
            record = analyze_frame(
                dataset, index, scene_name, scene_frame_index, prediction,
                args, allowed_label_ids)
            records.append(record)
            print('[{}/{}] {} frame={} index={} clean={} poor={} fp={}'.format(
                position, len(candidates), scene_name,
                scene_frame_index + 1, index,
                record['clean_successes'], record['poor_localizations'],
                record['false_positives']), flush=True)

    records.sort(key=lambda row: (
        -row['satisfies_all'], -row['rank_score'], row['dataset_index']))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys()) if records else []
    with open(str(output_dir / 'ranked_keyframes.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    result = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'dataset_name': args.dataset_name,
        'data_root': str(Path(cfg.data.test.data_root).resolve()),
        'ann_file': str(Path(cfg.data.test.ann_file).resolve()),
        'score_threshold': args.score_threshold,
        'selected_classes': args.classes or class_names,
        'clean_error_m': args.clean_error,
        'poor_error_m': args.poor_error,
        'max_match_distance_m': args.max_match_distance,
        'num_scanned': len(records),
        'selection': records[0] if records else None,
        'top_keyframes': records[:args.top_k],
        'notes': [
            'Matching is class-aware and based on BEV center distance.',
            'Visibility means the GT center projects inside at least two cameras.',
            'This is a diagnostic selector, not the official nuScenes metric.',
        ],
    }
    with open(str(output_dir / 'selection.json'), 'w') as handle:
        json.dump(result, handle, indent=2)
    if not records:
        raise RuntimeError('No candidate validation samples were found')
    print('\nSelected keyframe:\n' + json.dumps(records[0], indent=2))
    print('Results:', output_dir)


if __name__ == '__main__':
    main()
