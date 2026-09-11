"""Run PETR on a nuScenes scene and render GT/prediction camera canvases.

The upper row contains ground-truth boxes and the lower row predictions. Each
row contains the six cameras in the order stored in the nuScenes info file.
"""

import argparse
import importlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render a 2x6 GT/prediction nuScenes PETR canvas.')
    parser.add_argument(
        '--config', default='projects/configs/petr/'
        'petr_r50dcn_gridmask_p4_nuscenes_local.py')
    parser.add_argument(
        '--checkpoint', default='ckpts/'
        'petr_r50dcn_gridmask_p4_epoch_24.pth')
    parser.add_argument('--output-dir', default='visualizations/nuscenes_sanity')
    parser.add_argument('--sample-index', type=int, default=None,
                        help='Render this dataset index instead of a random scene.')
    parser.add_argument('--scene-name', default=None,
                        help='nuScenes scene name, for example scene-0103.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1,
                        help='0 means every available frame in the scene.')
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--tile-width', type=int, default=480)
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--model-label', default='PETR R50-P4 1408x512')
    parser.add_argument('--video', action='store_true')
    return parser.parse_args()


def import_plugin(cfg):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', '')
    module_path = os.path.dirname(plugin_dir).replace('/', '.')
    importlib.import_module(module_path)


def table_by_token(path):
    with open(path, 'r') as handle:
        return {record['token']: record for record in json.load(handle)}


def relocate_dataset_paths(dataset, data_root):
    """Repair absolute paths embedded by another checkout of nuScenes."""
    root = Path(data_root).resolve()

    def relocate(value):
        if not isinstance(value, str) or os.path.exists(value):
            return value
        normalized = value.replace('\\', '/')
        for directory in ('samples', 'sweeps', 'maps'):
            marker = '/' + directory + '/'
            if marker in normalized:
                relative = directory + '/' + normalized.split(marker, 1)[1]
                candidate = root / relative
                if candidate.exists():
                    return str(candidate)
        return value

    for info in dataset.data_infos:
        if 'lidar_path' in info:
            info['lidar_path'] = relocate(info['lidar_path'])
        for camera in info.get('cams', {}).values():
            if 'data_path' in camera:
                camera['data_path'] = relocate(camera['data_path'])
        for sweep in info.get('sweeps', []):
            if 'data_path' in sweep:
                sweep['data_path'] = relocate(sweep['data_path'])


def select_indices(dataset, data_root, sample_index, scene_name, seed):
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError('sample-index is outside the validation dataset')
        return [sample_index], 'sample-index-{:06d}'.format(sample_index)

    table_root = Path(data_root) / 'v1.0-trainval'
    samples = table_by_token(table_root / 'sample.json')
    scenes = table_by_token(table_root / 'scene.json')
    scene_to_indices = {}
    for index, info in enumerate(dataset.data_infos):
        token = info['token']
        if token not in samples:
            continue
        scene_token = samples[token]['scene_token']
        scene_to_indices.setdefault(scene_token, []).append(index)
    if not scene_to_indices:
        raise RuntimeError('No validation info tokens matched sample.json')

    if scene_name:
        matching = [token for token in scene_to_indices
                    if scenes[token]['name'] == scene_name]
        if not matching:
            raise ValueError('Scene is not represented in the validation infos: '
                             + scene_name)
        scene_token = matching[0]
    else:
        scene_token = random.Random(seed).choice(sorted(scene_to_indices))
    indices = sorted(scene_to_indices[scene_token],
                     key=lambda i: dataset.data_infos[i]['timestamp'])
    return indices, scenes[scene_token]['name']


def palette(label):
    colors = ((41, 121, 255), (65, 180, 75), (255, 120, 45),
              (190, 70, 210), (20, 190, 210), (230, 190, 40),
              (150, 100, 45), (255, 80, 120), (100, 210, 180),
              (190, 190, 190))
    return colors[int(label) % len(colors)]


def draw_boxes(image, boxes, labels, lidar2img, classes, scores=None,
               score_threshold=0.0):
    result = image.copy()
    if boxes is None or len(boxes) == 0:
        return result
    corners = boxes.corners.detach().cpu().numpy()
    labels = np.asarray(labels)
    if scores is not None:
        scores = np.asarray(scores)
    matrix = np.asarray(lidar2img, dtype=np.float32)
    for box_index, xyz in enumerate(corners):
        if scores is not None and scores[box_index] < score_threshold:
            continue
        homogeneous = np.concatenate(
            (xyz, np.ones((8, 1), dtype=np.float32)), axis=1)
        projected = homogeneous @ matrix.T
        depth = projected[:, 2]
        uv = projected[:, :2] / np.maximum(depth[:, None], 1e-5)
        color = palette(labels[box_index])
        for start, end in EDGES:
            if depth[start] <= 0.1 or depth[end] <= 0.1:
                continue
            p1 = tuple(np.round(uv[start]).astype(int))
            p2 = tuple(np.round(uv[end]).astype(int))
            cv2.line(result, p1, p2, color, 2, cv2.LINE_AA)
    return result


def make_tile(image, width, title):
    height = int(round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    header = np.full((38, width, 3), (241, 245, 249), dtype=np.uint8)
    text_width = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
    cv2.putText(header, title, ((width - text_width) // 2, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (51, 65, 85), 2,
                cv2.LINE_AA)
    return np.concatenate((header, resized), axis=0)


def put_text_fit(image, text, origin, max_width, scale, color,
                 thickness=1):
    """Draw a single line, shrinking it only when the badge requires it."""
    while scale > 0.34:
        text_width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            break
        scale -= 0.03
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def make_context_header(width, scene_name, frame_number, sample_token,
                        model_label, checkpoint, score_threshold):
    header = np.full((132, width, 3), (31, 41, 55), dtype=np.uint8)
    cv2.putText(header, 'PETR  |  MULTI-CAMERA 3D DETECTION', (28, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (248, 250, 252), 2,
                cv2.LINE_AA)
    cv2.putText(header, '{}  /  frame {:04d}'.format(
                    scene_name, frame_number), (30, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.76, (148, 197, 255), 2,
                cv2.LINE_AA)
    cv2.putText(header, 'sample  {}'.format(sample_token), (30, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (174, 187, 205), 1,
                cv2.LINE_AA)

    badge_specs = [
        ('DATASET', 'nuScenes val'),
        ('MODEL', model_label),
        ('CHECKPOINT', Path(checkpoint).name),
        ('SCORE', '>= {:.2f}'.format(score_threshold)),
    ]
    badge_widths = (230, 360, 410, 160)
    total_width = sum(badge_widths) + 18 * (len(badge_widths) - 1)
    x = width - total_width - 28
    for (label, value), badge_width in zip(badge_specs, badge_widths):
        cv2.rectangle(header, (x, 25), (x + badge_width, 108),
                      (45, 58, 76), -1)
        cv2.putText(header, label, (x + 14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (143, 161, 184), 1,
                    cv2.LINE_AA)
        put_text_fit(header, value, (x + 14, 83), badge_width - 28,
                     0.57, (244, 247, 251))
        x += badge_width + 18
    return header


def make_section_bar(width, title, detail, color):
    bar = np.full((52, width, 3), (255, 255, 255), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (9, 51), color, -1)
    cv2.putText(bar, title, (28, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (31, 41, 55), 2, cv2.LINE_AA)
    title_width = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0][0]
    cv2.putText(bar, detail, (48 + title_width, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 116, 139), 1,
                cv2.LINE_AA)
    cv2.line(bar, (0, 51), (width, 51), (222, 228, 236), 1)
    return bar


def make_class_legend(width, classes):
    legend = np.full((126, width, 3), (248, 250, 252), dtype=np.uint8)
    cv2.line(legend, (0, 0), (width, 0), (213, 221, 232), 1)
    cv2.putText(legend, 'CLASS LEGEND', (28, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (71, 85, 105), 2,
                cv2.LINE_AA)
    columns = 5
    left = 240
    cell_width = max((width - left - 30) // columns, 1)
    for label, class_name in enumerate(classes):
        row = label // columns
        column = label % columns
        x = left + column * cell_width
        y = 21 + row * 49
        color = palette(label)
        cv2.rectangle(legend, (x, y), (x + 28, y + 20), color, -1)
        cv2.putText(legend, class_name.replace('_', ' '),
                    (x + 42, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                    (42, 54, 72), 1,
                    cv2.LINE_AA)
    return legend


def render_canvas(raw_info, gt, prediction, classes, tile_width,
                  score_threshold, scene_name, frame_number, sample_token,
                  model_label, checkpoint):
    camera_names = list(raw_info.get('camera_names', []))
    if not camera_names:
        camera_names = ['CAM_{}'.format(i)
                        for i in range(len(raw_info['img_filename']))]
    gt_rows, pred_rows = [], []
    pred_boxes = prediction['boxes_3d']
    pred_scores = prediction['scores_3d'].detach().cpu().numpy()
    pred_labels = prediction['labels_3d'].detach().cpu().numpy()
    gt_labels = gt['gt_labels_3d']
    if torch.is_tensor(gt_labels):
        gt_labels = gt_labels.detach().cpu().numpy()
    for camera_index, image_path in enumerate(raw_info['img_filename']):
        image = mmcv.imread(image_path)
        projection = raw_info['lidar2img'][camera_index]
        gt_image = draw_boxes(
            image, gt['gt_bboxes_3d'], gt_labels, projection, classes)
        pred_image = draw_boxes(
            image, pred_boxes, pred_labels, projection, classes,
            scores=pred_scores, score_threshold=score_threshold)
        gt_rows.append(make_tile(
            gt_image, tile_width, camera_names[camera_index]))
        pred_rows.append(make_tile(
            pred_image, tile_width, camera_names[camera_index]))
    gt_row = np.concatenate(gt_rows, axis=1)
    pred_row = np.concatenate(pred_rows, axis=1)
    width = gt_row.shape[1]
    header = make_context_header(
        width, scene_name, frame_number, sample_token, model_label,
        checkpoint, score_threshold)
    gt_bar = make_section_bar(
        width, 'GROUND TRUTH', '{} annotated 3D boxes'.format(
            len(gt['gt_bboxes_3d'])), (210, 129, 42))
    pred_count = int((prediction['scores_3d'] >= score_threshold).sum())
    pred_bar = make_section_bar(
        width, 'PETR PREDICTIONS', '{} boxes above threshold'.format(
            pred_count), (69, 121, 245))
    legend = make_class_legend(width, classes)
    return np.concatenate(
        (header, gt_bar, gt_row, pred_bar, pred_row, legend), axis=0)


def prepare_one(dataset, index, device):
    sample = dataset.prepare_test_data(index)
    data = collate([sample], samples_per_gpu=1)
    return scatter(data, [device])[0]


def encode_mp4(frames_dir, output_path, fps, frame_count):
    """Encode numbered JPEG frames as a widely compatible H.264 MP4."""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError(
            'FFmpeg is required for --video but was not found in PATH')
    command = [
        ffmpeg, '-y', '-loglevel', 'warning',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%06d.jpg'),
        '-frames:v', str(frame_count),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output_path),
    ]
    subprocess.check_call(command)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This checkpoint sanity check requires a CUDA GPU')
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    relocate_dataset_paths(dataset, cfg.data.test.data_root)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    device = torch.cuda.current_device()
    model = model.cuda(device).eval()

    data_root = cfg.data.test.data_root
    indices, scene_name = select_indices(
        dataset, data_root, args.sample_index, args.scene_name, args.seed)
    if args.max_frames > 0:
        indices = indices[:args.max_frames]
    output_dir = Path(args.output_dir) / scene_name
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for frame_number, index in enumerate(indices):
        raw_info = dataset.get_data_info(index)
        raw_info['camera_names'] = list(dataset.data_infos[index]['cams'])
        gt = dataset.get_ann_info(index)
        data = prepare_one(dataset, index, device)
        with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, **data)
        prediction = outputs[0].get('pts_bbox', outputs[0])
        sample_token = dataset.data_infos[index]['token']
        canvas = render_canvas(
            raw_info, gt, prediction, dataset.CLASSES, args.tile_width,
            args.score_threshold, scene_name, frame_number, sample_token,
            args.model_label, args.checkpoint)
        frame_path = frames_dir / 'frame_{:06d}.jpg'.format(frame_number)
        cv2.imwrite(str(frame_path), canvas)
        kept = int((prediction['scores_3d'] >= args.score_threshold).sum())
        metadata.append(dict(frame=frame_number, dataset_index=index,
                             token=sample_token,
                             predictions_above_threshold=kept,
                             image=str(frame_path)))
        print('[{}/{}] {} predictions={} → {}'.format(
            frame_number + 1, len(indices), metadata[-1]['token'], kept,
            frame_path))
    if args.video:
        video_path = output_dir / '{}_gt_vs_prediction.mp4'.format(scene_name)
        encode_mp4(frames_dir, video_path, args.fps, len(metadata))
        print('H.264 video:', video_path)
    with open(output_dir / 'metadata.json', 'w') as handle:
        json.dump(dict(dataset='nuScenes v1.0-trainval', scene=scene_name,
                       model=args.model_label, config=args.config,
                       checkpoint=args.checkpoint,
                       score_threshold=args.score_threshold, frames=metadata),
                  handle, indent=2)
    print('Finished:', output_dir)


if __name__ == '__main__':
    main()
