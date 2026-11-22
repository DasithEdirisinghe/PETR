#!/usr/bin/env python3
"""Render official-filtered PETR 4 m TP pairs from saved nuScenes predictions."""
import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')
DIST = ((10, '0-10', '#2563EB'), (20, '10-20', '#06B6D4'),
        (30, '20-30', '#8B5CF6'), (40, '30-40', '#D946EF'),
        (float('inf'), '40-50', '#1E3A8A'))
ERROR = ((.5, '0-0.5', '#FDE047'), (1, '0.5-1', '#F59E0B'),
         (2, '1-2', '#EA580C'), (4, '2-4', '#78350F'))
GREEN, RED = '#16A34A', '#DC2626'
LEGEND_HEIGHT = 180
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
FIELDS = ('scene_name', 'scene_token', 'sample_token', 'timestamp', 'keyframe_index',
          'class', 'gt_instance_token', 'numeric_id', 'prediction_identifier',
          'score', 'gt_ego_distance_m', 'distance_bin', 'center_error_m',
          'error_bin', 'gt_center_global', 'prediction_center_global',
          'gt_center_ego', 'prediction_center_ego', 'camera_visibility')


def band(value, bins):
    return next((label, color) for upper, label, color in bins if value < upper)


def bgr(color):
    return tuple(bytes.fromhex(color[1:])[::-1])


def annotation(nusc, gt):
    sample = nusc.get('sample', gt.sample_token)
    matches = [nusc.get('sample_annotation', token) for token in sample['anns']
               if math.sqrt(sum((a - b) ** 2 for a, b in zip(
                   nusc.get('sample_annotation', token)['translation'],
                   gt.translation))) < 1e-4]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely resolve GT annotation: ' + gt.sample_token)
    return matches[0]


def official_pairs(evaluator, selected_tokens, classes):
    """Replay detection.algo.accumulate matching on evaluator-filtered boxes.

    The evaluator has already applied category conversion, class ranges,
    bike-rack filtering, zero-point removal and validation sample selection.
    Official accumulate sorts scores descending, finds the nearest unmatched
    same-class GT in its original order, and accepts only distance < 4 m.
    """
    import numpy as np
    for name in classes:
        predictions = [(token, index, box)
                       for token in evaluator.pred_boxes.sample_tokens
                       if token in selected_tokens
                       for index, box in enumerate(evaluator.pred_boxes[token])
                       if box.detection_name == name]
        scores = np.asarray([box.detection_score for _, _, box in predictions])
        taken = set()
        for rank in np.argsort(scores)[::-1]:
            token, index, pred = predictions[int(rank)]
            best, distance = None, float('inf')
            for gt_index, gt in enumerate(evaluator.gt_boxes[token]):
                if gt.detection_name != name or (token, gt_index) in taken:
                    continue
                d = evaluator.cfg.dist_fcn_callable(gt, pred)
                if d < distance:
                    best, distance = (gt_index, gt), d
            if best is not None and distance < 4.0:
                taken.add((token, best[0]))
                yield name, token, index, pred, best[1], distance


def choose_scenes(nusc, count, explicit):
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.eval.detection.utils import category_to_detection_name
    allowed = set(create_splits_scenes()['val'])
    scenes = [s for s in nusc.scene if s['name'] in allowed]
    if explicit:
        names = [x.strip() for x in explicit.split(',') if x.strip()]
        found = {s['name']: s for s in scenes}
        missing = set(names) - set(found)
        if missing:
            raise ValueError('Not validation scenes: ' + ', '.join(sorted(missing)))
        return [found[name] for name in names]
    # Greedy class coverage, with scene length as a tie breaker.
    counts = {}
    for scene in scenes:
        tally = Counter()
        sample = nusc.get('sample', scene['first_sample_token'])
        while True:
            for token in sample['anns']:
                category = nusc.get('sample_annotation', token)['category_name']
                name = category_to_detection_name(category)
                if name in CLASSES:
                    tally[name] += 1
            if not sample['next']:
                break
            sample = nusc.get('sample', sample['next'])
        counts[scene['token']] = tally
    chosen, covered = [], Counter()
    for _ in range(min(count, len(scenes))):
        scene = max((s for s in scenes if s not in chosen),
                    key=lambda s: (sum(min(counts[s['token']][c], 10) /
                                       (1 + covered[c]) for c in CLASSES),
                                   s['nbr_samples'], s['name']))
        chosen.append(scene)
        covered.update(counts[scene['token']])
    return chosen


def samples_for_scene(nusc, scene):
    sample = nusc.get('sample', scene['first_sample_token'])
    while True:
        yield sample
        if not sample['next']:
            break
        sample = nusc.get('sample', sample['next'])


def transform_box(box, translation, rotation):
    import numpy as np
    from pyquaternion import Quaternion
    box = box.copy()
    box.translate(-np.asarray(translation))
    box.rotate(Quaternion(rotation).inverse)
    return box


def projected_segments(box, calibration, pose, width, height):
    import numpy as np
    camera = transform_box(transform_box(box, pose['translation'], pose['rotation']),
                           calibration['translation'], calibration['rotation'])
    corners = camera.corners().T
    intrinsic = np.asarray(calibration['camera_intrinsic'])
    near = .1
    segments = []
    for i, j in EDGES:
        a, b = corners[i].copy(), corners[j].copy()
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near:
            a = a + (near - a[2]) / (b[2] - a[2]) * (b - a)
        if b[2] < near:
            b = b + (near - b[2]) / (a[2] - b[2]) * (a - b)
        p, q = intrinsic @ a, intrinsic @ b
        segments.append(((float(p[0] / p[2]), float(p[1] / p[2])),
                         (float(q[0] / q[2]), float(q[1] / q[2]))))
    center = camera.center
    uv = intrinsic @ center if center[2] > near else None
    center_uv = (float(uv[0] / uv[2]), float(uv[1] / uv[2])) if uv is not None else None
    visible = any(min(a[0], b[0]) < width and max(a[0], b[0]) >= 0 and
                  min(a[1], b[1]) < height and max(a[1], b[1]) >= 0
                  for a, b in segments)
    return segments, center_uv, visible


def id_badge(image, x, y, label, color, text_scale=.5):
    import cv2
    h, w = image.shape[:2]
    label = str(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, text_scale, 1)
    x = max(1, min(w - tw - 6, int(x)))
    y = max(th + 3, min(h - 3, int(y)))
    cv2.rectangle(image, (x, y - th - 3), (x + tw + 5, y + 2),
                  (255, 255, 255), -1)
    cv2.rectangle(image, (x, y - th - 3), (x + tw + 5, y + 2),
                  bgr(color), 1)
    cv2.putText(image, label, (x + 2, y), font, text_scale, (20, 20, 20), 1,
                cv2.LINE_AA)


def draw_projected(image, segments, center, color, label=None, bands=None):
    import cv2
    h, w = image.shape[:2]
    for a, b in segments:
        ok, p, q = cv2.clipLine((0, 0, w, h),
                               tuple(round(x) for x in a), tuple(round(x) for x in b))
        if ok:
            cv2.line(image, p, q, bgr(color), 1, cv2.LINE_AA)
    if center and 0 <= center[0] < w and 0 <= center[1] < h:
        point = tuple(round(x) for x in center)
        if color == GREEN:
            cv2.circle(image, point, 3, bgr(GREEN), 1, cv2.LINE_AA)
        else:
            cv2.drawMarker(image, point, bgr(RED),
                           cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)
    else:
        point = None
    visible_points = [p for edge in segments for p in edge if 0 <= p[0] < w and 0 <= p[1] < h]
    if not visible_points and not point:
        return
    anchor = (min(p[0] for p in visible_points), min(p[1] for p in visible_points)) if visible_points else point
    x, y = max(2, min(w - 60, int(anchor[0]))), max(24, min(h - 24, int(anchor[1])))
    if bands:
        for i, band_color in enumerate(bands):
            left = x + i * 17
            cv2.rectangle(image, (left - 1, y - 20), (left + 16, y - 8),
                          (255, 255, 255), -1)
            cv2.rectangle(image, (left + 1, y - 18), (left + 14, y - 10),
                          bgr(band_color), -1)
            cv2.rectangle(image, (left + 1, y - 18), (left + 14, y - 10),
                          (30, 30, 30), 1)
    if label is not None:
        id_badge(image, x + (36 if bands else 0), y - 8,
                 label, color)


def draw_bev(canvas, pairs, pose, scale=6.0):
    import cv2
    import numpy as np
    h, w = canvas.shape[:2]
    origin = (w // 2, h // 2)
    def point(xy):
        return (round(origin[0] - xy[1] * scale),
                round(origin[1] - xy[0] * scale))
    for distance in (10, 20, 30, 40, 50):
        radius = round(distance * scale)
        for angle in range(0, 360, 8):
            cv2.ellipse(canvas, origin, (radius, radius), 0,
                        angle, angle + 3, (205, 205, 205), 1, cv2.LINE_AA)
        label_x = round(origin[0] + radius * .71)
        label_y = round(origin[1] - radius * .71)
        cv2.putText(canvas, '{} m'.format(distance), (label_x + 3, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (125, 125, 125), 1,
                    cv2.LINE_AA)
    cv2.line(canvas, point((-50, 0)), point((50, 0)), (210, 210, 210), 1)
    cv2.line(canvas, point((0, -50)), point((0, 50)), (210, 210, 210), 1)
    cv2.circle(canvas, origin, 7, (20, 20, 20), -1)
    cv2.putText(canvas, 'front', (origin[0] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (25, 25, 25), 2)
    for pair in pairs:
        boxes = [transform_box(pair[key], pose['translation'], pose['rotation'])
                 for key in ('gt_box', 'pred_box')]
        centers = [point(box.center[:2]) for box in boxes]
        cv2.line(canvas, centers[0], centers[1], (110, 110, 110), 1, cv2.LINE_AA)
        for box, center, color in zip(boxes, centers, (GREEN, RED)):
            corners = box.bottom_corners()[:2].T
            polygon = np.asarray([point(c) for c in corners], dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, bgr(color), 1, cv2.LINE_AA)
            if color == GREEN:
                cv2.circle(canvas, center, 4, bgr(color), 2, cv2.LINE_AA)
            else:
                cv2.circle(canvas, center, 3, bgr(color), -1, cv2.LINE_AA)
        label_point = (round((centers[0][0] + centers[1][0]) / 2) - 18,
                       round((centers[0][1] + centers[1][1]) / 2) - 12)
        label_point = (max(2, min(w - 30, label_point[0])),
                       max(14, min(h - 2, label_point[1])))
        label = str(pair['id'])
        cv2.putText(canvas, label, label_point, cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, label_point, cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (20, 20, 20), 1, cv2.LINE_AA)
        x, y = centers[1]
        for i, color in enumerate(pair['bands']):
            left = x + 12 + i * 19
            cv2.rectangle(canvas, (left - 1, y - 20), (left + 18, y - 7),
                          (255, 255, 255), -1)
            cv2.rectangle(canvas, (left + 1, y - 18), (left + 16, y - 9),
                          bgr(color), -1)
            cv2.rectangle(canvas, (left + 1, y - 18), (left + 16, y - 9),
                          (20, 20, 20), 1)


def legend(width):
    import cv2
    import numpy as np
    image = np.full((LEGEND_HEIGHT, width, 3), 255, dtype=np.uint8)
    cv2.line(image, (0, 0), (width - 1, 0), (195, 195, 195), 1)
    def item(x, y, color, label):
        x = round(x)
        cv2.rectangle(image, (x, y - 12), (x + 17, y + 2), bgr(color), -1)
        cv2.rectangle(image, (x, y - 12), (x + 17, y + 2), (30, 30, 30), 1)
        cv2.putText(image, label, (x + 23, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .47, (25, 25, 25), 1, cv2.LINE_AA)
    item(12, 26, GREEN, 'GT box + ring center')
    item(width / 2, 26, RED, 'PETR box + cross center')
    cv2.putText(image, 'LEFT BAND  |  GT distance from ego', (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (25, 25, 25), 1, cv2.LINE_AA)
    for i, (_, label, color) in enumerate(DIST):
        item(12 + i * (width - 24) / 5, 87, color, label + ' m')
    cv2.putText(image, 'RIGHT BAND  |  BEV center error', (12, 130),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (25, 25, 25), 1, cv2.LINE_AA)
    for i, (_, label, color) in enumerate(ERROR):
        item(12 + i * (width - 24) / 4, 158, color, label + ' m')
    return image


def render_frame(nusc, sample, pairs, cameras, cell_width, cell_height, bev_width):
    import cv2
    import numpy as np
    cols = math.ceil(len(cameras) / 2)
    mosaic = np.full((2 * cell_height, cols * cell_width, 3), 255, dtype=np.uint8)
    visibility = {id(pair): [] for pair in pairs}
    for k, name in enumerate(cameras):
        sd = nusc.get('sample_data', sample['data'][name])
        calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        image = cv2.imread(str(Path(nusc.dataroot) / sd['filename']))
        if image is None:
            raise FileNotFoundError(sd['filename'])
        source_height, source_width = image.shape[:2]
        image = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        sx, sy = cell_width / source_width, cell_height / source_height
        def scaled(segments, center):
            lines = [((a[0] * sx, a[1] * sy), (b[0] * sx, b[1] * sy))
                     for a, b in segments]
            point = (center[0] * sx, center[1] * sy) if center else None
            return lines, point
        for pair in pairs:
            gt_segments, gt_center, gt_visible = projected_segments(
                pair['gt_box'], calibration, pose, source_width, source_height)
            pred_segments, pred_center, pred_visible = projected_segments(
                pair['pred_box'], calibration, pose, source_width, source_height)
            if gt_visible or pred_visible:
                visibility[id(pair)].append(name)
            gt_segments, gt_center = scaled(gt_segments, gt_center)
            pred_segments, pred_center = scaled(pred_segments, pred_center)
            draw_projected(image, gt_segments, gt_center, GREEN)
            draw_projected(image, pred_segments, pred_center, RED, pair['id'], pair['bands'])
        cv2.rectangle(image, (0, 0), (185, 25), (25, 25, 25), -1)
        cv2.putText(image, name, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y, x = divmod(k, cols)
        mosaic[y * cell_height:(y + 1) * cell_height,
               x * cell_width:(x + 1) * cell_width] = image
    bev_height = 2 * cell_height - LEGEND_HEIGHT
    bev = np.full((bev_height, bev_width, 3), 255, dtype=np.uint8)
    ref_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', ref_sd['ego_pose_token'])
    draw_bev(bev, pairs, pose, scale=min(bev_width, bev_height) / 110)
    cv2.putText(bev, 'BEV | 10 m range rings', (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .6, (25, 25, 25), 2)
    frame = np.hstack((mosaic, np.vstack((bev, legend(bev_width)))))
    return frame, visibility


def camera_visibility(nusc, sample, pairs, cameras):
    visibility = {id(pair): [] for pair in pairs}
    for name in cameras:
        sd = nusc.get('sample_data', sample['data'][name])
        calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        for pair in pairs:
            gt_visible = projected_segments(pair['gt_box'], calibration, pose,
                                            sd['width'], sd['height'])[2]
            pred_visible = projected_segments(pair['pred_box'], calibration, pose,
                                              sd['width'], sd['height'])[2]
            if gt_visible or pred_visible:
                visibility[id(pair)].append(name)
    return visibility


class VideoEncoder:
    """Encode BGR frames with H.264 rather than OpenCV's fragile mp4v writer."""

    def __init__(self, path, width, height, fps):
        executable = shutil.which('ffmpeg')
        if not executable:
            raise RuntimeError('ffmpeg is required for H.264 MP4 output; install it in '
                               'the PETR environment or add it to PATH')
        self.path = path
        self.process = subprocess.Popen([
            executable, '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', '{}x{}'.format(width, height),
            '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE)

    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            error = self.process.stderr.read().decode(errors='replace')
            raise RuntimeError('ffmpeg failed: ' + error)

    def release(self):
        self.process.stdin.close()
        error = self.process.stderr.read().decode(errors='replace')
        if self.process.wait() != 0:
            raise RuntimeError('ffmpeg failed to write {}: {}'.format(self.path, error))


def prepare(args, nusc, evaluator, scenes, samples):
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    sample_map = {s['token']: (scene, index, s)
                  for scene in scenes for index, s in enumerate(samples[scene['token']])}
    ids = {scene['token']: {} for scene in scenes}
    pairs_by_sample = defaultdict(list)
    for name, token, pred_index, pred, gt, error in official_pairs(
            evaluator, set(sample_map), args.classes):
        scene, index, sample = sample_map[token]
        ann = annotation(nusc, gt)
        instance = ann['instance_token']
        scene_ids = ids[scene['token']]
        if instance not in scene_ids:
            scene_ids[instance] = len(scene_ids) + 1
        distance = math.hypot(*gt.ego_translation[:2])
        distance_label, distance_color = band(distance, DIST)
        error_label, error_color = band(error, ERROR)
        pair = {
            'class': name, 'id': scene_ids[instance], 'ann': ann, 'pred': pred,
            'gt': gt, 'error': error, 'distance': distance,
            'distance_label': distance_label, 'error_label': error_label,
            'bands': (distance_color, error_color),
            'prediction_identifier': '{}:{}'.format(token, pred_index),
            'gt_box': nusc.get_box(ann['token']),
            'pred_box': Box(pred.translation, pred.size, Quaternion(pred.rotation)),
        }
        pairs_by_sample[token, name].append(pair)
    return pairs_by_sample


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent /
                        'outputs/petr_localization_visualization')
    parser.add_argument('--scene-count', type=int, default=10)
    parser.add_argument('--scenes', help='Comma-separated validation scene names')
    parser.add_argument('--classes', nargs='+', default=list(CLASSES), choices=CLASSES)
    parser.add_argument('--cameras', nargs='+', default=list(CAMERAS), choices=CAMERAS)
    parser.add_argument('--cell-width', type=int, default=640)
    parser.add_argument('--cell-height', type=int, default=360)
    parser.add_argument('--bev-width', type=int, default=720)
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    if args.scene_count < 1 or args.fps <= 0 or not args.cameras:
        parser.error('scene-count and fps must be positive; select a camera')
    if args.cell_height < 240 or args.bev_width < 600:
        parser.error('cell-height must be at least 240 and bev-width at least 600 for the BEV legend')
    if (math.ceil(len(args.cameras) / 2) * args.cell_width + args.bev_width) % 2:
        parser.error('combined camera and BEV width must be even for H.264')
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved predictions or nuScenes data root missing')
    import cv2
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    scenes = choose_scenes(nusc, args.scene_count, args.scenes)
    samples = {s['token']: list(samples_for_scene(nusc, s)) for s in scenes}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = args.output_dir / 'metadata'
    meta.mkdir(exist_ok=True)
    (meta / 'selected_scenes.json').write_text(json.dumps([
        {'name': s['name'], 'token': s['token'], 'keyframes': len(samples[s['token']])}
        for s in scenes], indent=2) + '\n')
    evaluator = NuScenesEval(nusc, config_factory('detection_cvpr_2019'),
                            str(args.results_json), 'val', str(meta / 'devkit_work'), verbose=False)
    pairs_by_sample = prepare(args, nusc, evaluator, scenes, samples)
    cv2.imwrite(str(args.output_dir / 'legend.png'), legend(args.bev_width))
    manifest = []
    with (meta / 'matched_4m_tps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for scene in scenes:
            scene_samples = samples[scene['token']]
            for name in args.classes:
                video = args.output_dir / scene['name'] / (name + '.mp4')
                video.parent.mkdir(exist_ok=True)
                encoder = None
                count = 0
                for index, sample in enumerate(scene_samples):
                    pairs = pairs_by_sample[sample['token'], name]
                    if args.metadata_only:
                        visibility = camera_visibility(nusc, sample, pairs, args.cameras)
                    else:
                        frame, visibility = render_frame(
                            nusc, sample, pairs, args.cameras,
                            args.cell_width, args.cell_height, args.bev_width)
                        if encoder is None:
                            cv2.imwrite(str(video.with_name(name + '_first_frame.png')), frame)
                            encoder = VideoEncoder(video, frame.shape[1], frame.shape[0], args.fps)
                        encoder.write(frame)
                    for pair in pairs:
                        writer.writerow({
                            'scene_name': scene['name'], 'scene_token': scene['token'],
                            'sample_token': sample['token'], 'timestamp': sample['timestamp'],
                            'keyframe_index': index, 'class': name,
                            'gt_instance_token': pair['ann']['instance_token'],
                            'numeric_id': pair['id'],
                            'prediction_identifier': pair['prediction_identifier'],
                            'score': pair['pred'].detection_score,
                            'gt_ego_distance_m': pair['distance'],
                            'distance_bin': pair['distance_label'],
                            'center_error_m': pair['error'], 'error_bin': pair['error_label'],
                            'gt_center_global': json.dumps(pair['gt'].translation),
                            'prediction_center_global': json.dumps(pair['pred'].translation),
                            'gt_center_ego': json.dumps(pair['gt'].ego_translation),
                            'prediction_center_ego': json.dumps(pair['pred'].ego_translation),
                            'camera_visibility': ';'.join(visibility[id(pair)]),
                        })
                        count += 1
                if encoder is not None:
                    encoder.release()
                manifest.append({'scene': scene['name'], 'class': name,
                                 'frames': len(scene_samples), 'matched_tps': count,
                                 'video': '' if args.metadata_only else str(video)})
                print(scene['name'], name, len(scene_samples), count, flush=True)
    with (meta / 'video_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print('Wrote', args.output_dir)


if __name__ == '__main__':
    main()
