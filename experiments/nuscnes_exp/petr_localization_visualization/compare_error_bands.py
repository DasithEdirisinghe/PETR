#!/usr/bin/env python3
"""Make distance-sliced, four-error-band PETR TP comparison videos.

The existing chronological pipeline.py is intentionally left unchanged.
This script reuses its evaluator-filtered 4 m matches and saved predictions.
Each output frame compares four independent keyframes, one per error band.
"""
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pipeline

CELL_W = 520
TITLE_H = 56
HEAD_H = 64
CROP_H = 292
CONTEXT_H = 292
BEV_H = 180
FOOT_H = 88
CELL_H = HEAD_H + CROP_H + CONTEXT_H + BEV_H + FOOT_H
SOURCE_CROP_W = 800
SOURCE_CROP_H = 450
VISIBILITY_RANGES = {'1': '0-40%', '2': '40-60%',
                     '3': '60-80%', '4': '80-100%'}
SELECTED_FIELDS = (
    'class', 'distance_bin', 'error_bin', 'slot', 'scene_name',
    'scene_token', 'sample_token', 'keyframe_index', 'timestamp',
    'gt_instance_token', 'numeric_id', 'prediction_identifier', 'camera',
    'visibility_token', 'other_gt_closer_than_assigned',
    'gt_ego_distance_m', 'center_error_m', 'score',
)
MANIFEST_FIELDS = (
    'class', 'distance_bin', 'scene_count', 'matched_tps', 'slots',
    'frames', 'lt_0_5_count', 'from_0_5_to_1_count',
    'from_1_to_2_count', 'from_2_to_4_count', 'video',
)


def put(canvas, text, x, y, size=.48, color=(35, 35, 35), thickness=1):
    import cv2
    cv2.putText(canvas, str(text), (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def label_distance(label):
    low, high = (int(value) for value in label.split('-'))
    return low, high


def closer_to_other_gt(evaluator, sample_token, pair):
    """Whether another same-class GT center beats the assigned center."""
    px, py = pair['pred'].translation[:2]
    return any(
        other is not pair['gt'] and other.detection_name == pair['class'] and
        math.hypot(px - other.translation[0], py - other.translation[1]) <
        pair['error']
        for other in evaluator.gt_boxes[sample_token])


def collect_candidates(scenes, samples, pairs_by_sample, classes, evaluator,
                       visibility_tokens):
    pools = defaultdict(list)
    allowed_visibility = set(visibility_tokens)
    for scene in scenes:
        for index, sample in enumerate(samples[scene['token']]):
            for name in classes:
                for pair in pairs_by_sample[sample['token'], name]:
                    if not 0 <= pair['error'] < 4:
                        raise ValueError('Comparison input contains a non-4 m TP')
                    if pair['ann']['visibility_token'] not in allowed_visibility:
                        continue
                    pools[name, pair['distance_label'], pair['error_label']].append({
                        'pair': pair, 'sample': sample, 'scene': scene,
                        'keyframe_index': index,
                        'other_gt_closer_than_assigned': closer_to_other_gt(
                            evaluator, sample['token'], pair),
                    })
    return pools


def best_camera(nusc, candidate, cameras):
    """Choose the camera with the largest clipped GT projection."""
    sample = candidate['sample']
    gt_box = candidate['pair']['gt_box']
    choice, best_area = None, 0.0
    for name in cameras:
        sd = nusc.get('sample_data', sample['data'][name])
        calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        segments, _, visible = pipeline.projected_segments(
            gt_box, calibration, pose, sd['width'], sd['height'])
        if not visible or not segments:
            continue
        points = [point for edge in segments for point in edge]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        left = max(0.0, min(x_values))
        right = min(float(sd['width']), max(x_values))
        top = max(0.0, min(y_values))
        bottom = min(float(sd['height']), max(y_values))
        area = max(0.0, right - left) * max(0.0, bottom - top)
        if area > best_area:
            choice, best_area = name, area
    return choice


def select_examples(nusc, pool, count, low, high, cameras, rng):
    """Fill each error lane with distinct examples near shared range targets."""
    shuffled = list(pool)
    rng.shuffle(shuffled)
    selected, used_predictions, used_instances = [], set(), set()
    scene_uses = Counter()
    camera_cache = {}
    for slot in range(count):
        target = low + (slot + .5) * (high - low) / count
        ranked = sorted(shuffled, key=lambda candidate: (
            1000 if candidate['pair']['ann']['instance_token'] in used_instances else 0,
            abs(candidate['pair']['distance'] - target) +
            1.5 * scene_uses[candidate['scene']['token']],
        ))
        chosen = None
        for candidate in ranked:
            identifier = candidate['pair']['prediction_identifier']
            if identifier in used_predictions:
                continue
            if identifier not in camera_cache:
                camera_cache[identifier] = best_camera(nusc, candidate, cameras)
            camera = camera_cache[identifier]
            if camera is not None:
                chosen = dict(candidate, camera=camera)
                break
        if chosen is None:
            break
        selected.append(chosen)
        used_predictions.add(chosen['pair']['prediction_identifier'])
        used_instances.add(chosen['pair']['ann']['instance_token'])
        scene_uses[chosen['scene']['token']] += 1
    return selected


def scaled_projection(segments, center, x0, y0, scale_x, scale_y):
    lines = [(((a[0] - x0) * scale_x, (a[1] - y0) * scale_y),
              ((b[0] - x0) * scale_x, (b[1] - y0) * scale_y))
             for a, b in segments]
    point = (((center[0] - x0) * scale_x,
              (center[1] - y0) * scale_y) if center else None)
    return lines, point


def camera_views(nusc, candidate, image):
    """Return fixed-source-pixel crop and full camera context for one pair."""
    import cv2
    import numpy as np
    pair = candidate['pair']
    sample = candidate['sample']
    name = candidate['camera']
    sd = nusc.get('sample_data', sample['data'][name])
    calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    h, w = image.shape[:2]
    gt_segments, gt_center, _ = pipeline.projected_segments(
        pair['gt_box'], calibration, pose, w, h)
    pred_segments, pred_center, _ = pipeline.projected_segments(
        pair['pred_box'], calibration, pose, w, h)
    if gt_center is None:
        visible = [p for edge in gt_segments for p in edge]
        if not visible:
            raise ValueError('Selected GT has no camera footprint')
        focus = (sum(p[0] for p in visible) / len(visible),
                 sum(p[1] for p in visible) / len(visible))
    elif pred_center is not None:
        focus = ((gt_center[0] + pred_center[0]) / 2,
                 (gt_center[1] + pred_center[1]) / 2)
    else:
        focus = gt_center
    crop_x = round(focus[0] - SOURCE_CROP_W / 2)
    crop_y = round(focus[1] - SOURCE_CROP_H / 2)
    source_crop = np.full((SOURCE_CROP_H, SOURCE_CROP_W, 3), 245, dtype=np.uint8)
    source_x0, source_x1 = max(0, crop_x), min(w, crop_x + SOURCE_CROP_W)
    source_y0, source_y1 = max(0, crop_y), min(h, crop_y + SOURCE_CROP_H)
    if source_x1 > source_x0 and source_y1 > source_y0:
        dest_x0, dest_y0 = source_x0 - crop_x, source_y0 - crop_y
        source_crop[dest_y0:dest_y0 + source_y1 - source_y0,
                    dest_x0:dest_x0 + source_x1 - source_x0] = image[
                        source_y0:source_y1, source_x0:source_x1]
    crop = cv2.resize(source_crop, (CELL_W, CROP_H), interpolation=cv2.INTER_AREA)
    crop_sx, crop_sy = CELL_W / SOURCE_CROP_W, CROP_H / SOURCE_CROP_H
    crop_gt = scaled_projection(gt_segments, gt_center, crop_x, crop_y,
                                crop_sx, crop_sy)
    crop_pred = scaled_projection(pred_segments, pred_center, crop_x, crop_y,
                                  crop_sx, crop_sy)
    pipeline.draw_projected(crop, crop_gt[0], crop_gt[1], pipeline.GREEN)
    pipeline.draw_projected(crop, crop_pred[0], crop_pred[1], pipeline.RED,
                            pair['id'], pair['bands'])
    context = cv2.resize(image, (CELL_W, CONTEXT_H), interpolation=cv2.INTER_AREA)
    sx, sy = CELL_W / w, CONTEXT_H / h
    context_gt = scaled_projection(gt_segments, gt_center, 0, 0, sx, sy)
    context_pred = scaled_projection(pred_segments, pred_center, 0, 0, sx, sy)
    pipeline.draw_projected(context, context_gt[0], context_gt[1], pipeline.GREEN)
    pipeline.draw_projected(context, context_pred[0], context_pred[1], pipeline.RED)
    cv2.rectangle(context, (round(crop_x * sx), round(crop_y * sy)),
                  (round((crop_x + SOURCE_CROP_W) * sx),
                   round((crop_y + SOURCE_CROP_H) * sy)),
                  (80, 80, 80), 1, cv2.LINE_AA)
    return crop, context


def local_bev(candidate):
    import cv2
    import numpy as np
    pair = candidate['pair']
    sample = candidate['sample']
    canvas = np.full((BEV_H, CELL_W, 3), 255, dtype=np.uint8)
    sd = candidate['nusc'].get('sample_data', sample['data']['LIDAR_TOP'])
    pose = candidate['nusc'].get('ego_pose', sd['ego_pose_token'])
    gt = pipeline.transform_box(pair['gt_box'], pose['translation'], pose['rotation'])
    pred = pipeline.transform_box(pair['pred_box'], pose['translation'], pose['rotation'])
    # Keep ego, GT and prediction visible even in distant GT-range bins.
    forward = (0.0, gt.center[0], pred.center[0])
    lateral = (0.0, gt.center[1], pred.center[1])
    x_low, x_high = min(forward) - 2.5, max(forward) + 2.5
    y_low, y_high = min(lateral) - 2.5, max(lateral) + 2.5
    scale = min((BEV_H - 28) / (x_high - x_low),
                (CELL_W - 28) / (y_high - y_low), BEV_H / 16.0)
    x_mid, y_mid = (x_low + x_high) / 2, (y_low + y_high) / 2
    def point(xy):
        return (round(CELL_W / 2 - (xy[1] - y_mid) * scale),
                round(BEV_H / 2 - (xy[0] - x_mid) * scale))
    grid_step = 2 if x_high - x_low <= 24 else 10
    for d in range(math.floor(y_low / grid_step) * grid_step,
                   math.ceil(y_high / grid_step) * grid_step + 1, grid_step):
        x = point((0, d))[0]
        cv2.line(canvas, (x, 0), (x, BEV_H - 1), (235, 235, 235), 1)
    for d in range(math.floor(x_low / grid_step) * grid_step,
                   math.ceil(x_high / grid_step) * grid_step + 1, grid_step):
        y = point((d, 0))[1]
        cv2.line(canvas, (0, y), (CELL_W - 1, y), (235, 235, 235), 1)
    gt_point, pred_point = point(gt.center[:2]), point(pred.center[:2])
    cv2.line(canvas, gt_point, pred_point, (95, 95, 95), 1, cv2.LINE_AA)
    for box, color in ((gt, pipeline.GREEN), (pred, pipeline.RED)):
        footprint = box.bottom_corners()[:2].T[[0, 1, 3, 2]]
        corners = np.asarray([point(xy) for xy in footprint],
                             dtype=np.int32)
        cv2.polylines(canvas, [corners], True, pipeline.bgr(color), 1,
                      cv2.LINE_AA)
    cv2.circle(canvas, gt_point, 4, pipeline.bgr(pipeline.GREEN), 2,
               cv2.LINE_AA)
    cv2.circle(canvas, pred_point, 3, pipeline.bgr(pipeline.RED), -1,
               cv2.LINE_AA)
    ego_point = point((0, 0))
    cv2.circle(canvas, ego_point, 5, (185, 90, 20), -1, cv2.LINE_AA)
    cv2.circle(canvas, ego_point, 6, (40, 40, 40), 1, cv2.LINE_AA)
    label_x = min(CELL_W - 84, ego_point[0] + 10)
    label_y = max(33, min(BEV_H - 8, ego_point[1] + 4))
    put(canvas, 'EGO (0,0)', label_x, label_y, .38, (45, 55, 80))
    put(canvas, 'LOCAL BEV | {} m grid | forward up'.format(grid_step),
        8, 18, .43)
    return canvas


def render_comparison(nusc, name, distance_label, slot, slots, selections,
                      counts, visibility_tokens):
    import cv2
    import numpy as np
    canvas = np.full((TITLE_H + CELL_H, CELL_W * 4, 3), 255, dtype=np.uint8)
    visibility_note = ('GT vis {}'.format(VISIBILITY_RANGES[visibility_tokens[0]])
                       if len(visibility_tokens) == 1 else 'GT vis selected')
    if len(visibility_tokens) == len(VISIBILITY_RANGES):
        visibility_note = 'all GT visibility'
    title = ('PETR 4 m TPs | {} | GT distance {} m | {} | '
             'example {}/{} | independent keyframes').format(
                 name.replace('_', ' '), distance_label, visibility_note,
                 slot + 1, slots)
    put(canvas, title, 18, 35, .75, thickness=2)
    row_total = sum(counts)
    for column, (_, error_label, color) in enumerate(pipeline.ERROR):
        x0 = column * CELL_W
        tile = canvas[TITLE_H:, x0:x0 + CELL_W]
        if column:
            cv2.line(canvas, (x0, TITLE_H),
                     (x0, TITLE_H + CELL_H - 1), (205, 205, 205), 1)
        cv2.rectangle(tile, (12, 15), (34, 37), pipeline.bgr(color), -1)
        cv2.rectangle(tile, (12, 15), (34, 37), (25, 25, 25), 1)
        put(tile, 'ERROR {} m'.format(error_label), 45, 33, .64,
            thickness=2)
        percentage = 100 * counts[column] / row_total if row_total else 0
        put(tile, 'n={} / {} ({:.1f}% of this bin)'.format(
            counts[column], row_total, percentage), 12, 55, .43)
        candidate = selections[column][slot] if slot < len(selections[column]) else None
        if candidate is None:
            put(tile, 'No further camera-visible example', 56, 310, .56)
            continue
        pair = candidate['pair']
        sample = candidate['sample']
        camera = candidate['camera']
        sd = nusc.get('sample_data', sample['data'][camera])
        image = cv2.imread(str(Path(nusc.dataroot) / sd['filename']))
        if image is None:
            raise FileNotFoundError(sd['filename'])
        crop, context = camera_views(nusc, candidate, image)
        crop_top = HEAD_H
        context_top = crop_top + CROP_H
        bev_top = context_top + CONTEXT_H
        tile[crop_top:context_top] = crop
        tile[context_top:bev_top] = context
        candidate['nusc'] = nusc
        tile[bev_top:bev_top + BEV_H] = local_bev(candidate)
        foot_top = bev_top + BEV_H
        put(tile, '{} | {} | frame {}'.format(
            candidate['scene']['name'], camera, candidate['keyframe_index']),
            12, foot_top + 20, .43)
        visibility_token = pair['ann']['visibility_token']
        put(tile, 'GT visibility {} ({}) | other GT closer {}'.format(
            visibility_token, VISIBILITY_RANGES.get(visibility_token, 'unknown'),
            int(candidate['other_gt_closer_than_assigned'])),
            12, foot_top + 44, .43)
        put(tile, 'GT {:.2f} m | error {:.2f} m | score {:.3f} | ID {}'.format(
            pair['distance'], pair['error'], pair['pred'].detection_score,
            pair['id']), 12, foot_top + 68, .43)
    return canvas


def main(argv=None, description=None, hide_visibility_option=False):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent /
                        'outputs/error_band_comparisons')
    parser.add_argument('--scenes', help='Comma-separated validation scene names')
    parser.add_argument('--scene-count', type=int, default=10)
    parser.add_argument('--all-val', action='store_true', help='Use all validation scenes')
    parser.add_argument('--classes', nargs='+', choices=pipeline.CLASSES,
                        default=list(pipeline.CLASSES))
    parser.add_argument('--distance-bins', nargs='+',
                        choices=[entry[1] for entry in pipeline.DIST],
                        default=[entry[1] for entry in pipeline.DIST])
    parser.add_argument('--cameras', nargs='+', choices=pipeline.CAMERAS,
                        default=list(pipeline.CAMERAS))
    parser.add_argument('--visibility-tokens', nargs='+',
                        choices=tuple(VISIBILITY_RANGES),
                        default=list(VISIBILITY_RANGES),
                        help=(argparse.SUPPRESS if hide_visibility_option else
                              'Keep TPs whose assigned GT has one of these nuScenes visibility tokens'))
    parser.add_argument('--examples-per-bin', type=int, default=12)
    parser.add_argument('--hold-frames', type=int, default=4)
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args(argv)
    if args.scenes and args.all_val:
        parser.error('--scenes and --all-val cannot be combined')
    if (args.scene_count < 1 or args.examples_per_bin < 1 or
            args.hold_frames < 1 or args.fps <= 0):
        parser.error('scene-count, examples-per-bin, hold-frames, and fps must be positive')
    if not args.results_json.is_file() or not (args.data_root / 'v1.0-trainval').is_dir():
        parser.error('Saved results or nuScenes root missing')
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.utils.splits import create_splits_scenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    if args.all_val:
        valid_names = set(create_splits_scenes()['val'])
        scenes = [scene for scene in nusc.scene if scene['name'] in valid_names]
    else:
        scenes = pipeline.choose_scenes(nusc, args.scene_count, args.scenes)
    samples = {scene['token']: list(pipeline.samples_for_scene(nusc, scene))
               for scene in scenes}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = args.output_dir / 'metadata'
    metadata.mkdir(exist_ok=True)
    (metadata / 'selected_scenes.json').write_text(json.dumps([
        {'name': scene['name'], 'token': scene['token'],
         'keyframes': len(samples[scene['token']])} for scene in scenes],
        indent=2) + '\n')
    (metadata / 'selection_config.json').write_text(json.dumps({
        'visibility_tokens': args.visibility_tokens,
        'classes': args.classes,
        'distance_bins': args.distance_bins,
        'all_val': args.all_val,
        'scenes': args.scenes,
        'scene_count': len(scenes),
        'examples_per_bin': args.examples_per_bin,
    }, indent=2) + '\n')
    evaluator = NuScenesEval(nusc, config_factory('detection_cvpr_2019'),
                            str(args.results_json), 'val',
                            str(metadata / 'devkit_work'), verbose=False)
    pairs_by_sample = pipeline.prepare(args, nusc, evaluator, scenes, samples)
    pools = collect_candidates(scenes, samples, pairs_by_sample, args.classes,
                               evaluator, args.visibility_tokens)
    manifest, selected_rows = [], []
    for name in args.classes:
        for distance_label in args.distance_bins:
            low, high = label_distance(distance_label)
            band_pools = [pools[name, distance_label, error_label]
                          for _, error_label, _ in pipeline.ERROR]
            counts = [len(pool) for pool in band_pools]
            if not sum(counts):
                print('Skipping empty {} {} m'.format(name, distance_label), flush=True)
                continue
            rng = random.Random(args.seed + sum(ord(c) for c in name + distance_label))
            selections = [select_examples(nusc, pool, args.examples_per_bin,
                                          low, high, args.cameras, rng)
                          for pool in band_pools]
            slots = max(len(selected) for selected in selections)
            if slots == 0:
                print('Skipping {} {} m: no GT visible in selected cameras'.format(
                    name, distance_label), flush=True)
                continue
            video = args.output_dir / name / ('{}m.mp4'.format(distance_label))
            video.parent.mkdir(exist_ok=True)
            encoder = None
            for slot in range(slots):
                frame = render_comparison(nusc, name, distance_label, slot,
                                          slots, selections, counts,
                                          args.visibility_tokens)
                if encoder is None:
                    import cv2
                    cv2.imwrite(str(video.with_suffix('.png')), frame)
                    encoder = pipeline.VideoEncoder(video, frame.shape[1],
                                                    frame.shape[0], args.fps)
                for _ in range(args.hold_frames):
                    encoder.write(frame)
            encoder.release()
            for column, selected in enumerate(selections):
                error_label = pipeline.ERROR[column][1]
                for slot, candidate in enumerate(selected):
                    pair = candidate['pair']
                    sample = candidate['sample']
                    selected_rows.append({
                        'class': name, 'distance_bin': distance_label,
                        'error_bin': error_label, 'slot': slot,
                        'scene_name': candidate['scene']['name'],
                        'scene_token': candidate['scene']['token'],
                        'sample_token': sample['token'],
                        'keyframe_index': candidate['keyframe_index'],
                        'timestamp': sample['timestamp'],
                        'gt_instance_token': pair['ann']['instance_token'],
                        'numeric_id': pair['id'],
                        'prediction_identifier': pair['prediction_identifier'],
                        'camera': candidate['camera'],
                        'visibility_token': pair['ann']['visibility_token'],
                        'other_gt_closer_than_assigned': int(
                            candidate['other_gt_closer_than_assigned']),
                        'gt_ego_distance_m': pair['distance'],
                        'center_error_m': pair['error'],
                        'score': pair['pred'].detection_score,
                    })
            manifest.append(dict(zip(MANIFEST_FIELDS, (
                name, distance_label, len(scenes), sum(counts), slots,
                slots * args.hold_frames, *counts, str(video)))))
            print('Wrote {} ({} TPs, {} examples)'.format(
                video, sum(counts), slots), flush=True)
    with (metadata / 'selected_examples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        writer.writerows(selected_rows)
    with (metadata / 'video_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    print('Wrote comparison videos to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
