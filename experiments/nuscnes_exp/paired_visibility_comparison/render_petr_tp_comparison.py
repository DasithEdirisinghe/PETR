#!/usr/bin/env python3
"""Render PETR 4 m TPs beside StreamPETR results for the same nuScenes GT."""

import argparse
import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'petr_localization_visualization'))
import pipeline
import compare_error_bands as original


VISIBILITY = original.VISIBILITY_RANGES
DISTANCE_BINS = tuple(item[1] for item in pipeline.DIST)
ERROR_BINS = tuple(item[1] for item in pipeline.ERROR)
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
OUTCOMES = ('better_band', 'same_band', 'worse_band', 'no_stream_4m_match')
GT_COLOR, PETR_COLOR, STREAM_COLOR = '#16A34A', '#DC2626', '#2563EB'
W, CROP_H, BEV_H = 520, 292, 190
TITLE_H, HEAD_H, LABEL_H, FOOT_H = 62, 91, 29, 83
TILE_H = HEAD_H + 2 * (LABEL_H + CROP_H) + BEV_H + FOOT_H


def error_band(error):
    if error is None:
        return None
    if not 0 <= error < 4:
        raise ValueError('Expected a fixed <4 m match, got {}'.format(error))
    return next(i for i, (upper, _, _) in enumerate(pipeline.ERROR)
                if error < upper)


def outcome(petr_error, stream_error):
    if stream_error is None:
        return 'no_stream_4m_match'
    a, b = error_band(petr_error), error_band(stream_error)
    if b < a:
        return 'better_band'
    if b > a:
        return 'worse_band'
    return 'same_band'


def read_cohort(path, classes, distance_bins, visibility_tokens):
    rows = []
    with path.open(newline='') as handle:
        for source in csv.DictReader(handle):
            distance_bin = (source['distance_bin'][:-1]
                            if source['distance_bin'].endswith('m')
                            else source['distance_bin'])
            if (source['class'] not in classes or
                    distance_bin not in distance_bins or
                    source['visibility_token'] not in visibility_tokens or
                    not source['petr_error_at_4m_m']):
                continue
            petr_error = float(source['petr_error_at_4m_m'])
            stream_error = (float(source['stream_error_at_4m_m'])
                            if source['stream_error_at_4m_m'] else None)
            row = {
                'sample_token': source['sample_token'],
                'annotation_token': source['annotation_token'],
                'instance_token': source['instance_token'],
                'class': source['class'],
                'distance_m': float(source['distance_m']),
                'distance_bin': distance_bin,
                'visibility_token': source['visibility_token'],
                'visibility_band': source['visibility_band'],
                'petr_error_m': petr_error,
                'stream_error_m': stream_error,
                'petr_score': float(source['petr_score_at_4m']),
                'stream_score': (float(source['stream_score_at_4m'])
                                 if source['stream_score_at_4m'] else None),
                'petr_band': ERROR_BINS[error_band(petr_error)],
                'stream_band': (ERROR_BINS[error_band(stream_error)]
                                if stream_error is not None else ''),
                'outcome': outcome(petr_error, stream_error),
            }
            rows.append(row)
    if not rows:
        raise ValueError('No PETR <4 m TPs in the selected cohort')
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(cohort):
    groups = defaultdict(list)
    for row in cohort:
        groups[row['class'], row['distance_bin'], row['visibility_token'],
               row['petr_band']].append(row)
    summary = []
    for (name, distance, visibility, band), rows in sorted(groups.items()):
        n = len(rows)
        paired = [r for r in rows if r['stream_error_m'] is not None]
        counts = Counter(r['outcome'] for r in rows)
        deltas = [r['stream_error_m'] - r['petr_error_m'] for r in paired]
        result = {'class': name, 'distance_bin': distance,
                  'visibility_token': visibility,
                  'visibility_band': VISIBILITY[visibility],
                  'petr_error_band': band, 'petr_tp_count': n}
        for label in OUTCOMES:
            result[label + '_count'] = counts[label]
            result[label + '_percent'] = 100 * counts[label] / n
        result['stream_lower_error_count'] = sum(d < -1e-8 for d in deltas)
        result['stream_higher_error_count'] = sum(d > 1e-8 for d in deltas)
        result['mean_stream_minus_petr_error_m_on_both'] = (
            sum(deltas) / len(deltas) if deltas else '')
        ordered = sorted(deltas)
        result['median_stream_minus_petr_error_m_on_both'] = (
            (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2
            if ordered else '')
        for threshold in THRESHOLDS:
            key = str(threshold).replace('.', '_')
            result['petr_lt_' + key + 'm_count'] = sum(
                r['petr_error_m'] < threshold for r in rows)
            result['stream_lt_' + key + 'm_count'] = sum(
                r['stream_error_m'] is not None and
                r['stream_error_m'] < threshold for r in rows)
            result['net_stream_gain_lt_' + key + 'm_count'] = (
                result['stream_lt_' + key + 'm_count'] -
                result['petr_lt_' + key + 'm_count'])
        summary.append(result)
    return summary


def draw_text(image, value, x, y, size=.44, color=(35, 35, 35), thickness=1):
    import cv2
    cv2.putText(image, str(value), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def choose_camera(nusc, row, cameras):
    sample = nusc.get('sample', row['sample_token'])
    candidate = {'sample': sample,
                 'pair': {'gt_box': nusc.get_box(row['annotation_token'])}}
    available = []
    for name in cameras:
        sd = nusc.get('sample_data', sample['data'][name])
        if (Path(nusc.dataroot) / sd['filename']).is_file():
            available.append(name)
    return original.best_camera(nusc, candidate, available)


def choose_examples(nusc, pool, limit, cameras, seed):
    """Cycle through outcomes while preferring distinct instances and scenes."""
    rng = random.Random(seed)
    buckets = {label: [r for r in pool if r['outcome'] == label]
               for label in OUTCOMES}
    for rows in buckets.values():
        rng.shuffle(rows)
    selected, used_instances, scene_counts, camera_cache = [], set(), Counter(), {}
    while len(selected) < limit and any(buckets.values()):
        progressed = False
        for label in OUTCOMES:
            candidates = buckets[label]
            if not candidates or len(selected) >= limit:
                continue
            target = (int(pool[0]['distance_bin'].split('-')[0]) +
                      (len(selected) + .5) * 10 / limit)
            ranked = sorted(candidates, key=lambda r: (
                r['instance_token'] in used_instances,
                scene_counts[r['scene_token']],
                abs(r['distance_m'] - target)))
            chosen = None
            for row in ranked:
                key = row['sample_token'], row['annotation_token']
                if key not in camera_cache:
                    camera_cache[key] = choose_camera(nusc, row, cameras)
                camera = camera_cache[key]
                if camera:
                    chosen = row, camera
                    break
                candidates.remove(row)
            if chosen is None:
                continue
            row, camera = chosen
            candidates.remove(row)
            selected.append(dict(row, camera=camera))
            used_instances.add(row['instance_token'])
            scene_counts[row['scene_token']] += 1
            progressed = True
        if not progressed:
            break
    return selected


def build_match_lookup(evaluator, sample_tokens, classes, nusc):
    lookup = {}
    for name, token, index, pred, gt, error in pipeline.official_pairs(
            evaluator, sample_tokens, classes):
        ann = pipeline.annotation(nusc, gt)
        key = token, ann['token']
        if key in lookup:
            raise AssertionError('GT matched twice: {}'.format(key))
        lookup[key] = pred, error, index
    return lookup


def source_crop(image, focus):
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    x0 = round(focus[0] - original.SOURCE_CROP_W / 2)
    y0 = round(focus[1] - original.SOURCE_CROP_H / 2)
    result = np.full((original.SOURCE_CROP_H, original.SOURCE_CROP_W, 3),
                     245, dtype=np.uint8)
    sx0, sx1 = max(0, x0), min(width, x0 + original.SOURCE_CROP_W)
    sy0, sy1 = max(0, y0), min(height, y0 + original.SOURCE_CROP_H)
    if sx1 > sx0 and sy1 > sy0:
        result[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return cv2.resize(result, (W, CROP_H), interpolation=cv2.INTER_AREA), x0, y0


def draw_crop(image, gt_projection, pred_projection, x0, y0, color):
    import cv2
    crop, _, _ = source_crop(image, (x0 + original.SOURCE_CROP_W / 2,
                                     y0 + original.SOURCE_CROP_H / 2))
    sx, sy = W / original.SOURCE_CROP_W, CROP_H / original.SOURCE_CROP_H
    for projection, shade in ((gt_projection, GT_COLOR),
                              (pred_projection, color)):
        if projection is None:
            continue
        segments, center = projection
        lines, point = original.scaled_projection(segments, center, x0, y0, sx, sy)
        pipeline.draw_projected(crop, lines, None, shade)
        if point is not None and 0 <= point[0] < W and 0 <= point[1] < CROP_H:
            pixel = tuple(round(value) for value in point)
            if shade == GT_COLOR:
                cv2.circle(crop, pixel, 3, pipeline.bgr(shade), 1, cv2.LINE_AA)
            else:
                cv2.drawMarker(crop, pixel, pipeline.bgr(shade),
                               cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)
    return crop


def bev_panel(nusc, row, gt_box, petr_box, stream_box, sample):
    import cv2
    import numpy as np
    canvas = np.full((BEV_H, W, 3), 255, dtype=np.uint8)
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    boxes = [(pipeline.transform_box(box, pose['translation'], pose['rotation']),
              color) for box, color in ((gt_box, GT_COLOR), (petr_box, PETR_COLOR))]
    if stream_box is not None:
        boxes.append((pipeline.transform_box(
            stream_box, pose['translation'], pose['rotation']), STREAM_COLOR))
    divider = W // 2
    cv2.line(canvas, (divider, 0), (divider, BEV_H), (190, 190, 190), 1)
    gt = boxes[0][0].center[:2]
    # Left: ego-to-object context. Right: fixed 12 m GT-centered zoom.
    max_forward = max([0.0] + [b.center[0] for b, _ in boxes]) + 3
    min_forward = min([0.0] + [b.center[0] for b, _ in boxes]) - 3
    max_lateral = max([0.0] + [b.center[1] for b, _ in boxes]) + 3
    min_lateral = min([0.0] + [b.center[1] for b, _ in boxes]) - 3
    context_scale = min((BEV_H - 32) / (max_forward - min_forward),
                        (divider - 24) / (max_lateral - min_lateral))
    def context(xy):
        return (round(divider / 2 - (xy[1] - (min_lateral + max_lateral) / 2) * context_scale),
                round(BEV_H / 2 - (xy[0] - (min_forward + max_forward) / 2) * context_scale))
    zoom_scale = min((BEV_H - 32) / 12, (divider - 26) / 12)
    def zoom(xy):
        return (round(divider + divider / 2 - (xy[1] - gt[1]) * zoom_scale),
                round(BEV_H / 2 - (xy[0] - gt[0]) * zoom_scale))
    for project in (context, zoom):
        gt_point = project(gt)
        for box, color in boxes:
            footprint = box.bottom_corners()[:2].T[[0, 1, 3, 2]]
            corners = np.asarray([project(xy) for xy in footprint], dtype=np.int32)
            cv2.polylines(canvas, [corners], True, pipeline.bgr(color), 1, cv2.LINE_AA)
            point = project(box.center[:2])
            cv2.line(canvas, gt_point, point, pipeline.bgr(color), 1, cv2.LINE_AA)
            if color == GT_COLOR:
                cv2.circle(canvas, point, 5, pipeline.bgr(color), 1,
                           cv2.LINE_AA)
            else:
                marker = (cv2.MARKER_CROSS if color == PETR_COLOR
                          else cv2.MARKER_DIAMOND)
                cv2.drawMarker(canvas, point, pipeline.bgr(color), marker,
                               7, 1, cv2.LINE_AA)
        if project is context:
            ego = project((0, 0))
            cv2.circle(canvas, ego, 5, (40, 40, 40), 1, cv2.LINE_AA)
            draw_text(canvas, 'EGO', min(divider - 45, ego[0] + 7),
                      min(BEV_H - 5, max(27, ego[1] + 5)), .34)
    draw_text(canvas, 'Ego context', 8, 18, .43)
    draw_text(canvas, 'GT zoom +/-6 m', divider + 8, 18, .43)
    return canvas


def render_tile(nusc, row, petr_match, stream_match):
    import cv2
    import numpy as np
    tile = np.full((TILE_H, W, 3), 255, dtype=np.uint8)
    sample = nusc.get('sample', row['sample_token'])
    sd = nusc.get('sample_data', sample['data'][row['camera']])
    image = cv2.imread(str(Path(nusc.dataroot) / sd['filename']))
    if image is None:
        raise FileNotFoundError(sd['filename'])
    calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    gt_box = nusc.get_box(row['annotation_token'])
    gt_projection = pipeline.projected_segments(
        gt_box, calibration, pose, image.shape[1], image.shape[0])[:2]
    focus = gt_projection[1]
    if focus is None:
        points = [p for edge in gt_projection[0] for p in edge]
        focus = (sum(p[0] for p in points) / len(points),
                 sum(p[1] for p in points) / len(points))
    _, x0, y0 = source_crop(image, focus)
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    def box_for(match):
        pred = match[0]
        return Box(pred.translation, pred.size, Quaternion(pred.rotation))
    petr_box = box_for(petr_match)
    stream_box = box_for(stream_match) if stream_match else None
    def projection(box):
        return (pipeline.projected_segments(
            box, calibration, pose, image.shape[1], image.shape[0])[:2]
                if box is not None else None)
    petr_crop = draw_crop(image, gt_projection, projection(petr_box),
                          x0, y0, PETR_COLOR)
    stream_crop = draw_crop(image, gt_projection, projection(stream_box),
                            x0, y0, STREAM_COLOR)
    if stream_box is None:
        cv2.rectangle(stream_crop, (3, 4), (W - 4, 31), (255, 255, 255), -1)
        draw_text(stream_crop, 'No StreamPETR <4 m assigned match', 10, 23, .54,
                  thickness=2)
    draw_text(tile, 'PETR ERROR {} m'.format(row['petr_band']), 12, 28, .62,
              thickness=2)
    draw_text(tile, 'n={} | {}'.format(row['cohort_count'], row['outcome']),
              12, 55, .44)
    draw_text(tile, 'GT vis {} | range {:.1f} m'.format(
        row['visibility_band'], row['distance_m']), 12, 78, .44)
    y = HEAD_H
    draw_text(tile, 'PETR {:.2f} m  score {:.3f}'.format(
        row['petr_error_m'], row['petr_score']), 12, y + 20, .47,
        pipeline.bgr(PETR_COLOR))
    y += LABEL_H
    tile[y:y + CROP_H] = petr_crop
    y += CROP_H
    if stream_match:
        delta = row['stream_error_m'] - row['petr_error_m']
        caption = 'StreamPETR {:.2f} m  delta {:+.2f} m  score {:.3f}'.format(
            row['stream_error_m'], delta, row['stream_score'])
    else:
        caption = 'StreamPETR: no <4 m matched TP'
    draw_text(tile, caption, 12, y + 20, .44, pipeline.bgr(STREAM_COLOR))
    y += LABEL_H
    tile[y:y + CROP_H] = stream_crop
    y += CROP_H
    tile[y:y + BEV_H] = bev_panel(nusc, row, gt_box, petr_box, stream_box, sample)
    y += BEV_H
    scene = nusc.get('scene', sample['scene_token'])
    draw_text(tile, '{} | {} | GT {}'.format(
        scene['name'], row['camera'], row['annotation_token'][:8]), 12, y + 25, .42)
    draw_text(tile, 'Green GT   Red PETR   Blue StreamPETR', 12, y + 49, .42)
    return tile


def render_frame(nusc, name, distance, visibility, selections, counts, slot, slots,
                 petr_lookup, stream_lookup):
    import cv2
    import numpy as np
    canvas = np.full((TITLE_H + TILE_H, W * 4, 3), 255, dtype=np.uint8)
    draw_text(canvas, '{} | GT {} m | GT visibility {} | example {}/{}'.format(
        name, distance, VISIBILITY[visibility], slot + 1, slots),
        15, 39, .73, thickness=2)
    for column, band in enumerate(ERROR_BINS):
        x0 = column * W
        if column:
            cv2.line(canvas, (x0, TITLE_H), (x0, TITLE_H + TILE_H),
                     (190, 190, 190), 1)
        if slot >= len(selections[column]):
            draw_text(canvas, '{} m | no further example'.format(band),
                      x0 + 22, TITLE_H + 170, .53)
            continue
        row = selections[column][slot]
        row['cohort_count'] = counts[column]
        key = row['sample_token'], row['annotation_token']
        petr_match = petr_lookup.get(key)
        stream_match = stream_lookup.get(key)
        if petr_match is None:
            raise AssertionError('Selected PETR TP absent from evaluator match: {}'.format(key))
        if abs(petr_match[1] - row['petr_error_m']) > 1e-5:
            raise AssertionError('PETR saved error and live matching disagree: {}'.format(key))
        if (stream_match is None) != (row['stream_error_m'] is None):
            raise AssertionError('StreamPETR saved and live match status disagree: {}'.format(key))
        if stream_match and abs(stream_match[1] - row['stream_error_m']) > 1e-5:
            raise AssertionError('StreamPETR saved error and live matching disagree: {}'.format(key))
        canvas[TITLE_H:, x0:x0 + W] = render_tile(
            nusc, row, petr_match, stream_match)
    return canvas


def main():
    dev_ws = HERE.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-gt', type=Path,
                        default=HERE / 'outputs/paired_gt.csv')
    parser.add_argument('--petr-results', type=Path,
                        default=HERE.parent / 'localization_oracle/outputs/baseline/'
                        'baseline/results_nusc.json')
    parser.add_argument('--stream-results', type=Path,
                        default=dev_ws / 'StreamPETR/test/'
                        'stream_petr_r50_noflash_428q_nui_60e/'
                        'Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json')
    parser.add_argument('--data-root', type=Path,
                        default=dev_ws / 'StreamPETR/data/nuscenes')
    parser.add_argument('--output-dir', type=Path,
                        default=HERE / 'outputs/petr_tp_videos')
    parser.add_argument('--classes', nargs='+', choices=pipeline.CLASSES,
                        default=list(pipeline.CLASSES))
    parser.add_argument('--distance-bins', nargs='+', choices=DISTANCE_BINS,
                        default=list(DISTANCE_BINS))
    parser.add_argument('--visibility-tokens', nargs='+', choices=tuple(VISIBILITY),
                        default=list(VISIBILITY))
    parser.add_argument('--cameras', nargs='+', choices=pipeline.CAMERAS,
                        default=list(pipeline.CAMERAS))
    parser.add_argument('--examples-per-band', type=int, default=8)
    parser.add_argument('--hold-frames', type=int, default=4)
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--metadata-only', action='store_true',
                        help='Write the full transition table without rendering videos')
    args = parser.parse_args()
    if args.examples_per_band < 1 or args.hold_frames < 1 or args.fps <= 0:
        parser.error('examples-per-band, hold-frames and fps must be positive')
    if not args.paired_gt.is_file():
        parser.error('Run compare.py first; missing {}'.format(args.paired_gt))
    if not args.metadata_only:
        for path in (args.petr_results, args.stream_results):
            if not path.is_file():
                parser.error('Missing saved predictions: {}'.format(path))
        if not (args.data_root / 'v1.0-trainval').is_dir():
            parser.error('Missing nuScenes metadata: {}'.format(args.data_root))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = read_cohort(args.paired_gt, set(args.classes),
                         set(args.distance_bins), set(args.visibility_tokens))
    write_csv(args.output_dir / 'petr_tp_transitions.csv', cohort)
    write_csv(args.output_dir / 'transition_summary.csv', summarize(cohort))
    print('PETR 4 m TP cohort: {:,}; transition table written'.format(len(cohort)),
          flush=True)
    if args.metadata_only:
        return

    import cv2
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    pools = defaultdict(list)
    for row in cohort:
        sample = nusc.get('sample', row['sample_token'])
        row['scene_token'] = sample['scene_token']
        pools[row['class'], row['distance_bin'], row['visibility_token'],
              row['petr_band']].append(row)
    selections = {}
    selected_rows = []
    for name in args.classes:
        for distance in args.distance_bins:
            for visibility in args.visibility_tokens:
                for band in ERROR_BINS:
                    key = name, distance, visibility, band
                    pool = pools[key]
                    if not pool:
                        selections[key] = []
                        continue
                    stable = int(hashlib.sha256('|'.join(key).encode()).hexdigest()[:12], 16)
                    chosen = choose_examples(nusc, pool, args.examples_per_band,
                                             args.cameras, args.seed + stable)
                    selections[key] = chosen
                    for slot, row in enumerate(chosen):
                        selected_rows.append({
                            'class': name, 'distance_bin': distance,
                            'visibility_token': visibility,
                            'visibility_band': VISIBILITY[visibility],
                            'petr_error_band': band, 'slot': slot,
                            'cohort_count': len(pool),
                            'sample_token': row['sample_token'],
                            'annotation_token': row['annotation_token'],
                            'instance_token': row['instance_token'],
                            'scene_token': row['scene_token'],
                            'camera': row['camera'], 'outcome': row['outcome'],
                            'petr_error_m': row['petr_error_m'],
                            'stream_error_m': row['stream_error_m']
                            if row['stream_error_m'] is not None else '',
                        })
    write_csv(args.output_dir / 'selected_examples.csv', selected_rows)
    if not selected_rows:
        raise ValueError('No camera-visible examples selected')
    sample_tokens = {row['sample_token'] for row in selected_rows}
    config = config_factory('detection_cvpr_2019')
    lookups = {}
    for name, path in (('PETR', args.petr_results),
                       ('StreamPETR', args.stream_results)):
        print('Loading {} saved predictions...'.format(name), flush=True)
        evaluator = NuScenesEval(nusc, config, str(path), 'val',
                                str(args.output_dir / ('devkit_' + name.lower())),
                                verbose=False)
        lookups[name] = build_match_lookup(evaluator, sample_tokens,
                                           args.classes, nusc)
        del evaluator
    manifest = []
    for name in args.classes:
        for distance in args.distance_bins:
            for visibility in args.visibility_tokens:
                columns = [selections[name, distance, visibility, band]
                           for band in ERROR_BINS]
                slots = max(map(len, columns))
                if not slots:
                    continue
                counts = [len(pools[name, distance, visibility, band])
                          for band in ERROR_BINS]
                video = (args.output_dir / name / VISIBILITY[visibility] /
                         '{}m.mp4'.format(distance))
                video.parent.mkdir(parents=True, exist_ok=True)
                encoder = None
                for slot in range(slots):
                    frame = render_frame(nusc, name, distance, visibility,
                                         columns, counts, slot, slots,
                                         lookups['PETR'], lookups['StreamPETR'])
                    if encoder is None:
                        cv2.imwrite(str(video.with_suffix('.png')), frame)
                        encoder = pipeline.VideoEncoder(video, frame.shape[1],
                                                        frame.shape[0], args.fps)
                    for _ in range(args.hold_frames):
                        encoder.write(frame)
                encoder.release()
                manifest.append({'class': name, 'distance_bin': distance,
                                 'visibility_token': visibility,
                                 'visibility_band': VISIBILITY[visibility],
                                 'total_petr_tps': sum(counts),
                                 'selected_examples': sum(map(len, columns)),
                                 'video': str(video)})
                print('Wrote {}'.format(video), flush=True)
    write_csv(args.output_dir / 'manifest.csv', manifest)


if __name__ == '__main__':
    main()
