#!/usr/bin/env python3
"""Render crowded and non-crowded keyframes with baseline and oracle views."""
import argparse
import base64
import csv
import html
import json
import math
import random
from collections import defaultdict
from pathlib import Path


CATEGORIES = (
    'missed_gt_close_prediction_claimed_elsewhere',
    'missed_gt_no_prediction_within_threshold',
    'matched_gt_better_overlapping_prediction_elsewhere',
    'matched_gt_low_overlap',
    'matched_gt_overlap',
)

CROWDING_DISTANCE = {'pedestrian': 2.0, 'bicycle': 2.0, 'car': 4.0}


def gt_crowding_indices(boxes, distance):
    """Indices of GTs with another same-class GT strictly within distance."""
    crowded = set()
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if math.hypot(a.translation[0] - b.translation[0],
                          a.translation[1] - b.translation[1]) < distance:
                crowded.update((i, j))
    return crowded


def visible_context_gt_indices(nusc, token, class_gt, camera, ego, max_distance):
    """All filtered same-class GTs visible in this camera and close to ego."""
    annotations = [nusc.get('sample_annotation', ann_token)
                   for ann_token in nusc.get('sample', token)['anns']]
    width, height = camera['record']['width'], camera['record']['height']
    visible = []
    for gi, gt in enumerate(class_gt):
        if math.hypot(gt.translation[0] - ego[0],
                      gt.translation[1] - ego[1]) > max_distance:
            continue
        if not any(str(ann['visibility_token']) == '4' and
                   max(abs(a - b) for a, b in zip(
                       gt.translation, ann['translation'])) < 1e-4
                   for ann in annotations):
            continue
        uv = project(gt.translation, camera)
        if uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height:
            visible.append(gi)
    return visible

def iter_json_array(path):
    """Read large nuScenes JSON tables without loading the entire array."""
    decoder = json.JSONDecoder()
    with path.open() as handle:
        buffer = ''
        position = 0
        finished = False
        while True:
            if not finished and len(buffer) - position < 1024 * 1024:
                buffer = buffer[position:]
                position = 0
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    finished = True
            while position < len(buffer) and buffer[position] in ' \r\n\t,[]':
                position += 1
            if position == len(buffer):
                if finished:
                    return
                continue
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if finished:
                    raise
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    finished = True
                continue
            yield record
            position = end


def ego_poses(data_root, sample_tokens, pose_token_cache=None):
    table = data_root / 'v1.0-trainval'
    wanted = set(sample_tokens)
    sample_to_pose = json.loads(pose_token_cache.read_text()) \
        if pose_token_cache and pose_token_cache.is_file() else {}
    missing = wanted - set(sample_to_pose)
    if missing:
        for row in iter_json_array(table / 'sample_data.json'):
            if row['sample_token'] in missing and row['is_key_frame'] and \
                    '/LIDAR_TOP/' in row['filename']:
                sample_to_pose[row['sample_token']] = row['ego_pose_token']
                if not (missing - set(sample_to_pose)):
                    break
        if pose_token_cache:
            pose_token_cache.write_text(json.dumps(sample_to_pose, indent=2) + '\n')
    if not wanted <= set(sample_to_pose):
        raise ValueError('LIDAR_TOP ego pose missing for {} samples'.format(
            len(wanted - set(sample_to_pose))))
    pose_tokens = set(sample_to_pose.values())
    poses = {}
    for row in iter_json_array(table / 'ego_pose.json'):
        if row['token'] in pose_tokens:
            poses[row['token']] = {'center_global': row['translation'],
                                   'rotation_wxyz': row['rotation']}
            if len(poses) == len(pose_tokens):
                break
    if len(poses) != len(pose_tokens):
        raise ValueError('Missing {} ego poses'.format(len(pose_tokens) - len(poses)))
    return {sample: poses[pose] for sample, pose in sample_to_pose.items()}


def camera_metadata(data_root, sample_tokens, cache_dir):
    """Resolve six keyframe cameras, their poses, and calibrations."""
    table = data_root / 'v1.0-trainval'
    records_path = cache_dir / 'camera_records.json'
    pose_path = cache_dir / 'camera_poses.json'
    records = json.loads(records_path.read_text()) if records_path.is_file() else {}
    missing = set(sample_tokens) - set(records)
    if missing:
        found = defaultdict(list)
        for row in iter_json_array(table / 'sample_data.json'):
            if row['sample_token'] in missing and row['is_key_frame'] and \
                    '/CAM_' in row['filename']:
                found[row['sample_token']].append({key: row[key] for key in
                    ('filename', 'ego_pose_token', 'calibrated_sensor_token',
                     'width', 'height')})
                if all(len(found[token]) >= 6 for token in missing):
                    break
        for token in missing:
            if not found[token]:
                raise ValueError('No camera image found for ' + token)
            records[token] = found[token]
        records_path.write_text(json.dumps(records, indent=2) + '\n')
    poses = json.loads(pose_path.read_text()) if pose_path.is_file() else {}
    needed_poses = {row['ego_pose_token'] for token in sample_tokens
                    for row in records[token]} - set(poses)
    if needed_poses:
        for row in iter_json_array(table / 'ego_pose.json'):
            if row['token'] in needed_poses:
                poses[row['token']] = {'translation': row['translation'],
                                       'rotation': row['rotation']}
                if needed_poses <= set(poses):
                    break
        if not needed_poses <= set(poses):
            raise ValueError('Camera ego pose records are incomplete')
        pose_path.write_text(json.dumps(poses, indent=2) + '\n')
    calib_tokens = {row['calibrated_sensor_token'] for token in sample_tokens
                    for row in records[token]}
    with (table / 'calibrated_sensor.json').open() as handle:
        calibrations = {row['token']: row for row in json.load(handle)
                        if row['token'] in calib_tokens}
    if set(calibrations) != calib_tokens:
        raise ValueError('Camera calibration records are incomplete')
    return records, poses, calibrations


def inverse_rotate(q, vector):
    w, x, y, z = q
    x, y, z = -x, -y, -z
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty,
            vy+w*ty+z*tx-x*tz,
            vz+w*tz+x*ty-y*tx)


def rotate(q, vector):
    return inverse_rotate((q[0], -q[1], -q[2], -q[3]), vector)


def camera_bev_axes(camera):
    forward = rotate(camera['pose']['rotation'],
                     rotate(camera['calibration']['rotation'], (0, 0, 1)))
    length = math.hypot(forward[0], forward[1])
    if length < 1e-6:
        raise ValueError('Selected camera has no horizontal forward direction')
    forward = (forward[0] / length, forward[1] / length)
    right = (forward[1], -forward[0])
    return right, forward


def camera_point(global_point, pose, calibration):
    local = inverse_rotate(pose['rotation'], [a-b for a,b in
                           zip(global_point, pose['translation'])])
    return inverse_rotate(calibration['rotation'], [a-b for a,b in
                          zip(local, calibration['translation'])])


def project(point, camera):
    pose, calibration = camera['pose'], camera['calibration']
    x, y, z = camera_point(point, pose, calibration)
    if z <= 0.1:
        return None
    intrinsic = calibration['camera_intrinsic']
    return ((intrinsic[0][0]*x + intrinsic[0][1]*y + intrinsic[0][2]*z)/z,
            (intrinsic[1][0]*x + intrinsic[1][1]*y + intrinsic[1][2]*z)/z)


def box_corners_3d(box):
    width, length, height = box['size_wlh']
    center = box['center_global']
    corners = []
    for z in (-height/2, height/2):
        for x, y in ((-length/2, -width/2), (length/2, -width/2),
                     (length/2, width/2), (-length/2, width/2)):
            rotated = rotate(box['rotation_wxyz'], (x, y, z))
            corners.append(tuple(center[i] + rotated[i] for i in range(3)))
    return corners


def choose_camera(group, records, poses, calibrations):
    candidates = []
    for row in records:
        camera = {'record': row, 'pose': poses[row['ego_pose_token']],
                  'calibration': calibrations[row['calibrated_sensor_token']]}
        width, height = row['width'], row['height']
        visible = 0
        centered = 0.0
        for gt in group['ground_truth']:
            uv = project(gt['center_global'], camera)
            if uv and 0 <= uv[0] < width and 0 <= uv[1] < height:
                visible += 1
                centered += 1 - min(1, math.hypot((uv[0]-width/2)/(width/2),
                                                   (uv[1]-height/2)/(height/2)))
        candidates.append((visible, centered, camera))
    best = max(candidates, key=lambda item: (item[0], item[1]))
    return best[2], best[0]


TP_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def threshold_tag(threshold):
    return '{:g}'.format(threshold).replace('.', 'p') + 'm'


def official_assignments(group, threshold, mode='baseline'):
    if mode != 'baseline':
        outcome = group[mode + '_assignments'].get(str(float(threshold)))
        if outcome is None:
            raise ValueError('Missing oracle {} m assignments for {}'.format(
                threshold, group['sample_token']))
        return {int(pi): (entry['assigned_group_gt'] if entry['assigned_group_gt']
                         is not None else -1)
                for pi, entry in outcome.items()
                if entry['assigned_group_gt'] is not None or entry['tp_outside_group']}
    outcome = group['thresholds'].get(str(float(threshold)))
    if outcome is None:
        raise ValueError('Missing official {} m assignments for {}'.format(
            threshold, group['sample_token']))
    return {int(pi): int(gi) for pi, gi in
            outcome['official_prediction_to_gt'].items() if gi is not None}


def prediction_frame_rank(group, prediction, threshold, mode):
    """Rank among all retained same-class predictions in this frame."""
    if mode == 'baseline':
        return prediction.get('frame_score_rank',
                              prediction['prediction_index'] + 1)
    entry = group[mode + '_assignments'][str(float(threshold))][
        str(prediction['prediction_index'])]
    return entry.get('oracle_frame_rank', entry['oracle_local_rank'])


def render_camera(group, camera, image_data=None, include_predictions=True,
                  include_gt=True, tp_threshold=4.0, mode='baseline'):
    row = camera['record']
    width, height = row['width'], row['height']
    edges = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7))
    view_box = '0 0 {} {}'.format(width,height)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="{}">'.format(view_box)]
    if image_data is not None:
        parts.append('<image href="data:image/jpeg;base64,{}" width="{}" height="{}"/>'.format(
            image_data,width,height))
    official4 = official_assignments(group, tp_threshold, mode) if include_predictions else {}
    gt_by_index = {gt['gt_index']: gt for gt in group['ground_truth']}
    if include_predictions:
        for pred in group['predictions']:
            gt = gt_by_index.get(official4.get(pred['prediction_index']))
            if gt is None:
                continue
            start = project(pred['center_global'], camera)
            end = project(gt['center_global'], camera)
            if start and end:
                parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#16a34a" stroke-width="3" stroke-opacity=".8" stroke-dasharray="5 4"/>'.format(
                    start[0], start[1], end[0], end[1]))
    boxes = (group['ground_truth'] if include_gt else []) + \
            (group['predictions'] if include_predictions else [])
    for box in boxes:
        is_gt = 'gt_index' in box
        if is_gt:
            color, stroke = '#2563eb', '4'
        else:
            color = ('#16a34a' if box['prediction_index'] in official4
                     else '#7c3aed')
            stroke = '4' if box['prediction_index'] in official4 else '3'
        camera_corners = [camera_point(point,camera['pose'],camera['calibration'])
                          for point in box_corners_3d(box)]
        center = project(box['center_global'],camera)
        segments = []
        for a,b in edges:
            start,end = camera_corners[a],camera_corners[b]
            if start[2]<=0.1 and end[2]<=0.1:
                continue
            if start[2]<=0.1:
                fraction = (0.1-start[2])/(end[2]-start[2])
                start = tuple(start[i]+fraction*(end[i]-start[i]) for i in range(3))
            if end[2]<=0.1:
                fraction = (0.1-end[2])/(start[2]-end[2])
                end = tuple(end[i]+fraction*(start[i]-end[i]) for i in range(3))
            intrinsic = camera['calibration']['camera_intrinsic']
            def uv(point):
                x,y,z=point
                return ((intrinsic[0][0]*x+intrinsic[0][1]*y+intrinsic[0][2]*z)/z,
                        (intrinsic[1][0]*x+intrinsic[1][1]*y+intrinsic[1][2]*z)/z)
            first,second=uv(start),uv(end)
            if min(first[0],second[0]) < width and max(first[0],second[0]) >= 0 and \
                    min(first[1],second[1]) < height and max(first[1],second[1]) >= 0:
                segments.append((first,second))
        if not segments and not (center and 0<=center[0]<width and 0<=center[1]<height):
            continue
        dash = '' if is_gt else ' stroke-dasharray="7 5"'
        for a,b in segments:
            parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="{}"{} />'.format(
                a[0],a[1],b[0],b[1],color,stroke,dash))
            if not is_gt and len(box.get('submitted_classes', ())) > 1:
                parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#f97316" stroke-width="1.5"/>'.format(
                    a[0], a[1], b[0], b[1]))
        if is_gt and center and 0 <= center[0] < width and 0 <= center[1] < height:
            offset = rotate(box['rotation_wxyz'], (0, 0, -box['size_wlh'][2] / 2))
            bottom = project([box['center_global'][i] + offset[i] for i in range(3)],
                             camera)
            if bottom and 0 <= bottom[0] < width and 0 <= bottom[1] < height:
                parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="2" stroke-dasharray="4 4"/>'.format(
                    center[0], center[1], bottom[0], bottom[1], color))
                parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="6" fill="#fff" stroke="{}" stroke-width="3"/>'.format(
                    bottom[0], bottom[1], color))
            parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="6" fill="{}" stroke="#fff" stroke-width="2"/>'.format(
                center[0], center[1], color))
            parts.append('<text x="{:.1f}" y="{:.1f}" font-size="18" font-weight="bold" fill="{}" stroke="#fff" stroke-width="3" paint-order="stroke">G{}</text>'.format(
                center[0] + 8, center[1] - 8, color, box['gt_index']))
        if not is_gt and box['prediction_index'] in official4 and segments:
            gi = official4[box['prediction_index']]
            label = 'P{}→{}'.format(box['prediction_index'],
                                    'G{}'.format(gi) if gi >= 0 else 'outside')
            x = max(2., min(width - 100., min(point[0] for edge in segments for point in edge)))
            y = max(18., min(height - 2., min(point[1] for edge in segments for point in edge) - 3))
            parts.append('<rect x="{:.1f}" y="{:.1f}" width="96" height="19" rx="3" fill="#fff" fill-opacity=".88"/>'.format(
                x, y-16))
            parts.append('<text x="{:.1f}" y="{:.1f}" fill="#15803d" font-family="Arial" font-size="15" font-weight="bold">{}</text>'.format(
                x+3, y-2, label))
        # Labels cover small pedestrians in crowded camera views. The paired BEV
        # carries GT IDs and prediction score ranks without hiding image pixels.
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


def rank_color(rank, count):
    fraction = (rank - 1) / max(count - 1, 1)
    return 'hsl(275,75%,{:.1f}%)'.format(28 + 42 * fraction)


def esc(value):
    return html.escape(str(value), quote=True)


def load_groups(path):
    with path.open() as handle:
        return {group['sample_token']: group for group in map(json.loads, handle)}


def load_diagnostics(path, threshold):
    by_sample = defaultdict(list)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if float(row['threshold_m']) == threshold:
                by_sample[row['sample_token']].append(row)
    return by_sample


def load_visibility(path):
    visibility = {}
    if path.is_file():
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['class'] == 'pedestrian':
                    visibility[row['annotation_token']] = int(row['visibility_token'])
    return visibility


def eligible_group_tokens(groups, visibility):
    return {token for token, group in groups.items()
            if 2 <= len(group['ground_truth']) <= 6 and
            2 <= len(group['predictions']) <= 15 and
            any(math.hypot(a['center_global'][0] - b['center_global'][0],
                           a['center_global'][1] - b['center_global'][1]) <
                CROWDING_DISTANCE['pedestrian']
                for i, a in enumerate(group['ground_truth'])
                for b in group['ground_truth'][i + 1:]) and
            all(visibility.get(gt['annotation_token']) == 4
                for gt in group['ground_truth'])}


def max_gt_ego_distance(group, ego):
    return max(math.hypot(gt['center_global'][0] - ego['center_global'][0],
                          gt['center_global'][1] - ego['center_global'][1])
               for gt in group['ground_truth'])


def select_groups(groups, diagnostics, count, eligible_tokens):
    selected = []
    quota = max(1, count // len(CATEGORIES))
    for category in CATEGORIES:
        candidates = []
        for token, rows in diagnostics.items():
            if token in {item[0] for item in selected}:
                continue
            if token not in eligible_tokens:
                continue
            group = groups[token]
            n_gt, n_pred = len(group['ground_truth']), len(group['predictions'])
            n_category = sum(row['category'] == category for row in rows)
            if n_category:
                candidates.append((-n_category, n_pred, -n_gt, token))
        for _, _, _, token in sorted(candidates)[:quota]:
            selected.append((token, category))
    if len(selected) < count:
        for token in sorted(groups):
            if len(selected) >= count:
                break
            if token in {item[0] for item in selected}:
                continue
            if token in eligible_tokens:
                selected.append((token, 'additional_crowded_group'))
    return selected[:count]


def load_assignment_changes(path, threshold, change_filter):
    """GT-level final greedy assignment changes, keyed by class/frame/annotation."""
    changes = defaultdict(dict)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if float(row['threshold_m']) != threshold:
                continue
            if change_filter == 'b1':
                selected = row['subtype'] == 'B1'
            elif change_filter == 'improved':
                selected = row['category'] == 'B'
            else:
                selected = row['category'] in ('B', 'C', 'D')
            if selected:
                if not row['baseline_prediction_id'] or not row['oracle_prediction_id'] or \
                        row['baseline_prediction_id'] == row['oracle_prediction_id']:
                    raise ValueError('Invalid changed-assignment record for ' +
                                     row['gt_id'])
                changes[row['class'], row['sample_token']][row['gt_id']] = row
    return changes


def polygon(box):
    w, qx, qy, qz = box['rotation_wxyz']
    yaw = math.atan2(2 * (w*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
    x, y = box['center_global'][:2]
    width, length = box['size_wlh'][:2]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + u*c - v*s, y + u*s + v*c)
            for u, v in ((-length/2, -width/2), (length/2, -width/2),
                         (length/2, width/2), (-length/2, width/2))]


def render(group, rows, threshold, ego, camera, tp_threshold=4.0,
           mode='baseline'):
    token = group['sample_token']
    gt_by_index = {g['gt_index']: g for g in group['ground_truth']}
    official4 = official_assignments(group, tp_threshold, mode)
    visible_predictions = group['predictions']
    pred_by_index = {p['prediction_index']: p for p in group['predictions']}
    gt_owner4 = {gi: pi for pi, gi in official4.items() if gi >= 0}
    pairs = {(p['prediction_index'], p['gt_index']): p for p in group['pair_matrix']}
    right, forward = camera_bev_axes(camera)
    def oriented(point):
        return (point[0]*right[0] + point[1]*right[1],
                point[0]*forward[0] + point[1]*forward[1])
    coords = [oriented(corner) for box in group['ground_truth'] + visible_predictions
              for corner in polygon(box)]
    min_x, max_x = min(x for x, _ in coords), max(x for x, _ in coords)
    min_y, max_y = min(y for _, y in coords), max(y for _, y in coords)
    extent = max(max_x - min_x, max_y - min_y, 8.) + 2.
    mid_x, mid_y = (min_x + max_x)/2, (min_y + max_y)/2
    scale = 900. / extent
    def xy(point):
        u, v = oriented(point)
        return 50 + 450 + (u-mid_x)*scale, 90 + 450 - (v-mid_y)*scale
    def shape(points):
        return ' '.join('{:.1f},{:.1f}'.format(*xy(point)) for point in points)
    canvas_height = max(1040, 180 + 27 * len(rows) + 40 * len(visible_predictions))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1710 {}">'.format(canvas_height),
             '<rect width="1710" height="{}" fill="#fff"/>'.format(canvas_height),
             '<style>text{font-family:Arial,sans-serif;fill:#111827}.muted{fill:#475569}.small{font-size:13px}.medium{font-size:16px}.title{font-size:22px;font-weight:bold}</style>',
             '<text x="48" y="35" class="title">{} {} keyframe</text>'.format(
                 esc(group.get('model_name', 'PETR')),
                 esc(group.get('class', 'pedestrian').replace('_', ' ').title())),
             '<text x="48" y="62" class="medium">{}</text>'.format(esc(token)),
             '<text x="1010" y="34" class="medium">Prediction distribution by {} frame rank</text>'.format(
                 '3D IoU oracle' if mode == 'iou3d' else
                 'Distance oracle' if mode == 'distance' else
                 'GT-informed oracle' if mode == 'oracle' else 'PETR confidence'),
             '<text x="1010" y="56" class="small muted">GT nearby count uses &lt;{} m; green links show {} TP at {:g} m.</text>'.format(threshold, mode, tp_threshold),
             '<rect x="50" y="90" width="900" height="900" fill="#f8fafc" stroke="#cbd5e1"/>',
             '<text x="62" y="116" class="small">↑ {} forward · camera right →</text>'.format(
                 esc(Path(camera['record']['filename']).parent.name))]
    for k in range(1, 5):
        position = 50 + k*900/5
        parts.append('<line x1="{0:.1f}" y1="90" x2="{0:.1f}" y2="990" stroke="#e2e8f0"/>'.format(position))
        position = 90 + k*900/5
        parts.append('<line x1="50" y1="{0:.1f}" x2="950" y2="{0:.1f}" stroke="#e2e8f0"/>'.format(position))
    for pi, gi in official4.items():
        if gi not in gt_by_index or pi not in pred_by_index:
            continue
        px, py = xy(pred_by_index[pi]['center_global'])
        gx, gy = xy(gt_by_index[gi]['center_global'])
        parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#16a34a" stroke-width="2.5" stroke-opacity=".8" stroke-dasharray="6 4"/>'.format(
            px, py, gx, gy))
    for gt in group['ground_truth']:
        gi = gt['gt_index']
        gx, gy = xy(gt['center_global'])
        color = '#2563eb'
        parts.append('<polygon points="{}" fill="{}" fill-opacity=".10" stroke="{}" stroke-width="2.5"/>'.format(
            shape(polygon(gt)), color, color))
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="3" fill="{}"/>'.format(gx, gy, color))
        parts.append('<text x="{:.1f}" y="{:.1f}" font-size="15" font-weight="bold" fill="{}">G{}</text>'.format(
            gx+5, gy-7, color, gi))
    for pred in visible_predictions:
        pi = pred['prediction_index']
        rank = prediction_frame_rank(group, pred, tp_threshold, mode)
        color = '#16a34a' if pi in official4 else '#7c3aed'
        px, py = xy(pred['center_global'])
        parts.append('<polygon points="{}" fill="none" stroke="{}" stroke-width="2.2" stroke-dasharray="5 3"/>'.format(shape(polygon(pred)), color))
        if len(pred.get('submitted_classes', ())) > 1:
            parts.append('<polygon points="{}" fill="none" stroke="#f97316" stroke-width="1.5" stroke-opacity=".9"/>'.format(shape(polygon(pred))))
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="2.5" fill="{}"/>'.format(px, py, color))
        parts.append('<text x="{:.1f}" y="{:.1f}" font-size="12" fill="{}">P{} #{}{}</text>'.format(
            px+5, py+14, color, pi, rank,
            ' TP→{}'.format('G{}'.format(official4[pi])
                            if official4[pi] >= 0 else 'outside')
            if pi in official4 else ' FP'))
    # Keep the local view readable when ego is far away; place a direction
    # indicator on its boundary and an exact relative-position inset at right.
    ex, ey = xy(ego['center_global'])
    ego_visible = 50 <= ex <= 950 and 90 <= ey <= 990
    if ego_visible:
        heading = math.atan2(2*(ego['rotation_wxyz'][0]*ego['rotation_wxyz'][3] +
                                ego['rotation_wxyz'][1]*ego['rotation_wxyz'][2]),
                             1-2*(ego['rotation_wxyz'][2]**2 + ego['rotation_wxyz'][3]**2))
        parts.append('<polygon points="{}" fill="#111827" stroke="#fff" stroke-width="1.5"/>'.format(
            shape([(ego['center_global'][0] + 2.5*math.cos(heading), ego['center_global'][1] + 2.5*math.sin(heading)),
                   (ego['center_global'][0] + 1.5*math.cos(heading+2.5), ego['center_global'][1] + 1.5*math.sin(heading+2.5)),
                   (ego['center_global'][0] + 1.5*math.cos(heading-2.5), ego['center_global'][1] + 1.5*math.sin(heading-2.5))])))
        parts.append('<text x="{:.1f}" y="{:.1f}" font-size="14" font-weight="bold">EGO</text>'.format(ex+8, ey-8))
    else:
        dx, dy = ex-500, ey-540
        factor = min(420/abs(dx) if dx else float('inf'),
                     420/abs(dy) if dy else float('inf'))
        bx, by = 500+dx*factor, 540+dy*factor
        bx, by = max(70,min(930,bx)), max(110,min(970,by))
        parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="#111827" stroke-width="2"/>'.format(bx-18*dx/math.hypot(dx,dy), by-18*dy/math.hypot(dx,dy), bx, by))
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="5" fill="#111827"/>'.format(bx,by))
        parts.append('<text x="{:.1f}" y="{:.1f}" font-size="13" font-weight="bold">EGO direction</text>'.format(min(bx+8,830), max(by-8,120)))
    parts.extend(['<text x="50" y="1015" class="small muted">Blue = GT. Green = TP at {:g} m; purple = FP; orange outline = same box submitted in multiple classes.</text>'.format(tp_threshold),
                  '<text x="50" y="1034" class="small muted">Black ego marker is exact when in view; boundary marker points toward ego when it lies outside.</text>',
                  '<text x="1010" y="92" class="medium">GT and nearest prediction</text>'])
    y = 115
    for row in sorted(rows, key=lambda item: int(item['gt_index'])):
        gi = int(row['gt_index'])
        nearest = 'P' + row['nearest_prediction_id'].rsplit(':', 1)[1] if row['nearest_prediction_id'] else 'none'
        distance = '{:.2f}'.format(float(row['nearest_prediction_distance_m'])) if row['nearest_prediction_distance_m'] else '—'
        owner = 'TP{:g} P{}'.format(tp_threshold, gt_owner4[gi]) if gi in gt_owner4 else 'FN{:g}'.format(tp_threshold)
        parts.append('<text x="1010" y="{}" class="small"><tspan font-weight="bold">G{}</tspan> | {} | nearest {} ({} m) | {} pred &lt;{} m</text>'.format(
            y, gi, owner, esc(nearest), distance, row['predictions_within_threshold'], threshold))
        y += 27
    y += 8
    parts.append('<text x="1010" y="{}" class="medium">Every selected-camera prediction (fixed set)</text>'.format(y))
    y += 24
    for pred in sorted(visible_predictions, key=lambda item:
                       prediction_frame_rank(group, item, tp_threshold, mode)):
        pi = pred['prediction_index']
        nearest = min((pairs[pi, gi] for gi in gt_by_index),
                      key=lambda pair: pair['center_distance_m'])
        assignment = ('TP{:g}→{}'.format(tp_threshold,
                      'G{}'.format(official4[pi]) if official4[pi] >= 0 else 'outside')
                      if pi in official4 else 'FP{:g}'.format(tp_threshold))
        rank = prediction_frame_rank(group, pred, tp_threshold, mode)
        parts.append('<text x="1010" y="{}" class="small"><tspan fill="{}">P{}</tspan>  #{:02d}  score {:.2f}  | {} | nearest G{} {:.2f} m</text>'.format(
            y, '#16a34a' if pi in official4 else '#7c3aed', pi,
            rank, pred['score'], assignment, nearest['gt_index'],
            nearest['center_distance_m']))
        if len(pred.get('submitted_classes', ())) > 1:
            labels = ' | '.join('{} {:.2f}'.format(item['class'], item['score'])
                                for item in pred['submitted_classes'])
            parts.append('<text x="1031" y="{}" class="small" fill="#c2410c">also submitted: {}</text>'.format(
                y + 18, esc(labels)))
        y += 40
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def render_assignment_table(group, mode='baseline'):
    assignments = {threshold: official_assignments(group, threshold, mode)
                   for threshold in TP_THRESHOLDS}
    gt_ids = {gt['gt_index'] for gt in group['ground_truth']}
    pairs = {(pair['prediction_index'], pair['gt_index']): pair
             for pair in group['pair_matrix']}
    rows = []
    ranking_threshold = 4.0
    def display_rank(pred):
        return prediction_frame_rank(group, pred, ranking_threshold, mode)
    for pred in sorted(group['predictions'], key=display_rank):
        pi = pred['prediction_index']
        nearest = min((pairs[pi, gi] for gi in gt_ids),
                      key=lambda pair: pair['center_distance_m'])
        cells = []
        for threshold in TP_THRESHOLDS:
            assigned = assignments[threshold].get(pi)
            if assigned is None:
                status = 'FP'
            elif assigned in gt_ids:
                status = 'TP → G{} ({:.2f} m)'.format(
                    assigned, pairs[pi, assigned]['center_distance_m'])
            else:
                status = 'TP → GT outside group'
            if mode != 'baseline':
                rank_here = prediction_frame_rank(group, pred, threshold, mode)
                status = '#{} {}'.format(rank_here, status)
            cells.append('<td class="threshold t{} {}">{}</td>'.format(
                threshold_tag(threshold), 'tp' if assigned is not None else 'fp', esc(status)))
        rows.append('<tr><td>P{}</td><td>#{}</td><td>{:.3f}</td>{}<td>G{} ({:.2f} m)</td></tr>'.format(
            pi, display_rank(pred), pred['score'], ''.join(cells),
            nearest['gt_index'], nearest['center_distance_m']))
    return '<table><thead><tr><th>Prediction</th><th>{} frame rank</th><th>Original PETR score</th><th>TP @ 0.5 m</th><th>TP @ 1 m</th><th>TP @ 2 m</th><th>TP @ 4 m</th><th>Nearest group GT</th></tr></thead><tbody>{}</tbody></table>'.format(
        '{} 4 m'.format(mode.title()) if mode != 'baseline' else 'PETR', ''.join(rows))


def render_comparison_table(group):
    group_gt = {gt['gt_index'] for gt in group['ground_truth']}
    rows = []
    for threshold in TP_THRESHOLDS:
        baseline = official_assignments(group, threshold)
        oracle = official_assignments(group, threshold, 'oracle')
        distance = official_assignments(group, threshold, 'distance')
        iou3d = official_assignments(group, threshold, 'iou3d')
        baseline_gt = {gi for gi in baseline.values() if gi in group_gt}
        oracle_gt = {gi for gi in oracle.values() if gi in group_gt}
        distance_gt = {gi for gi in distance.values() if gi in group_gt}
        iou3d_gt = {gi for gi in iou3d.values() if gi in group_gt}
        def group_assignment(mapping, pi):
            if pi not in mapping:
                return None
            return mapping[pi] if mapping[pi] in group_gt else -1
        changed = [pi for pi in sorted(set(baseline) | set(oracle) |
                                    set(distance) | set(iou3d))
                   if len({group_assignment(mapping, pi) for mapping in
                           (baseline, oracle, distance, iou3d)}) > 1]
        rows.append('<tr><td>{:g} m</td><td>{}/{}</td><td>{}/{}</td><td>{}/{}</td><td>{}/{}</td><td>{}</td></tr>'.format(
            threshold, len(baseline_gt), len(group_gt), len(oracle_gt), len(group_gt),
            len(distance_gt), len(group_gt), len(iou3d_gt), len(group_gt),
            esc(', '.join('P{}'.format(pi) for pi in changed) or '—')))
    return ('<h3>Assignment changes within this group</h3>'
            '<table><thead><tr><th>Threshold</th><th>Baseline GT TPs</th>'
            '<th>Cardinality re-rank GT TPs</th><th>Distance re-rank GT TPs</th>'
            '<th>3D IoU re-rank GT TPs</th>'
            '<th>Predictions with differing TP/group-GT assignment</th></tr></thead>'
            '<tbody>{}</tbody></table>').format(''.join(rows))


def render_card(item, group):
    image = esc(item['camera_svg'])
    gt_overlay = esc(item['camera_gt_overlay_svg'])
    pred_overlay = esc(item['camera_prediction_overlay_svg'])
    bev = esc(item['svg'])
    change = group.get('selection_change')
    change_html = ('<p style="background:#fff7ed;padding:10px;border-left:4px solid #ea580c">'
                   'Changed target G{} at {:g} m ({}{}): baseline P{} score {:.3f}, '
                   '3D IoU {:.3f}, XY {:.2f} m → IoU rerank P{} score {:.3f}, '
                   '3D IoU {:.3f}, XY {:.2f} m. Scores are original PETR scores.</p>'.format(
                       change['target_gt_index'], change['threshold_m'],
                       esc(change['category']), esc('/' + change['subtype']
                           if change['subtype'] else ''),
                       change['baseline_prediction_index'],
                       change['baseline_original_score'], change['baseline_3d_iou'],
                       change['baseline_xy_error_m'],
                       change['iou3d_prediction_index'],
                       change['iou3d_original_score'], change['iou3d_3d_iou'],
                       change['iou3d_xy_error_m'])) if change else ''
    early = group.get('selection_early_fp')
    if early:
        change_html += ('<p style="background:#fff7ed;padding:10px;border-left:4px solid #ea580c">'
                        'Baseline at {:g} m: {} full-frame FPs before the last TP '
                        '(rank #{}, score {:.3f}); {} full-frame TPs. Near target G{}, '
                        '{} early FPs versus {} TPs within {:.1f} m '
                        '({} early FP centers in camera). '
                        'These counts select the case; box colors follow the active '
                        'threshold and matching method.</p>'.format(
                            early['threshold_m'], early['frame_early_fp_count'],
                            early['last_tp_rank'], early['last_tp_score'],
                            early['frame_tp_count'], early['target_gt_index'],
                            early['local_early_fp_count'], early['local_tp_count'],
                            early['radius_m'],
                            early['camera_visible_early_fp_count']))
    return '''<article data-class="{}" data-crowding="{}" hidden><h2>{:02d}. {} · {} target · {}</h2><p>{} visible GTs ({} target), {} nearby predictions · {} camera ({} GT centers projected in frame) · farthest GT {:.1f} m from ego</p>{}
<div class="views"><div><h3>Camera</h3><div class="camera-stack"><img class="camera-base" src="{}" loading="lazy" alt="Camera keyframe {}"><img class="layer gt-layer" src="{}" loading="lazy" alt=""><img class="layer pred-layer threshold-image" data-base="{}" src="{}" loading="lazy" alt=""></div></div>
<div><h3>Camera-aligned BEV and ego position</h3><a class="threshold-link" data-base="{}" href="{}"><img class="threshold-image" data-base="{}" src="{}" loading="lazy" alt="BEV keyframe {}"></a></div></div>
<div class="assignment-table baseline-table"><h3>Baseline assignments at 0.5, 1, 2 and 4 m</h3>{}</div>
<div class="assignment-table oracle-table"><h3>Cardinality re-ranked assignments at 0.5, 1, 2 and 4 m</h3>{}</div>
<div class="assignment-table distance-table"><h3>Distance-minimizing re-ranked assignments at 0.5, 1, 2 and 4 m</h3>{}</div>
<div class="assignment-table iou3d-table"><h3>3D IoU re-ranked assignments at 0.5, 1, 2 and 4 m</h3>{}</div>{}</article>'''.format(
        esc(item['class']), esc(item['crowding']), item['index'],
        esc(item['class'].replace('_', ' ').title()),
        esc(item['crowding'].replace('_', ' ').title()),
        esc(item['sample_token']), item['gt_count'], item['target_gt_count'],
        item['prediction_count'], esc(item['camera_name']),
        item['camera_visible_gt_centers'], item['max_gt_ego_distance_m'],
        change_html, image, item['index'], gt_overlay, pred_overlay, pred_overlay,
        bev, bev, bev, bev, item['index'], render_assignment_table(group),
        render_assignment_table(group, 'oracle'),
        render_assignment_table(group, 'distance'),
        render_assignment_table(group, 'iou3d'), render_comparison_table(group))


def early_fp_metrics(token, ground_truth, predictions, threshold, radius,
                     min_early_fp=2):
    """Count baseline FPs ahead of the last TP, globally and by GT vicinity.

    `predictions` is in descending PETR score order. Matching uses every
    filtered same-class GT and prediction in the frame.
    """
    from reranking_oracle import greedy_match_records

    records = greedy_match_records({token: ground_truth}, predictions,
                                   range(len(predictions)), threshold)
    tp_indices = {pi for (pi, _) in records.values()}
    if len(tp_indices) < 2:
        return None
    last_index = max(tp_indices)
    early_fp_indices = [pi for pi in range(last_index)
                        if pi not in tp_indices]
    if len(early_fp_indices) <= len(tp_indices):
        return None
    per_gt = {}
    for gi, gt in enumerate(ground_truth):
        def close(pi):
            pred = predictions[pi]
            return math.hypot(pred.translation[0] - gt.translation[0],
                              pred.translation[1] - gt.translation[1]) <= radius
        local_fp = [pi for pi in early_fp_indices if close(pi)]
        local_tp = [pi for pi in tp_indices if close(pi)]
        if len(local_fp) > len(local_tp) and len(local_fp) >= min_early_fp:
            per_gt[gi] = {
                'threshold_m': threshold, 'radius_m': radius,
                'target_gt_index': gi, 'frame_early_fp_count': len(early_fp_indices),
                'frame_tp_count': len(tp_indices), 'last_tp_rank': last_index + 1,
                'last_tp_score': predictions[last_index].detection_score,
                'local_early_fp_count': len(local_fp),
                'local_tp_count': len(local_tp),
                'early_fp_prediction_indices': local_fp,
            }
    return per_gt


def extra_gallery_groups(args, cache, used_tokens, changes=None):
    """Sample GT-defined crowding strata and replay full-frame matching."""
    from nuscenes import NuScenes
    from confidence_quality_correlation import load_filtered_boxes
    from reranking_oracle import (distance_oracle_order, greedy_match_records,
                                  iou3d_oracle_order, oracle_order)

    pred_boxes, gt_boxes, _ = load_filtered_boxes(args.results_json, args.data_root)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    results = []
    for class_index, class_name in enumerate(('pedestrian', 'car', 'bicycle')):
      for stratum in (('crowded', 'non_crowded') if args.selection_mode != 'random' or
                      class_name != 'pedestrian' else ('non_crowded',)):
        requested = ((args.crowded_count if stratum == 'crowded' else
                      args.count - args.crowded_count) if class_name == 'pedestrian'
                     else args.extra_count)
        if not requested:
            continue
        candidates = []
        for token in gt_boxes.sample_tokens:
            if token in used_tokens:
                continue
            if changes is not None and (class_name, token) not in changes:
                continue
            class_gt = [box for box in gt_boxes[token]
                        if box.detection_name == class_name]
            if not class_gt:
                continue
            crowded = gt_crowding_indices(class_gt, CROWDING_DISTANCE[class_name])
            if stratum == 'crowded' and not crowded:
                continue
            if stratum == 'non_crowded' and len(crowded) == len(class_gt):
                continue
            annotations = [nusc.get('sample_annotation', ann_token)
                           for ann_token in nusc.get('sample', token)['anns']]
            change_rows = changes.get((class_name, token), {}) if changes is not None else None
            if changes is not None and not change_rows:
                continue
            visible = []
            for gi, gt in enumerate(class_gt):
                if (gi in crowded) != (stratum == 'crowded'):
                    continue
                if any(str(ann['visibility_token']) == '4' and
                       (changes is None or ann['token'] in change_rows) and
                       max(abs(a - b) for a, b in zip(
                           gt.translation, ann['translation'])) < 1e-4
                       for ann in annotations):
                    visible.append(gi)
            if visible:
                if args.selection_mode == 'early_fp':
                    full_pred = [box for box in pred_boxes[token]
                                 if box.detection_name == class_name]
                    full_pred = [full_pred[i] for i in sorted(
                        range(len(full_pred)),
                        key=lambda i: (full_pred[i].detection_score, i), reverse=True)]
                    metrics = early_fp_metrics(
                        token, class_gt, full_pred, args.selection_threshold,
                        args.extra_prediction_radius, args.min_early_fp)
                    if not metrics:
                        continue
                    visible = [gi for gi in visible if gi in metrics]
                    if visible:
                        visible.sort(key=lambda gi: (-metrics[gi]['local_early_fp_count'],
                                                     metrics[gi]['local_tp_count'], gi))
                        candidates.append((token, visible, metrics))
                else:
                    candidates.append((token, visible, None))
        rng = random.Random(args.extra_seed + class_index * 2 +
                            (stratum == 'non_crowded'))
        rng.shuffle(candidates)
        if args.selection_mode == 'early_fp':
            candidates.sort(key=lambda item: (
                -item[2][item[1][0]]['local_early_fp_count'],
                -item[2][item[1][0]]['frame_early_fp_count'],
                item[2][item[1][0]]['local_tp_count']))
        # Camera and ego-distance checks may reject candidates.
        candidates = candidates[:min(max(requested * 30, 200), 800)]
        if not candidates:
            raise ValueError('No visible {} {} GT candidates'.format(
                stratum, class_name))
        pose_cache = args.output_dir / 'extra_ego_pose_tokens.json'
        extra_ego = ego_poses(args.data_root, {token for token, _, _ in candidates},
                              pose_cache)
        cache.update(extra_ego)
        eligible = []
        for token, visible, early_info in candidates:
            ego = cache[token]['center_global']
            class_gt = [box for box in gt_boxes[token]
                        if box.detection_name == class_name]
            close = [gi for gi in visible if math.hypot(
                class_gt[gi].translation[0] - ego[0],
                class_gt[gi].translation[1] - ego[1]) <= args.max_ego_distance]
            if close:
                if args.selection_mode == 'early_fp':
                    close.sort(key=lambda gi: (
                        -early_info[gi]['local_early_fp_count'],
                        early_info[gi]['local_tp_count'], gi))
                else:
                    close.sort(key=lambda gi: math.hypot(
                        class_gt[gi].translation[0] - ego[0],
                        class_gt[gi].translation[1] - ego[1]))
                if stratum == 'crowded' and args.selection_mode == 'random':
                    pairs = [(i, j) for i in close for j in close
                             if i < j and math.hypot(
                                 class_gt[i].translation[0] - class_gt[j].translation[0],
                                 class_gt[i].translation[1] - class_gt[j].translation[1]) <
                             CROWDING_DISTANCE[class_name]]
                    if not pairs:
                        continue
                    pair = min(pairs, key=lambda ij: close.index(ij[0]) +
                               close.index(ij[1]))
                    eligible.append((token, list(pair), None))
                else:
                    selected_info = dict(early_info[close[0]]) if early_info else None
                    eligible.append((token, close[:1], selected_info))
        if not eligible:
            raise ValueError('No close-range visible GT candidates for ' + class_name)
        camera_records, camera_poses, calibrations = camera_metadata(
            args.data_root, {token for token, _, _ in eligible}, args.output_dir)
        class_count = 0
        for token, preliminary_gt, early_info in eligible:
            if class_count >= requested:
                break
            if token in used_tokens:
                continue
            full_gt = [box for box in gt_boxes[token]
                       if box.detection_name == class_name]
            camera_group = {'ground_truth': [
                {'center_global': full_gt[gi].translation} for gi in preliminary_gt]}
            camera, _ = choose_camera(camera_group, camera_records[token],
                                      camera_poses, calibrations)
            width, height = camera['record']['width'], camera['record']['height']
            selected_gt = []
            for gi in preliminary_gt:
                uv = project(full_gt[gi].translation, camera)
                if uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height:
                    selected_gt.append(gi)
            if stratum == 'crowded' and args.selection_mode == 'random':
                selected_gt = [gi for gi in selected_gt if any(
                    gj != gi and math.hypot(
                        full_gt[gi].translation[0] - full_gt[gj].translation[0],
                        full_gt[gi].translation[1] - full_gt[gj].translation[1]) <
                    CROWDING_DISTANCE[class_name] for gj in selected_gt)]
            else:
                selected_gt = selected_gt[:1]
            if not selected_gt or (stratum == 'crowded' and args.selection_mode == 'random' and
                                   len(selected_gt) < 2) or not (args.data_root /
                                       camera['record']['filename']).is_file():
                continue
            target_gt = list(selected_gt)
            context_gt = visible_context_gt_indices(
                nusc, token, full_gt, camera, cache[token]['center_global'],
                args.max_ego_distance)
            if not set(target_gt) <= set(context_gt):
                raise ValueError('Selected GT missing from visible context for ' + token)
            if args.selection_mode != 'random' and stratum == 'crowded' and not any(
                    gi != target_gt[0] and math.hypot(
                        full_gt[gi].translation[0] - full_gt[target_gt[0]].translation[0],
                        full_gt[gi].translation[1] - full_gt[target_gt[0]].translation[1]) <
                    CROWDING_DISTANCE[class_name] for gi in context_gt):
                continue
            selected_gt = context_gt
            change_row = None
            if changes is not None:
                target_box = full_gt[target_gt[0]]
                matching_ids = [ann_token for ann_token in
                                nusc.get('sample', token)['anns']
                                if max(abs(a - b) for a, b in zip(
                                    nusc.get('sample_annotation', ann_token)['translation'],
                                    target_box.translation)) < 1e-4]
                if len(matching_ids) != 1:
                    raise ValueError('Cannot identify selected changed GT for ' + token)
                change_row = changes[class_name, token][matching_ids[0]]
            full_pred = [box for box in pred_boxes[token]
                         if box.detection_name == class_name]
            full_pred = [full_pred[i] for i in sorted(
                range(len(full_pred)),
                key=lambda i: (full_pred[i].detection_score, i), reverse=True)]
            if early_info:
                visible_early_fp = 0
                for pi in early_info['early_fp_prediction_indices']:
                    uv = project(full_pred[pi].translation, camera)
                    if uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height:
                        visible_early_fp += 1
                if not visible_early_fp:
                    continue
                early_info['camera_visible_early_fp_count'] = visible_early_fp
            nearby = [pi for pi, pred in enumerate(full_pred)
                      if min(math.hypot(pred.translation[0] - full_gt[gi].translation[0],
                                        pred.translation[1] - full_gt[gi].translation[1])
                             for gi in selected_gt) <= args.extra_prediction_radius]
            if not nearby:
                continue
            group = {
                'sample_token': token, 'class': class_name,
                'crowding': stratum,
                'crowding_threshold_m': CROWDING_DISTANCE[class_name],
                'target_gt_indices': target_gt,
                'ground_truth': [{
                    'gt_index': gi, 'center_global': list(full_gt[gi].translation),
                    'size_wlh': list(full_gt[gi].size),
                    'rotation_wxyz': list(full_gt[gi].rotation)}
                    for gi in selected_gt],
                'predictions': [{
                    'prediction_index': pi, 'center_global': list(full_pred[pi].translation),
                    'size_wlh': list(full_pred[pi].size),
                    'rotation_wxyz': list(full_pred[pi].rotation),
                    'score': full_pred[pi].detection_score,
                    'local_score_rank': rank,
                    'frame_score_rank': pi + 1}
                    for rank, pi in enumerate(nearby, 1)],
            }
            if early_info:
                group['selection_early_fp'] = early_info
            group['pair_matrix'] = [
                {'prediction_index': pi, 'gt_index': gi,
                 'center_distance_m': math.hypot(
                     full_pred[pi].translation[0] - full_gt[gi].translation[0],
                     full_pred[pi].translation[1] - full_gt[gi].translation[1])}
                for pi in nearby for gi in selected_gt]
            group['thresholds'] = {}
            for mode in ('oracle', 'distance', 'iou3d'):
                group[mode + '_assignments'] = {}
            gt_frame = {token: full_gt}
            selected_records = {}
            for threshold in TP_THRESHOLDS:
                orders = {'baseline': list(range(len(full_pred))),
                          'oracle': oracle_order(gt_frame, full_pred, threshold)[0],
                          'distance': distance_oracle_order(
                              gt_frame, full_pred, threshold)[0],
                          'iou3d': iou3d_oracle_order(
                              gt_frame, full_pred, threshold)[0]}
                for mode, order in orders.items():
                    records = greedy_match_records(gt_frame, full_pred, order,
                                                   threshold)
                    if changes is not None and threshold == args.selection_threshold and \
                            mode in ('baseline', 'iou3d'):
                        selected_records[mode] = records.get((token, target_gt[0]))
                    owner = {pi: gi for (_, gi), (pi, _) in records.items()}
                    if mode == 'baseline':
                        group['thresholds'][str(threshold)] = {
                            'official_prediction_to_gt': {
                                str(pi): (owner[pi] if owner[pi] in selected_gt else -1)
                                if pi in owner else None for pi in nearby}}
                    else:
                        global_rank = {pi: rank for rank, pi in enumerate(order, 1)}
                        local_rank = {pi: rank for rank, pi in enumerate(
                            sorted(nearby, key=lambda pi: global_rank[pi]), 1)}
                        group[mode + '_assignments'][str(threshold)] = {
                            str(pi): {'oracle_local_rank': local_rank[pi],
                                      'oracle_frame_rank': global_rank[pi],
                                      'assigned_group_gt': owner.get(pi)
                                      if owner.get(pi) in selected_gt else None,
                                      'tp_outside_group': (pi in owner and
                                                           owner[pi] not in selected_gt)}
                            for pi in nearby}
            if changes is not None:
                baseline_pair = selected_records['baseline']
                iou3d_pair = selected_records['iou3d']
                if baseline_pair is None or iou3d_pair is None or \
                        baseline_pair[0] == iou3d_pair[0]:
                    raise ValueError('Saved change not reproduced for {} G{}'.format(
                        token, target_gt[0]))
                if baseline_pair[0] not in nearby or iou3d_pair[0] not in nearby:
                    raise ValueError('Changed prediction omitted by display radius')
                for mode, pair, field in (
                        ('baseline', baseline_pair, 'baseline_prediction_box_json'),
                        ('iou3d', iou3d_pair, 'oracle_prediction_box_json')):
                    saved = json.loads(change_row[field])
                    actual = full_pred[pair[0]]
                    if max(abs(a - b) for a, b in zip(
                            saved['translation'], actual.translation)) > 1e-4 or \
                            abs(saved['detection_score'] -
                                actual.detection_score) > 1e-6:
                        raise ValueError('Saved {} prediction differs from replay for {}'.format(
                            mode, token))
                group['selection_change'] = {
                    'threshold_m': args.selection_threshold,
                    'category': change_row['category'],
                    'subtype': change_row['subtype'],
                    'target_gt_index': target_gt[0],
                    'baseline_prediction_index': baseline_pair[0],
                    'iou3d_prediction_index': iou3d_pair[0],
                    'baseline_original_score': float(change_row['baseline_original_score']),
                    'iou3d_original_score': float(change_row['oracle_original_score']),
                    'baseline_3d_iou': float(change_row['baseline_3d_iou']),
                    'iou3d_3d_iou': float(change_row['oracle_3d_iou']),
                    'baseline_xy_error_m': float(change_row['baseline_xy_error_m']),
                    'iou3d_xy_error_m': float(change_row['oracle_xy_error_m']),
                }
            diagnostics = []
            for gi in selected_gt:
                nearest = min(nearby, key=lambda pi: math.hypot(
                    full_pred[pi].translation[0] - full_gt[gi].translation[0],
                    full_pred[pi].translation[1] - full_gt[gi].translation[1]))
                distance = math.hypot(
                    full_pred[nearest].translation[0] - full_gt[gi].translation[0],
                    full_pred[nearest].translation[1] - full_gt[gi].translation[1])
                diagnostics.append({
                    'gt_index': gi, 'nearest_prediction_id': 'P:{}'.format(nearest),
                    'nearest_prediction_distance_m': distance,
                    'predictions_within_threshold': sum(
                        math.hypot(full_pred[pi].translation[0] - full_gt[gi].translation[0],
                                   full_pred[pi].translation[1] - full_gt[gi].translation[1])
                        < args.threshold for pi in nearby)})
            results.append((token, class_name, stratum, group, diagnostics, camera))
            used_tokens.add(token)
            class_count += 1
        if args.selection_mode != 'random' and class_count < requested and class_count:
            print('Selected {} usable {} {} {} frames (requested {})'.format(
                class_count, args.selection_mode, stratum, class_name,
                requested), flush=True)
        if class_count == 0 or (args.selection_mode == 'random' and class_count < requested):
            raise ValueError('Only {} usable {} {} frames found; requested {}'.format(
                class_count, stratum, class_name, requested))
    return results


def main():
    base = Path(__file__).parent / 'outputs'
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', type=Path, default=base / 'groups.jsonl')
    parser.add_argument('--gt-diagnostics', type=Path, default=base / 'gt_diagnostics.csv')
    parser.add_argument('--visibility-csv', type=Path, default=root /
                        'experiments/nuscnes_exp/paired_visibility_comparison/outputs/paired_gt.csv')
    parser.add_argument('--output-dir', type=Path, default=base / 'visualizations')
    parser.add_argument('--oracle-json', type=Path, default=base /
                        'reranking_oracle/gallery_oracle_assignments.json')
    parser.add_argument('--data-root', type=Path,
                        default=root.parent / 'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--selection-mode', choices=('random', 'changed', 'early_fp'),
                        default='random',
                        help='random gallery, changed 3D IoU assignment, or early-FP cases')
    parser.add_argument('--transitions-csv', type=Path, default=base /
                        'assignment_transition/gt_transitions.csv')
    parser.add_argument('--selection-threshold', type=float, default=2.0,
                        choices=TP_THRESHOLDS,
                        help='evaluation threshold used to select changed or early-FP cases')
    parser.add_argument('--min-early-fp', type=int, default=2,
                        help='early-FP mode: minimum FPs before last TP near target')
    parser.add_argument('--change-filter', choices=('any', 'improved', 'b1'),
                        default='any',
                        help='changed mode: any different assignment, higher 3D IoU, or B1')
    parser.add_argument('--threshold', type=float, default=2.0, choices=(0.5, 1.0, 2.0, 4.0))
    parser.add_argument('--count', type=int, default=60)
    parser.add_argument('--crowded-count', type=int, default=None,
                        help='crowded pedestrian frames; default half of --count')
    parser.add_argument('--max-ego-distance', type=float, default=20.0,
                        help='maximum BEV distance in metres for every selected GT')
    parser.add_argument('--extra-count', type=int, default=10,
                        help='random car and bicycle keyframes per crowding group per class')
    parser.add_argument('--extra-seed', type=int, default=42,
                        help='reproducible car and bicycle frame selection')
    parser.add_argument('--extra-prediction-radius', type=float, default=8.0,
                        help='display retained same-class predictions within this XY radius of a visible GT')
    args = parser.parse_args()
    if args.crowded_count is None:
        args.crowded_count = args.count // 2
    if args.count < 2 or not 0 < args.crowded_count < args.count or \
            args.max_ego_distance <= 0 or args.extra_count < 0 or \
            args.extra_prediction_radius < max(TP_THRESHOLDS) or args.min_early_fp < 1:
        parser.error('count must be at least 2, crowded-count between 1 and count-1; '
                     'max-ego-distance positive; extra-count nonnegative; '
                     'extra-prediction-radius at least 4 m; min-early-fp positive')
    if not args.results_json.is_file():
        parser.error('PETR results JSON missing: {}'.format(args.results_json))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / 'ego_positions.json'
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    groups, diagnostics, camera_choices, chosen = {}, {}, {}, []
    changes = None
    if args.selection_mode != 'random':
        if args.selection_mode == 'changed':
            if not args.transitions_csv.is_file():
                parser.error('GT transition CSV missing: {}'.format(args.transitions_csv))
            changes = load_assignment_changes(
                args.transitions_csv, args.selection_threshold, args.change_filter)
            if not changes:
                parser.error('No assignment changes at the selected threshold')
    else:
        if not args.oracle_json.is_file():
            parser.error('Missing {}. Run pedestrian reranking_oracle.py first'.format(
                args.oracle_json))
        groups = load_groups(args.groups)
        diagnostics = load_diagnostics(args.gt_diagnostics, args.threshold)
        visibility = load_visibility(args.visibility_csv)
        if not visibility:
            parser.error('Pedestrian visibility records missing from {}'.format(
                args.visibility_csv))
        eligible = eligible_group_tokens(groups, visibility)
        needed = eligible - set(cache)
        if needed:
            cache.update(ego_poses(args.data_root, needed,
                                   args.output_dir / 'ego_pose_tokens.json'))
            cache_path.write_text(json.dumps(cache, indent=2) + '\n')
        eligible = {token for token in eligible
                    if max_gt_ego_distance(groups[token], cache[token]) <=
                    args.max_ego_distance}
        candidate_groups = select_groups(groups, diagnostics,
                                         args.crowded_count * 5, eligible)
        if not candidate_groups:
            parser.error('No crowded groups meet visibility and ego distance limits')
        camera_records, camera_poses, calibrations = camera_metadata(
            args.data_root, {token for token, _ in candidate_groups},
            args.output_dir)
        camera_choices = {token: choose_camera(
            groups[token], camera_records[token], camera_poses, calibrations)
            for token, _ in candidate_groups}
        camera_visible = {token for token, _ in candidate_groups
                          if camera_choices[token][1] ==
                          len(groups[token]['ground_truth'])}
        chosen = select_groups(groups, diagnostics, args.crowded_count,
                               camera_visible)
        if len(chosen) < args.crowded_count:
            parser.error('Only {} crowded groups pass gallery requirements'.format(
                len(chosen)))
        export = json.loads(args.oracle_json.read_text())
        oracle_export = export['sample_tokens']
        distance_export = export.get('distance_sample_tokens', {})
        iou3d_export = export.get('iou3d_sample_tokens', {})
        for token, _ in chosen:
            if token not in oracle_export or set(oracle_export[token]) != \
                    {str(t) for t in TP_THRESHOLDS}:
                parser.error('Oracle export missing for ' + token)
            if token not in distance_export or set(distance_export[token]) != \
                    {str(t) for t in TP_THRESHOLDS}:
                parser.error('Distance oracle export missing for ' + token)
            if token not in iou3d_export or set(iou3d_export[token]) != \
                    {str(t) for t in TP_THRESHOLDS}:
                parser.error('3D IoU oracle export missing for ' + token)
            groups[token]['oracle_assignments'] = oracle_export[token]
            groups[token]['distance_assignments'] = distance_export[token]
            groups[token]['iou3d_assignments'] = iou3d_export[token]
            groups[token]['class'] = 'pedestrian'
            groups[token]['crowding'] = 'crowded'
            groups[token]['crowding_threshold_m'] = CROWDING_DISTANCE['pedestrian']
            groups[token]['target_gt_indices'] = [
                gt['gt_index'] for gt in groups[token]['ground_truth']]
    extra = extra_gallery_groups(args, cache, {token for token, _ in chosen},
                                 changes=changes)
    for token, class_name, stratum, group, rows, camera in extra:
        groups[token] = group
        diagnostics[token] = rows
        camera_choices[token] = (camera, len(group['ground_truth']))
        chosen.append((token, '{}_{}_{}_frame'.format(
            args.selection_mode, stratum, class_name)))
    if (args.output_dir / 'comparison_models.json').is_file() and \
            (args.output_dir / 'manifest.csv').is_file():
        with (args.output_dir / 'manifest.csv').open(newline='') as handle:
            previous = [(row['sample_token'], row['class'], row['crowding'])
                        for row in csv.DictReader(handle)]
        current = [(token, groups[token]['class'], groups[token]['crowding'])
                   for token, _ in chosen]
        if current != previous:
            raise ValueError('Selected frames differ from the existing model gallery; '
                             'use a fresh --output-dir to change case selection')
    cache_path.write_text(json.dumps(cache, indent=2) + '\n')
    for old_zoom in args.output_dir.glob('*_camera_zoom.svg'):
        if not old_zoom.name.startswith('._') and old_zoom.is_file():
            old_zoom.unlink()
    manifest = []
    for index, (token, category) in enumerate(chosen, 1):
        camera, visible_gt_count = camera_choices[token]
        filename = '{:02d}_{}.svg'.format(index, token)
        camera_filename = '{:02d}_{}_camera.svg'.format(index, token)
        camera_output = args.output_dir / camera_filename
        existing = camera_output.read_text() if camera_output.is_file() else ''
        marker = 'data:image/jpeg;base64,'
        if marker in existing:
            start = existing.index(marker) + len(marker)
            image_data = existing[start:existing.index('"', start)]
        else:
            image_path = args.data_root / camera['record']['filename']
            if not image_path.is_file():
                raise FileNotFoundError('Camera image missing: {}'.format(image_path))
            image_data = base64.b64encode(image_path.read_bytes()).decode('ascii')
        gt_camera_filename = '{:02d}_{}_camera_gt.svg'.format(index, token)
        (args.output_dir / gt_camera_filename).write_text(render_camera(
            groups[token], camera, image_data, include_predictions=False))
        gt_overlay_filename = '{:02d}_{}_camera_gt_overlay.svg'.format(index, token)
        (args.output_dir / gt_overlay_filename).write_text(render_camera(
            groups[token], camera, include_predictions=False))
        camera_output.write_text(render_camera(
            groups[token], camera, image_data, include_predictions=False,
            include_gt=False))
        for tp_threshold in TP_THRESHOLDS:
            for mode in ('baseline', 'oracle', 'distance', 'iou3d'):
                suffix = ('_' + mode if mode != 'baseline' else '') + (
                    '' if tp_threshold == 4.0 else '_tp{}'.format(threshold_tag(tp_threshold)))
                (args.output_dir / filename.replace('.svg', suffix + '.svg')).write_text(
                    render(groups[token], diagnostics[token], args.threshold,
                           cache[token], camera, tp_threshold=tp_threshold,
                           mode=mode))
                pred_overlay = camera_filename.replace('.svg', '_pred' + suffix + '.svg')
                (args.output_dir / pred_overlay).write_text(render_camera(
                    groups[token], camera, include_gt=False, tp_threshold=tp_threshold,
                    mode=mode))
        manifest.append({'index': index, 'sample_token': token,
                         'class': groups[token]['class'],
                         'crowding': groups[token]['crowding'],
                         'crowding_threshold_m': groups[token]['crowding_threshold_m'],
                         'selection_category': category,
                         'change_category': groups[token].get(
                             'selection_change', {}).get('category', ''),
                         'change_subtype': groups[token].get(
                             'selection_change', {}).get('subtype', ''),
                         'changed_target_gt_index': groups[token].get(
                             'selection_change', {}).get('target_gt_index', ''),
                         'baseline_target_prediction_index': groups[token].get(
                             'selection_change', {}).get(
                                 'baseline_prediction_index', ''),
                         'iou3d_target_prediction_index': groups[token].get(
                             'selection_change', {}).get(
                                 'iou3d_prediction_index', ''),
                         'early_fp_selection_threshold_m': groups[token].get(
                             'selection_early_fp', {}).get('threshold_m', ''),
                         'early_fp_before_last_tp_frame': groups[token].get(
                             'selection_early_fp', {}).get('frame_early_fp_count', ''),
                         'early_fp_before_last_tp_near_target': groups[token].get(
                             'selection_early_fp', {}).get('local_early_fp_count', ''),
                         'tp_near_target': groups[token].get(
                             'selection_early_fp', {}).get('local_tp_count', ''),
                         'early_fp_centers_in_camera': groups[token].get(
                             'selection_early_fp', {}).get(
                                 'camera_visible_early_fp_count', ''),
                         'threshold_m': args.threshold,
                         'gt_visibility_token': 4,
                         'max_gt_ego_distance_m': round(max_gt_ego_distance(
                             groups[token], cache[token]), 3),
                         'gt_count': len(groups[token]['ground_truth']),
                         'target_gt_count': len(groups[token]['target_gt_indices']),
                         'prediction_count': len(groups[token]['predictions']),
                         'svg': filename, 'camera_gt_svg': gt_camera_filename,
                         'camera_svg': camera_filename,
                         'camera_gt_overlay_svg': gt_overlay_filename,
                         'camera_prediction_overlay_svg': camera_filename.replace('.svg', '_pred.svg'),
                         'camera_name': Path(camera['record']['filename']).parent.name,
                         'camera_visible_gt_centers': visible_gt_count})
    with (args.output_dir / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    cards = '\n'.join(render_card(item, groups[item['sample_token']])
                      for item in manifest)
    class_buttons = ''.join(
        '<button type="button" data-class="{}" aria-pressed="{}">{} ({})</button>'.format(
            name, 'true' if name == 'pedestrian' else 'false',
            name.replace('_', ' ').title(),
            sum(item['class'] == name for item in manifest))
        for name in (('pedestrian', 'car', 'bicycle') if args.extra_count else ('pedestrian',)))
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>PETR BEV review</title><style>
body{{font:16px Arial,sans-serif;max-width:1530px;margin:32px auto;color:#111827}}article{{margin:28px 0;border-top:1px solid #cbd5e1;padding-top:16px}}p{{color:#475569}}table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{padding:7px 10px;border:1px solid #cbd5e1;text-align:left}}th{{background:#f1f5f9}}td.tp{{background:#dcfce7}}.controls{{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #cbd5e1;z-index:2}}.research-summary img{{display:block;width:100%;border:1px solid #cbd5e1;box-sizing:border-box}}.views{{display:grid;grid-template-columns:minmax(0,1fr);gap:20px}}.views img{{display:block;width:100%}}.views h3{{margin:8px 0}}.views .threshold-link img{{border:1px solid #cbd5e1;box-sizing:border-box}}.camera-stack{{position:relative;width:100%;border:1px solid #cbd5e1;box-sizing:border-box}}.camera-stack img{{display:block;width:100%;margin:0}}.camera-stack .layer{{position:absolute;inset:0;height:100%;pointer-events:none}}body.hide-gt .gt-layer,body.hide-pred .pred-layer,[hidden]{{display:none!important}}body.mode-baseline .oracle-table,body.mode-baseline .distance-table,body.mode-baseline .iou3d-table,body.mode-oracle .baseline-table,body.mode-oracle .distance-table,body.mode-oracle .iou3d-table,body.mode-distance .baseline-table,body.mode-distance .oracle-table,body.mode-distance .iou3d-table,body.mode-iou3d .baseline-table,body.mode-iou3d .oracle-table,body.mode-iou3d .distance-table{{display:none}}.threshold.active{{outline:2px solid #16a34a;outline-offset:-2px}}button{{font:inherit;padding:5px 14px;margin-right:6px;cursor:pointer}}button[aria-pressed="true"]{{background:#dcfce7;border:2px solid #16a34a}}
</style></head><body class="mode-baseline"><h1>PETR validation keyframes</h1><p>Displayed GT annotations have nuScenes visibility token 4 (80–100%) and are within {} m BEV distance of ego. Car and bicycle frames are reproducibly sampled; matches use every retained same-class prediction and GT in the full frame, while the display shows predictions near the selected GTs. Each case shows its camera image followed by its camera-aligned BEV at full width. Switch among PETR confidence order, maximum-cardinality re-ranking, minimum-distance re-ranking, and maximum-3D-IoU re-ranking. The colored TP links show the actual greedy assignments after each ordering. Boxes are identical in both views. Blue shows GT; purple shows predictions ranked by the selected mode; green marks TPs and their assigned GTs. Re-ranked scores use GT information and are diagnostic. Each table compares all four thresholds. Click a BEV map to open it at full size. The &lt;{} m value sets only the nearby-count column.</p><div class="controls">Class: {}<br>Matching: <button type="button" data-mode="baseline" aria-pressed="true">PETR baseline</button><button type="button" data-mode="oracle" aria-pressed="false">Cardinality re-rank</button><button type="button" data-mode="distance" aria-pressed="false">Distance re-rank</button><button type="button" data-mode="iou3d" aria-pressed="false">3D IoU re-rank</button> Threshold: <button type="button" data-threshold="0p5m" aria-pressed="false">0.5 m</button><button type="button" data-threshold="1m" aria-pressed="false">1 m</button><button type="button" data-threshold="2m" aria-pressed="false">2 m</button><button type="button" data-threshold="4m" aria-pressed="true">4 m</button> <span>Camera boxes:</span> <button type="button" data-overlay="gt" aria-pressed="true">GT boxes</button><button type="button" data-overlay="pred" aria-pressed="true">Prediction boxes</button></div><section class="research-summary"><h2>Research summary</h2><a href="pedestrian_ap_scene_summary.svg"><img src="pedestrian_ap_scene_summary.svg" alt="Official pedestrian AP and crowded-scene matching summary"></a><p>The 60 visible close-range groups are diagnostic examples. Their GT match fraction is not AP. At 4 m, 260 nearby retained predictions in these groups are baseline FPs; their effect on AP depends on confidence rank. A non-nearest assignment flags a case for image-level review; it does not establish which pedestrian a prediction visually represents. <a href="pedestrian_ap_scene_summary.csv">Summary values</a> · <a href="non_nearest_tp_review.csv">Non-nearest TP cases</a>.</p></section>{}<script>
let selectedThreshold='4m',selectedMode='baseline',selectedClass='pedestrian',selectedCrowding='all',pageIndex=0;
const pageSize=5;
function variant(base){{const suffix=(selectedMode==='baseline'?'':'_'+selectedMode)+(selectedThreshold==='4m'?'':'_tp'+selectedThreshold);return base.replace(/\\.svg$/,suffix+'.svg')}}
function refresh(){{
 const cards=[...document.querySelectorAll('article[data-class]')];
 const matching=cards.filter(card=>card.dataset.class===selectedClass&&(selectedCrowding==='all'||card.dataset.crowding===selectedCrowding));
 const pageCount=Math.max(1,Math.ceil(matching.length/pageSize));
 pageIndex=Math.min(pageIndex,pageCount-1);
 const visible=new Set(matching.slice(pageIndex*pageSize,(pageIndex+1)*pageSize));
 cards.forEach(card=>card.hidden=!visible.has(card));
 document.querySelector('#page-status').textContent='Page '+(pageIndex+1)+' of '+pageCount+' · '+matching.length+' cases';
 document.querySelector('#page-prev').disabled=pageIndex===0;
 document.querySelector('#page-next').disabled=pageIndex>=pageCount-1;
 document.querySelector('.research-summary').hidden=selectedClass!=='pedestrian'||selectedCrowding==='non_crowded';
 document.querySelectorAll('.controls button[data-class]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.class===selectedClass?'true':'false'));
 document.querySelectorAll('.controls button[data-crowding]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.crowding===selectedCrowding?'true':'false'));
 for(const mode of ['baseline','oracle','distance','iou3d'])document.body.classList.toggle('mode-'+mode,selectedMode===mode);
 document.querySelectorAll('.controls button[data-mode]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.mode===selectedMode?'true':'false'));
 document.querySelectorAll('.controls button[data-threshold]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.threshold===selectedThreshold?'true':'false'));
 document.querySelectorAll('article:not([hidden]) .threshold-image').forEach(image=>{{const next=variant(image.dataset.base);if(image.getAttribute('src')!==next)image.setAttribute('src',next)}});
 document.querySelectorAll('article:not([hidden]) .threshold-link').forEach(link=>{{const next=variant(link.dataset.base);if(link.getAttribute('href')!==next)link.setAttribute('href',next)}});
 document.querySelectorAll('article:not([hidden]) .threshold').forEach(cell=>cell.classList.toggle('active',cell.classList.contains('t'+selectedThreshold)));
}}
document.querySelectorAll('.controls button[data-class]').forEach(button=>button.addEventListener('click',()=>{{selectedClass=button.dataset.class;pageIndex=0;refresh()}}));document.querySelectorAll('.controls button[data-crowding]').forEach(button=>button.addEventListener('click',()=>{{selectedCrowding=button.dataset.crowding;pageIndex=0;refresh()}}));document.querySelectorAll('.controls button[data-mode]').forEach(button=>button.addEventListener('click',()=>{{selectedMode=button.dataset.mode;refresh()}}));document.querySelectorAll('.controls button[data-threshold]').forEach(button=>button.addEventListener('click',()=>{{selectedThreshold=button.dataset.threshold;refresh()}}));document.querySelectorAll('.controls button[data-overlay]').forEach(button=>button.addEventListener('click',()=>{{document.body.classList.toggle('hide-'+button.dataset.overlay);button.setAttribute('aria-pressed',button.getAttribute('aria-pressed')==='true'?'false':'true')}}));document.querySelector('#page-prev').addEventListener('click',()=>{{pageIndex--;refresh()}});document.querySelector('#page-next').addEventListener('click',()=>{{pageIndex++;refresh()}});refresh();
</script></body></html>'''.format(args.max_ego_distance, args.threshold, class_buttons, cards)
    page = page.replace(
        '</div><section class="research-summary">',
        '<div class="case-pages"><button id="page-prev" type="button">Previous cases</button>'
        '<span id="page-status" aria-live="polite"></span> '
        '<button id="page-next" type="button">Next cases</button></div>'
        '</div><section class="research-summary">')
    page = page.replace(
        '<br>Matching:',
        '<br>Crowding: <button type="button" data-crowding="all" aria-pressed="true">All</button>'
        '<button type="button" data-crowding="crowded" aria-pressed="false">Crowded</button>'
        '<button type="button" data-crowding="non_crowded" aria-pressed="false">Non-crowded</button>'
        '<br>Matching:')
    page = page.replace(
        'Car and bicycle frames are reproducibly sampled;',
        'Each class has crowded and non-crowded cases sampled with fixed same-class GT '
        'center-distance cutoffs: 2 m for pedestrian and bicycle, 4 m for car. '
        'The crowding label stays fixed across evaluation thresholds;')
    page = page.replace(
        'Blue shows GT; purple shows predictions ranked by the selected mode;',
        'GT boxes are blue in both views. The nearby prediction set stays fixed '
        'when controls change. Full-frame TPs are green, including those assigned '
        'to a GT outside the displayed group; FPs are purple. Prediction labels '
        'show frame-wide same-class rank;')
    page = page.replace(
        '<h2>Research summary</h2>',
        '<h2>Previous crowded-only research summary</h2>')
    page = page.replace(
        'The 60 visible close-range groups are diagnostic examples. Their GT match fraction is not AP.',
        'This summary describes the earlier 60 crowded-group sample, not the current gallery. '
        'Its GT match fraction is not AP.')
    if args.selection_mode == 'changed':
        page = page.replace(
            "let selectedThreshold='4m'",
            "let selectedThreshold='{}'".format(
                threshold_tag(args.selection_threshold)))
        page = page.replace(
            'Each class has crowded and non-crowded cases sampled with fixed same-class GT',
            'Each class has crowded and non-crowded changed-assignment targets '
            'selected with fixed same-class GT')
        page = page.replace(
            '<h1>PETR validation keyframes</h1>',
            '<h1>PETR baseline versus 3D IoU rerank assignment changes</h1>'
            '<p>Cases have different final greedy prediction assignments for a target GT '
            'at {:g} m. Filter: {}. Target assignments are listed above each image.</p>'.format(
                args.selection_threshold, esc(args.change_filter)))
        page = page.replace(
            '<section class="research-summary">',
            '<section class="research-summary" style="display:none">')
    elif args.selection_mode == 'early_fp':
        page = page.replace(
            "let selectedThreshold='4m'",
            "let selectedThreshold='{}'".format(
                threshold_tag(args.selection_threshold)))
        page = page.replace(
            '<h1>PETR validation keyframes</h1>',
            '<h1>PETR early false-positive cases</h1>'
            '<p>Cases are selected at {:g} m using baseline PETR ranking. '
            'Full-frame FPs before the last TP exceed full-frame TPs; near '
            'the selected visible GT, at least {} early FPs outnumber nearby '
            'TPs. Selection counts are shown above each image. '
            'Prediction colors follow the active matching threshold.</p>'.format(
                args.selection_threshold, args.min_early_fp))
        page = page.replace(
            '<section class="research-summary">',
            '<section class="research-summary" style="display:none">')
    (args.output_dir / 'index.html').write_text(page)
    print('Wrote {} BEV frames to {}'.format(len(manifest), args.output_dir))


if __name__ == '__main__':
    main()
