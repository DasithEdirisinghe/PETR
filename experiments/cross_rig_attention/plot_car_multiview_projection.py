#!/usr/bin/env python3
"""Project GT cars and top-ranked PETR car boxes onto original rig images."""

import argparse
import base64
import csv
import html
import json
import math
from pathlib import Path

from plot_car_frame_bev import ROOT, DEFAULT_INPUT, DEFAULT_TOKEN, car_gt


EDGES = ((0, 1), (1, 3), (3, 2), (2, 0),
         (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2-x1*x2-y1*y2-z1*z2,
            w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,
            w1*z2+x1*y2-y1*x2+z1*w2)


def rotate(quaternion, vector):
    q = tuple(float(value) for value in quaternion)
    norm = math.sqrt(sum(value*value for value in q))
    q = tuple(value/norm for value in q)
    conjugate = (q[0], -q[1], -q[2], -q[3])
    return qmul(qmul(q, (0.0, *vector)), conjugate)[1:]


def inverse_rotate(quaternion, vector):
    q = tuple(float(value) for value in quaternion)
    return rotate((q[0], -q[1], -q[2], -q[3]), vector)


def corners(box):
    center = tuple(float(value) for value in box['translation'])
    length, width, height = (float(value) for value in box['size'])
    output = []
    for dz in (-height/2, height/2):
        for dy in (-width/2, width/2):
            for dx in (-length/2, length/2):
                offset = rotate(box['rotation'], (dx, dy, dz))
                output.append(tuple(center[i]+offset[i] for i in range(3)))
    return output


def camera_point(world, ego, sensor):
    relative = tuple(float(world[i])-float(ego['translation'][i]) for i in range(3))
    in_ego = inverse_rotate(ego['rotation'], relative)
    relative = tuple(in_ego[i]-float(sensor['translation'][i]) for i in range(3))
    return inverse_rotate(sensor['rotation'], relative)


def near_clip(first, second, near=0.05):
    if first[2] < near and second[2] < near:
        return None
    if first[2] < near or second[2] < near:
        fraction = (near-first[2])/(second[2]-first[2])
        crossing = tuple(first[i]+fraction*(second[i]-first[i]) for i in range(3))
        if first[2] < near:
            first = crossing
        else:
            second = crossing
    return first, second


def project(point, sensor):
    intrinsic = sensor['camera_intrinsic']
    return (intrinsic[0][0]*point[0]/point[2]+intrinsic[0][2],
            intrinsic[1][1]*point[1]/point[2]+intrinsic[1][2])


def image_clip(first, second, width, height):
    """Liang–Barsky segment clipping to the original image rectangle."""
    x0, y0 = first
    x1, y1 = second
    dx, dy = x1-x0, y1-y0
    low, high = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width-x0), (-dy, y0), (dy, height-y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        value = q/p
        if p < 0:
            low = max(low, value)
        else:
            high = min(high, value)
        if low > high:
            return None
    return ((x0+low*dx, y0+low*dy), (x0+high*dx, y0+high*dy))


def lines_for_box(box, ego, sensor, width, height):
    points = [camera_point(world, ego, sensor) for world in corners(box)]
    segments = []
    for start, end in EDGES:
        visible = near_clip(points[start], points[end])
        if visible is None:
            continue
        clipped = image_clip(project(visible[0], sensor),
                             project(visible[1], sensor), width, height)
        if clipped is not None:
            segments.append(clipped)
    return segments


def records(token):
    tables = ROOT / 'data/pccr/R1-f/v1.0-trainval'
    with (tables / 'sample_data.json').open() as handle:
        cameras = [row for row in json.load(handle)
                   if row['sample_token'] == token and row['is_key_frame']
                   and row['filename'].lower().endswith(('.jpg', '.jpeg', '.png'))]
    with (tables / 'calibrated_sensor.json').open() as handle:
        calibration = {row['token']: row for row in json.load(handle)}
    cameras = [row for row in cameras
               if calibration[row['calibrated_sensor_token']]['camera_intrinsic']]
    pose_tokens = {row['ego_pose_token'] for row in cameras}
    if len(pose_tokens) != 1:
        raise ValueError('Cameras do not share one synchronized ego pose')
    with (tables / 'ego_pose.json').open() as handle:
        ego = next(row for row in json.load(handle) if row['token'] in pose_tokens)
    return sorted(cameras, key=lambda row: row['filename']), calibration, ego


def plot(output, token, model, cameras, calibration, ego, gt_boxes, predictions, statuses):
    if len(cameras) != 8:
        raise ValueError('Expected eight R1-f cameras; got {}'.format(len(cameras)))
    cell_w, image_h, label_h, gap, margin = 1280, 720, 44, 20, 24
    width = margin*2 + 2*cell_w + gap
    height = 110 + 4*(label_h+image_h+gap) + 60
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height),
        '<rect width="{}" height="{}" fill="#f3f6f9"/>'.format(width, height),
        '<text x="24" y="46" font-family="Arial" font-size="32" '
        'font-weight="bold" fill="#183047">Original R1-f camera views · '
        'GT and {} car-box projections</text>'.format(model),
        '<text x="24" y="82" font-family="Arial" font-size="19" '
        'fill="#5b7185">Frame {} · green GT · blue TP · red FP · '
        'same top 10 {} car outputs as BEV plot</text>'.format(token, model),
    ]
    image_root = ROOT / 'data/pccr/R1-f'
    for index, camera in enumerate(cameras):
        sensor = calibration[camera['calibrated_sensor_token']]
        image_path = image_root / camera['filename']
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_width, image_height = camera['width'], camera['height']
        if (image_width, image_height) != (cell_w, image_h):
            raise ValueError('Unexpected original camera dimensions: {}'.format(image_path))
        left = margin+(index % 2)*(cell_w+gap)
        top = 110+(index // 2)*(label_h+image_h+gap)
        name = image_path.parent.name
        elements.append('<rect x="{}" y="{}" width="{}" height="{}" '
                        'rx="7" fill="white"/>'.format(left, top, cell_w, label_h+image_h))
        elements.append('<text x="{}" y="{}" font-family="Arial" font-size="22" '
                        'font-weight="bold" fill="#183047">{}</text>'.format(
                            left+10, top+30, html.escape(name)))
        encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
        elements.append('<image x="{}" y="{}" width="{}" height="{}" '
                        'xlink:href="data:image/jpeg;base64,{}"/>'.format(
                            left, top+label_h, cell_w, image_h, encoded))
        for gt_index, box in gt_boxes.items():
            segments = lines_for_box(box, ego, sensor, image_width, image_height)
            for first, second in segments:
                elements.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                                'stroke="#18b982" stroke-width="4" '
                                'stroke-opacity="0.90"/>'.format(
                                    left+first[0], top+label_h+first[1],
                                    left+second[0], top+label_h+second[1]))
            if segments:
                x = min(p[0] for segment in segments for p in segment)
                y = min(p[1] for segment in segments for p in segment)
                elements.append('<text x="{:.1f}" y="{:.1f}" font-family="Arial" '
                                'font-size="18" font-weight="bold" fill="#18b982" '
                                'stroke="black" stroke-width="3" paint-order="stroke">'
                                'GT {}</text>'.format(left+x, top+label_h+max(20,y), gt_index))
        for rank, (box, status) in enumerate(zip(predictions, statuses), 1):
            color = '#32aaff' if status['is_tp'] == '1' else '#ff554c'
            segments = lines_for_box(box, ego, sensor, image_width, image_height)
            for first, second in segments:
                elements.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                                'stroke="{}" stroke-width="4" stroke-opacity="0.9" '
                                '{}/>'.format(
                                    left+first[0], top+label_h+first[1],
                                    left+second[0], top+label_h+second[1], color,
                                    '' if status['is_tp'] == '1' else 'stroke-dasharray="9 6" '))
            if segments:
                x = min(p[0] for segment in segments for p in segment)
                y = min(p[1] for segment in segments for p in segment)
                elements.append('<text x="{:.1f}" y="{:.1f}" font-family="Arial" '
                                'font-size="18" font-weight="bold" fill="{}" '
                                'stroke="black" stroke-width="3" paint-order="stroke">'
                                '#{} {:.2f} {}</text>'.format(
                                    left+x, top+label_h+max(20,y), color, rank,
                                    float(box['detection_score']),
                                    'TP' if status['is_tp'] == '1' else 'FP'))
    elements.append('<text x="24" y="{}" font-family="Arial" font-size="18" '
                    'fill="#50677a">Original, uncropped camera JPEGs. Boxes are '
                    'projected from saved global 3D boxes with camera intrinsics '
                    'and poses; lines are clipped to the image.</text>'.format(height-24))
    elements.append('</svg>')
    output.write_text('\n'.join(elements))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--sample-token', default=DEFAULT_TOKEN)
    parser.add_argument('--model', choices=('R1', 'R1-f'), default='R1')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    token = args.sample_token
    cameras, calibration, ego = records(token)
    gt_boxes = car_gt(args.input_dir, token)
    with (args.input_dir / args.model / 'formatted/results_pccr.json').open() as handle:
        boxes = json.load(handle)['results'][token]
    selected = sorted(((i, box) for i, box in enumerate(boxes)
                       if box['detection_name'] == 'car'),
                      key=lambda item: -float(item[1]['detection_score']))[:10]
    ledger = {}
    with (args.input_dir / 'car_ap_rank_audit/ranked_car_predictions.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['model'] == args.model and row['sample_token'] == token:
                ledger[int(row['prediction_index'])] = row
    output = args.output or args.input_dir / 'car_ap_rank_audit' / (
        'frame_{}_{}_top10_car_cameras.svg'.format(token, args.model.replace('-', '')))
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(output, token, args.model, cameras, calibration, ego, gt_boxes,
         [box for _, box in selected], [ledger[index] for index, _ in selected])
    print('Saved', output)


if __name__ == '__main__':
    main()
