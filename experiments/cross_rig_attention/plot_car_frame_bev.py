#!/usr/bin/env python3
"""Plot one frame's highest-score car outputs with official TP/FP decisions."""

import argparse
import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/cross_rig_attention/output/r1f_val'
DEFAULT_TOKEN = '44b18378388647449563a98b86c8c3d6'


def yaw(quaternion):
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def box_corners(box):
    x, y = (float(value) for value in box['translation'][:2])
    length, width = (float(value) for value in box['size'][:2])
    angle = yaw(box['rotation'])
    cosine, sine = math.cos(angle), math.sin(angle)
    return [(x+dx*cosine-dy*sine, y+dx*sine+dy*cosine)
            for dx, dy in ((-length/2, -width/2), (length/2, -width/2),
                           (length/2, width/2), (-length/2, width/2))]


def car_gt(input_dir, token):
    centers = {}
    with (input_dir / 'analysis/car_per_gt.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['sample_token'] == token:
                centers[int(row['gt_index'])] = (float(row['gt_x']), float(row['gt_y']))
    annotations_path = ROOT / 'data/pccr/R1-f/v1.0-trainval/sample_annotation.json'
    with annotations_path.open() as handle:
        annotations = [row for row in json.load(handle)
                       if row['sample_token'] == token]
    gt_boxes = {}
    for index, center in centers.items():
        matches = [row for row in annotations
                   if math.dist(center, tuple(float(v) for v in row['translation'][:2])) < .05]
        if len(matches) != 1:
            raise ValueError('Cannot identify evaluator-valid GT car {}'.format(index))
        gt_boxes[index] = matches[0]
    return gt_boxes


def ego_pose(token):
    tables = ROOT / 'data/pccr/R1-f/v1.0-trainval'
    with (tables / 'sample_data.json').open() as handle:
        pose_tokens = {row['ego_pose_token'] for row in json.load(handle)
                       if row['sample_token'] == token and row['is_key_frame']}
    if len(pose_tokens) != 1:
        raise ValueError('Expected one synchronized keyframe ego pose for {}'.format(token))
    with (tables / 'ego_pose.json').open() as handle:
        poses = [row for row in json.load(handle) if row['token'] in pose_tokens]
    if len(poses) != 1:
        raise ValueError('Missing ego pose for {}'.format(token))
    return poses[0]


def plot(path, token, model, top, ledger, gt_boxes, ego, bounds=None):
    all_points = [corner for box in list(gt_boxes.values())+top
                  for corner in box_corners(box)]
    all_points.append(tuple(float(value) for value in ego['translation'][:2]))
    xs, ys = [point[0] for point in all_points], [point[1] for point in all_points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    span = max(xmax-xmin, ymax-ymin, 20.0)+16
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    xmin, xmax, ymin, ymax = cx-span/2, cx+span/2, cy-span/2, cy+span/2
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        span = xmax-xmin
    left, top_px, size = 72, 125, 780

    def point(world):
        return (left+(world[0]-xmin)*size/span,
                top_px+(ymax-world[1])*size/span)

    def polygon(box):
        return ' '.join('{:.1f},{:.1f}'.format(*point(corner))
                        for corner in box_corners(box))

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1000" '
             'viewBox="0 0 1500 1000">',
             '<rect width="1500" height="1000" fill="#f3f6f9"/>',
             '<text x="70" y="56" font-family="Arial" font-size="31" '
             'font-weight="bold" fill="#183047">{} on R1-f: top 10 car outputs in one frame</text>'.format(model),
             '<text x="70" y="87" font-family="Arial" font-size="17" '
             'fill="#60758a">Frame {} · TP/FP uses official 4 m car matching</text>'.format(token),
             '<rect x="60" y="112" width="805" height="805" rx="14" fill="white"/>',
             '<rect x="890" y="112" width="550" height="805" rx="14" fill="white"/>']
    for step in range(5):
        grid = left+size*step/4
        parts.append('<path d="M{:.1f} {}V{} M{} {:.1f}H{}" '
                     'stroke="#e8edf1"/>'.format(grid, top_px, top_px+size,
                                                   left, grid, left+size))
    ex, ey = (float(value) for value in ego['translation'][:2])
    angle = yaw(ego['rotation'])
    forward = (math.cos(angle), math.sin(angle))
    side = (-forward[1], forward[0])
    arrow = [(ex+4*forward[0], ey+4*forward[1]),
             (ex-1.5*forward[0]+1.7*side[0],
              ey-1.5*forward[1]+1.7*side[1]),
             (ex-1.5*forward[0]-1.7*side[0],
              ey-1.5*forward[1]-1.7*side[1])]
    parts.append('<polygon points="{}" fill="#8154b4" fill-opacity="0.82" '
                 'stroke="white" stroke-width="2"/>'.format(
                     ' '.join('{:.1f},{:.1f}'.format(*point(world)) for world in arrow)))
    ego_x, ego_y = point((ex, ey))
    parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="white"/>'.format(
        ego_x, ego_y))
    parts.append('<text x="{:.1f}" y="{:.1f}" font-family="Arial" '
                 'font-size="17" font-weight="bold" fill="#8154b4">Ego</text>'.format(
                     ego_x+10, ego_y-10))
    for index, box in gt_boxes.items():
        parts.append('<polygon points="{}" fill="none" stroke="#168176" '
                     'stroke-width="3"/>'.format(polygon(box)))
        x, y = point(box['translation'][:2])
        parts.append('<text x="{:.1f}" y="{:.1f}" font-family="Arial" '
                     'font-size="16" font-weight="bold" fill="#168176">GT {}</text>'.format(
                         x+6, y-6, index))
    parts.append('<text x="918" y="158" font-family="Arial" font-size="20" '
                 'font-weight="bold" fill="#183047">Rank · score · AP result · nearest GT</text>')
    for position, box in enumerate(top, 1):
        row = ledger[position-1]
        is_tp = row['is_tp'] == '1'
        color = '#2563a6' if is_tp else '#c9574c'
        parts.append('<polygon points="{}" fill="{}" fill-opacity="0.12" '
                     'stroke="{}" stroke-width="3" {} />'.format(
                         polygon(box), color, color,
                         '' if is_tp else 'stroke-dasharray="7 5"'))
        x, y = point(box['translation'][:2])
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="13" fill="{}"/>'.format(x, y, color))
        parts.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle" '
                     'font-family="Arial" font-size="14" font-weight="bold" '
                     'fill="white">{}</text>'.format(x, y+5, position))
        distance = row['nearest_car_gt_distance_m']
        distance_text = 'no GT car' if not distance else '{:.1f} m'.format(float(distance))
        if row['fp_reason'] == 'duplicate_or_competing_car':
            reason = 'duplicate / already matched'
        elif row['fp_reason'] == 'near_other_class_gt':
            reason = 'near other-class GT'
        elif row['fp_reason'] == 'no_gt_within_threshold':
            reason = 'no GT within 4 m'
        else:
            reason = 'matched GT car'
        yy = 208+(position-1)*67
        parts.append('<rect x="912" y="{}" width="505" height="58" rx="8" '
                     'fill="#f5f8fa"/>'.format(yy-29))
        parts.append('<text x="928" y="{}" font-family="Arial" font-size="17" '
                     'font-weight="bold" fill="{}">#{} · {:.3f} · {}</text>'.format(
                         yy-4, color, position, float(box['detection_score']),
                         'TP' if is_tp else 'FP'))
        parts.append('<text x="928" y="{}" font-family="Arial" font-size="14" '
                     'fill="#52687d">Nearest car: {} · {}</text>'.format(
                         yy+18, distance_text, html.escape(reason)))
    parts.append('<text x="70" y="960" font-family="Arial" font-size="16" '
                 'fill="#60758a">Green: evaluator-valid GT car · blue: TP · red dashed: FP. '
                 'Numbers rank car outputs by score within this frame.</text>')
    parts.append('<text x="70" y="982" font-family="Arial" font-size="15" '
                 'fill="#8154b4">Purple arrow: ego position and forward heading '
                 '(pose marker, not a scaled vehicle box).</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--sample-token', default=DEFAULT_TOKEN)
    parser.add_argument('--model', choices=('R1', 'R1-f'), default='R1')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    token = args.sample_token
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
    top = [box for _, box in selected]
    statuses = [ledger[index] for index, _ in selected]
    gt_boxes = car_gt(args.input_dir, token)
    ego = ego_pose(token)
    output = args.output or args.input_dir / 'car_ap_rank_audit' / (
        'frame_{}_{}_top10_car_bev.svg'.format(token, args.model.replace('-', '')))
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(output, token, args.model, top, statuses, gt_boxes, ego)
    print('Saved', output)


if __name__ == '__main__':
    main()
