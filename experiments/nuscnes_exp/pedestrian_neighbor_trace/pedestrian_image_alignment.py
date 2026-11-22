#!/usr/bin/env python3
"""Compare PETR's official 4 m pedestrian TP with available geometric boxes.

Every nuScenes validation pedestrian GT receives one row, including unmatched
GTs. Projected 2D IoU is the IoU of image-clipped, axis-aligned rectangles
around projected 3D boxes in a camera chosen using the GT alone. It describes
projected geometry, not the image features attended to by a PETR query.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from analyze import overlaps
from localization_capacity import load_filtered_class
from reranking_oracle import greedy_match_records
from visualize import box_corners_3d, camera_point, rotate

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
         (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
EPS = 1e-6
CLASS_CATEGORIES = {
    'car': {'vehicle.car'}, 'truck': {'vehicle.truck'},
    'bus': {'vehicle.bus.bendy', 'vehicle.bus.rigid'},
    'trailer': {'vehicle.trailer'},
    'construction_vehicle': {'vehicle.construction'},
    'pedestrian': {'human.pedestrian.adult', 'human.pedestrian.child',
                   'human.pedestrian.police_officer',
                   'human.pedestrian.construction_worker'},
    'motorcycle': {'vehicle.motorcycle'}, 'bicycle': {'vehicle.bicycle'},
    'traffic_cone': {'movable_object.trafficcone'},
    'barrier': {'movable_object.barrier'},
}


def center_distance(a, b):
    return math.hypot(a.translation[0] - b.translation[0],
                      a.translation[1] - b.translation[1])


def corners(box):
    return box_corners_3d({'center_global': box.translation,
                           'size_wlh': box.size,
                           'rotation_wxyz': box.rotation})


def camera_records(nusc, token):
    sample = nusc.get('sample', token)
    records = []
    for name in CAMERAS:
        sd = nusc.get('sample_data', sample['data'][name])
        calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        records.append((name, {'pose': pose, 'calibration': calibration,
                               'width': sd['width'], 'height': sd['height']}))
    return records


def projected_rect(box, camera):
    """Clip 3D edges at the camera near plane, then clip their image AABB."""
    points = [camera_point(p, camera['pose'], camera['calibration'])
              for p in corners(box)]
    projected = []
    intrinsics = camera['calibration']['camera_intrinsic']

    def add(point):
        x, y, z = point
        projected.append(((intrinsics[0][0]*x + intrinsics[0][1]*y +
                           intrinsics[0][2]*z)/z,
                          (intrinsics[1][0]*x + intrinsics[1][1]*y +
                           intrinsics[1][2]*z)/z))

    for point in points:
        if point[2] >= .1:
            add(point)
    for ia, ib in EDGES:
        a, b = points[ia], points[ib]
        if (a[2] < .1) == (b[2] < .1):
            continue
        t = (.1-a[2])/(b[2]-a[2])
        add(tuple(a[j] + t*(b[j]-a[j]) for j in range(3)))
    if not projected:
        return None
    x0 = max(0.0, min(x for x, _ in projected))
    y0 = max(0.0, min(y for _, y in projected))
    x1 = min(float(camera['width']), max(x for x, _ in projected))
    y1 = min(float(camera['height']), max(y for _, y in projected))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def rect_area(rect):
    return (rect[2]-rect[0])*(rect[3]-rect[1]) if rect else 0.0


def rect_iou(a, b):
    if not a or not b:
        return 0.0
    overlap = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * \
              max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    union = rect_area(a) + rect_area(b) - overlap
    return overlap/union if union else 0.0


def reference_camera(records, gt):
    # Keep this compatible with the PETR Python 3.7 environment (no :=).
    choices = []
    for name, camera in records:
        rect = projected_rect(gt, camera)
        if rect is not None:
            choices.append((rect_area(rect), name, camera, rect))
    if not choices:
        return '', None, None
    _, name, camera, rect = max(choices, key=lambda item: item[0])
    return name, camera, rect


def radial_transverse(pred, gt, camera):
    if camera is None:
        return None, None
    pose, calibration = camera['pose'], camera['calibration']
    offset = rotate(pose['rotation'], calibration['translation'])
    origin = [pose['translation'][i] + offset[i] for i in range(3)]
    rx, ry = gt.translation[0]-origin[0], gt.translation[1]-origin[1]
    length = math.hypot(rx, ry)
    if length <= EPS:
        return None, None
    dx, dy = pred.translation[0]-gt.translation[0], pred.translation[1]-gt.translation[1]
    return (dx*rx + dy*ry)/length, (dx*(-ry) + dy*rx)/length


def measurements(pred, gt, camera, gt_rect):
    radial, transverse = radial_transverse(pred, gt, camera)
    return {'score': pred.detection_score,
            'center_m': center_distance(pred, gt),
            'iou3d': overlaps(pred, gt)[1],
            'projected_iou2d': rect_iou(projected_rect(pred, camera), gt_rect)
            if camera else None,
            'radial_error_m': radial, 'transverse_error_m': transverse}


def preferred_alternative_better(alternative, selected):
    """2D evidence is mandatory; use 3D IoU, or center when IoU is absent."""
    if alternative is None or selected is None:
        return False
    if alternative['projected_iou2d'] <= selected['projected_iou2d'] + EPS:
        return False
    if alternative['iou3d'] <= EPS and selected['iou3d'] <= EPS:
        return alternative['center_m'] + EPS < selected['center_m']
    return alternative['iou3d'] > selected['iou3d'] + EPS


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_row(token, gi, selected, alternative, best_2d, preferred_alt,
               predictions, camera_name,
               gt_rect, candidate_count, pred_tp_owner, threshold, visibility):
    selected_i, selected_m = selected if selected else (None, None)
    alt_i, alt_m = alternative if alternative else (None, None)
    image_i, image_m = best_2d if best_2d else (None, None)
    preferred_i, preferred_m = preferred_alt if preferred_alt else (None, None)
    row = {'sample_token': token, 'gt_index': gi, 'gt_id': f'{token}:{gi}',
           'official_match': int(selected_i is not None),
           'camera': camera_name, 'visibility_token': visibility,
           'visibility_percent': {1: '0-40%', 2: '40-60%',
                                  3: '60-80%', 4: '80-100%'}[visibility],
           'gt_projected_area_px': rect_area(gt_rect),
           'candidate_count_within_threshold': candidate_count,
           'selected_prediction_id': selected_i if selected_i is not None else '',
           'best_3d_prediction_id': alt_i if alt_i is not None else '',
           'best_2d_prediction_id': image_i if image_i is not None else '',
           'preferred_alt_prediction_id': preferred_i if preferred_i is not None else '',
           'preferred_alt_status': (
               'unmatched' if preferred_i not in pred_tp_owner else
               'TP_same_GT' if pred_tp_owner[preferred_i] == (token, gi) else
               'TP_other_GT') if preferred_i is not None else '',
           'preferred_alt_other_gt_index': (
               pred_tp_owner[preferred_i][1]
               if preferred_i in pred_tp_owner and
               pred_tp_owner[preferred_i] != (token, gi) else ''),
           'best_3d_status': ('unmatched' if alt_i not in pred_tp_owner else
                              'TP_same_GT' if pred_tp_owner[alt_i] == (token, gi)
                              else 'TP_other_GT') if alt_i is not None else '',
           'best_3d_other_gt_index': (pred_tp_owner[alt_i][1]
                                      if alt_i in pred_tp_owner and
                                      pred_tp_owner[alt_i] != (token, gi) else ''),
           'selected_box_json': json.dumps(predictions[selected_i].serialize(),
                                           separators=(',', ':')) if selected_i is not None else '',
           'best_3d_box_json': json.dumps(predictions[alt_i].serialize(),
                                          separators=(',', ':')) if alt_i is not None else '',
           'preferred_alt_box_json': json.dumps(predictions[preferred_i].serialize(),
                                                separators=(',', ':'))
           if preferred_i is not None else ''}
    for prefix, metrics in (('selected', selected_m), ('best_3d', alt_m),
                            ('best_2d', image_m),
                            ('preferred_alt', preferred_m)):
        for key in ('score', 'center_m', 'iou3d', 'projected_iou2d',
                    'radial_error_m', 'transverse_error_m'):
            row[prefix+'_'+key] = metrics[key] if metrics and metrics[key] is not None else ''
    both = selected_m is not None and alt_m is not None and selected_i != alt_i
    image_ok = both and selected_m['projected_iou2d'] is not None and \
               alt_m['projected_iou2d'] is not None
    row['different_3d_candidate'] = int(both)
    row['better_3d_iou'] = int(both and alt_m['iou3d'] > selected_m['iou3d'] + EPS)
    row['better_center'] = int(both and alt_m['center_m'] + EPS < selected_m['center_m'])
    row['better_projected_2d'] = int(image_ok and
                                    alt_m['projected_iou2d'] >
                                    selected_m['projected_iou2d'] + EPS)
    row['lower_confidence'] = int(both and alt_m['score'] + EPS < selected_m['score'])
    row['higher_confidence'] = int(both and alt_m['score'] > selected_m['score'] + EPS)
    radial_ok = both and selected_m['radial_error_m'] is not None and \
                alt_m['radial_error_m'] is not None
    row['better_absolute_radial_error'] = int(
        radial_ok and abs(alt_m['radial_error_m']) + EPS <
        abs(selected_m['radial_error_m']))
    row['geometrically_better'] = int(row['better_3d_iou'] and
                                      row['better_projected_2d'] and
                                      row['better_absolute_radial_error'])
    row['geometrically_better_unmatched_alt'] = int(
        row['geometrically_better'] and row['best_3d_status'] == 'unmatched')
    row['delta_iou3d'] = alt_m['iou3d']-selected_m['iou3d'] if both else ''
    row['delta_projected_iou2d'] = (alt_m['projected_iou2d']-
                                    selected_m['projected_iou2d']) if image_ok else ''
    row['preferred_alt_better'] = int(
        preferred_alternative_better(preferred_m, selected_m))
    row['preferred_alt_rule_version_3'] = 3
    row['threshold_m'] = threshold
    return row


def visibility_distribution(selected):
    labels = {1: '0-40%', 2: '40-60%', 3: '60-80%', 4: '80-100%'}
    return {str(v): {'range': labels[v],
                     'count': sum(r['visibility_token'] == v for r in selected),
                     'percent': 100*sum(r['visibility_token'] == v for r in selected)/len(selected)
                     if selected else 0} for v in range(1, 5)}


def summarize(rows, prediction_count, threshold):
    matched = [r for r in rows if r['official_match']]
    compared = [r for r in matched if r['different_3d_candidate']]
    projected = [r for r in compared if r['camera']]
    counts = Counter()
    for row in compared:
        for key in ('better_3d_iou', 'better_center', 'better_projected_2d',
                    'better_absolute_radial_error', 'lower_confidence',
                    'higher_confidence', 'geometrically_better',
                    'geometrically_better_unmatched_alt'):
            counts[key] += row[key]
    counts['best_3d_already_tp_other_gt'] = sum(
        r['best_3d_status'] == 'TP_other_GT' for r in compared)
    preferred_better = [r for r in matched if r['preferred_alt_better']]
    counts['preferred_any_better'] = len(preferred_better)
    counts['preferred_unmatched_better'] = sum(
        r['preferred_alt_status'] == 'unmatched' for r in preferred_better)
    counts['preferred_assigned_other_better'] = sum(
        r['preferred_alt_status'] == 'TP_other_GT' for r in preferred_better)
    if counts['preferred_any_better'] != \
            counts['preferred_unmatched_better'] + \
            counts['preferred_assigned_other_better']:
        raise AssertionError('Preferred-alternative status decomposition failed')
    geometric = [r for r in rows if r['geometrically_better']]
    clean = [r for r in rows if r['geometrically_better_unmatched_alt']]
    unmatched = [r for r in rows if not r['official_match']]
    summary = {'threshold_m': threshold, 'gt_count': len(rows),
               'prediction_count': prediction_count,
               'official_matched_gt': len(matched),
               'official_unmatched_gt': len(rows)-len(matched),
               'gt_with_candidate': sum(r['candidate_count_within_threshold'] > 0
                                        for r in rows),
               'matched_gt_with_different_best_3d': len(compared),
               'compared_gt_with_projected_camera': len(projected),
               'counts': dict(counts),
               'denominator_for_geometric_case': len(projected),
               'geometric_case_percent_of_projected_compared':
                   100*counts['geometrically_better']/len(projected)
                   if projected else None,
               'geometric_case_percent_of_all_matched_gt':
                   100*counts['geometrically_better']/len(matched)
                   if matched else None,
               'visibility_distribution': {
                   'all_gt': visibility_distribution(rows),
                   'official_unmatched_gt': visibility_distribution(unmatched),
                   'different_best_3d_candidate': visibility_distribution(compared),
                   'geometrically_better': visibility_distribution(geometric),
                   'geometrically_better_unmatched_alternative':
                       visibility_distribution(clean)},
               'geometric_case_confidence': {
                   'lower': sum(r['lower_confidence'] for r in geometric),
                   'higher': sum(r['higher_confidence'] for r in geometric),
                   'tied': sum(not r['lower_confidence'] and
                               not r['higher_confidence'] for r in geometric)}}
    return summary


def plot_pairs(path, rows, class_name='pedestrian', model_name='PETR'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; CSV and summary were still written', flush=True)
        return
    paired = [r for r in rows if r['different_3d_candidate'] and r['camera']]
    if not paired:
        return
    x = [r['delta_iou3d'] for r in paired]
    y = [r['delta_projected_iou2d'] for r in paired]
    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(x, y, gridsize=55, mincnt=1, bins='log', cmap='viridis')
    fig.colorbar(hb, ax=ax, label='GT count (log scale)')
    ax.axhline(0, color='black', linewidth=.8)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set(xlabel='Best-3D candidate minus official TP: 3D IoU',
           ylabel='Best-3D candidate minus official TP: projected 2D IoU',
           title=f'{model_name} {class_name} validation: geometry of competing predictions')
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def write_summary(path, summary):
    class_name = summary.get('class_name', 'pedestrian')
    model_name = summary.get('model_name', 'PETR')
    c = summary['counts']
    matched = summary['official_matched_gt']
    compared = summary['matched_gt_with_different_best_3d']
    projected = summary['compared_gt_with_projected_camera']
    geometric = c.get('geometrically_better', 0)
    clean = c.get('geometrically_better_unmatched_alt', 0)
    visibility = summary['visibility_distribution']
    visibility_lines = [
        '\n## Visibility distribution\n\n'
        '| Group | V1 0-40% | V2 40-60% | V3 60-80% | V4 80-100% |\n'
        '|---|---:|---:|---:|---:|']
    for key, label in (
            ('official_unmatched_gt', 'Official unmatched GT'),
            ('different_best_3d_candidate', 'Different best-3D candidate'),
            ('geometrically_better', 'Geometrically better'),
            ('geometrically_better_unmatched_alternative',
             'Geometrically better, alternative unmatched')):
        values = visibility[key]
        visibility_lines.append('| {} | {} | {} | {} | {} |'.format(
            label, *[f"{values[str(v)]['count']:,} "
                     f"({values[str(v)]['percent']:.1f}%)" for v in range(1, 5)]))
    path.write_text(
        f'# {model_name} {class_name} hypothesis alignment, validation set\n\n'
        f"Matching threshold: {summary['threshold_m']:g} m. "
        f"GTs: {summary['gt_count']:,}; predictions: {summary['prediction_count']:,}.\n\n"
        '| Measure | Count | Denominator |\n|---|---:|---:|\n'
        f"| Officially matched GTs | {matched:,} | {summary['gt_count']:,} GTs |\n"
        f"| Unmatched GTs | {summary['official_unmatched_gt']:,} | {summary['gt_count']:,} GTs |\n"
        f"| Matched GTs with a different best-3D candidate | {compared:,} | {matched:,} matched GTs |\n"
        f"| Better 3D IoU, projected 2D IoU, and absolute radial error | {geometric:,} | {projected:,} comparable GTs |\n"
        f"| Same geometric pattern with alternative officially unmatched | {clean:,} | {projected:,} comparable GTs |\n"
        f"| Geometric cases whose alternative has lower confidence | {summary['geometric_case_confidence']['lower']:,} | {geometric:,} geometric cases |\n"
        f"| Best-3D candidate already TP for another GT | {c.get('best_3d_already_tp_other_gt', 0):,} | {compared:,} compared GTs |\n\n"
        '\nConfidence is reported descriptively and is not part of the geometric '
        'case definition.\n\n'
        'The geometric pattern is a GT-informed diagnosis. Projected 2D '
        'IoU compares image-clipped rectangles around projected 3D boxes in '
        'one camera selected from the GT projection alone. It does not measure '
        'visual feature correspondence. The alternative may depict another '
        f'{class_name}; inspect such cases before treating them as missed ranking '
        'opportunities.\n\n'
        'See `all_gt.csv` for every GT, `geometrically_better_cases.csv` for the '
        'geometric pattern, and `paired_iou_deltas.png` for the distribution.\n'
        + '\n'.join(visibility_lines) + '\n')


def gt_visibility(nusc, gt_by_sample, class_name='pedestrian'):
    """Resolve official filtered GT boxes to nuScenes visibility tokens."""
    annotations = {}
    wanted = set(gt_by_sample)
    instance_category = {
        instance['token']: nusc.get('category', instance['category_token'])['name']
        for instance in nusc.instance
    }
    allowed = CLASS_CATEGORIES[class_name]
    for ann in nusc.sample_annotation:
        if ann['sample_token'] in wanted and \
                instance_category[ann['instance_token']] in allowed:
            annotations.setdefault(ann['sample_token'], []).append(ann)
    result = {}
    for token, boxes in gt_by_sample.items():
        candidates = annotations.get(token, [])
        for gi, gt in enumerate(boxes):
            distances = [(sum((a-b)**2 for a, b in
                              zip(gt.translation, ann['translation'])), ann)
                         for ann in candidates]
            distance, ann = min(distances, default=(float('inf'), None),
                                key=lambda item: item[0])
            if ann is None or distance > 1e-8:
                raise ValueError('Cannot resolve GT visibility for '
                                 f'{token}:{gi}')
            result[token, gi] = int(ann['visibility_token'])
    return result


def write_visibility_csv(path, summary):
    rows = []
    for group, values in summary['visibility_distribution'].items():
        total = sum(item['count'] for item in values.values())
        for token, item in values.items():
            rows.append({'group': group, 'visibility_token': token,
                         'visibility_percent': item['range'],
                         'count': item['count'], 'group_total': total,
                         'percent_within_group': item['percent']})
    write_csv(path, rows, list(rows[0]))


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path,
                        default=base/'outputs/pedestrian_image_alignment_4m')
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--class-name', default='pedestrian',
                        choices=('car', 'truck', 'bus', 'trailer',
                                 'construction_vehicle', 'pedestrian',
                                 'motorcycle', 'bicycle', 'traffic_cone',
                                 'barrier'))
    parser.add_argument('--model-name', default='PETR')
    parser.add_argument('--max-gt', type=int, default=0,
                        help='Pilot only: stop after this many GTs; 0 means full validation')
    args = parser.parse_args()
    if args.threshold <= 0 or args.max_gt < 0:
        parser.error('threshold must be positive and max-gt nonnegative')
    if not args.results_json.is_file() or not (args.data_root/'v1.0-trainval').is_dir():
        parser.error('Prediction JSON or nuScenes metadata missing')
    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    gt_by_sample, predictions = load_filtered_class(
        args.results_json, args.data_root, args.class_name)
    visibility = gt_visibility(nusc, gt_by_sample, args.class_name)
    matches = greedy_match_records(gt_by_sample, predictions,
                                   range(len(predictions)), args.threshold)
    pred_tp_owner = {pi: key for key, (pi, _) in matches.items()}
    by_frame = defaultdict(list)
    for pi, pred in enumerate(predictions):
        by_frame[pred.sample_token].append(pi)
    rows = []
    limit_reached = False
    for token, gt_boxes in gt_by_sample.items():
        frame_indices = by_frame[token]
        records = camera_records(nusc, token) if gt_boxes else []
        for gi, gt in enumerate(gt_boxes):
            name, camera, gt_rect = reference_camera(records, gt)
            candidates = []
            for pi in frame_indices:
                pred = predictions[pi]
                if center_distance(pred, gt) < args.threshold:
                    candidates.append((pi, measurements(pred, gt, camera, gt_rect)))
            key = (token, gi)
            selected = next((pair for pair in candidates
                             if key in matches and pair[0] == matches[key][0]), None)
            if key in matches and selected is None:
                raise AssertionError('Official TP is absent from its candidate set')
            best_3d = max(candidates, key=lambda pair: (
                pair[1]['iou3d'], -pair[1]['center_m'], pair[1]['score'])) \
                if candidates else None
            best_2d = max(candidates, key=lambda pair: (
                pair[1]['projected_iou2d'], pair[1]['iou3d'],
                 -pair[1]['center_m'])) if camera and candidates else None
            alternatives = [pair for pair in candidates
                            if selected is not None and pair[0] != selected[0] and
                            preferred_alternative_better(pair[1], selected[1])]
            preferred_alt = max(alternatives, key=lambda pair: (
                pair[1]['projected_iou2d'], pair[1]['iou3d'],
                -pair[1]['center_m'],
                pair[1]['score'])) if alternatives else None
            rows.append(report_row(token, gi, selected, best_3d, best_2d,
                                   preferred_alt,
                                   predictions, name, gt_rect,
                                   len(candidates), pred_tp_owner, args.threshold,
                                   visibility[token, gi]))
            if args.max_gt and len(rows) >= args.max_gt:
                limit_reached = True
                break
        if limit_reached:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(args.output_dir/'all_gt.csv', rows, list(rows[0]))
        review = sorted((r for r in rows if r['geometrically_better']),
                        key=lambda r: (r['delta_iou3d']+r['delta_projected_iou2d']),
                        reverse=True)
        write_csv(args.output_dir/'geometrically_better_cases.csv', review,
                  list(rows[0]))
    summary = summarize(rows, len(predictions), args.threshold)
    summary['class_name'] = args.class_name
    summary['model_name'] = args.model_name
    summary['input_results_json'] = str(args.results_json.resolve())
    summary['data_root'] = str(args.data_root.resolve())
    summary['pilot_max_gt'] = args.max_gt
    summary['method'] = ('Official nuScenes filtering and score-ordered greedy '
                         f'{args.class_name} matching; all same-class predictions within '
                         'the selected ground-plane threshold are candidates. '
                         'Best 3D candidate maximizes yaw-box 3D IoU; ties use '
                         'center error then score. A single camera is selected '
                         'per GT by largest image-clipped GT projected area. '
                         'Projected 2D IoU uses axis-aligned rectangles enclosing '
                         'projected 3D corners in that fixed camera. Candidate '
                         'geometric preference requires higher 3D IoU, higher '
                         'projected 2D IoU, and smaller absolute prediction-center '
                         'displacement along the camera-to-GT ray. Confidence is '
                         'reported but is not a selection condition. Candidate '
                         'identity is geometric only; a candidate may be an '
                         'official TP for another GT. Image-feature correspondence '
                         'is not measured.')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_summary(args.output_dir/'summary.md', summary)
    write_visibility_csv(args.output_dir/'visibility_distribution.csv', summary)
    plot_pairs(args.output_dir/'paired_iou_deltas', rows, args.class_name,
               args.model_name)
    print('Wrote', args.output_dir/'all_gt.csv', flush=True)
    print('Wrote', args.output_dir/'geometrically_better_cases.csv', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
