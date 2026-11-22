#!/usr/bin/env python3
"""Run a qualitative SAM 2 pedestrian-mask pilot on selected camera frames.

Preparation needs only the standard library. Inference needs SAM 2, torch,
numpy, and Pillow in the environment where this script is run.
"""
import argparse
import csv
import html
import json
import math
import os
import re
import shutil
from pathlib import Path

import visualize as viz


def esc(value):
    return html.escape(str(value), quote=True)


def projected_box(box, camera, width, height):
    points = [viz.project(point, camera) for point in viz.box_corners_3d(box)]
    points = [point for point in points if point is not None]
    if not points:
        return None
    x1 = max(0., min(point[0] for point in points))
    y1 = max(0., min(point[1] for point in points))
    x2 = min(float(width - 1), max(point[0] for point in points))
    y2 = min(float(height - 1), max(point[1] for point in points))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def load_selected_groups(path, tokens):
    groups = {}
    with path.open() as handle:
        for line in handle:
            group = json.loads(line)
            if group['sample_token'] in tokens:
                groups[group['sample_token']] = group
                if len(groups) == len(tokens):
                    break
    missing = tokens - set(groups)
    if missing:
        raise ValueError('Missing selected groups: {}'.format(', '.join(sorted(missing))))
    return groups


def prepare(args):
    with args.manifest.open(newline='') as handle:
        manifest = list(csv.DictReader(handle))[:args.max_frames]
    if not manifest:
        raise ValueError('No frames found in {}'.format(args.manifest))
    tokens = {row['sample_token'] for row in manifest}
    groups = load_selected_groups(args.groups, tokens)
    records, poses, calibrations = viz.camera_metadata(
        args.data_root, tokens, args.manifest.parent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, prompts, projections = [], [], {}
    for row in manifest:
        token = row['sample_token']
        group = groups[token]
        camera, visible_count = viz.choose_camera(
            group, records[token], poses, calibrations)
        camera_name = Path(camera['record']['filename']).parent.name
        if camera_name != row['camera_name'] or visible_count != len(group['ground_truth']):
            raise ValueError('Selected camera changed for sample {}'.format(token))
        width, height = camera['record']['width'], camera['record']['height']
        image_path = args.data_root / camera['record']['filename']
        image_name = os.path.relpath(image_path, args.output_dir)
        frame = {'index': row['index'], 'sample_token': token,
                 'camera_name': camera_name, 'image': image_name,
                 'gt_svg': '../visualizations/' + row['camera_gt_svg'],
                 'width': width, 'height': height}
        frames.append(frame)
        projected = {'gt': [], 'prediction': []}
        for kind, boxes in (('gt', group['ground_truth']),
                            ('prediction', group['predictions'])):
            for item in boxes:
                center = viz.project(item['center_global'], camera)
                envelope = projected_box(item, camera, width, height)
                if center is None or envelope is None:
                    continue
                entry = {'index': item['gt_index'] if kind == 'gt' else item['prediction_index'],
                         'center_xy': center, 'box_xyxy': envelope}
                if kind == 'prediction':
                    entry.update({'score': item['score'],
                                  'local_score_rank': item['local_score_rank'],
                                  'official_gt_by_threshold': {
                                      threshold: outcome['official_prediction_to_gt'].get(
                                          str(item['prediction_index']))
                                      for threshold, outcome in group['thresholds'].items()}})
                projected[kind].append(entry)
        projections[token] = projected
        for gt in group['ground_truth']:
            point = viz.project(gt['center_global'], camera)
            box = projected_box(gt, camera, width, height)
            if point is None or box is None or not 0 <= point[0] < width or not 0 <= point[1] < height:
                continue
            prompts.append({'sample_token': token, 'frame_index': row['index'],
                            'gt_index': gt['gt_index'], 'camera_name': camera_name,
                            'auto_x': round(point[0], 2), 'auto_y': round(point[1], 2),
                            'box_x1': round(box[0], 2), 'box_y1': round(box[1], 2),
                            'box_x2': round(box[2], 2), 'box_y2': round(box[3], 2),
                            'manual_x': '', 'manual_y': ''})
    fields = list(prompts[0])
    with (args.output_dir / 'prompt_suggestions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prompts)
    manual_path = args.output_dir / 'manual_prompts.csv'
    if not manual_path.exists():
        shutil.copyfile(args.output_dir / 'prompt_suggestions.csv', manual_path)
    people_path = args.output_dir / 'manual_people.csv'
    if not people_path.exists():
        with people_path.open('w', newline='') as handle:
            csv.writer(handle).writerow(('sample_token', 'person_id', 'x', 'y', 'notes'))
    (args.output_dir / 'frames.json').write_text(json.dumps(frames, indent=2) + '\n')
    (args.output_dir / 'box_projections.json').write_text(
        json.dumps(projections, indent=2) + '\n')
    cards = []
    for frame in frames:
        cards.append('<section><h2>Frame {} · {}</h2><p>{} · Click image to read pixel coordinates: <output>—</output></p><img class="clickable" width="{}" height="{}" src="{}"><p><a href="{}">GT wireframe</a></p></section>'.format(
            esc(frame['index']), esc(frame['sample_token']), esc(frame['camera_name']),
            frame['width'], frame['height'], esc(frame['image']), esc(frame['gt_svg'])))
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>SAM prompt preview</title>
<style>body{font:16px Arial,sans-serif;max-width:1600px;margin:24px auto}section{border-top:1px solid #ccc;padding:16px 0}img{max-width:100%;height:auto;cursor:crosshair}output{font-weight:bold}</style></head>
<body><h1>SAM prompt preview</h1><p>Click each visible pedestrian's torso to get pixel coordinates. Add one row per person to manual_people.csv using an independent person_id such as A or B. Automatic GT-center prompts are in prompt_suggestions.csv and can be tested separately; they should not define person identity.</p>'''
    page += '\n'.join(cards)
    page += '''<script>document.querySelectorAll('img.clickable').forEach(img => img.addEventListener('click', e => {const r=img.getBoundingClientRect();const x=Math.round((e.clientX-r.left)*img.naturalWidth/r.width);const y=Math.round((e.clientY-r.top)*img.naturalHeight/r.height);img.closest('section').querySelector('output').textContent=x+', '+y;}));</script></body></html>'''
    (args.output_dir / 'prepare.html').write_text(page)
    print('Prepared {} frames and {} GT prompts in {}'.format(len(frames), len(prompts), args.output_dir))
    return frames, prompts


def manual_points(path):
    result = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['manual_x'] and row['manual_y']:
                result[row['sample_token'], int(row['gt_index'])] = (
                    float(row['manual_x']), float(row['manual_y']))
    return result


def manual_people(path, frames):
    sizes = {frame['sample_token']: (frame['width'], frame['height']) for frame in frames}
    people = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if not row['person_id'] or not row['x'] or not row['y']:
                continue
            token = row['sample_token']
            if token not in sizes:
                raise ValueError('Manual person references an unselected sample: ' + token)
            person_id = row['person_id']
            if not re.fullmatch(r'[A-Za-z0-9_]+', person_id):
                raise ValueError('person_id must contain only letters, digits, or underscore')
            point = (float(row['x']), float(row['y']))
            if not 0 <= point[0] < sizes[token][0] or not 0 <= point[1] < sizes[token][1]:
                raise ValueError('Manual point outside image for ' + token)
            key = (token, person_id)
            if key in people:
                raise ValueError('Duplicate manual person {}'.format(key))
            people[key] = point
    return people


def pixel_inside(mask, point):
    x, y = (int(round(value)) for value in point)
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def save_overlay(image, mask, positive, negatives, box, output):
    from PIL import Image, ImageDraw
    import numpy as np
    base = image.convert('RGBA')
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask] = (20, 180, 80, 115)
    base.alpha_composite(Image.fromarray(rgba, 'RGBA'))
    draw = ImageDraw.Draw(base)
    if box is not None:
        draw.rectangle(box, outline=(255, 190, 0, 255), width=3)
    x, y = positive
    draw.ellipse((x-6, y-6, x+6, y+6), fill=(255, 255, 255, 255),
                 outline=(0, 0, 0, 255), width=2)
    for x, y in negatives:
        draw.line((x-5, y-5, x+5, y+5), fill=(220, 30, 40, 255), width=3)
        draw.line((x-5, y+5, x+5, y-5), fill=(220, 30, 40, 255), width=3)
    base.convert('RGB').save(output)


def run_sam(args, frames, prompts):
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not args.sam_checkpoint or not args.sam_checkpoint.is_file():
        raise ValueError('Provide an existing --sam-checkpoint')
    if not args.sam_config:
        raise ValueError('Provide --sam-config from the installed SAM 2 package')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SAM2ImagePredictor(build_sam2(
        args.sam_config, str(args.sam_checkpoint), device=device))
    manually_selected = manual_points(args.output_dir / 'manual_prompts.csv')
    people = manual_people(args.output_dir / 'manual_people.csv', frames)
    projections = json.loads((args.output_dir / 'box_projections.json').read_text())
    by_frame = {}
    for row in prompts:
        by_frame.setdefault(row['sample_token'], []).append(row)
    results, person_results, correspondence = [], [], []
    with torch.inference_mode():
        for frame in frames:
            token = frame['sample_token']
            image = Image.open(args.output_dir / frame['image']).convert('RGB')
            predictor.set_image(np.asarray(image))
            frame_prompts = by_frame[token]
            for row in frame_prompts:
                gt_index = int(row['gt_index'])
                auto = (float(row['auto_x']), float(row['auto_y']))
                box = [float(row[key]) for key in ('box_x1','box_y1','box_x2','box_y2')]
                manual = manually_selected.get((token, gt_index))
                for mode, positive, use_box in (
                    ('point', auto, False), ('point_box', auto, True),
                    *((('manual_point', manual, False),) if manual else ())):
                    negatives = [manually_selected.get((token, int(other['gt_index'])),
                                 (float(other['auto_x']), float(other['auto_y'])))
                                 for other in frame_prompts if other['gt_index'] != row['gt_index']]
                    points = np.array([positive] + negatives, dtype=np.float32)
                    labels = np.array([1] + [0]*len(negatives), dtype=np.int32)
                    masks, scores, _ = predictor.predict(
                        point_coords=points, point_labels=labels,
                        box=np.array(box, dtype=np.float32) if use_box else None,
                        multimask_output=True, normalize_coords=False)
                    viable = [i for i, mask in enumerate(masks)
                              if pixel_inside(mask, positive) and
                              not any(pixel_inside(mask, point) for point in negatives)]
                    pool = viable or list(range(len(scores)))
                    best = max(pool, key=lambda i: float(scores[i]))
                    mask = masks[best].astype(bool)
                    stem = '{}_G{}_{}'.format(frame['index'], gt_index, mode)
                    mask_file, overlay_file = stem + '_mask.png', stem + '_overlay.png'
                    Image.fromarray((mask*255).astype(np.uint8), 'L').save(
                        args.output_dir / mask_file)
                    save_overlay(image, mask, positive, negatives, box,
                                 args.output_dir / overlay_file)
                    ys, xs = np.nonzero(mask)
                    result = {'sample_token': token, 'frame_index': frame['index'],
                              'gt_index': gt_index, 'camera_name': frame['camera_name'],
                              'prompt_mode': mode, 'prompt_x': positive[0],
                              'prompt_y': positive[1], 'sam_predicted_iou': float(scores[best]),
                              'mask_area_px': int(mask.sum()),
                              'mask_bbox_xyxy': json.dumps([int(xs.min()), int(ys.min()),
                                                             int(xs.max()), int(ys.max())]) if len(xs) else '',
                              'positive_in_mask': int(pixel_inside(mask, positive)),
                              'neighbor_points_in_mask': sum(pixel_inside(mask, pt) for pt in negatives),
                              'mask_file': mask_file, 'overlay_file': overlay_file,
                              'manual_review': ''}
                    results.append(result)
            frame_people = [(person_id, point) for (sample, person_id), point in people.items()
                            if sample == token]
            for person_id, positive in frame_people:
                negatives = [point for other_id, point in frame_people if other_id != person_id]
                points = np.array([positive] + negatives, dtype=np.float32)
                labels = np.array([1] + [0]*len(negatives), dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=points, point_labels=labels, multimask_output=True,
                    normalize_coords=False)
                viable = [i for i, mask in enumerate(masks)
                          if pixel_inside(mask, positive) and
                          not any(pixel_inside(mask, point) for point in negatives)]
                best = max(viable or range(len(scores)), key=lambda i: float(scores[i]))
                mask = masks[best].astype(bool)
                stem = '{}_person_{}'.format(frame['index'], person_id)
                mask_file, overlay_file = stem + '_mask.png', stem + '_overlay.png'
                Image.fromarray((mask*255).astype(np.uint8), 'L').save(
                    args.output_dir / mask_file)
                save_overlay(image, mask, positive, negatives, None,
                             args.output_dir / overlay_file)
                person_results.append({'sample_token': token, 'frame_index': frame['index'],
                                       'person_id': person_id, 'prompt_x': positive[0],
                                       'prompt_y': positive[1],
                                       'sam_predicted_iou': float(scores[best]),
                                       'mask_area_px': int(mask.sum()),
                                       'other_person_points_in_mask': sum(
                                           pixel_inside(mask, pt) for pt in negatives),
                                       'mask_file': mask_file, 'overlay_file': overlay_file,
                                       'manual_review': ''})
                for kind in ('gt', 'prediction'):
                    for item in projections[token][kind]:
                        x1, y1, x2, y2 = item['box_xyxy']
                        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
                        x2 = min(mask.shape[1], math.ceil(x2) + 1)
                        y2 = min(mask.shape[0], math.ceil(y2) + 1)
                        overlap = int(mask[y1:y2, x1:x2].sum())
                        correspondence.append({
                            'sample_token': token, 'person_id': person_id,
                            'box_kind': kind, 'box_index': item['index'],
                            'projected_center_in_mask': int(pixel_inside(
                                mask, item['center_xy'])),
                            'mask_fraction_inside_box': round(overlap / max(1, int(mask.sum())), 4),
                            'box_rectangle_fraction_covered': round(
                                overlap / max(1, (x2-x1)*(y2-y1)), 4),
                            'prediction_score': item.get('score', ''),
                            'local_score_rank': item.get('local_score_rank', ''),
                            'official_gt_by_threshold': json.dumps(
                                item.get('official_gt_by_threshold', {})),
                            'sam_mask_file': mask_file})
            print('Segmented frame {} ({})'.format(frame['index'], frame['camera_name']), flush=True)
    with (args.output_dir / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    if person_results:
        for filename, rows in (('person_results.csv', person_results),
                               ('correspondence.csv', correspondence)):
            with (args.output_dir / filename).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    cards = []
    for frame in frames:
        subset = [row for row in results if row['sample_token'] == frame['sample_token']]
        independent = [row for row in person_results if row['sample_token'] == frame['sample_token']]
        person_items = ''.join('<div><h3>Visible person {} · SAM score {:.3f}</h3><a href="{}"><img src="{}"></a></div>'.format(
            esc(row['person_id']), row['sam_predicted_iou'], esc(row['overlay_file']),
            esc(row['overlay_file'])) for row in independent)
        items = ''.join('<div><h3>GT {} · {} · SAM score {:.3f}</h3><p>Mask area {} px · neighbor prompts inside mask: {}</p><a href="{}"><img src="{}"></a></div>'.format(
            row['gt_index'], esc(row['prompt_mode']), row['sam_predicted_iou'],
            row['mask_area_px'], row['neighbor_points_in_mask'],
            esc(row['overlay_file']), esc(row['overlay_file'])) for row in subset)
        cards.append('<section><h2>Frame {} · {}</h2><p>{} · <a href="{}">raw image</a> · <a href="{}">GT wireframes</a></p><h3>Independently clicked people</h3><div class="grid">{}</div><h3>GT-derived prompts (exploratory)</h3><div class="grid">{}</div></section>'.format(
            frame['index'], esc(frame['sample_token']), esc(frame['camera_name']),
            esc(frame['image']), esc(frame['gt_svg']), person_items, items))
    page = '<!doctype html><html><head><meta charset="utf-8"><title>SAM pedestrian pilot</title><style>body{{font:16px Arial,sans-serif;max-width:1500px;margin:24px auto}}section{{border-top:1px solid #ccc;margin:24px 0}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}img{{width:100%}}p{{color:#475569}}</style></head><body><h1>SAM 2 pedestrian pilot</h1><p>Green areas are SAM masks. White dots are positive prompts; red crosses are neighboring pedestrian negative prompts; yellow rectangles are projected nuScenes 3D boxes. Independently clicked person masks are the main experiment; compare each with every GT and prediction in correspondence.csv. SAM predicted IoU is a model quality estimate, not measured accuracy. Inspect masks and record success or failure manually.</p>{}</body></html>'.format('\n'.join(cards))
    (args.output_dir / 'index.html').write_text(page)
    print('Wrote {} masks to {}'.format(len(results), args.output_dir))


def main():
    base = Path(__file__).parent
    root = base.resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=base / 'outputs/visualizations/manifest.csv')
    parser.add_argument('--groups', type=Path, default=base / 'outputs/groups.jsonl')
    parser.add_argument('--data-root', type=Path,
                        default=root.parent / 'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs/sam_pilot')
    parser.add_argument('--max-frames', type=int, default=10)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--sam-config', help='SAM 2 Hydra config, e.g. configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--sam-checkpoint', type=Path)
    parser.add_argument('--device', choices=('cuda','cpu','mps'))
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error('max-frames must be positive')
    frames, prompts = prepare(args)
    if not args.prepare_only:
        run_sam(args, frames, prompts)


if __name__ == '__main__':
    main()
