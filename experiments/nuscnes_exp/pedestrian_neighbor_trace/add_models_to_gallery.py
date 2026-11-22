#!/usr/bin/env python3
"""Add independently evaluated models to the fixed PETR gallery frames.

Run after visualize.py. Repeat --model NAME=RESULTS_JSON for additional models.
The script preserves the gallery's frame selection, GTs, cameras, and controls.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

from confidence_quality_correlation import load_filtered_boxes
from reranking_oracle import (distance_oracle_order, greedy_match_records,
                              iou3d_oracle_order, oracle_order)
from visualize import (TP_THRESHOLDS, camera_metadata,
                       render, render_assignment_table, render_camera,
                       render_comparison_table, threshold_tag,
                       visible_context_gt_indices)


def model_slug(name):
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    if not slug or slug == 'petr':
        raise ValueError('Model name must be distinct from PETR: ' + name)
    return slug


def parse_model(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Use NAME=RESULTS_JSON')
    name, path = value.split('=', 1)
    try:
        slug = model_slug(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    return {'name': name, 'slug': slug, 'results_json': str(Path(path).expanduser().resolve())}


def build_group(row, reference_gt, full_gt, full_pred, radius, nearby_threshold):
    """Match the model on the full frame, then display boxes near fixed GTs."""
    token, class_name = row['sample_token'], row['class']
    display_gt = [int(gt['gt_index']) for gt in reference_gt]
    for gt in reference_gt:
        source = full_gt[int(gt['gt_index'])]
        if max(abs(a - b) for a, b in zip(
                source.translation, gt['center_global'])) > 1e-4:
            raise ValueError('Filtered GT order differs for {} {}'.format(token, class_name))
    full_pred = [full_pred[i] for i in sorted(
        range(len(full_pred)),
        key=lambda i: (full_pred[i].detection_score, i), reverse=True)]
    nearby = [pi for pi, pred in enumerate(full_pred)
              if min(math.hypot(pred.translation[0] - full_gt[gi].translation[0],
                                pred.translation[1] - full_gt[gi].translation[1])
                     for gi in display_gt) <= radius]
    group = {
        'sample_token': token, 'class': class_name,
        'crowding': row['crowding'],
        'ground_truth': reference_gt,
        'predictions': [{
            'prediction_index': pi,
            'center_global': list(full_pred[pi].translation),
            'size_wlh': list(full_pred[pi].size),
            'rotation_wxyz': list(full_pred[pi].rotation),
            'score': full_pred[pi].detection_score,
            'local_score_rank': rank,
            'frame_score_rank': pi + 1}
            for rank, pi in enumerate(nearby, 1)],
        'pair_matrix': [{
            'prediction_index': pi, 'gt_index': gi,
            'center_distance_m': math.hypot(
                full_pred[pi].translation[0] - full_gt[gi].translation[0],
                full_pred[pi].translation[1] - full_gt[gi].translation[1])}
            for pi in nearby for gi in display_gt],
        'thresholds': {},
    }
    for mode in ('oracle', 'distance', 'iou3d'):
        group[mode + '_assignments'] = {}
    gt_frame = {token: full_gt}
    for threshold in TP_THRESHOLDS:
        orders = {'baseline': list(range(len(full_pred))),
                  'oracle': oracle_order(gt_frame, full_pred, threshold)[0],
                  'distance': distance_oracle_order(gt_frame, full_pred,
                                                    threshold)[0],
                  'iou3d': iou3d_oracle_order(gt_frame, full_pred,
                                              threshold)[0]}
        for mode, order in orders.items():
            records = greedy_match_records(gt_frame, full_pred, order, threshold)
            owner = {pi: gi for (_, gi), (pi, _) in records.items()}
            if mode == 'baseline':
                group['thresholds'][str(threshold)] = {
                    'official_prediction_to_gt': {
                        str(pi): (owner[pi] if owner[pi] in display_gt else -1)
                        if pi in owner else None for pi in nearby}}
            else:
                rank = {pi: position for position, pi in enumerate(order, 1)}
                local_rank = {pi: position for position, pi in enumerate(
                    sorted(nearby, key=lambda pi: rank[pi]), 1)}
                group[mode + '_assignments'][str(threshold)] = {
                    str(pi): {
                        'oracle_local_rank': local_rank[pi],
                        'oracle_frame_rank': rank[pi],
                        'assigned_group_gt': owner.get(pi)
                        if owner.get(pi) in display_gt else None,
                        'tp_outside_group': pi in owner and owner[pi] not in display_gt}
                    for pi in nearby}
    diagnostics = []
    for gi in display_gt:
        distances = [(math.hypot(
            full_pred[pi].translation[0] - full_gt[gi].translation[0],
            full_pred[pi].translation[1] - full_gt[gi].translation[1]), pi)
            for pi in nearby]
        nearest_distance, nearest_pi = min(distances, default=(None, None))
        diagnostics.append({
            'gt_index': gi,
            'nearest_prediction_id': '' if nearest_pi is None else 'P:{}'.format(nearest_pi),
            'nearest_prediction_distance_m': '' if nearest_pi is None else nearest_distance,
            'predictions_within_threshold': sum(
                distance < nearby_threshold for distance, _ in distances)})
    return group, diagnostics


def add_model_to_html(base_html, models, groups_by_model, manifest):
    """Add a model selector and model-specific tables to the existing UI."""
    if 'data-model="petr"' in base_html:
        raise ValueError('Base gallery already contains model controls')
    buttons = ''.join('<button type="button" data-model="{}" aria-pressed="{}">{}</button>'.format(
        slug, 'true' if slug == 'petr' else 'false', escape(name))
        for slug, name in [('petr', 'PETR')] + [
            (model['slug'], model['name']) for model in models])
    html = base_html.replace('<div class="controls">',
                             '<div class="controls">Model: {}<br>'.format(buttons), 1)
    by_token = {row['sample_token']: row for row in manifest}
    article_pattern = re.compile(r'(<article\b[^>]*>)(.*?)(</article>)', re.S)

    def article(match):
        start, body, end = match.groups()
        token_match = re.search(r'\b([0-9a-f]{32})\b', body)
        if not token_match or token_match.group(1) not in by_token:
            raise ValueError('Cannot map gallery article to manifest row')
        token = token_match.group(1)
        # The original four tables and comparison table remain PETR-specific.
        table_start = body.find('<div class="assignment-table baseline-table">')
        if table_start < 0:
            raise ValueError('PETR assignment tables missing for ' + token)
        body = body[:table_start] + '<div class="model-tables" data-model="petr">' + \
            body[table_start:] + '</div>'
        for model in models:
            slug, name = model['slug'], model['name']
            group = groups_by_model[slug][token]
            tables = []
            for mode, title in (('baseline', 'Baseline'),
                                ('oracle', 'Cardinality re-ranked'),
                                ('distance', 'Distance re-ranked'),
                                ('iou3d', '3D IoU re-ranked')):
                table = render_assignment_table(group, mode)
                table = table.replace('Original PETR score',
                                      'Original {} score'.format(escape(name)))
                table = table.replace('<th>PETR frame rank</th>',
                                      '<th>{} frame rank</th>'.format(escape(name)))
                tables.append('<div class="assignment-table {}-table"><h3>{} assignments '
                              'at 0.5, 1, 2 and 4 m</h3>{}</div>'.format(
                                  mode, title, table))
            tables.append(render_comparison_table(group))
            body += '<div class="model-tables" data-model="{}" hidden>{}</div>'.format(
                slug, ''.join(tables))
        old_count = by_token[token]['prediction_count'] + ' nearby predictions'
        counts = '<span class="model-count" data-model="petr">{}</span>'.format(old_count)
        counts += ''.join('<span class="model-count" data-model="{}" hidden>{} nearby predictions</span>'.format(
            model['slug'], len(groups_by_model[model['slug']][token]['predictions']))
            for model in models)
        if old_count not in body:
            raise ValueError('Prediction count missing for ' + token)
        body = body.replace(old_count, counts, 1)
        return start + body + end

    html, count = article_pattern.subn(article, html)
    if count != len(manifest):
        raise ValueError('Expected {} articles; found {}'.format(len(manifest), count))
    html = html.replace("let selectedThreshold=", "let selectedModel='petr';\nlet selectedThreshold=", 1)
    pattern = re.compile(r'function variant\(base\)\{[^\n]+\}')
    replacement = ("function variant(base){const model=(selectedModel==='petr'?'':'_'+selectedModel);"
                   "const suffix=model+(selectedMode==='baseline'?'':'_'+selectedMode)+"
                   "(selectedThreshold==='4m'?'':'_tp'+selectedThreshold);"
                   "return base.replace(/\\.svg$/,suffix+'.svg')}")
    html, count = pattern.subn(lambda _: replacement, html, count=1)
    if count != 1:
        raise ValueError('Gallery variant function missing')
    needle = "\n}\ndocument.querySelectorAll('.controls button[data-class]')"
    update = ("\n document.querySelectorAll('.model-tables,.model-count').forEach(item=>"
              "item.hidden=item.dataset.model!==selectedModel);"
              "\n document.querySelectorAll('.controls button[data-model]').forEach(button=>"
              "button.setAttribute('aria-pressed',button.dataset.model===selectedModel?'true':'false'));"
              "\n}\ndocument.querySelectorAll('.controls button[data-class]')")
    if needle not in html:
        raise ValueError('Gallery refresh function has changed')
    html = html.replace(needle, update, 1)
    needle = 'refresh();\n</script>'
    listener = ("document.querySelectorAll('.controls button[data-model]').forEach(button=>"
                "button.addEventListener('click',()=>{selectedModel=button.dataset.model;refresh()}));"
                "refresh();\n</script>")
    if needle not in html:
        raise ValueError('Gallery script ending has changed')
    html = html.replace(needle, listener, 1)
    note = ('<p>Model switch uses the same selected frames, camera, and GTs. '
            'Each model has its own filtered predictions and full-frame matching. '
            'Case selection and early-FP counts refer to PETR only.</p>')
    html = html.replace('<div class="controls">', note + '<div class="controls">', 1)
    return html


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    default_stream = (root.parent / 'StreamPETR/test/'
                      'stream_petr_r50_noflash_428q_nui_60e/'
                      'Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery-dir', type=Path,
                        default=base / 'outputs/visualizations')
    parser.add_argument('--data-root', type=Path,
                        default=root.parent / 'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--model', type=parse_model, action='append',
                        help='NAME=nuScenes-results-json; repeat for ViT models')
    parser.add_argument('--prediction-radius', type=float, default=8.0)
    parser.add_argument('--max-ego-distance', type=float, default=20.0,
                        help='Same displayed GT range used for the existing gallery')
    args = parser.parse_args()
    if args.model is None:
        args.model = [parse_model('StreamPETR={}'.format(default_stream))]
    if len({model['slug'] for model in args.model}) != len(args.model):
        parser.error('Model names must be unique')
    for model in args.model:
        if not Path(model['results_json']).is_file():
            parser.error('Model results missing: ' + model['results_json'])
    if args.prediction_radius < max(TP_THRESHOLDS) or args.max_ego_distance <= 0:
        parser.error('prediction-radius must be at least 4 m and max-ego-distance positive')
    gallery = args.gallery_dir
    manifest_path, index_path = gallery / 'manifest.csv', gallery / 'index.html'
    if not manifest_path.is_file() or not index_path.is_file():
        parser.error('Run visualize.py first; gallery manifest or index is missing')
    manifest = list(csv.DictReader(manifest_path.open(newline='')))
    if not manifest:
        parser.error('Gallery manifest is empty')
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    base_path, metadata_path = gallery / 'index_petr_only.html', gallery / 'comparison_models.json'
    existing_meta = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if existing_meta and existing_meta.get('manifest_sha256') == manifest_hash:
        current_html = index_path.read_text()
        if 'data-model="petr"' in current_html:
            base_html = base_path.read_text()
        else:
            # visualize.py has regenerated PETR assets for these same frames.
            base_html = current_html
            base_path.write_text(base_html)
        # Preserve previously added models when a new model is supplied.
        all_models = {m['slug']: m for m in existing_meta['models']}
        all_models.update({m['slug']: m for m in args.model})
        models = list(all_models.values())
    else:
        base_html = index_path.read_text()
        if 'data-model="petr"' in base_html:
            parser.error('Gallery changed after model augmentation; regenerate with visualize.py')
        base_path.write_text(base_html)
        models = args.model

    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    tokens = {row['sample_token'] for row in manifest}
    camera_records, poses, calibrations = camera_metadata(args.data_root, tokens, gallery)
    ego = json.loads((gallery / 'ego_positions.json').read_text())
    groups_by_model = {}
    for model in models:
        print('Loading {} predictions from {}'.format(model['name'], model['results_json']),
              flush=True)
        pred_boxes, gt_boxes, _ = load_filtered_boxes(
            Path(model['results_json']), args.data_root)
        if not tokens <= set(pred_boxes.sample_tokens):
            raise ValueError('{} results lack {} gallery frames'.format(
                model['name'], len(tokens - set(pred_boxes.sample_tokens))))
        group_map = {}
        for row in manifest:
            token, class_name = row['sample_token'], row['class']
            camera_rows = [item for item in camera_records[token]
                           if Path(item['filename']).parent.name == row['camera_name']]
            if len(camera_rows) != 1:
                raise ValueError('Gallery camera not unique for ' + token)
            camera_row = camera_rows[0]
            camera = {'record': camera_row,
                      'pose': poses[camera_row['ego_pose_token']],
                      'calibration': calibrations[camera_row['calibrated_sensor_token']]}
            full_gt = [box for box in gt_boxes[token]
                       if box.detection_name == class_name]
            visible_gt = visible_context_gt_indices(
                nusc, token, full_gt, camera, ego[token]['center_global'],
                args.max_ego_distance)
            if len(visible_gt) != int(row['gt_count']):
                raise ValueError('Displayed GT count differs for {}: {} vs {}'.format(
                    token, len(visible_gt), row['gt_count']))
            reference_gt = [{
                'gt_index': gi, 'center_global': list(full_gt[gi].translation),
                'size_wlh': list(full_gt[gi].size),
                'rotation_wxyz': list(full_gt[gi].rotation)}
                for gi in visible_gt]
            full_pred = [box for box in pred_boxes[token]
                         if box.detection_name == class_name]
            group, diagnostics = build_group(
                row, reference_gt, full_gt, full_pred,
                args.prediction_radius, float(row['threshold_m']))
            group['model_name'] = model['name']
            group_map[token] = group
            stem = '{:02d}_{}'.format(int(row['index']), token)
            for threshold in TP_THRESHOLDS:
                for mode in ('baseline', 'oracle', 'distance', 'iou3d'):
                    suffix = '_' + model['slug'] + (
                        '_' + mode if mode != 'baseline' else '') + (
                        '' if threshold == 4.0 else '_tp' + threshold_tag(threshold))
                    (gallery / (stem + suffix + '.svg')).write_text(render(
                        group, diagnostics, float(row['threshold_m']),
                        ego[token], camera, tp_threshold=threshold, mode=mode))
                    (gallery / (stem + '_camera_pred' + suffix + '.svg')).write_text(
                        render_camera(group, camera, include_gt=False,
                                      tp_threshold=threshold, mode=mode))
        groups_by_model[model['slug']] = group_map
        print('Rendered {} frames for {}'.format(len(group_map), model['name']), flush=True)
    html = add_model_to_html(base_html, models, groups_by_model, manifest)
    temp = gallery / 'index.html.tmp'
    temp.write_text(html)
    temp.replace(index_path)
    metadata_path.write_text(json.dumps({
        'manifest_sha256': manifest_hash, 'models': models,
        'data_root': str(args.data_root.resolve()),
        'prediction_radius_m': args.prediction_radius,
        'note': 'Frames selected by PETR; every model matched independently on the full frame.'
    }, indent=2) + '\n')
    print('Updated {} with {} models'.format(index_path, len(models) + 1), flush=True)


if __name__ == '__main__':
    main()
