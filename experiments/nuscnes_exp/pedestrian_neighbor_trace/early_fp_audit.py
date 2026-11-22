#!/usr/bin/env python3
"""Validation-wide pedestrian FP audit with a fixed camera-review queue.

Automatic labels describe geometry only. Physical correspondence requires a
reviewer to fill review_labels.csv; unreviewed cases remain unreviewed.
"""

import argparse
import bisect
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from localization_capacity import load_filtered_pedestrians
from reranking_oracle import greedy_match_records, official_ap
from visualize import box_corners_3d, project

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
LABELS = ('genuine_duplicate', 'association_ambiguity',
          'likely_localization_error', 'background_spurious', 'uncertain')


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_kind(near, threshold):
    if any(x['distance_m'] < threshold and x['claimant_id'] is not None for x in near):
        return 'claimed_gt_within_threshold'
    if near and near[0]['distance_m'] >= threshold:
        return 'no_gt_within_threshold'
    return 'other_unmatched'


def audit(gt, predictions, threshold):
    order = list(range(len(predictions)))
    matches = greedy_match_records(gt, predictions, order, threshold)
    by_pred = {pi: (token, gi, d) for (token, gi), (pi, d) in matches.items()}
    last_tp = max(by_pred) if by_pred else -1
    tp_ranks = sorted(by_pred)
    rows, relations = [], []
    for pi, pred in enumerate(predictions):
        if pi in by_pred:
            continue
        token = pred.sample_token
        near = []
        for gi, target in enumerate(gt[token]):
            d = math.hypot(pred.translation[0]-target.translation[0],
                           pred.translation[1]-target.translation[1])
            claim = matches.get((token, gi))
            near.append(dict(gt_id=f'{token}:{gi}', distance_m=d,
                             claimant_id=claim[0] if claim else None,
                             claimant_distance_m=claim[1] if claim else None))
        near.sort(key=lambda x: x['distance_m'])
        kind = candidate_kind(near, threshold)
        claimed_near = [x for x in near if x['distance_m'] < threshold
                        and x['claimant_id'] is not None]
        row = dict(prediction_id=pi, sample_token=token, rank=pi+1,
                   score=pred.detection_score, before_last_tp=int(pi < last_tp),
                   later_tp_count=len(tp_ranks)-bisect.bisect_right(tp_ranks, pi),
                   candidate_kind=kind, nearest_gt_id=near[0]['gt_id'] if near else '',
                   nearest_gt_distance_m=near[0]['distance_m'] if near else '',
                   gt_within_threshold=sum(x['distance_m'] < threshold for x in near),
                   claimed_gt_within_threshold=len(claimed_near),
                   better_localized_than_claimant=int(any(
                       x['distance_m'] < x['claimant_distance_m'] for x in claimed_near)),
                   pred_box_json=json.dumps(pred.serialize(), separators=(',', ':')))
        rows.append(row)
        if pi < last_tp:
            for item in near:
                relations.append(dict(prediction_id=pi, **item))
    return rows, relations, matches, last_tp


def review_queue(rows, per_kind):
    """Fixed high/mid/low rank selection within each geometry stratum."""
    groups = defaultdict(list)
    for row in rows:
        if row['before_last_tp']:
            groups[row['candidate_kind']].append(row)
    selected = []
    for kind, group in sorted(groups.items()):
        group.sort(key=lambda r: r['rank'])
        n = min(per_kind, len(group))
        for j in range(n):
            selected.append(group[round(j*(len(group)-1)/max(n-1, 1))])
    return sorted({r['prediction_id']: r for r in selected}.values(),
                  key=lambda r: r['rank'])


def box_dict(box):
    return dict(size_wlh=box.size, center_global=box.translation,
                rotation_wxyz=box.rotation)


def camera_cards(nusc, data_root, html_dir, pred, nearby, claimants):
    sample = nusc.get('sample', pred.sample_token)
    boxes = [('FP', pred, '#ef4444')] + [
        (f'GT {gi}', gtbox, '#38bdf8') for gi, gtbox in nearby]
    boxes += [(f'claim {pi}', box, '#facc15') for pi, box in claimants]
    cards = []
    for channel in CAMERAS:
        sd = nusc.get('sample_data', sample['data'][channel])
        calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        camera = dict(pose=pose, calibration=calibration)
        shapes = []
        for label, box, color in boxes:
            pts = [project(p, camera) for p in box_corners_3d(box_dict(box))]
            if not all(p is not None for p in pts):
                continue
            center = project(box.translation, camera)
            if center is None or not (0 <= center[0] < sd['width'] and
                                      0 <= center[1] < sd['height']):
                continue
            for a, b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                         (0,4),(1,5),(2,6),(3,7)):
                shapes.append(f'<line x1="{pts[a][0]:.1f}" y1="{pts[a][1]:.1f}" '
                              f'x2="{pts[b][0]:.1f}" y2="{pts[b][1]:.1f}" '
                              f'stroke="{color}" stroke-width="3"/>')
            shapes.append(f'<text x="{center[0]:.1f}" y="{center[1]:.1f}" '
                          f'fill="{color}" font-size="28">{html.escape(label)}</text>')
        if shapes:
            image = quote(os.path.relpath(Path(data_root) / sd['filename'],
                                          start=html_dir), safe='/')
            cards.append(f'<div><h4>{channel}</h4><svg viewBox="0 0 {sd["width"]} '
                         f'{sd["height"]}"><image href="{html.escape(image)}" '
                         f'width="100%" height="100%"/>{"".join(shapes)}</svg></div>')
    return ''.join(cards) or '<p>No box center projects inside a camera.</p>'


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, default=root /
                        'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root', type=Path, default=root.parent /
                        'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=base/'outputs/early_fp_audit')
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--review-per-kind', type=int, default=30)
    parser.add_argument('--no-images', action='store_true')
    args = parser.parse_args()
    if args.threshold <= 0 or args.review_per_kind < 1:
        parser.error('threshold and review-per-kind must be positive')
    if not args.results_json.is_file() or not (args.data_root/'v1.0-trainval').is_dir():
        parser.error('Prediction JSON or nuScenes metadata missing')
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    gt, predictions = load_filtered_pedestrians(args.results_json, args.data_root)
    cfg = config_factory('detection_cvpr_2019')
    rows, relations, matches, last_tp = audit(gt, predictions, args.threshold)
    out = args.output_dir/str(args.threshold).replace('.', 'p')
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'fps.csv', rows, list(rows[0]))
    write_csv(out/'fp_gt_relations.csv', relations,
              ['prediction_id','gt_id','distance_m','claimant_id','claimant_distance_m'])
    queue = review_queue(rows, args.review_per_kind)
    write_csv(out/'review_queue.csv', queue, list(rows[0]))
    labels_path = out/'review_labels.csv'
    if not labels_path.exists():
        write_csv(labels_path,
                  [dict(prediction_id=r['prediction_id'], label='',
                        corresponding_gt_id='', image_evidence='', notes='')
                   for r in rows if r['before_last_tp']],
                  ['prediction_id','label','corresponding_gt_id','image_evidence','notes'])
    labels = {}
    early_by_id = {r['prediction_id']: r for r in rows if r['before_last_tp']}
    with labels_path.open(newline='') as handle:
        for item in csv.DictReader(handle):
            pi = int(item['prediction_id'])
            if pi not in early_by_id or pi in labels:
                raise ValueError(f'Unknown or repeated early FP ID: {pi}')
            if item['label'] and item['label'] not in LABELS:
                raise ValueError(f'Invalid label for prediction {item["prediction_id"]}')
            if item['label'] in ('genuine_duplicate', 'association_ambiguity') and \
                    not early_by_id[pi]['claimed_gt_within_threshold']:
                raise ValueError(f'No claimed GT within threshold for {pi}')
            if item['label'] == 'likely_localization_error' and \
                    early_by_id[pi]['gt_within_threshold']:
                raise ValueError(f'Localization-error center is within threshold for {pi}')
            labels[pi] = item['label']
    early = [r for r in rows if r['before_last_tp']]
    counts = Counter(labels.get(r['prediction_id']) or 'unreviewed' for r in early)
    reviewed = sum(v for k, v in counts.items() if k != 'unreviewed')
    queue_counts = Counter(labels.get(r['prediction_id']) or 'unreviewed' for r in queue)
    category_rows = []
    for label in LABELS:
        count = counts[label]
        category_rows.append(dict(category=label, reviewed_count=count,
            pct_of_reviewed=100*count/reviewed if reviewed else '',
            pct_of_all_early_fps=100*count/len(early) if reviewed == len(early) and early else '',
            population='all_early_fps' if reviewed == len(early) else 'reviewed_only'))
    write_csv(out/'category_counts.csv', category_rows, list(category_rows[0]))
    summary = dict(threshold_m=args.threshold, gt_count=sum(map(len, gt.values())),
                   prediction_count=len(predictions), tp_count=len(matches),
                   fp_count=len(rows), early_fp_count=len(early),
                   late_fp_count=len(rows)-len(early), last_tp_rank=last_tp+1,
                   baseline_ap_percent=100*official_ap(gt, predictions, None, cfg, args.threshold),
                   geometry_candidate_counts=dict(Counter(r['candidate_kind'] for r in early)),
                   review_queue_count=len(queue), reviewed_count=reviewed,
                   category_counts=category_rows,
                   reviewed_label_counts=dict(counts), queue_label_counts=dict(queue_counts),
                   review_scope='category prevalence and category AP require labels for every early FP')
    reference = base/'outputs/reranking_oracle/pedestrian_oracle_comparison.csv'
    if reference.is_file():
        with reference.open(newline='') as handle:
            previous = next((r for r in csv.DictReader(handle)
                             if float(r['threshold_m']) == args.threshold), None)
        if previous:
            expected_ap = float(previous['baseline_ap_percent'])
            expected_tp = int(previous['baseline_full_set_tp'])
            if abs(summary['baseline_ap_percent']-expected_ap) > 1e-4 or \
                    summary['tp_count'] != expected_tp:
                raise AssertionError('Baseline AP/TP differs from saved oracle baseline')
            summary['baseline_verified_against'] = str(reference)
    if reviewed == len(early):
        category_oracles = {}
        for label in LABELS:
            selected = {pi for pi, value in labels.items() if value == label}
            reordered = [i for i in range(len(predictions)) if i not in selected]
            reordered.extend(i for i in range(len(predictions)) if i in selected)
            new_matches = greedy_match_records(gt, predictions, reordered, args.threshold)
            ap = 100*official_ap(gt, predictions, reordered, cfg, args.threshold)
            category_oracles[label] = dict(count=len(selected), ap_percent=ap,
                ap_gain_points=ap-summary['baseline_ap_percent'], tp_count=len(new_matches))
        summary['category_oracles'] = category_oracles
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if not args.no_images:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
        by_pred_rel = defaultdict(list)
        for relation in relations:
            by_pred_rel[relation['prediction_id']].append(relation)
        cards = []
        for row in queue:
            pi = row['prediction_id']
            local_rel = [x for x in by_pred_rel[pi]
                         if x['distance_m'] <= max(args.threshold+4, 8)]
            nearby = [(int(x['gt_id'].rsplit(':',1)[1]),
                       gt[row['sample_token']][int(x['gt_id'].rsplit(':',1)[1])])
                      for x in local_rel]
            claimant_ids = sorted({x['claimant_id'] for x in local_rel
                                   if x['claimant_id'] is not None})
            cards.append(f'<section><h3>FP {pi} · rank {row["rank"]} · '
                         f'{html.escape(row["candidate_kind"])}</h3><p>Score '
                         f'{row["score"]:.4f}; nearest GT {row["nearest_gt_id"]} '
                         f'at {row["nearest_gt_distance_m"]} m</p>' +
                         camera_cards(nusc, args.data_root, out, predictions[pi], nearby,
                                      [(ci, predictions[ci]) for ci in claimant_ids]) +
                         '</section>')
        (out/'review.html').write_text('<!doctype html><meta charset="utf-8">'
            '<style>body{font:16px sans-serif;margin:2rem;background:#111827;color:white}'
            'section{border-bottom:2px solid #555;padding:2rem 0}svg{width:100%;max-width:1100px}'
            'h4{margin-bottom:.2rem}</style><h1>Early FP image review</h1>'+
            '<p>Red: FP; blue: nearby GT; yellow: GT claimant. Record conclusions '
            'in review_labels.csv. Camera images load from the local nuScenes data root.</p>'+
            ''.join(cards))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
