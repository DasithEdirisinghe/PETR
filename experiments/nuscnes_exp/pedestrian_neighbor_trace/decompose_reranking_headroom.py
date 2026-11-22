#!/usr/bin/env python3
"""Decompose fixed-box train AP headroom by high-ranked FP geometry.

Each class and nuScenes distance threshold is evaluated independently. FP
categories are diagnostic GT-informed labels, not training targets or an
inference-time rule. A counterfactual moves only one category of baseline FPs
to the end of that class's score list, preserving all other relative order.
Matching and AP are then recomputed with the official nuScenes implementation.
Category gains interact and must not be summed.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from all_class_reranking_oracle import class_boxes
from localization_capacity import THRESHOLDS
from pedestrian_image_alignment import (camera_records, projected_rect,
                                        radial_transverse, rect_iou,
                                        reference_camera)
from reranking_oracle import greedy_match_records, official_ap, oracle_order

CATEGORIES = ('recoverable_alternative', 'same_gt_duplicate', 'right_ray_wrong_depth',
              'competing_gt', 'other_class_overlap', 'unsupported', 'ambiguous')


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def demote(order, indices):
    """Stable move-to-end, without removing any prediction."""
    indices = set(indices)
    return [i for i in order if i not in indices] + [i for i in order if i in indices]


def promote(order, indices):
    indices = set(indices)
    return [i for i in order if i in indices] + [i for i in order if i not in indices]


class ProjectionContext:
    def __init__(self, nusc, all_gt):
        self.nusc = nusc
        self.all_gt = all_gt
        self.cameras = {}
        self.reference = {}
        self.rects = {}

    def view(self, token, gt_index):
        key = token, gt_index
        if key not in self.reference:
            if token not in self.cameras:
                self.cameras[token] = camera_records(self.nusc, token)
            self.reference[key] = reference_camera(
                self.cameras[token], self.all_gt[token][gt_index])
        return self.reference[key]

    def rect(self, box, token, camera_name, camera, box_id):
        key = token, camera_name, box_id
        if key not in self.rects:
            self.rects[key] = projected_rect(box, camera)
        return self.rects[key]


def classify_fp(pred, pred_index, class_name, threshold, gt_same, all_gt,
                oracle_owner, context, min_iou, owner_margin):
    """Conservative ownership label for one baseline FP.

    Candidate owner is the same-class GT with largest projected overlap in
    its best-visible camera. A competitor within owner_margin is ambiguous.
    This avoids assigning a same-ray duplicate by nearest center alone.
    """
    token = pred.sample_token
    same = gt_same[token]
    all_frame = all_gt[token]
    # EvalBoxes retains each class's GT objects; identify them geometrically
    # so their indices are valid in the full per-frame list used by the view.
    candidates = []
    for gi, gt in enumerate(all_frame):
        if gt.detection_name != class_name:
            continue
        camera_name, camera, gt_rect = context.view(token, gi)
        if camera is None:
            continue
        pred_rect = context.rect(pred, token, camera_name, camera,
                                 ('pred', class_name, pred_index))
        iou = rect_iou(pred_rect, gt_rect)
        if iou > 0:
            dist = math.hypot(pred.translation[0]-gt.translation[0],
                              pred.translation[1]-gt.translation[1])
            radial, tangent = radial_transverse(pred, gt, camera)
            candidates.append((iou, gi, gt, camera_name, camera, pred_rect,
                               dist, radial, tangent))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0] if candidates else None
    if best and len(candidates) > 1 and \
            candidates[1][0] >= min_iou and \
            best[0] - candidates[1][0] < owner_margin:
        return 'competing_gt', '', best[0], candidates[1][0], '', '', '', ''
    if best and best[0] >= min_iou:
        iou, gi, gt, camera_name, camera, pred_rect, dist, radial, tangent = best
        other_iou = 0.0
        for other_gi, other in enumerate(all_frame):
            if other.detection_name == class_name:
                continue
            other_rect = context.rect(other, token, camera_name, camera,
                                      ('gt', other_gi))
            other_iou = max(other_iou, rect_iou(pred_rect, other_rect))
        if other_iou >= min_iou and other_iou > iou + owner_margin:
            return ('other_class_overlap', '', iou, other_iou, dist,
                    radial if radial is not None else '',
                    tangent if tangent is not None else '', camera_name)
        if other_iou >= min_iou and abs(other_iou-iou) <= owner_margin:
            return ('ambiguous', '', iou, other_iou, dist,
                    radial if radial is not None else '',
                    tangent if tangent is not None else '', camera_name)
        owner_key = next((j for j, other in enumerate(same)
                          if other is gt or (other.translation == gt.translation
                                              and other.detection_name == gt.detection_name)),
                         None)
        if owner_key is None:
            raise AssertionError('Same-class GT missing from class view')
        if dist < threshold and (token, owner_key) in oracle_owner:
            category = 'same_gt_duplicate'
        elif radial is not None and abs(radial) > abs(tangent) and \
                dist >= threshold:
            category = 'right_ray_wrong_depth'
        elif dist < threshold:
            # A projected, in-range FP without an oracle owner should not
            # silently become a background target.
            category = 'ambiguous'
        else:
            category = 'ambiguous'
        return (category, owner_key, iou, '', dist,
                radial if radial is not None else '',
                tangent if tangent is not None else '', camera_name)
    # No credible same-class projection: check whether another class explains
    # the predicted geometry in the most visible view of its GT.
    for gi, gt in enumerate(all_frame):
        if gt.detection_name == class_name:
            continue
        camera_name, camera, gt_rect = context.view(token, gi)
        if camera is None:
            continue
        pred_rect = context.rect(pred, token, camera_name, camera,
                                 ('pred', class_name, pred_index))
        if rect_iou(pred_rect, gt_rect) >= min_iou:
            return 'other_class_overlap', '', '', '', '', '', '', camera_name
    return 'unsupported', '', best[0] if best else 0, '', '', '', '', ''


def analyze_class(evaluator, nusc, class_name, context, thresholds, min_iou,
                  owner_margin, output_dir, max_detail_rows):
    gt, original, predictions = class_boxes(evaluator, class_name)
    if not predictions or not sum(map(len, gt.values())):
        return [], []
    base_order = list(range(len(predictions)))
    rows, details = [], []
    for threshold in thresholds:
        detail_count = 0
        oracle_rank, selected = oracle_order(gt, predictions, threshold)
        baseline_records = greedy_match_records(gt, predictions, base_order,
                                                threshold)
        baseline_tp = {pi for pi, _ in baseline_records.values()}
        # Oracle-selected predictions that were FP in the original order are
        # the direct ranking reversals of interest.
        selected_fp = set(selected) - baseline_tp
        last_selected_rank = max(selected) if selected else -1
        early_fp = [i for i in base_order if i <= last_selected_rank and
                    i not in baseline_tp]
        owner = {(predictions[i].sample_token, gi): i
                 for i, (gi, _, _) in selected.items()}
        groups = defaultdict(list)
        # Number of oracle representatives currently below this FP. This is
        # a rank-pressure measure, not a count of lost TPs.
        selected_prefix = [0] * (len(predictions) + 1)
        for i in range(len(predictions)-1, -1, -1):
            selected_prefix[i] = selected_prefix[i+1] + int(i in selected)
        score_sum, pressure_sum = Counter(), Counter()
        for i in early_fp:
            pred = predictions[i]
            category, owner_gi, proj_iou, competitor_iou, distance, radial, \
                tangent, camera = classify_fp(
                    pred, i, class_name, threshold, gt,
                    evaluator.gt_boxes.boxes, owner, context, min_iou,
                    owner_margin)
            if i in selected_fp:
                # A baseline FP selected as an oracle representative is a
                # candidate to promote; suppressing it would invert the audit.
                category = 'recoverable_alternative'
            groups[category].append(i)
            score_sum[category] += pred.detection_score
            pressure_sum[category] += selected_prefix[i+1]
            if detail_count < max_detail_rows:
                details.append({
                    'class': class_name, 'threshold_m': threshold,
                    'sample_token': pred.sample_token, 'prediction_rank': i+1,
                    'prediction_score': pred.detection_score,
                    'category': category, 'owner_gt_index': owner_gi,
                    'owner_projected_iou': proj_iou,
                    'competing_projected_iou': competitor_iou,
                    'owner_xy_error_m': distance,
                    'owner_radial_error_m': radial,
                    'owner_tangential_error_m': tangent,
                    'camera': camera,
                    'oracle_representatives_below': selected_prefix[i+1],
                    'oracle_selected_but_baseline_fp': int(i in selected_fp),
                })
                detail_count += 1
        baseline_ap = official_ap(gt, original, None, evaluator.cfg,
                                  threshold, class_name)
        oracle_ap = official_ap(gt, predictions, oracle_rank, evaluator.cfg,
                                threshold, class_name)
        combined = [i for cat, group in groups.items()
                    if cat != 'recoverable_alternative' for i in group]
        combined_order = promote(demote(base_order, combined),
                                 groups['recoverable_alternative'])
        combined_ap = official_ap(gt, predictions, combined_order,
                                  evaluator.cfg, threshold, class_name)
        combined_tp = len(greedy_match_records(gt, predictions,
                                               combined_order, threshold))
        for category in CATEGORIES:
            group = groups[category]
            if group:
                moved_order = (promote(base_order, group)
                               if category == 'recoverable_alternative'
                               else demote(base_order, group))
                moved_ap = official_ap(gt, predictions, moved_order,
                                       evaluator.cfg, threshold, class_name)
                moved_tp = len(greedy_match_records(gt, predictions,
                                                    moved_order, threshold))
            else:
                moved_ap, moved_tp = baseline_ap, len(baseline_records)
            rows.append({
                'class': class_name, 'threshold_m': threshold,
                'category': category, 'gt_count': sum(map(len, gt.values())),
                'prediction_count': len(predictions),
                'baseline_tp_count': len(baseline_records),
                'oracle_selected_count': len(selected),
                'oracle_selected_baseline_fp_count': len(selected_fp),
                'early_fp_count': len(early_fp),
                'category_early_fp_count': len(group),
                'category_mean_score': score_sum[category]/len(group)
                if group else '',
                'category_oracle_representatives_below_sum': pressure_sum[category],
                'baseline_ap_percent': 100*baseline_ap,
                'oracle_ap_percent': 100*oracle_ap,
                'oracle_gap_pp': 100*(oracle_ap-baseline_ap),
                'category_action_ap_percent': 100*moved_ap,
                'category_action_gain_pp': 100*(moved_ap-baseline_ap),
                'category_action_tp_count': moved_tp,
                'combined_action_ap_percent': 100*combined_ap,
                'combined_action_gain_pp': 100*(combined_ap-baseline_ap),
                'combined_action_tp_count': combined_tp,
            })
        print('{} @ {:g} m: baseline {:.2f}, oracle {:.2f}, all-group {:.2f}; '
              '{} early FPs'.format(class_name, threshold, 100*baseline_ap,
                                    100*oracle_ap, 100*combined_ap,
                                    len(early_fp)), flush=True)
        context.rects.clear()  # bound memory across thresholds
    return rows, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eval-set', choices=('train', 'val'), default='train')
    parser.add_argument('--restrict-to-result-tokens', action='store_true')
    parser.add_argument('--classes', nargs='+')
    parser.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
    parser.add_argument('--min-projected-iou', type=float, default=0.1)
    parser.add_argument('--owner-margin', type=float, default=0.05)
    parser.add_argument('--max-detail-rows-per-class', type=int, default=20000)
    args = parser.parse_args()
    if not args.results_json.is_file() or not (args.data_root/'v1.0-trainval').is_dir():
        parser.error('Missing predictions or nuScenes metadata')
    if any(t not in THRESHOLDS for t in args.thresholds) or \
            not 0 < args.min_projected_iou <= 1 or \
            not 0 <= args.owner_margin <= 1:
        parser.error('Use nuScenes thresholds and valid projected-IoU settings')
    from nuscenes import NuScenes
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    cfg = config_factory('detection_cvpr_2019')
    classes = list(cfg.class_names) if args.classes is None else args.classes
    if not classes or len(set(classes)) != len(classes) or \
            set(classes) - set(cfg.class_names):
        parser.error('Choose unique nuScenes detection classes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root),
                    verbose=False)
    result_path = args.results_json
    analysis_tokens = None
    if args.restrict_to_result_tokens:
        from nuscenes.eval.common.loaders import load_gt
        from nuscenes.eval.detection.data_classes import DetectionBox
        submission = json.loads(args.results_json.read_text())
        analysis_tokens = set(submission['results'])
        expected = set(load_gt(nusc, args.eval_set, DetectionBox,
                               verbose=False).sample_tokens)
        if analysis_tokens - expected:
            parser.error('Results contain tokens outside the evaluation split')
        completed = dict(submission)
        completed['results'] = dict(submission['results'])
        completed['results'].update({token: [] for token in expected-analysis_tokens})
        source = args.output_dir/'source_eval'
        source.mkdir(exist_ok=True)
        result_path = source/'completed_for_loader_results_nusc.json'
        result_path.write_text(json.dumps(completed))
    evaluator = NuScenesEval(nusc, config=cfg, result_path=str(result_path),
                             eval_set=args.eval_set,
                             output_dir=str(args.output_dir/'source_eval'),
                             verbose=False)
    if analysis_tokens is not None:
        evaluator.gt_boxes.boxes = {token: evaluator.gt_boxes.boxes[token]
                                    for token in analysis_tokens}
        evaluator.pred_boxes.boxes = {token: evaluator.pred_boxes.boxes[token]
                                      for token in analysis_tokens}
    context = ProjectionContext(nusc, evaluator.gt_boxes.boxes)
    rows, details = [], []
    for name in classes:
        part_rows, part_details = analyze_class(
            evaluator, nusc, name, context, args.thresholds,
            args.min_projected_iou, args.owner_margin, args.output_dir,
            args.max_detail_rows_per_class)
        rows.extend(part_rows)
        details.extend(part_details)
        # Also release camera/GT projection state between classes.
        context.cameras.clear()
        context.reference.clear()
    if not rows:
        parser.error('No GT/prediction class pairs could be analyzed')
    write_csv(args.output_dir/'category_counterfactual_ap.csv', rows,
              list(rows[0]))
    if details:
        write_csv(args.output_dir/'early_fp_examples.csv', details,
                  list(details[0]))
    by_key = defaultdict(list)
    for row in rows:
        by_key[row['category']].append(row)
    summary = {
        'source_results_json': str(args.results_json.resolve()),
        'eval_set': args.eval_set,
        'analyzed_sample_count': len(analysis_tokens) if analysis_tokens is not None
        else len(evaluator.gt_boxes.sample_tokens),
        'classes': classes, 'thresholds_m': args.thresholds,
        'baseline_mean_ap_percent': sum(r['baseline_ap_percent'] for r in rows)/len(rows),
        'oracle_mean_ap_percent': sum(r['oracle_ap_percent'] for r in rows)/len(rows),
        'combined_action_mean_ap_percent': sum(
            r['combined_action_ap_percent'] for r in rows)/len(rows),
        'category_counterfactual_mean_gain_pp': {
            cat: sum(r['category_action_gain_pp'] for r in group)/len(group)
            for cat, group in by_key.items()},
        'category_early_fp_count_across_thresholds': {
            cat: sum(r['category_early_fp_count'] for r in group)
            for cat, group in by_key.items()},
        'note': ('AP means use one row per class/threshold; category gains interact '
                 'and cannot be added. GT-informed labels are diagnostics. The '
                 'recoverable oracle representatives are promoted; other '
                 'baseline FP categories are demoted.'),
    }
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Wrote {}'.format(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
