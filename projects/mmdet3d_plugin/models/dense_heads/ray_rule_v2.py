"""Opt-in rule-guided PETR assignment and class-score ranking.

The rule geometry is detached. A certified pair may replace an ordinary PETR
match, while unmatched GTs are filled with the original PETR cost. The same
role labels drive the ranking loss in original-match and rule-match runs.
"""

import torch
import torch.nn.functional as F
from mmdet.core.bbox.assigners import AssignResult
from scipy.optimize import linear_sum_assignment

from ...core.bbox.util import denormalize_bbox, normalize_bbox
from .ray_rule_loss import _camera_geometry


def _original_cost(assigner, encoded, logits, gt, labels):
    """Use the exact classification and regression costs of PETR's assigner."""
    normalized = normalize_bbox(gt, assigner.pc_range)
    return (assigner.cls_cost(logits, labels) +
            assigner.reg_cost(encoded[:, :8], normalized[:, :8]))


def _assign_result(gt_count, query_count, pairs, labels, device):
    assigned = torch.zeros(query_count, dtype=torch.long, device=device)
    assigned_labels = torch.full((query_count,), -1, dtype=torch.long, device=device)
    for q, g in pairs:
        assigned[q] = g + 1
        assigned_labels[q] = labels[g]
    return AssignResult(gt_count, assigned, None, labels=assigned_labels)


@torch.no_grad()
def prepare_rule_v2_layer(cls_scores, bbox_preds, gt_boxes, gt_labels,
                          img_metas, assigner, cfg):
    """Return rule assignments, certified positives, negatives and audit data.

    All indices in positives and negatives refer to (batch, query, class).
    The original assigner is called independently for the comparison audit.
    """
    positives, duplicates, unsupported = [], [], []
    rule_results, original_results, examples = [], [], []
    stats = {name: 0.0 for name in (
        'gt', 'certified', 'hm_fallback', 'changed', 'changed_better',
        'changed_worse', 'changed_equal', 'changed_certified',
        'changed_fallback', 'certified_compat_rejected',
        'duplicates', 'unsupported', 'wrong_class', 'changed_old_center_sum',
        'changed_new_center_sum', 'positive_hm_background',
        'negative_hm_positive')}
    # MMCV's distributed loss parser requires identical log keys on every
    # rank, even when their batches contain different object classes.
    for class_id in range(cls_scores.size(-1)):
        stats['class_{}_changed'.format(class_id)] = 0.0
    min_iou = float(cfg.get('min_projected_iou', 0.15))
    owner_margin = float(cfg.get('owner_iou_margin', 0.05))
    max_center = float(cfg.get('max_certified_center', 4.0))
    max_tangent = float(cfg.get('max_certified_tangent', 2.0))
    min_class_score = float(cfg.get('min_class_score', 0.02))
    class_logit_margin = float(cfg.get('class_logit_margin', 1.5))
    duplicate_gap = float(cfg.get('duplicate_center_gap', 0.5))
    unsupported_iou = float(cfg.get('unsupported_max_iou', 0.05))
    unsupported_distance = float(cfg.get('unsupported_min_center', 4.0))
    unsupported_score = float(cfg.get('unsupported_min_score', 0.05))
    example_limit = int(cfg.get('audit_examples_per_layer', 8))
    for batch_id, (logits, encoded, gt, labels, meta) in enumerate(
            zip(cls_scores, bbox_preds, gt_boxes, gt_labels, img_metas)):
        logits = logits.detach()
        encoded = encoded.detach()
        n, m = encoded.size(0), gt.size(0)
        stats['gt'] += m
        original = assigner.assign(encoded, logits, gt, labels)
        original_results.append(original)
        if not n or not m:
            rule_results.append(original)
            if n and not m:
                probabilities = logits.sigmoid()
                for c in range(logits.size(1)):
                    indices = torch.nonzero(
                        probabilities[:, c] >= unsupported_score,
                        as_tuple=False).flatten()
                    unsupported.extend((batch_id, int(q), c) for q in indices)
                stats['unsupported'] += sum(1 for x in unsupported if x[0] == batch_id)
            continue

        pred = denormalize_bbox(encoded, None)
        overlap, radial, tangent = _camera_geometry(pred, gt, meta)
        distance = torch.linalg.vector_norm(pred[:, None, :2] - gt[None, :, :2],
                                            dim=-1)
        best_overlap, owner = overlap.max(dim=1)
        second = (overlap.topk(2, dim=1).values[:, 1] if m > 1
                  else torch.zeros_like(best_overlap))
        clear = ((best_overlap >= min_iou) &
                 ((best_overlap - second) >= owner_margin))
        largest_logit = logits.max(dim=1).values
        class_logits = logits[:, labels]
        compatible = ((class_logits.sigmoid() >= min_class_score) &
                      (class_logits >= largest_logit[:, None] - class_logit_margin))
        owner_compatible = compatible.gather(1, owner[:, None]).squeeze(1)
        owner_distance = distance.gather(1, owner[:, None]).squeeze(1)
        owner_tangent = tangent.gather(1, owner[:, None]).squeeze(1).abs()
        certified = (clear & owner_compatible &
                     (owner_distance <= max_center) &
                     (owner_tangent <= max_tangent))
        stats['certified_compat_rejected'] += int((clear & ~owner_compatible).sum())

        # Unique owner for each query makes the GT-local minimum globally
        # one-to-one. Fallback is solved jointly on the remaining submatrix.
        pairs = []
        winners = {}
        for g in range(m):
            candidates = torch.nonzero(certified & (owner == g),
                                       as_tuple=False).flatten()
            if not candidates.numel():
                continue
            q = int(candidates[distance[candidates, g].argmin()])
            pairs.append((q, g))
            winners[g] = q
            positives.append((batch_id, q, int(labels[g])))
        stats['certified'] += len(winners)
        used_queries = {q for q, _ in pairs}
        remaining_gt = [g for g in range(m) if g not in winners]
        remaining_queries = [q for q in range(n) if q not in used_queries]
        if remaining_gt and remaining_queries:
            fallback_cost = _original_cost(
                assigner, encoded[remaining_queries], logits[remaining_queries],
                gt[remaining_gt], labels[remaining_gt])
            fallback_cost = torch.nan_to_num(
                fallback_cost.detach().cpu(), nan=100., posinf=100., neginf=-100.)
            row, col = linear_sum_assignment(fallback_cost)
            pairs.extend((remaining_queries[int(r)], remaining_gt[int(c)])
                         for r, c in zip(row, col))
            stats['hm_fallback'] += len(row)
        result = _assign_result(m, n, pairs, labels, encoded.device)
        rule_results.append(result)

        original_positive = torch.nonzero(
            original.gt_inds > 0, as_tuple=False).flatten()
        original_by_gt = {
            int(original.gt_inds[q]) - 1: int(q) for q in original_positive}
        for q, g in pairs:
            old_q = original_by_gt.get(g)
            if old_q is None or old_q == q:
                continue
            old_d = float(distance[old_q, g])
            new_d = float(distance[q, g])
            stats['changed'] += 1
            stats['changed_certified' if g in winners else 'changed_fallback'] += 1
            stats['changed_old_center_sum'] += old_d
            stats['changed_new_center_sum'] += new_d
            class_key = 'class_{}_changed'.format(int(labels[g]))
            stats[class_key] += 1
            stats['changed_better' if new_d < old_d - 1e-4 else
                  'changed_worse' if new_d > old_d + 1e-4 else
                  'changed_equal'] += 1
            if len(examples) < example_limit:
                examples.append(dict(
                    sample_idx=str(meta.get('sample_idx', batch_id)),
                    gt_index=g, class_id=int(labels[g]),
                    original_query=old_q, rule_query=q,
                    rule_pair_type='certified' if g in winners else 'hm_fallback',
                    original_center_m=old_d, rule_center_m=new_d,
                    original_projected_iou=float(overlap[old_q, g]),
                    rule_projected_iou=float(overlap[q, g]),
                    original_radial_m=float(radial[old_q, g]),
                    rule_radial_m=float(radial[q, g]),
                    original_tangential_m=float(tangent[old_q, g]),
                    rule_tangential_m=float(tangent[q, g]),
                    original_class_score=float(logits[old_q, labels[g]].sigmoid()),
                    rule_class_score=float(logits[q, labels[g]].sigmoid())))

        # A duplicate is inferior to an actually certified winner. A query
        # matched to another GT is protected from negative rule labels.
        for g, winner in winners.items():
            candidates = torch.nonzero(
                clear & (owner == g) & owner_compatible &
                (distance[:, g] >= distance[winner, g] + duplicate_gap),
                as_tuple=False).flatten()
            for candidate in candidates:
                q = int(candidate)
                if ((int(result.gt_inds[q]) > 0 and
                     int(result.gt_inds[q]) != g + 1) or
                    (int(original.gt_inds[q]) > 0 and
                     int(original.gt_inds[q]) != g + 1)):
                    continue
                duplicates.append((batch_id, q, int(labels[g])))
                if int(original.gt_inds[q]) > 0:
                    stats['negative_hm_positive'] += 1
        stats['duplicates'] += sum(1 for x in duplicates if x[0] == batch_id)
        for g, winner in winners.items():
            if int(original.gt_inds[winner]) == 0:
                stats['positive_hm_background'] += 1

        max_iou = overlap.max(dim=1).values
        min_distance = distance.min(dim=1).values
        background = ((result.gt_inds == 0) & (original.gt_inds == 0) &
                      (max_iou <= unsupported_iou) &
                      (min_distance >= unsupported_distance))
        probabilities = logits.sigmoid()
        for c in range(logits.size(1)):
            indices = torch.nonzero(
                background & (probabilities[:, c] >= unsupported_score),
                as_tuple=False).flatten()
            unsupported.extend((batch_id, int(q), c) for q in indices)
            # A query projecting clearly to a different-class GT can be a
            # wrong-class FP, provided no GT of class c plausibly owns it.
            same_class_gt = labels == c
            if same_class_gt.any():
                supported_as_c = ((overlap[:, same_class_gt] > unsupported_iou) |
                                  (distance[:, same_class_gt] < unsupported_distance)).any(1)
            else:
                supported_as_c = torch.zeros(n, dtype=torch.bool, device=logits.device)
            wrong_class = (clear & ~supported_as_c &
                           (labels[owner] != c) &
                           (result.labels != c) &
                           (probabilities[:, c] >= unsupported_score))
            wrong_indices = torch.nonzero(wrong_class, as_tuple=False).flatten()
            unsupported.extend((batch_id, int(q), c) for q in wrong_indices)
            stats['wrong_class'] += int(wrong_indices.numel())
            stats['negative_hm_positive'] += int(
                (original.labels[wrong_indices] == c).sum())
        stats['unsupported'] += sum(1 for x in unsupported if x[0] == batch_id)
    return (rule_results, original_results, positives, duplicates,
            unsupported, stats, examples)


def rule_v2_rank_loss(cls_scores, positives, duplicates, unsupported, cfg):
    """Score only; compare certified positives with confirmed negatives."""
    zero = cls_scores.sum() * 0
    negatives = list(dict.fromkeys(duplicates + unsupported))
    if not positives and not negatives:
        return zero, {'ranking_pairs': 0.0, 'ranking_inversions': 0.0}
    classes = sorted({entry[2] for entry in positives + negatives})
    positive_weight = float(cfg.get('positive_weight', 1.0))
    negative_weight = float(cfg.get('negative_weight', 1.0))
    pair_weight = float(cfg.get('pair_weight', 1.0))
    margin = float(cfg.get('ranking_margin', 0.5))
    hard_negative_max = int(cfg.get('hard_negative_max', 64))
    total = zero
    pair_count, inversion_count = 0, 0
    for c in classes:
        pos = [entry for entry in positives if entry[2] == c]
        neg = [entry for entry in negatives if entry[2] == c]
        pos_logits = (torch.stack([cls_scores[b, q, c] for b, q, _ in pos])
                      if pos else None)
        neg_logits = (torch.stack([cls_scores[b, q, c] for b, q, _ in neg])
                      if neg else None)
        if pos_logits is not None:
            total = total + positive_weight * F.softplus(-pos_logits).mean()
        if neg_logits is not None:
            total = total + negative_weight * F.softplus(neg_logits).mean()
        if pos_logits is not None and neg_logits is not None:
            hard = neg_logits.topk(min(hard_negative_max, neg_logits.numel())).values
            gap = margin + hard[None, :] - pos_logits[:, None]
            total = total + pair_weight * F.softplus(gap).mean()
            pair_count += gap.numel()
            inversion_count += int((gap.detach() > margin).sum())
    return total / len(classes), {
        'ranking_pairs': float(pair_count),
        'ranking_inversions': float(inversion_count)}
