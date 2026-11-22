"""Opt-in, detached geometric targets for PETR class-score supervision.

This module does not alter matching or box regression. All role decisions are
made without gradients; only the selected class logits receive gradients.
"""

import torch
import torch.nn.functional as F

from ...core.bbox.util import denormalize_bbox


def _project_rectangles(boxes, lidar2img, image_shape):
    """Project gravity-centred [x,y,z,w,l,h,yaw,...] boxes into one view."""
    if not boxes.numel():
        return boxes.new_zeros((0, 4)), boxes.new_zeros((0,), dtype=torch.bool)
    device, dtype = boxes.device, boxes.dtype
    signs = boxes.new_tensor([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    half = boxes[:, None, 3:6].clamp(min=0) / 2
    xyz = signs[None] * half
    angle = boxes[:, 6]
    ca, sa = angle.cos()[:, None], angle.sin()[:, None]
    x = xyz[..., 0] * ca - xyz[..., 1] * sa + boxes[:, None, 0]
    y = xyz[..., 0] * sa + xyz[..., 1] * ca + boxes[:, None, 1]
    z = xyz[..., 2] + boxes[:, None, 2]
    corners = torch.stack((x, y, z, torch.ones_like(x)), -1)
    matrix = torch.as_tensor(lidar2img, device=device, dtype=dtype)
    projected = corners @ matrix.T
    depth = projected[..., 2]
    # Exclude boxes that cross the near plane: their clipped rectangle is
    # unstable and can create a spurious owner.
    valid_depth = (depth > 0.1).all(-1)
    uv = projected[..., :2] / depth.clamp(min=0.1)[..., None]
    height, width = image_shape[:2]
    xy_min = uv.amin(1)
    xy_max = uv.amax(1)
    rect = torch.cat((xy_min, xy_max), -1)
    rect[:, 0::2] = rect[:, 0::2].clamp(0, float(width))
    rect[:, 1::2] = rect[:, 1::2].clamp(0, float(height))
    valid = valid_depth & ((rect[:, 2] - rect[:, 0]) >= 2) & (
        (rect[:, 3] - rect[:, 1]) >= 2) & torch.isfinite(rect).all(-1)
    return rect, valid


def _pairwise_iou(a, b):
    low = torch.maximum(a[:, None, :2], b[None, :, :2])
    high = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    area_inter = (high - low).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return area_inter / (area_a[:, None] + area_b[None, :] - area_inter).clamp(min=1e-6)


def _camera_geometry(pred, gt, meta):
    """Best projected overlap and corresponding horizontal ray errors."""
    n, m = pred.size(0), gt.size(0)
    overlap = pred.new_zeros((n, m))
    radial = pred.new_zeros((n, m))
    tangent = pred.new_zeros((n, m))
    for matrix, shape in zip(meta['lidar2img'], meta['img_shape']):
        pred_rect, pred_valid = _project_rectangles(pred, matrix, shape)
        gt_rect, gt_valid = _project_rectangles(gt, matrix, shape)
        iou = _pairwise_iou(pred_rect, gt_rect)
        iou = iou * (pred_valid[:, None] & gt_valid[None, :])
        view = torch.as_tensor(matrix, device=pred.device, dtype=pred.dtype)
        origin = torch.linalg.inv(view)[:3, 3]
        direction = gt[:, :2] - origin[None, :2]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        delta = pred[:, None, :2] - gt[None, :, :2]
        view_radial = (delta * direction[None]).sum(-1)
        view_tangent = delta[..., 0] * -direction[None, :, 1] + delta[..., 1] * direction[None, :, 0]
        better = iou > overlap
        overlap = torch.where(better, iou, overlap)
        radial = torch.where(better, view_radial, radial)
        tangent = torch.where(better, view_tangent, tangent)
    return overlap, radial, tangent


def _batch_ranks(scores):
    """One-based batch ranks; equal scores are not compared as inversions."""
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device)
    return ranks


def ray_rule_layer_loss(cls_scores, bbox_preds, gt_boxes, gt_labels,
                        img_metas, assigner, cfg):
    """One scalar loss and detached diagnostics for one decoder layer.

    A query must have one clear geometric owner before it receives a target.
    Hungarian matches are recomputed with the same assigner solely to measure
    disagreement; the usual PETR target path remains unchanged.
    """
    min_iou = float(cfg.get('min_projected_iou', 0.15))
    owner_margin = float(cfg.get('owner_iou_margin', 0.05))
    min_quality_gap = float(cfg.get('min_quality_gap', 0.1))
    sigma_r = float(cfg.get('sigma_radial', 2.0))
    sigma_t = float(cfg.get('sigma_tangential', 1.0))
    power_r = float(cfg.get('power_radial', 2.0))
    power_t = float(cfg.get('power_tangential', 2.0))
    if min(sigma_r, sigma_t, power_r, power_t) <= 0:
        raise ValueError('Ray-rule scales and powers must be positive')

    entries = []
    stats = {'preferred': 0, 'inferior': 0, 'hm_conflicts': 0,
             'rank_violations': 0, 'eligible': 0,
             'preferred_hm_conflict': 0, 'inferior_hm_conflict': 0,
             'preferred_bce_sum': 0.0, 'inferior_bce_sum': 0.0,
             'preferred_weighted_bce_sum': 0.0,
             'inferior_weighted_bce_sum': 0.0,
             'preferred_conflict_bce_sum': 0.0,
             'inferior_conflict_bce_sum': 0.0,
             'radial_inferior': 0, 'tangential_inferior': 0,
             'radial_inferior_bce_sum': 0.0,
             'tangential_inferior_bce_sum': 0.0,
             'pair_penalty_sum': 0.0, 'pair_count': 0}
    if cfg.get('collect_diagnostics', False):
        for class_id in range(cls_scores.size(-1)):
            for role in ('preferred', 'inferior'):
                stats[f'class_{class_id}_{role}_count'] = 0
                stats[f'class_{class_id}_{role}_bce_sum'] = 0.0
    with torch.no_grad():
        for batch_id, (pred_encoded, logits, gt, labels, meta) in enumerate(
                zip(bbox_preds, cls_scores, gt_boxes, gt_labels, img_metas)):
            if not gt.numel():
                continue
            pred = denormalize_bbox(pred_encoded.detach(), None)
            overlap, radial, tangent = _camera_geometry(pred, gt, meta)
            best_overlap, owner = overlap.max(1)
            if gt.size(0) > 1:
                second = overlap.topk(2, dim=1).values[:, 1]
            else:
                second = torch.zeros_like(best_overlap)
            clear = (best_overlap >= min_iou) & ((best_overlap - second) >= owner_margin)
            quality = torch.exp(-((radial.abs() / sigma_r) ** power_r +
                                  (tangent.abs() / sigma_t) ** power_t))
            hm = assigner.assign(pred_encoded.detach(), logits.detach(), gt, labels).gt_inds - 1
            # The only winner for each GT is its best localized clear query.
            for g in range(gt.size(0)):
                class_id = int(labels[g])
                if class_id < 0 or class_id >= cls_scores.size(-1):
                    continue
                candidates = torch.nonzero(clear & (owner == g), as_tuple=False).flatten()
                if not candidates.numel():
                    continue
                values = quality[candidates, g]
                winner = candidates[values.argmax()]
                chosen = [winner]
                targets = [quality[winner, g]]
                roles = [True]
                for q in candidates:
                    if q == winner:
                        continue
                    if quality[winner, g] - quality[q, g] < min_quality_gap:
                        continue
                    chosen.append(q)
                    targets.append(quality.new_zeros(()))
                    roles.append(False)
                # The exact same PETR assigner exposes GT identity, not only
                # positive/background. No gradients pass through this step.
                for q, target, preferred in zip(chosen, targets, roles):
                    assigned = int(hm[q])
                    if not preferred and assigned >= 0 and assigned != int(g):
                        # A different GT may legitimately own this query.
                        continue
                    conflict = assigned != int(g) if preferred else assigned == int(g)
                    radial_dominant = bool(radial[q, g].abs() >= tangent[q, g].abs())
                    entries.append((batch_id, int(q), class_id, target,
                                    preferred, conflict, radial_dominant))
                    stats['preferred' if preferred else 'inferior'] += 1
                    if not preferred:
                        stats['radial_inferior' if radial_dominant else
                              'tangential_inferior'] += 1
                    if cfg.get('collect_diagnostics', False):
                        stats[f'class_{class_id}_{"preferred" if preferred else "inferior"}_count'] += 1
                    stats['hm_conflicts'] += int(conflict)
                    if conflict:
                        stats['preferred_hm_conflict' if preferred else
                              'inferior_hm_conflict'] += 1

    if not entries:
        return cls_scores.sum() * 0, stats

    batch_index = torch.tensor([x[0] for x in entries], device=cls_scores.device)
    query_index = torch.tensor([x[1] for x in entries], device=cls_scores.device)
    class_index = torch.tensor([x[2] for x in entries], device=cls_scores.device)
    targets = torch.stack([x[3] for x in entries]).detach()
    preferred = torch.tensor([x[4] for x in entries], device=cls_scores.device, dtype=torch.bool)
    conflict = torch.tensor([x[5] for x in entries], device=cls_scores.device, dtype=cls_scores.dtype)
    radial_dominant = torch.tensor([x[6] for x in entries], device=cls_scores.device,
                                   dtype=torch.bool)
    selected_logits = cls_scores[batch_index, query_index, class_index]

    with torch.no_grad():
        # The actual score rank includes *all* class-c query predictions in
        # the optimizer batch. The violation fraction compares only reliable
        # rule-labelled entries, since other queries have unknown quality.
        selected_ranks = torch.zeros_like(batch_index)
        for class_id in class_index.unique():
            same_class = class_index == class_id
            all_scores = cls_scores[..., class_id].detach().sigmoid().flatten()
            ranks = _batch_ranks(all_scores)
            selected_ranks[same_class] = ranks[
                batch_index[same_class] * cls_scores.size(1) + query_index[same_class]]
        quality_gap = targets[:, None] - targets[None, :]
        comparable = (quality_gap.abs() > min_quality_gap) & (
            class_index[:, None] == class_index[None, :])
        score_gap = selected_logits.detach()[:, None] - selected_logits.detach()[None, :]
        inverted = comparable & (quality_gap * score_gap < 0)
        denom = comparable.sum(1).clamp(min=1)
        rank_fraction = inverted.sum(1).to(cls_scores.dtype) / denom
        stats['rank_violations'] = int(inverted.sum()) // 2
        stats['eligible'] = len(entries)
        weights = (1 + float(cfg.get('hm_weight', 1.0)) * conflict +
                   float(cfg.get('rank_weight', 1.0)) * rank_fraction)
        weights = weights.clamp(max=float(cfg.get('max_weight', 3.0)))

    raw_bce = F.binary_cross_entropy_with_logits(
        selected_logits, targets, reduction='none')
    per_entry = raw_bce * weights
    if cfg.get('collect_diagnostics', False):
        for mask, name in ((preferred, 'preferred'), (~preferred, 'inferior')):
            stats[name + '_bce_sum'] = float(raw_bce[mask].detach().sum())
            stats[name + '_weighted_bce_sum'] = float(per_entry[mask].detach().sum())
            conflict_mask = mask & conflict.bool()
            stats[name + '_conflict_bce_sum'] = float(
                raw_bce[conflict_mask].detach().sum())
            for class_id in range(cls_scores.size(-1)):
                class_mask = mask & (class_index == class_id)
                stats[f'class_{class_id}_{name}_bce_sum'] = float(
                    raw_bce[class_mask].detach().sum())
        for mask, name in (((~preferred) & radial_dominant, 'radial_inferior'),
                           ((~preferred) & (~radial_dominant), 'tangential_inferior')):
            stats[name + '_bce_sum'] = float(raw_bce[mask].detach().sum())
    loss = selected_logits.sum() * 0
    active_classes = class_index.unique()
    for class_id in active_classes:
        same_class = class_index == class_id
        for mask, group_weight in ((preferred & same_class, 'preferred_group_weight'),
                                   (~preferred & same_class, 'negative_group_weight')):
            if mask.any():
                loss = loss + per_entry[mask].mean() * float(cfg.get(group_weight, 1.0))
        # A direct ranking gradient helps when an HM-negative preferred query
        # already scores above its soft quality target: BCE alone would lower it.
        pair_mask = ((targets[:, None] - targets[None, :]) > min_quality_gap) & (
            same_class[:, None] & same_class[None, :])
        if pair_mask.any():
            logit_gap = selected_logits[:, None] - selected_logits[None, :]
            margin = float(cfg.get('ranking_margin', 0.5))
            pair_values = F.softplus(margin - logit_gap)[pair_mask]
            pair_loss = pair_values.mean()
            loss = loss + float(cfg.get('pair_weight', 0.25)) * pair_loss
            if cfg.get('collect_diagnostics', False):
                stats['pair_penalty_sum'] += float(pair_values.detach().sum())
                stats['pair_count'] += int(pair_values.numel())
    return loss / active_classes.numel(), stats
