"""Synthetic assignment/ranking check for V6 and V7 in the PETR torch env.

Run from repository root: python experiments/nuscnes_exp/pedestrian_neighbor_trace/test_rule_v2_smoke.py
"""

from pathlib import Path
import sys

import torch
from mmcv import Config
from mmdet.core.bbox.assigners import AssignResult
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox  # noqa: E402
from projects.mmdet3d_plugin.models.dense_heads import ray_rule_v2  # noqa: E402
from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D  # noqa: E402


class SyntheticPETRAssigner:
    """Small classification-plus-regression cost with ordinary HM behavior."""

    pc_range = None

    @staticmethod
    def cls_cost(logits, labels):
        return -4 * logits.sigmoid()[:, labels]

    @staticmethod
    def reg_cost(encoded, gt_encoded):
        return 0.1 * torch.cdist(encoded[:, :2], gt_encoded[:, :2], p=1)

    def assign(self, encoded, logits, gt, labels):
        assigned = torch.zeros(encoded.size(0), dtype=torch.long,
                               device=encoded.device)
        assigned_labels = torch.full_like(assigned, -1)
        if gt.numel() and encoded.numel():
            cost = self.cls_cost(logits, labels) + self.reg_cost(
                encoded[:, :8], normalize_bbox(gt, None)[:, :8])
            rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
            for row, col in zip(rows, cols):
                assigned[row] = int(col) + 1
                assigned_labels[row] = labels[col]
        return AssignResult(gt.size(0), assigned, None, labels=assigned_labels)


def box(x, y=0):
    return torch.tensor([x, y, 0., 1., 1., 2., 0., 0., 0.])


def main():
    class MetaCheckingHead:
        ray_rule_cfg = None
        rule_v2_cfg = {'matching': 'original'}

        def __call__(self, features, metadata):
            return None

        def loss(self, *args, **kwargs):
            assert kwargs['img_metas'] == [{'sample_idx': 'smoke'}]
            return {'meta_forwarded': True}

    class DummyDetector:
        pts_bbox_head = MetaCheckingHead()

    assert Petr3D.forward_pts_train(
        DummyDetector(), [], [], [], [{'sample_idx': 'smoke'}]
    )['meta_forwarded']

    config_dir = Path(__file__).resolve().parents[3] / 'projects/configs/petr'
    v6 = Config.fromfile(str(config_dir /
        'petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v6.py'))
    v7 = Config.fromfile(str(config_dir /
        'petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v7.py'))
    assert v6.model.pts_bbox_head.rule_v2_cfg.matching == 'original'
    assert v7.model.pts_bbox_head.rule_v2_cfg.matching == 'rule'
    assert v6.model.pts_bbox_head.rule_v2_cfg.loss_weight == (
        v7.model.pts_bbox_head.rule_v2_cfg.loss_weight)
    assert v6.load_from == v7.load_from
    assert v6.data.samples_per_gpu == v7.data.samples_per_gpu == 4

    gt = torch.stack([box(10), box(20), box(30)])
    labels = torch.tensor([0, 8, 0])
    predictions = torch.stack([
        box(10.2), box(13.), box(20.2), box(30.5), box(14.), box(50.)])
    encoded = normalize_bbox(predictions, None)[None]
    probabilities = torch.full((1, 6, 10), 0.001)
    probabilities[0, 0, 0] = .2  # well-localized car, lower score
    probabilities[0, 1, 0] = .9  # wrong-depth car duplicate, higher score
    probabilities[0, 2, 0] = .01  # car-incompatible pedestrian query
    probabilities[0, 2, 8] = .7
    probabilities[0, 3, 0] = .6  # zero-overlap fallback
    probabilities[0, 4, 0] = .01  # projects to car, but predicts pedestrian
    probabilities[0, 4, 8] = .4
    probabilities[0, 5, 0] = .7  # unsupported car FP
    logits = torch.logit(probabilities).requires_grad_()
    overlap = torch.tensor([
        [.35, 0., 0.], [.60, 0., 0.], [.50, .70, 0.],
        [0., 0., 0.], [.40, 0., 0.], [0., 0., 0.]])
    radial = torch.zeros_like(overlap)
    tangent = torch.zeros_like(overlap)
    old_geometry = ray_rule_v2._camera_geometry
    ray_rule_v2._camera_geometry = lambda pred, target, meta: (
        overlap.to(pred.device), radial.to(pred.device), tangent.to(pred.device))
    try:
        results, originals, positives, duplicates, unsupported, stats, examples = (
            ray_rule_v2.prepare_rule_v2_layer(
                logits, encoded, [gt], [labels], [{}], SyntheticPETRAssigner(),
                v7.model.pts_bbox_head.rule_v2_cfg))
    finally:
        ray_rule_v2._camera_geometry = old_geometry

    assert int(results[0].gt_inds[0]) == 1  # rule car winner
    assert int(originals[0].gt_inds[1]) == 1  # PETR selects high-score duplicate
    assert int(results[0].gt_inds[2]) == 2  # pedestrian winner
    assert int(results[0].gt_inds[3]) == 3  # HM fallback for car GT 3
    assert (0, 0, 0) in positives and (0, 2, 8) in positives
    assert all(q != 3 for _, q, _ in positives)  # fallback is not certified
    assert (0, 1, 0) in duplicates
    assert (0, 4, 8) in unsupported  # wrong-class projected FP
    assert (0, 5, 0) in unsupported
    assert (0, 2, 0) not in positives  # class-incompatible car pair
    assert stats['changed_better'] >= 1
    assert examples and examples[0]['rule_pair_type'] == 'certified'
    empty_stats = ray_rule_v2.prepare_rule_v2_layer(
        logits, encoded, [gt[:0]], [labels[:0]], [{}],
        SyntheticPETRAssigner(), v7.model.pts_bbox_head.rule_v2_cfg)[5]
    assert set(stats) == set(empty_stats)  # stable MMCV log keys across ranks
    assert all('class_{}_changed'.format(c) in stats for c in range(10))

    loss, rank_stats = ray_rule_v2.rule_v2_rank_loss(
        logits, positives, duplicates, unsupported,
        v7.model.pts_bbox_head.rule_v2_cfg)
    assert torch.isfinite(loss) and rank_stats['ranking_pairs'] > 0
    loss.backward()
    assert logits.grad[0, 0, 0] < 0  # gradient descent raises valid car
    assert logits.grad[0, 1, 0] > 0  # lowers wrong-depth duplicate
    assert logits.grad[0, 2, 8] < 0  # raises valid pedestrian
    assert logits.grad[0, 4, 8] > 0  # lowers wrong-class pedestrian FP
    assert logits.grad[0, 5, 0] > 0  # lowers unsupported car FP
    assert logits.grad[0, 2, 0] == 0  # does not relabel pedestrian as car
    print('PASS: V6/V7 config parity, rule/HM assignment difference, fallback, '
          'class gate, duplicate/background roles, and score gradient signs')


if __name__ == '__main__':
    main()
