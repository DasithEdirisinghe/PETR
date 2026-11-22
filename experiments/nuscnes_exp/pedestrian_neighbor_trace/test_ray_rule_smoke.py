"""Synthetic forward/backward smoke test for opt-in ray-rule supervision.

Run from PETR repository root in its training environment:
    python experiments/nuscnes_exp/pedestrian_neighbor_trace/test_ray_rule_smoke.py
"""

from pathlib import Path
import sys

import torch
from mmcv import Config
from mmcv.runner import HOOKS

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from projects.mmdet3d_plugin.models.dense_heads.ray_rule_loss import (  # noqa: E402
    ray_rule_layer_loss,
)
from projects.mmdet3d_plugin.models.utils.parameter_freeze import (  # noqa: E402
    configure_parameter_freeze, enforce_frozen_module_eval,
)


class AssignedInferior:
    """Stand-in for HM: it selects the worse-depth query in each frame."""

    def assign(self, bbox_pred, cls_score, gt_bboxes, gt_labels):
        class Result:
            pass

        result = Result()
        result.gt_inds = torch.tensor([0, 1, 0], device=bbox_pred.device)
        return result


def encoded_box(x):
    # PETR order: x, y, log(w), log(l), z, log(h), sin(yaw), cos(yaw), vx, vy.
    return torch.tensor([x, 0., -0.22314355, -0.22314355, 0.,
                         0.58778666, 0., 1., 0., 0.])


def main():
    repo_root = Path(__file__).resolve().parents[3]
    base_config = (repo_root / 'projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py').read_text()
    rule_config = (repo_root / 'projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule.py').read_text()
    assert 'ray_rule_cfg' not in base_config
    assert 'ray_rule_cfg' in rule_config
    assert 'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py' in rule_config
    assert 'epoch_24.pth' in rule_config
    parsed_config = Config.fromfile(str(
        repo_root / 'projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule.py'))
    assert parsed_config.data.samples_per_gpu == 4
    assert parsed_config.runner.max_epochs == 1
    assert HOOKS.get('RayRuleDiagnosticsHook') is not None
    frozen_config = Config.fromfile(str(repo_root /
        'projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule_frozen_backbone.py'))
    assert frozen_config.model.parameter_freeze_cfg.freeze_modules == ['img_backbone']
    assert frozen_config.load_from == parsed_config.load_from
    assert frozen_config.work_dir != parsed_config.work_dir
    decoder_config = Config.fromfile(str(repo_root /
        'projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only.py'))
    assert decoder_config.model.parameter_freeze_cfg.freeze_modules == [
        'img_backbone', 'img_neck', 'pts_bbox_head.input_proj',
        'pts_bbox_head.position_encoder', 'pts_bbox_head.adapt_pos3d']
    assert decoder_config.load_from == parsed_config.load_from
    assert decoder_config.work_dir not in (parsed_config.work_dir, frozen_config.work_dir)
    config_dir = repo_root / 'projects/configs/petr'
    v1 = Config.fromfile(str(config_dir /
        'petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v1.py'))
    v2 = Config.fromfile(str(config_dir /
        'petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v2.py'))
    v3 = Config.fromfile(str(config_dir /
        'petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v3.py'))
    assert v1.model.pts_bbox_head.ray_rule_cfg.loss_weight == 0.05
    assert v2.model.pts_bbox_head.ray_rule_cfg.loss_weight == 0.075
    assert v3.model.pts_bbox_head.ray_rule_cfg is None
    assert v3.custom_hooks == []
    assert v1.model.parameter_freeze_cfg == v2.model.parameter_freeze_cfg
    assert v2.model.parameter_freeze_cfg == v3.model.parameter_freeze_cfg
    assert v1.load_from == v2.load_from == v3.load_from
    assert v1.optimizer.lr == v2.optimizer.lr == v3.optimizer.lr
    assert v1.runner.max_epochs == v2.runner.max_epochs == v3.runner.max_epochs == 1
    assert len({v1.work_dir, v2.work_dir, v3.work_dir}) == 3
    v4 = Config.fromfile(str(config_dir /
        'petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v4.py'))
    v5 = Config.fromfile(str(config_dir /
        'petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v5.py'))
    assert v4.runner.max_epochs == v5.runner.max_epochs == 4
    assert v4.model.pts_bbox_head.ray_rule_cfg.loss_weight == 0.075
    assert v5.model.pts_bbox_head.ray_rule_cfg is None
    assert v5.custom_hooks == []
    assert v4.model.parameter_freeze_cfg == v5.model.parameter_freeze_cfg
    assert v4.load_from == v5.load_from == v2.load_from
    assert v4.optimizer.lr == v5.optimizer.lr == v2.optimizer.lr
    assert v4.data.samples_per_gpu == v5.data.samples_per_gpu == 4
    assert v4.custom_hooks[0].filename == 'ray_rule_diagnostics_epoch_{epoch}.json'

    # Exercise module freezing without constructing the full PETR detector.
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
            self.neck = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 2)

    tiny = TinyModel()
    tiny.backbone[1].weight.requires_grad_(False)
    state = configure_parameter_freeze(tiny, dict(
        freeze_modules=['backbone'], unfreeze_modules=['backbone.0']))
    assert tiny.backbone[0].weight.requires_grad
    assert not tiny.backbone[1].weight.requires_grad
    assert not tiny.backbone[1].bias.requires_grad
    assert tiny.neck.weight.requires_grad and tiny.head.weight.requires_grad
    tiny.train()
    enforce_frozen_module_eval(tiny, state)
    assert tiny.backbone[0].training and not tiny.backbone[1].training
    assert tiny.neck.training and tiny.head.training
    try:
        configure_parameter_freeze(tiny, dict(freeze_modules=['missing']))
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown freeze module was accepted')
    # Camera looks along +LiDAR-x. A depth error retains image overlap.
    lidar2img = torch.tensor([
        [0., 50., 0., 100.],
        [0., 0., -50., 100.],
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
    ])
    metas = [{'lidar2img': [lidar2img], 'img_shape': [(200, 200, 3)]}
             for _ in range(3)]
    gt_boxes = [torch.tensor([[10., 0., 0., 0.8, 0.8, 1.8, 0.]]),
                torch.tensor([[12., 0., 0., 0.8, 0.8, 1.8, 0.]]),
                torch.tensor([[14., 0., 0., 0.8, 0.8, 1.8, 0.]])]
    gt_labels = [torch.tensor([8]), torch.tensor([8]), torch.tensor([0])]
    boxes = torch.stack([
        torch.stack([encoded_box(10.), encoded_box(11.5), encoded_box(30.)]),
        torch.stack([encoded_box(12.), encoded_box(13.5), encoded_box(30.)]),
        torch.stack([encoded_box(14.), encoded_box(15.5), encoded_box(30.)]),
    ])
    logits = torch.full((3, 3, 10), -5.)
    # In both classes the poorer depth candidate initially outranks the winner.
    logits[:2, 0, 8] = -1.
    logits[:2, 1, 8] = 2.
    logits[2, 0, 0] = -1.
    logits[2, 1, 0] = 2.
    logits.requires_grad_()
    cfg = dict(min_projected_iou=0.15,
               owner_iou_margin=0.05, min_quality_gap=0.10,
               sigma_radial=2.0, sigma_tangential=1.0,
               power_radial=2.0, power_tangential=2.0,
               hm_weight=1.0, rank_weight=1.0, max_weight=3.0,
               pair_weight=0.25, ranking_margin=0.5,
               collect_diagnostics=True)
    loss, stats = ray_rule_layer_loss(
        logits, boxes, gt_boxes, gt_labels, metas, AssignedInferior(), cfg)
    assert torch.isfinite(loss), loss
    assert stats['preferred'] == 3, stats
    assert stats['inferior'] == 3, stats
    assert stats['hm_conflicts'] == 6, stats
    assert stats['rank_violations'] > 0, stats
    assert stats['radial_inferior'] == 3, stats
    assert stats['tangential_inferior'] == 0, stats
    assert stats['class_8_preferred_count'] == 2, stats
    assert stats['class_0_preferred_count'] == 1, stats
    assert stats['preferred_bce_sum'] > 0, stats
    assert stats['inferior_conflict_bce_sum'] > 0, stats
    assert stats['pair_penalty_sum'] > 0, stats
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:2, 0, 8] < 0).all(), logits.grad[:2, 0, 8]
    assert (logits.grad[:2, 1, 8] > 0).all(), logits.grad[:2, 1, 8]
    assert logits.grad[2, 0, 0] < 0, logits.grad[2, 0, 0]
    assert logits.grad[2, 1, 0] > 0, logits.grad[2, 1, 0]
    expected_nonzero = torch.zeros_like(logits.grad, dtype=torch.bool)
    expected_nonzero[:2, :2, 8] = True
    expected_nonzero[2, :2, 0] = True
    assert (logits.grad[~expected_nonzero] == 0).all()
    print('PASS: opt-in config, finite loss, three winners and three inferiors, HM/rank conflicts,')
    print('      pedestrian and car score gradients, and no unrelated gradients')
    print('loss:', float(loss.detach()), 'stats:', stats)


if __name__ == '__main__':
    main()
