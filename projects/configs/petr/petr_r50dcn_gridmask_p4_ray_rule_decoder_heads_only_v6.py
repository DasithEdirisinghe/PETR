"""V6: one-epoch original PETR matching plus V2 certified ranking loss.

Base: V5 decoder/head-only control, original epoch-24 checkpoint, 2 A100 GPUs
with global batch eight. Compared with V5, run a one-epoch pilot and enable V2
geometric role analysis, ranking loss and assignment/gradient audit. Original
PETR matching, classification and regression supervision remain unchanged.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v5.py']

model = dict(pts_bbox_head=dict(rule_v2_cfg=dict(
    matching='original',
    enable_ranking=True,
    min_projected_iou=0.15,
    owner_iou_margin=0.05,
    max_certified_center=4.0,
    max_certified_tangent=2.0,
    min_class_score=0.02,
    class_logit_margin=1.5,
    duplicate_center_gap=0.5,
    unsupported_max_iou=0.05,
    unsupported_min_center=4.0,
    unsupported_min_score=0.05,
    hard_negative_max=64,
    positive_weight=1.0,
    negative_weight=1.0,
    pair_weight=1.0,
    ranking_margin=0.5,
    loss_weight=0.05)))

total_epochs = 1
runner = dict(max_epochs=total_epochs)
evaluation = dict(interval=1)
custom_hooks = [dict(
    type='RuleV2DiagnosticsHook',
    grad_interval=50,
    filename='rule_v2_audit_epoch_{epoch}.json')]
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v6'
