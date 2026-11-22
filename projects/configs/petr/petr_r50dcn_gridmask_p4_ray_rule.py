"""Fine-tune the trained 2-GPU PETR-R50 with opt-in ray rules for all classes."""

_base_ = ['./petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py']

# This first run keeps the base model's trainable parameters unchanged.
# Set model.parameter_freeze_cfg in a derived config to freeze selected modules.

model = dict(pts_bbox_head=dict(ray_rule_cfg=dict(
    min_projected_iou=0.15,
    owner_iou_margin=0.05,
    min_quality_gap=0.10,
    sigma_radial=2.0,
    sigma_tangential=1.0,
    power_radial=2.0,
    power_tangential=2.0,
    hm_weight=1.0,
    rank_weight=1.0,
    max_weight=3.0,
    preferred_group_weight=1.0,
    negative_group_weight=1.0,
    ranking_margin=0.5,
    pair_weight=0.25,
    loss_weight=0.05)))

load_from = (
    'results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/'
    'epoch_24.pth')
resume_from = None

# First run: one epoch to inspect the supervision before choosing weights.
optimizer = dict(lr=2e-5)
total_epochs = 1
runner = dict(max_epochs=total_epochs)
evaluation = dict(interval=1)
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule'

# Available only in this opt-in config. The hook writes a first-epoch JSON
# summary with case counts, mean loss values, and sampled decoder gradients.
custom_imports = dict(
    imports=['projects.mmdet3d_plugin.hooks.ray_rule_diagnostics'],
    allow_failed_imports=False)
custom_hooks = [dict(type='RayRuleDiagnosticsHook', max_epochs=1, grad_interval=50)]
