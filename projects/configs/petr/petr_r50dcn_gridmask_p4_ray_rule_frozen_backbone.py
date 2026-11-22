"""One-epoch ray-rule fine-tuning with the image backbone frozen.

This starts from the same original epoch-24 checkpoint as the all-trainable
ray-rule run, so their results can be compared directly.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule.py']

# Exact names from Petr3D.named_modules(). Descendant names may be placed in
# unfreeze_modules to restore their original trainability after parent freeze.
# Existing architecture-frozen parameters, such as backbone BatchNorm weights,
# remain frozen when restored.
model = dict(parameter_freeze_cfg=dict(
    freeze_modules=['img_backbone'],
    unfreeze_modules=[]))

work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_frozen_backbone'
