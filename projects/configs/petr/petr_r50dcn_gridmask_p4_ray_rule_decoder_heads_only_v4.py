"""V4: four-epoch decoder/head-only ray-rule run at weight 0.075.

Base: original PETR-R50-P4 epoch-24 checkpoint, global batch eight, decoder
      and prediction heads trainable; same rule and weight as V2.
Compared with V2: train for four epochs instead of one, validate every epoch,
      and write to a separate V4 directory. No rule parameters change.
Compared with V3 control: the ray-rule loss remains enabled.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v2.py']

total_epochs = 4
runner = dict(max_epochs=total_epochs)
evaluation = dict(interval=1)
custom_hooks = [dict(
    type='RayRuleDiagnosticsHook',
    max_epochs=total_epochs,
    grad_interval=50,
    filename='ray_rule_diagnostics_epoch_{epoch}.json')]
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v4'
