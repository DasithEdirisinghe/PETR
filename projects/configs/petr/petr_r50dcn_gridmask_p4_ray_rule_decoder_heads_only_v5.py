"""V5: matched four-epoch decoder/head-only control without ray rules.

Base: original PETR-R50-P4 epoch-24 checkpoint, global batch eight, decoder
      and prediction heads trainable; same no-rule setup as V3.
Compared with V3: train for four epochs instead of one, validate every epoch,
      and write to a separate V5 directory.
Compared with V4: only the ray-rule loss and diagnostics hook are absent.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v3.py']

total_epochs = 4
runner = dict(max_epochs=total_epochs)
evaluation = dict(interval=1)
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v5'
