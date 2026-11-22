"""V3: decoder/head-only control with no ray-rule loss.

Base: same original PETR-R50-P4 epoch-24 checkpoint and V2 fine-tuning setup:
      2 GPUs x 4 frames, one epoch, learning rate 2e-5, decoder/head-only.
Compared with V2: disable ray_rule_cfg and its diagnostics hook; retain the
      same data, seed, freeze policy, optimizer, and schedule. The work
      directory is separate. V2 versus V3 isolates adding the rule loss.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v2.py']

model = dict(pts_bbox_head=dict(ray_rule_cfg=None))
custom_hooks = []
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v3'
