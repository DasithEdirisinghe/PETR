"""V2: decoder/head-only ray-rule fine-tuning, rule weight 0.075.

Base: same original PETR-R50-P4 epoch-24 checkpoint and V1 fine-tuning setup:
      2 GPUs x 4 frames, one epoch, learning rate 2e-5, decoder/head-only.
Compared with V1: raise only ray_rule_cfg.loss_weight from 0.05 to 0.075 and
      write to a separate V2 work directory. All other settings are inherited.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v1.py']

model = dict(pts_bbox_head=dict(ray_rule_cfg=dict(loss_weight=0.075)))
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v2'
