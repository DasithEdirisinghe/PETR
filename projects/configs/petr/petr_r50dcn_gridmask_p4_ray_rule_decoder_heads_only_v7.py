"""V7: one-epoch rule-guided assignment plus identical V6 ranking loss.

Base: V6, including checkpoint, data, frozen modules, schedule and rule labels.
Compared with V6, certified rule pairs receive PETR's ordinary classification
and box-regression targets; remaining GTs use original PETR cost as fallback.
The V2 ranking loss and its weight are unchanged.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v6.py']

model = dict(pts_bbox_head=dict(rule_v2_cfg=dict(matching='rule')))
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v7'
