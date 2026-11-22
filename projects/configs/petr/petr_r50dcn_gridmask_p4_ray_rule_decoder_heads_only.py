"""One-epoch ray-rule fine-tuning of decoder, queries, and prediction heads.

All image feature extraction and learned 3D position embedding modules are
frozen. This starts from the same original epoch-24 checkpoint as the other
ray-rule variants and writes to a separate directory.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule.py']

model = dict(parameter_freeze_cfg=dict(
    freeze_modules=[
        'img_backbone',
        'img_neck',
        'pts_bbox_head.input_proj',
        'pts_bbox_head.position_encoder',
        'pts_bbox_head.adapt_pos3d',
    ],
    unfreeze_modules=[]))

# The unfrozen learned groups are pts_bbox_head.reference_points,
# query_embedding, transformer.decoder, cls_branches, and reg_branches.
work_dir = 'results/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only'
