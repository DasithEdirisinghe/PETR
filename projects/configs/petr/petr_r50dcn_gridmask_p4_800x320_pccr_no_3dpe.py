"""PCCR PETR ablation without PETR 3DPE or URoPE.

The standard multiview 2D positional encoding is retained. Only the 3D
geometry encoding supplied by PETR's learned 3DPE or URoPE is disabled.
"""

_base_ = './petr_r50dcn_gridmask_p4_800x320_pccr.py'

work_dir = './results/petr_r50dcn_gridmask_p4_800x320_pccr_no_3dpe/R1'

model = dict(
    pts_bbox_head=dict(
        with_position=False,
        position_embedding_mode='petr',
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            urope=False,
                            dropout=0.1),
                    ])))))
