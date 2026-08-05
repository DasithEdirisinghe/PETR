"""PCCR PETR with URoPE replacing PETR's learned absolute 3DPE."""

_base_ = './petr_r50dcn_gridmask_p4_800x320_pccr.py'

work_dir = './results/petr_r50dcn_gridmask_p4_800x320_pccr_urope/R1'

model = dict(
    pts_bbox_head=dict(
        with_position=False,
        position_embedding_mode='urope',
        urope_cfg=dict(
            depth_num=4,
            min_depth=1.0,
            max_depth=61.2,
            lid=False),
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
                            urope=True,
                            urope_freq_base=100.0,
                            urope_freq_scale=1.0,
                            dropout=0.1),
                    ])))))
