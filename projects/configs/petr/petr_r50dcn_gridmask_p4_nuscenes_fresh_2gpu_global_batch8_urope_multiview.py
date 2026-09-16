"""Train PETR on nuScenes with URoPE and multiview PE, without learned 3DPE.

This is the nuScenes counterpart of the PCC-R URoPE+multiview experiment.
URoPE injects explicit 3D query/key geometry inside decoder cross-attention,
while the additive camera/row/column sine encoding is retained as key PE.
PETR's learned frustum position encoder is disabled.
"""

_base_ = './petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py'

work_dir = (
    'results/'
    'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_'
    'urope_multiview')

model = dict(
    pts_bbox_head=dict(
        # Disable PETR's learned absolute frustum 3DPE.
        with_position=False,
        position_embedding_mode='urope',
        # Keep additive camera/row/column sine positional encoding.
        urope_with_multiview_pe=True,
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
