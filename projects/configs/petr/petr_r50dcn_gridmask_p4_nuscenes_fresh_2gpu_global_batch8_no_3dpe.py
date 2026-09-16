"""nuScenes PETR R50-P4 ablation: no learned 3DPE, global batch size 8.

All training and evaluation settings are inherited from the fresh two-GPU
global-batch-8 run. Standard multiview 2D positional encoding is retained.
"""

_base_ = './petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py'

model = dict(pts_bbox_head=dict(with_position=False))

work_dir = (
    'results/'
    'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_no_3dpe')
