"""Fresh two-GPU nuScenes training for vanilla PETR R50-P4.

The official recipe uses 8 GPUs × 1 sample/GPU with lr=2e-4. This config uses
2 GPUs × 1 sample/GPU and linearly scales the learning rate to 5e-5.
"""

_base_ = './petr_r50dcn_gridmask_p4_nuscenes_local.py'

optimizer = dict(lr=5e-5)

# Fresh PETR detector training. The ResNet-50 ImageNet initialization declared
# by the base model config is intentionally retained.
load_from = None
resume_from = None

work_dir = 'results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu'

