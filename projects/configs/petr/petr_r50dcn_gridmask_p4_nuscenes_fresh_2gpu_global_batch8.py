"""Fresh PETR R50-P4 training on two GPUs with global batch size 8.

This matches the authors' 8-GPU x 1-sample training recipe's global batch
size by using 2 GPUs x 4 samples per GPU, so the original 2e-4 learning rate
is retained.
"""

_base_ = './petr_r50dcn_gridmask_p4_nuscenes_local.py'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

optimizer = dict(lr=2e-4)

# Train the detector from scratch while retaining the ImageNet-pretrained
# ResNet-50 initialization declared in the shared PETR base config.
load_from = None
resume_from = None

work_dir = (
    'results/'
    'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8')
