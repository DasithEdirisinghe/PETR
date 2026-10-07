"""Base config for supervised R1 -> R1-f oracle-adapter diagnostics."""

_base_ = ["../petr_r50dcn_gridmask_p4_800x320_pccr_r1f.py"]

# Load the frozen R1 source detector. Override this on the command line when
# the best checkpoint is not latest.pth.
load_from = "./results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth"
resume_from = None

# Child configs set the mode.  A zero-initialized residual makes epoch zero
# exactly equivalent to the loaded source model.
model = dict(
    pts_bbox_head=dict(
        oracle_adapter=dict(
            mode="query",
            hidden_dims=256,
            calibration_dims=128,
            residual_scale=1.0,
            translation_scale=50.0,
        )))

# Only adapter parameters have requires_grad=True.  There is consequently no
# backbone LR multiplier in this diagnostic optimizer.
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1e-4,
    weight_decay=1e-4,
)
lr_config = dict(
    _delete_=True,
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=0.1,
    min_lr_ratio=0.01,
)

total_epochs = 20
runner = dict(type="EpochBasedRunner", max_epochs=20)
evaluation = dict(interval=5)
checkpoint_config = dict(interval=5, max_keep_ckpts=4)
find_unused_parameters = False

work_dir = "./experiments/oracle_adapters/output/R1_to_R1-f/query"
