"""PETR with EVA02-L, retaining PETR's 900-query head and decoder.

The EVA backbone's internal SimpleFeaturePyramid emits one stride-16 P4 map
with 256 channels. PETR's ResNet/CPFPN neck is not used in this variant.
"""

_base_ = './petr_r50dcn_gridmask_p4_nuscenes_local.py'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_root = '../RoboBEV/data/sets/nuscenes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
ida_aug_conf = dict(
    resize_lim=(0.47, 0.625),
    final_dim=(320, 800),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)

model = dict(
    img_backbone=dict(
        _delete_=True,
        type='EVAViT',
        img_size=320,
        patch_size=16,
        window_size=16,
        global_window_size=20,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        window_block_indexes=(
            list(range(0, 2)) + list(range(3, 5)) +
            list(range(6, 8)) + list(range(9, 11)) +
            list(range(12, 14)) + list(range(15, 17)) +
            list(range(18, 20)) + list(range(21, 23))),
        sim_fpn=dict(
            scale_factors=[1.0],
            in_channels=1024,
            out_channels=256,
            out_indices=[4]),
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cp=True,
        flash_attn=False),
    img_neck=None)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925], translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05], reverse_angle=True, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800),
         pts_scale_ratio=1, flip=False, transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names,
                  with_label=False),
             dict(type='Collect3D', keys=['img']),
         ]),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(data_root=data_root,
               ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
               pipeline=train_pipeline),
    val=dict(data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline, test_mode=True),
    test=dict(data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, test_mode=True))

# The bash runner scales this 8-GPU reference LR to its selected GPU count.
optimizer = dict(lr=2e-4)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic',
                        grad_clip=dict(max_norm=35, norm_type=2))
evaluation = dict(interval=4, pipeline=test_pipeline)
load_from = 'ckpts/petr_eva02_l_p4_backbone_init.pth'
resume_from = None
