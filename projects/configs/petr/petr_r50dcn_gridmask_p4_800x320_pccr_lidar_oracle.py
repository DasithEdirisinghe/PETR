"""PCCR PETR using current-scan LiDAR as an oracle image 3D PE.

The dataset modality deliberately remains camera-only so annotation filtering
is identical to the PETR baseline. LiDAR is loaded explicitly only to build
the experimental positional encoding.
"""

_base_ = './petr_r50dcn_gridmask_p4_800x320_pccr.py'

class_names = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'adult', 'child',
    'traffic_light', 'traffic_sign'
]
point_cloud_range = [-51.2, -51.2, -6.0, 51.2, 51.2, 6.0]
image_aug = dict(
    resize_lim=(0.5875, 0.78125),
    final_dim=(320, 800),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=720,
    W=1280,
    rand_flip=True)
image_norm = dict(
    mean=[103.53, 116.28, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)
load_points = dict(
    type='LoadPointsFromFile',
    coord_type='LIDAR',
    load_dim=5,
    use_dim=5,
    file_client_args=dict(backend='disk'))

train_pipeline = [
    load_points,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=image_aug,
        training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        transform_points=True,
        training=True),
    dict(type='NormalizeMultiviewImage', **image_norm),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points'])
]

test_pipeline = [
    load_points,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=image_aug,
        training=False),
    dict(type='NormalizeMultiviewImage', **image_norm),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'points'])
        ])
]

model = dict(
    pts_bbox_head=dict(
        with_position=False,
        with_multiview=True,
        position_embedding_mode='lidar_oracle',
        lidar_oracle_cfg=dict(
            x_num_feats=84,
            y_num_feats=84,
            z_num_feats=88,
            temperature=10000,
            interpolation='nearest_depth',
            fallback_depth=20.0,
            clamp=True,
            with_multiview_pe=True)))

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

evaluation = dict(pipeline=test_pipeline)

work_dir = (
    './results/'
    'petr_r50dcn_gridmask_p4_800x320_pccr_lidar_oracle/R1')
