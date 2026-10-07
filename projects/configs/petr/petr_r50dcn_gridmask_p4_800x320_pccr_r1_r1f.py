"""Compute-matched PETR training on both PCCR R1 and R1-f.

Each child dataset keeps its own image paths and camera calibration. The two
rigs share the same training pipeline and are concatenated before sampling.
Validation and test remain on R1 by default; evaluate each rig separately.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_800x320_pccr.py']

classes = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'adult', 'child',
    'traffic_light', 'traffic_sign'
]
modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False,
    use_external=False)

# Match the vanilla R1/R1-f training transforms. Keep this in sync with the
# baseline config if its pipeline changes.
mixed_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -6.0, 51.2, 51.2, 6.0]),
    dict(type='ObjectNameFilter', classes=classes),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.5875, 0.78125),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=720,
            W=1280,
            rand_flip=True),
        training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=classes),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

r1_train = dict(
    type='PCCRDataset',
    data_root='data/pccr/R1/',
    ann_file='data/pccr/R1/R1_infos_train.pkl',
    pipeline=mixed_train_pipeline,
    classes=classes,
    modality=modality,
    test_mode=False,
    box_type_3d='LiDAR',
    with_velocity=False,
    use_valid_flag=True)
r1f_train = dict(
    r1_train,
    data_root='data/pccr/R1-f/',
    ann_file='data/pccr/R1-f/R1-f_infos_train.pkl')


data = dict(
    train=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[r1_train, r1f_train]))

# With similarly sized rig datasets, 60 concatenated epochs have about the
# same optimizer-step budget as 120 epochs on one rig.
total_epochs = 60
runner = dict(type='EpochBasedRunner', max_epochs=60)
work_dir = './results/petr_r50dcn_gridmask_p4_800x320_pccr/R1_R1f'
