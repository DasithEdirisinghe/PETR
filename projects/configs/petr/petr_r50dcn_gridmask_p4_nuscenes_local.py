"""Official PETR R50-P4 checkpoint with the workspace nuScenes dataset."""

_base_ = './petr_r50dcn_gridmask_p4.py'

data_root = 'data/nuscenes/'
train_ann_file = data_root + 'nuscenes_infos_temporal_train.pkl'
val_ann_file = data_root + 'nuscenes_infos_temporal_val.pkl'

data = dict(
    train=dict(data_root=data_root, ann_file=train_ann_file),
    val=dict(data_root=data_root, ann_file=val_ann_file, test_mode=True),
    test=dict(data_root=data_root, ann_file=val_ann_file, test_mode=True))
