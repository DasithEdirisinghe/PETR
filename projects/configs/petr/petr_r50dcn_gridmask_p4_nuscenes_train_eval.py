"""PETR-R50 deterministic inference over the nuScenes training split.

This configuration is evaluation-only.  It reuses the normal PETR test
pipeline but points ``data.test`` at the training annotations.  Keeping this
separate prevents train diagnostics from overwriting validation settings.
"""

_base_ = './petr_r50dcn_gridmask_p4_nuscenes_local.py'

data_root = 'data/nuscenes/'

data = dict(
    test=dict(
        data_root=data_root,
        # Use the complete official train split.  The temporal train info file
        # contains only 25,433 of the 28,130 official train keyframes.
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        test_mode=True))

