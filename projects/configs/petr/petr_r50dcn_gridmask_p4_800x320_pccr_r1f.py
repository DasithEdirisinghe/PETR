"""Vanilla PETR trained and evaluated in-domain on the PCCR R1-f rig.

This inherits the established R1 vanilla baseline unchanged and overrides only
the dataset identity, split paths, and work directory. R1-f has the same eight
camera poses as R1 and changes their horizontal FOV from 65 to 90 degrees.
"""

_base_ = ["./petr_r50dcn_gridmask_p4_800x320_pccr.py"]

dataset_name = "R1-f"
data_root = "data/pccr/R1-f/"
train_ann_file = "data/pccr/R1-f/R1-f_infos_train.pkl"
val_ann_file = "data/pccr/R1-f/R1-f_infos_val.pkl"
test_ann_file = "data/pccr/R1-f/R1-f_infos_test.pkl"

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=train_ann_file,
    ),
    val=dict(
        data_root=data_root,
        ann_file=val_ann_file,
    ),
    test=dict(
        data_root=data_root,
        ann_file=test_ann_file,
    ),
)

work_dir = "./results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f"
