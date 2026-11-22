_base_ = './petr_eva02_l_p4_800x320_nuscenes.py'

data = dict(samples_per_gpu=4)
work_dir = 'results/petr_eva02_l_p4_800x320_nuscenes_global_batch_8'
