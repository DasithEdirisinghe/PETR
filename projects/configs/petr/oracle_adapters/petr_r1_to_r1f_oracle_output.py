_base_ = ["./petr_r1_to_r1f_oracle_base.py"]

model = dict(pts_bbox_head=dict(oracle_adapter=dict(mode="output")))
work_dir = "./experiments/oracle_adapters/output/R1_to_R1-f/output"
