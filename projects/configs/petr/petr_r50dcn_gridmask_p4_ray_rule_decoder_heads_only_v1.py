"""V1: decoder/head-only ray-rule fine-tuning, rule weight 0.05.

Base: petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8.py.
Compared with base: load its original epoch-24 checkpoint instead of training
      from scratch; use one epoch and learning rate 2e-5; freeze image
      backbone, neck, input projection, and learned 3DPE modules; add ray-rule
      loss at weight 0.05. Batch remains 2 GPUs x 4 frames.
Previous version: none. This aliases the existing decoder_heads_only run and
      deliberately retains its work directory for provenance.
"""

_base_ = ['./petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only.py']
