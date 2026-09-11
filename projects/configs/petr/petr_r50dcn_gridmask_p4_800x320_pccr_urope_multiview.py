"""PCCR PETR with URoPE and additive camera/row/column sine PE."""

_base_ = './petr_r50dcn_gridmask_p4_800x320_pccr_urope.py'

work_dir = (
    './results/'
    'petr_r50dcn_gridmask_p4_800x320_pccr_urope_multiview/R1')

model = dict(
    pts_bbox_head=dict(
        urope_with_multiview_pe=True))
