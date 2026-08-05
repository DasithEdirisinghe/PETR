# URoPE integration for PETR/PCCR

## What is integrated

PETR's original 3D positional encoding samples a dense frustum, converts all
sampled points to LiDAR coordinates, concatenates `3 * depth_num` absolute
coordinates, and learns a convolutional position encoder. This is implemented
by `PETRHead.position_embeding`.

URoPE is not an alternative coordinate tensor for that convolution. It is a
relative rotation of attention Q/K features. For the paper's 2D--3D case:

1. each image feature location is converted to a calibrated camera ray;
2. a fixed metric depth is assigned to each attention-head group;
3. the resulting point is transformed into the LiDAR frame;
4. the PETR object reference point is converted from normalized coordinates to
   the same LiDAR frame;
5. axial 3D RoPE is applied to projected Q and K before their dot product.

The implementation is in:

- `projects/mmdet3d_plugin/models/utils/urope.py`: geometry and 3D rotary math;
- `projects/mmdet3d_plugin/models/utils/petr_transformer.py`: URoPE-aware PETR
  cross-attention while retaining the original `nn.MultiheadAttention` weights;
- `projects/mmdet3d_plugin/models/dense_heads/petr_head.py`: configuration,
  query conversion, and one-time key geometry construction;
- `projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr_urope.py`: isolated
  experiment config that leaves the baseline config unchanged.

`position_embedding_mode` supports:

- `petr` (default): unchanged PETR behavior;
- `urope`: URoPE replaces additive PETR 3DPE;
- `hybrid`: PETR 3DPE and URoPE are both enabled for an ablation.

## Train

On the configured PBS cluster:

```bash
conda activate petr
python tools/smoke_test_urope.py

# On a GPU node with the PCCR R1 data linked, verify a real training batch.
python tools/smoke_test_urope.py --full-forward

qsub tools/train_pccr_r1_urope_2xa100.pbs
```

For an inexpensive integration check before the full run, override the runner
and output directory:

```bash
bash tools/dist_train.sh \
  projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr_urope.py 2 \
  --work-dir results/urope_smoke/R1 \
  --seed 0 \
  --cfg-options runner.max_epochs=1 checkpoint_config.interval=1
```

Train from scratch for the controlled comparison. A PETR checkpoint can still
initialize the shared attention/backbone weights, but removal of the learned
`position_encoder` means it is not a like-for-like resumed optimization run.

## Evaluate FoV/camera-rig robustness

Use the existing cross-rig evaluator with URoPE-specific paths:

```bash
CONFIG=projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr_urope.py \
CHECKPOINT=results/petr_r50dcn_gridmask_p4_800x320_pccr_urope/R1/latest.pth \
OUTPUT_ROOT=results/petr_r50dcn_gridmask_p4_800x320_pccr_urope/R1/cross_rig \
bash tools/eval_pccr_r1_all_rigs.sh
```

Compare both in-domain R1 and the generalization gap for every target rig:

```text
absolute_drop(rig) = mAP(R1) - mAP(rig)
relative_retention(rig) = mAP(rig) / mAP(R1)
URoPE_gain(rig) = mAP_URoPE(rig) - mAP_PETR(rig)
```

The main claim is supported only if URoPE improves average out-of-rig
retention, not merely R1 mAP. Report results separately for the controlled
variants (`R1-c10`, `R1-c6`, `R1-f`, `R1-r`, `R1-t`) and alternative layouts
(`R2`--`R9`). Use the same seed, schedule, augmentation, checkpoint-selection
rule, and evaluator for PETR and URoPE; ideally run three seeds.

## Recommended ablations

1. Replacement: `position_embedding_mode='urope'` (primary experiment).
2. Hybrid: `position_embedding_mode='hybrid'`, `with_position=True`.
3. Depth anchors: 1, 2, 4, and 8 with eight attention heads.
4. Spacing: uniform versus LID.
5. Intrinsic perturbation: focal scaling within and outside the training range.

The current config uses four uniform camera-z anchors over `[1.0, 61.2)` m,
shared by eight heads. This follows the paper's preferred four-anchor,
head-wise design while adapting its range to the PCCR detection volume.

## Paper/code caveats

The public URoPE repository implements its 2D--2D novel-view attention, not the
PETR experiment. The paper specifies the 2D--3D extension only at the method
level (skip projection and compare the 3D query to lifted image points). The
implementation here follows that equation directly.

The paper's PETR numbers use nuScenes, ResNet-50, 256x704 images, 900 queries,
24 epochs, batch size 16, and report 34.9 to 37.3 NDS and 30.9 to 32.2 mAP.
Those figures are evidence for the method, not expected PCCR target values.
