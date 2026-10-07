# Vanilla PETR: image cells, calibration, positional encoding and decoding

Diagram: [PNG](PETR_VANILLA_FEATURE_TO_DECODER.png) · [editable SVG](PETR_VANILLA_FEATURE_TO_DECODER.svg).

This traces the vanilla path in the local [PCCR baseline config](projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py): `with_position=True`, `with_multiview=True`, no URoPE and no LiDAR-oracle input. Architecture mechanisms are vanilla PETR; the numerical example uses this repository's R50-DCN/CPFPN configuration. Numbered sections correspond to the diagram. Matching symbols connect sections; arrows within sections show local computation.

## Symbols and dimensions

| Symbol | Meaning | Baseline/example |
| --- | --- | --- |
| B, N | Batch size, number of cameras | N=6 is an example, not a hard-coded requirement |
| Hi, Wi | Padded image height and width | 320, 800 |
| H, W | Selected feature-map height and width | 20, 50 |
| c, r, s | Zero-based camera, feature row, feature column | One cell index |
| C | Decoder / image-feature channel dimension | 256 |
| D | Candidate depths per cell | 64 |
| T | Object queries | 900 |
| L | Image tokens across all cameras, N·H·W | 6000 for six cameras |
| X, G, M, P | Content, ray PE, multiview PE, total image PE | Each token has C channels; P=G+M |
| R, E | Learned reference coordinates, encoded query position | R: T×3; E: T×256 before broadcast |

`[L]` means trainable parameters; `[F]` means a parameter-free formula. A parameter-free transform can still propagate gradients and produce changing outputs. Attention equations omit biases and dropout for clarity.

## 1. What is the feature that receives positional encoding?

The detector folds cameras into the batch dimension, processes the images with the same backbone and neck, and restores the camera dimension. The head selects feature level 0 and applies `input_proj`, a learned 1×1 convolution.

The vector `X[b,c,:,r,s]` contains 256 appearance features. It summarizes a CNN receptive field, not a single original image pixel or an exact, non-overlapping patch. PETR constructs a matching position vector `P[b,c,:,r,s]`. There is no separate XYZ assignment to every original image pixel, and no known surface depth attached to the feature cell.

The head also builds an image-validity mask from `img_shape` within `pad_shape`, then downsamples it to H×W using nearest interpolation. This mask later excludes padded tokens from cross-attention.

Code: [image feature extraction](projects/mmdet3d_plugin/models/detectors/petr3d.py), [head input projection and forward](projects/mmdet3d_plugin/models/dense_heads/petr_head.py).

## 2. Calibration and image-side 3DPE

### 2a. What calibration means

Using column-vector notation, a camera's extrinsic matrix maps a point in the common LiDAR coordinate frame to camera coordinates:

```text
X_camera = R_c X_common + t_c
E_c = [R_c  t_c; 0 0 0 1]                    (4×4)
K_c = [fx skew cx; 0 fy cy; 0 0 1]            (3×3)
P_c = Kbar_c E_c                              (4×4)
```

`Kbar_c` embeds K in a 4×4 identity. Intrinsics describe focal lengths and principal point; extrinsics describe the camera pose relative to the common frame. For a point with camera-z depth d, projection produces `[u*d, v*d, d, 1]`. Dividing the first two components by d yields image coordinates u,v.

The dataset stores this combined matrix as `lidar2img[c]`. Its separate `extrinsics` field uses a transposed storage convention, so the actual dataset expression is `intrinsics[c] @ extrinsics[c].T`, consistent with P above. Do not multiply the stored matrices as though both already used column-vector conventions.

**The name “LiDAR frame” identifies a coordinate system. Vanilla PETR does not need LiDAR point returns.** Its box coordinates and backprojected rays share that frame, potentially transformed by training-time 3D augmentation. It is not automatically a world-global/map coordinate frame.

### 2b. Calibration must follow augmentation

The image pipeline updates intrinsics with the resize/crop/flip affine transform and recomputes `lidar2img`. Virtual 3D rotation/scale also updates projection matrices by multiplying with the inverse 3D transform. Conceptually:

```text
P_updated = Abar_image · P_original · inverse(G_3D)
```

Backprojection uses this updated matrix from `img_metas`, so the feature locations and geometry agree. Ordinary right/bottom padding does not shift the principal point, but padded dimensions determine feature-to-image grid scaling and validity masking.

Code: [dataset calibration construction](projects/mmdet3d_plugin/datasets/pccr_dataset.py), [augmentation updates](projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py).

### 2c. One cell becomes 64 possible XYZ locations

For feature row r and column s, vanilla `position_embeding` uses:

```text
u = s * pad_width / W
v = r * pad_height / H
```

There is **no +0.5 cell-center offset in this vanilla implementation**. For H=20, W=50 and a 320×800 image, cell (r=7,s=12) maps to (u=192,v=112).

With `LID=True`, the nonuniform depth schedule is:

```text
d_k = depth_start + bin_size * k * (k + 1),   k=0,...,D-1
bin_size = (position_range[3] - depth_start) / (D * (D+1))
D=64; depth_start=1
```

These are candidate camera-z depths, not predicted depths, probabilities, or Euclidean distances from the camera. The configuration uses the x-maximum of `position_range` to set this schedule's upper scale; the last sample does not reach that endpoint.

For each depth, PETR computes:

```text
[x_k, y_k, z_k, 1]ᵀ = inverse(lidar2img[c]) @ [u*d_k, v*d_k, d_k, 1]ᵀ
```

With unaugmented rigid extrinsics, this is equivalently `X_common = R_c.T @ (d_k * inverse(K_c) @ [u,v,1] - t_c)`. Thus each camera has its own ray origin and direction. The 64 candidate points describe where the cell's observed content could lie; none is explicitly selected here.

### 2d. A ray becomes one learned 256D embedding

Each candidate XYZ is normalized per axis using `position_range` (not the different box-decoding `pc_range`). Coordinates outside this range are possible. `inverse_sigmoid` clamps values to avoid infinite logits.

The code packs depth-major triples into channels and applies:

```text
[x0,y0,z0, x1,y1,z1, ..., x63,y63,z63]        192 channels
    → inverse_sigmoid of normalized coordinates
    → Conv1×1(192,1024) → ReLU → Conv1×1(1024,256)
    → G[b,c,:,r,s]                            256 channels
```

These convolutions act as a shared per-cell MLP: they mix coordinate/depth channels, not neighboring cells. The output is learned to be useful for detection. It is not an explicit depth estimate and does not guarantee recovery of a physical surface position. **This image-side PETR 3DPE does not use the query-side sine encoding.**

Implementation nuance: `position_embeding` also computes an out-of-range geometry mask, but the vanilla head discards the returned mask. The decoder receives the image-padding mask, not that geometry mask.

Code: [`position_embeding`, `position_encoder`](projects/mmdet3d_plugin/models/dense_heads/petr_head.py).

## 3. Multiview encoding and token alignment

Independently, `SinePositionalEncoding3D(masks)` generates camera-index, row and column encodings. Despite the class name, its three axes are **camera, row, column—not metric XYZ**.

With no padding, ignoring epsilon, the phases are `2π(c+1)/N`, `2π(r+1)/H`, and `2π(s+1)/W`. With padding, the implementation uses cumulative counts of valid entries and mask-dependent denominators. Each axis gets 128 channels (64 sine/cosine pairs):

```text
S128(camera) ⊕ S128(row) ⊕ S128(column)     384
    → Conv1×1(384,1024) → ReLU → Conv1×1(1024,256)
    → M[b,c,:,r,s]                          256
P = G + M                                  256
```

Camera index distinguishes ordered views but does not itself encode calibration, camera orientation or physical location. Those enter through G. There are six different camera-axis codes in a six-camera example, not six separate learned PE networks.

X and P are passed separately into the transformer and flattened in identical order:

```text
grid [B,N,C,H,W] → permute [N,H,W,B,C] → reshape [L,B,C]
j = (c*H + r)*W + s
```

Thus position token P[j] describes exactly the same camera cell as content token X[j]. For c=2,r=7,s=12 with H=20,W=50, j=2362. Six cameras produce 6000 tokens, with one global attention sequence per batch item.

Code: [multiview encoding](projects/mmdet3d_plugin/models/utils/positional_encoding.py), [token flattening and decoder call](projects/mmdet3d_plugin/models/utils/petr_transformer.py).

## 4. Query position is a separate branch

900 learned reference points R define object-query locations. `pos2posemb3d` scales coordinates by 2π and uses fixed sine/cosine frequencies, returning 384 channels concatenated in **y,x,z** order. The learned `query_embedding` MLP maps 384→256→ReLU→256, producing E. The 256 output matches decoder channels; it is not three decoded spatial coordinates.

Fixed frequencies do not mean constant outputs: changing R changes the sine values. Gradients flow through that transform into R. R is initialized in [0,1]; the raw embedding weights are not explicitly sigmoid-constrained before this sine calculation.

Initial query content is zero. Query PE E and reference R are reused across all six decoder layers in this vanilla path. Decoder layers update query **content**, not the stored reference points. Optimizer updates can change R and the MLP weights between training steps, so the next forward pass recomputes a different E.

Code: [`pos2posemb3d`, `reference_points`, `query_embedding`](projects/mmdet3d_plugin/models/dense_heads/petr_head.py).

## 5. Where PE actually affects attention

Each decoder layer applies self-attention, norm, cross-attention, norm, FFN, norm, with residual connections around attention and FFN blocks. Query self-attention adds E to Q/K inputs, not V. Let u be its normalized output. Cross-attention then uses:

```text
Q = Wq(u + E)
K = Wk(X + P) = Wk(X + G + M)
V = Wv(X)
```

After splitting into eight heads, each head has 32 channels. For a particular query i:

```text
A[i,j] = softmax over j of (Q[i] dot K[j] / sqrt(32) + padding_mask[j])
o[i] = sum over j of A[i,j] * V[j]
```

The softmax spans **all L image tokens, across all cameras**. PETR does not first choose a camera or hard-project each object query into a particular image cell. Position and content jointly determine attention scores. The 8 weighted-value outputs are concatenated, projected, and used to update query content via residual/norm/FFN operations.

The memory X and image PE P are reused by every layer; the queries change as they accumulate information. There is no active transformer encoder in this baseline path.

For visualization, an actual attention row `A[b,head,i,:]` can be reshaped into `[N,H,W]` and overlaid on the camera images. This is different from a cosine-similarity map of G or P: a PE similarity map measures embedding geometry, not what a decoder query actually attended to. The current attention wrapper does not expose the attention weights as its normal output; capturing them requires instrumentation.

Code: [`PETRTransformer.forward`, `PETRMultiheadAttention.forward`](projects/mmdet3d_plugin/models/utils/petr_transformer.py).

## 6. From query features to 3D detections

Each layer's query features feed class and box-regression heads. Center residuals are added to the inverse-sigmoid **original reference coordinates**, passed through sigmoid, then rescaled by `pc_range` to metric XYZ. In the PCCR eight-channel box code, x,y are slots 0,1 and z is slot 4; they are not stored as three consecutive center channels. Other channels encode log dimensions and sine/cosine yaw.

Training applies matching and detection losses to decoder outputs, with gradients reaching the backbone, both image PE adapters, query MLP and learned reference points. At inference the final layer is decoded by `NMSFreeCoder`: sigmoid class scores, top-K query/class pairs, box decoding and range filtering (max_num=300 in this config). Predicted centers are not fed back into the next decoder layer to rebuild E.

Code: [box/class heads and center offsets](projects/mmdet3d_plugin/models/dense_heads/petr_head.py), [NMS-free decoding](projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py).

## Regenerate the figure

From the PETR repository root:

```bash
python3 tools/analysis_tools/draw_petr_vanilla_flow.py
rsvg-convert PETR_VANILLA_FEATURE_TO_DECODER.svg -o PETR_VANILLA_FEATURE_TO_DECODER.png
```

Only documentation and the diagram generator are added; model, config, training and inference behavior are unchanged.
