# PETR architecture: symbols, legend and teardown guide

[PNG diagram](PETR_ARCHITECTURE_ABLATION_MAP.png) · [Editable SVG](PETR_ARCHITECTURE_ABLATION_MAP.svg)

The diagram uses short labels and dimension formulas. Matching symbols connect
panels; this document holds the longer explanations. `[L]` marks trainable
parameters and `[F]` marks a parameter-free operation. **Parameter-free does
not mean that the output is constant.**

## Symbols

| Symbol | Meaning |
|---|---|
| B, N | Batch size and camera count |
| Hi, Wi; Hf, Wf | Input-image dimensions; selected feature-map dimensions |
| C = 256 | Transformer embedding width |
| T = 900 | Number of object queries |
| D = 64 | PETR depth candidates per image ray; 3D in the formula means 3 × D |
| L = N Hf Wf | Total image tokens; 6000 for six cameras and a 20×50 feature grid |
| a = 8; dh = C/a = 32 | Attention heads; channels per head |
| X, P | Image appearance and image position, each [B,L,C] |
| G, H | Grid PE and geometry PE; P = G + H |
| Rₜ | Trainable reference coordinates [T,3] at optimizer step t |
| S, Mθₜ | Parameter-free sine/cosine transform; learned query MLP |
| Eₜ = Mθₜ(S(Rₜ)) | Query PE [T,C], broadcast to [B,T,C] |
| hℓ, uℓ, vℓ | Content after a decoder layer, self-attention/norm, and cross-attention/norm |
| Pcam | Augmented 4×4 lidar2img matrix; distinct from image PE P |
| ℛq, ℛk | URoPE rotations from query/key 3D coordinates |
| ⊕; + | Channel concatenation; elementwise addition |
| σ; logitε | Sigmoid; epsilon-clamped inverse sigmoid |

Shapes are shown batch-first for readability. The actual transformer passes
image memory as [L,B,C] and queries as [T,B,C]. Projection biases, dropout and
layout transposes are omitted; Wq/Wk/Wv/Wo denote affine projections.

## Section C: fixed function, changing embedding

The old label “fixed sine encoding” referred to the frequencies. The revised
label is “Sine / cosine transform S [F]”. For one scalar coordinate s:

```text
ωk = 10000^(-2k/128), k = 0…63
S128(s)[2k]   = sin(2π s ωk)
S128(s)[2k+1] = cos(2π s ωk)

S(Rₜ) = S128(yₜ) ⊕ S128(xₜ) ⊕ S128(zₜ)  → 384 channels
Eₜ = Linear256→256(ReLU(Linear384→256(S(Rₜ))))
```

S has no learned parameters, but gradients pass through it. After backward
and an optimizer update, both Rₜ and the query MLP weights θₜ can change.
The next forward pass recomputes S(Rₜ₊₁), so the sinusoidal values generally
change. E also changes through the updated MLP. Within a forward pass E is
computed once and reused across all six layers. At ordinary evaluation,
stored references and MLP weights produce the same E across samples.

References are initialized uniformly in [0,1]. The code directly uses
`reference_points.weight` before S, without a sigmoid constraint; this range
is an initialization convention, not a guaranteed bound during training.
The centre-regression path uses epsilon-clamped inverse sigmoid separately.

Code: [sinusoidal function and query MLP](projects/mmdet3d_plugin/models/dense_heads/petr_head.py).

## Sections A and B: image content and position

The head selects feature level 0 from CPFPN and applies a 1×1 projection.
Calibration is used to build geometry; it is not supplied to ResNet.

Grid PE G selects conventional per-camera 2D sine PE (row + column) or MPE
(camera index + row + column). MPE's fixed sine transform yields 384 channels
and its learned adapter maps 384→1024→256 with ReLU. For 2D PE combined with
PETR 3DPE, the adapter maps 256→1024→256. The 2D-only branch bypasses this
adapter. MPE is not metric XYZ despite the class name `SinePositionalEncoding3D`.

PETR geometry H takes feature-cell coordinates and 64 candidate camera depths,
back-projects them with inverse(lidar2img), normalizes XYZ by `position_range`,
applies epsilon-clamped inverse sigmoid, flattens to 192 channels, then uses
learned convolutions 192→1024→256 with ReLU. It encodes the calibrated ray's
candidates without assigning attention weights to individual depths.

Oracle geometry H instead projects the current LiDAR scan into each camera.
The closest positive-depth return in a cell supplies its exact XYZ. Empty
cells copy the nearest occupied cell's depth and back-project through their
own centre ray. The utility filters points by `position_range`; cameras with
no valid returns use fallback depth 20. XYZ is normalized by `position_range`
and clamped to [0,1] in the current config. Fixed sine blocks (y84,x84,z88)
produce 256 channels without a learned geometry adapter. LiDAR is required at
both training and inference.

Query references correspond to `pc_range`, while image geometry normalization
uses `position_range`; those ranges differ in the PCCR config. Matching channel
widths do not imply identical coordinate normalization or encoding functions.

Code: [PETR head](projects/mmdet3d_plugin/models/dense_heads/petr_head.py),
[MPE](projects/mmdet3d_plugin/models/utils/positional_encoding.py),
[Oracle geometry](projects/mmdet3d_plugin/models/utils/lidar_oracle_pe.py).

## Sections D and E: attention, prediction and supervision

The layer order is self-attention, norm, cross-attention, norm, FFN, norm.
Attention/FFN wrappers include residual additions, represented as Add + Norm.
Query PE is added in both attention calls. Image PE enters the cross-attention
key; values contain image appearance alone.

URoPE is inside cross-attention, after Q/K projection and before the dot
product. Its query coordinates come from references converted to metres;
key coordinates come from calibrated rays at head-specific fixed depths.
These are not the Oracle LiDAR coordinates. Without URoPE, Q/K bypass rotation.

Class and regression branches operate on every decoder layer's output.
The box-centre prediction reuses R from C:

```text
ĉℓ = sigmoid(inverse_sigmoid(R) + Δℓ)
cℓ = pc_range_min + ĉℓ × (pc_range_max − pc_range_min)
```

PCCR's eight regression values encode centre, three log-size values and
sine/cosine yaw, without velocity. Final decoding uses the last layer's
sigmoid class scores, top-K query/class entries, box denormalization and
range filtering. Section E groups this flow: class scores bypass the centre
calculation, rather than being transformed with box coordinates.

Training applies Hungarian matching and classification/regression losses to
layer predictions. Gradients reach R and the query MLP. Predicted centres
do not replace reference points inside the decoder.

Code: [transformer](projects/mmdet3d_plugin/models/utils/petr_transformer.py),
[prediction and losses](projects/mmdet3d_plugin/models/dense_heads/petr_head.py),
[coder](projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py).

## Component legend for current experiments

| Variant | 2D PE | MPE | PETR image 3DPE | LiDAR PE | URoPE | Query PE |
|---|---|---|---|---|---|---|
| Baseline | — | ✓ | ✓ | — | — | ✓ |
| No image 3DPE | — | ✓ | — | — | — | ✓ |
| Pure URoPE | — | — | — | — | ✓ | ✓ |
| URoPE + MPE | — | ✓ | — | — | ✓ | ✓ |
| Hybrid | — | ✓ | ✓ | — | ✓ | ✓ |
| LiDAR Oracle | — | ✓ | — | ✓ | — | ✓ |

Insertion points: change G for grid encoding, H for image geometry, S or Mθ
for query PE, or projected Q/K for relative attention geometry. Removing E
and removing R as a box anchor are separate ablations.

## Regeneration

```bash
python tools/analysis_tools/draw_petr_architecture.py
rsvg-convert PETR_ARCHITECTURE_ABLATION_MAP.svg -o PETR_ARCHITECTURE_ABLATION_MAP.png
```

The generator uses the Python standard library. SVG elements remain editable.

This diagram follows the current PCCR PETR implementation and separates four
places that are easy to conflate:

1. Image appearance features from ResNet and CPFPN.
2. Image-grid positional encoding: conventional per-camera 2D PE **or** multiview PE
   (MPE).
3. Image-side geometry: vanilla PETR ray 3DPE, LiDAR Oracle PE, or neither.
4. Query-side 3D reference embedding, which remains present in all current
   variants and is also reused to anchor box-centre regression.

## Vanilla PETR configuration mapping

| `with_position` | `with_multiview` | Positional encoder type | Image-side PE |
|---:|---:|---|---|
| `False` | `False` | `SinePositionalEncoding` | 2D PE only |
| `False` | `True` | `SinePositionalEncoding3D` | MPE only |
| `True` | `False` | `SinePositionalEncoding` | PETR 3DPE + 2D PE |
| `True` | `True` | `SinePositionalEncoding3D` | PETR 3DPE + MPE |

Simply changing `with_multiview=False` while leaving
`SinePositionalEncoding3D` configured is not a valid conventional-2D-PE
ablation: the false branch supplies a three-dimensional `[B,H,W]` mask, while
`SinePositionalEncoding3D` expects `[B,N,H,W]`. Use MMDetection's
`SinePositionalEncoding` for that branch.

## Cross-attention boundary

The effective vanilla cross-attention inputs are:

```text
Q = Wq(query content + query 3D positional embedding)
K = Wk(image appearance + image positional embedding)
V = Wv(image appearance)
```

URoPE is different from an additive PE: it rotates projected Q/K using their
3D coordinates inside cross-attention. The LiDAR Oracle experiment remains an
additive image-key PE and does not replace image appearance values.
