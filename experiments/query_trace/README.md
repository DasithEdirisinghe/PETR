# PETR matched-query trace

This experiment traces one clean and one poorly localized final-layer
Hungarian GT/query match through every vanilla PETR decoder layer. It does not
change PETR's normal model implementation.

## Precheck and explicit query selection

First list all final-layer Hungarian-matched queries for the selected frame:

```bash
python experiments/query_trace/trace_query.py \
  --scene-name scene-0101 \
  --scene-frame-index 28 \
  --score-threshold 0.35 \
  --classes car pedestrian truck bus \
  --list-queries
```

This runs the baseline forward pass and writes `query_candidates.csv` and
`query_candidates.json` below the sample-token output directory. Each record
contains query index, GT and predicted classes, class correctness, matched
class confidence, threshold status, and final BEV center error.

Then trace one or more exact query indices:

```bash
python experiments/query_trace/trace_query.py \
  --scene-name scene-0101 \
  --scene-frame-index 28 \
  --classes car pedestrian truck bus \
  --query-indices 127 604 \
  --plot-bev-rays
```

Outputs are grouped under directories such as `query_0127_car/`, making the
query identity and matched class explicit. If `--query-indices` is omitted,
the previous automatic clean/poor selection policy remains available, but its
outputs also use query-number directories.

It reads the sample token from the keyframe selector by default:

```bash
python experiments/query_trace/trace_query.py
```

Prefer specifying the selected scene and zero-based keyframe index explicitly:

```bash
python experiments/query_trace/trace_query.py \
  --scene-name scene-0018 \
  --scene-frame-index 0 \
  --score-threshold 0.35
```

Replace `0` with the `scene_frame_index` reported by the keyframe selector.

An explicit sample can also be used:

```bash
python experiments/query_trace/trace_query.py \
  --sample-token f6486e3c8a90429dbf1d3d66a91239d3 \
  --score-threshold 0.35
```

Restrict both clean and poorly localized query selection to chosen classes:

```bash
python experiments/query_trace/trace_query.py \
  --score-threshold 0.35 \
  --classes car pedestrian
```

Class names must match the checkpoint's nuScenes classes. If `--classes` is
omitted, all classes are eligible.

Outputs are written below `experiments/query_trace/outputs/<sample-token>/`:

- `trace_summary.json`: selected GT/query pairs, confidence, layerwise boxes,
  confidence, and replay validation errors;
- `trace_tensors.pt`: query states, query PE, image features, image PE,
  per-head projected Q/K, logits, attention, references, and predictions;
- `query_<index>_<class>/attention_all_layers.png`: one
  seven-row by six-camera grid per target. The first row shows the matched
  ground-truth box, followed by decoder layers 1-6. Columns are cameras; all
  attention cells share one heatmap scale and show the layer's predicted 3D
  box. Each layer also projects the learned reference point (cyan circle),
  predicted center (yellow X), and GT center (magenta diamond) when the point
  lies inside that camera. The header reports the GT and final predicted
  classes, while each layer row reports that layer's predicted class. A
  high-resolution summary panel in every layer row
  reports BEV distances, confidence, and the normalized leave-one-component-
  out attention impact percentages for H-X, H-G3D, H-GMV, E-X, E-G3D, and
  E-GMV. Here H is query content/state, E is query positional embedding, X is
  image appearance, G3D is image 3DPE, and GMV is multiview PE.

Component impact uses every valid image token. For each component, the script
computes the Jensen-Shannon divergence between full attention and attention
with that logit component removed, averages it across heads, and normalizes the
six impacts to 100% independently for each query and layer.

## Optional BEV ray-attention visualization

Plot the final decoder layer's 200 highest-attention image-token rays in the
shared LiDAR bird's-eye view:

```bash
python experiments/query_trace/trace_query.py \
  --scene-name scene-0101 \
  --scene-frame-index 28 \
  --score-threshold 0.35 \
  --classes car pedestrian truck bus \
  --plot-bev-rays
```

Layer 6 is the default. Select multiple one-based decoder layers and a
different ray count with, for example:

```text
--bev-ray-layers 1 3 6 --bev-ray-top-k 300
```

The plot shows camera origins, attention-colored rays, query reference,
predicted and GT centers, and predicted and GT BEV boxes. Camera triangles
are rotated toward their calibrated image-centre rays; the legend maps their
`C0` through `C5` abbreviations to the full nuScenes camera names. Matching
camera-colored dotted boundaries show each camera's horizontal BEV viewing
angle. Ray colors use a labelled logarithmic attention-weight
scale so that variation among the selected rays remains visible. The plot is
saved as `query_<index>_<class>/bev_ray_attention_L06.png`.
The title reports the matched GT class and that decoder layer's predicted
class.
Because a PETR image token represents a sampled camera ray rather than one
depth, the plot shows attended 3D directions and does not claim a unique
attended 3D point.

The tracer runs PETR twice on cached image features: once to choose or verify
the final matches and once with diagnostic attention capture. It checks that
both model outputs agree and validates the diagnostic attention against the
original cross-attention output.
