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

## Optional per-head camera-attention visualization

Plot the final decoder layer as one comparison canvas:

```bash
python experiments/query_trace/trace_query.py \
  --scene-name scene-0101 \
  --scene-frame-index 28 \
  --query-indices 127 \
  --plot-per-head-attention
```

The default is decoder layer 6. Select one or more one-based layers with, for
example:

```text
--plot-per-head-attention --per-head-layers 3 6
```

Each selected layer produces
`query_<index>_<class>/per_head_attention_L<layer>.png`. The canvas has nine
rows (head average followed by heads 1 through 8) and one column for each
camera. All panels use a shared 99.5-percentile attention scale. Each panel
reports that head's attention mass assigned to the camera and overlays the GT
box, predicted box, query reference, predicted center, and GT center.

Every head is independently softmax-normalized over all image tokens and sums
to one. Therefore, this canvas compares spatial concentration and allocation
across cameras; it does not by itself measure how strongly one head contributes
to the final decoder output.

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
attended 3D point. Ray back-projection uses the centre of each feature cell,
`u=(column+0.5)*image_width/feature_width` and
`v=(row+0.5)*image_height/feature_height`, rather than the cell boundary.

The tracer runs PETR twice on cached image features: once to choose or verify
the final matches and once with diagnostic attention capture. It checks that
both model outputs agree and validates the diagnostic attention against the
original cross-attention output. For URoPE, validation reconstructs all 900
queries with the native batch/head tensor shapes before selecting the requested
queries. This avoids shape-dependent CUDA rounding differences while retaining
the strict replay check.

## nuScenes URoPE trace

The same generic `trace_query.py` entry point supports PETR-URoPE. Pass the
URoPE config and checkpoint just as for baseline PETR. For sample token
`22ce804db92749a1aee3a08ffbecfbbd`, baseline PETR query 163 is matched to GT
index 57 (car). To trace the URoPE query assigned to that same physical object:

```bash
python experiments/query_trace/trace_query.py \
  --config projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_urope_multiview.py \
  --checkpoint results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_urope_multiview/epoch_24.pth \
  --sample-token 22ce804db92749a1aee3a08ffbecfbbd \
  --gt-indices 57 \
  --dataset-name nuScenes-URoPE \
  --output-dir experiments/query_trace/outputs/nuscenes_urope \
  --plot-per-head-attention \
  --per-head-layers 6 \
  --plot-bev-rays \
  --bev-ray-layers 1 2 3 4 5 6
```

Use `--query-indices 163` instead only if the intention is specifically to
inspect URoPE's own query 163. Query IDs are model-specific, so that is not
guaranteed to be the object represented by baseline PETR query 163.

URoPE outputs record the config, checkpoint and positional-encoding settings.
They save full per-head attention plus relative rotary-geometry impact;
`G3D` is absent because learned PETR frustum 3DPE is disabled, while `GMV`
remains because this model retains multiview PE.

## PCCR R1 to R1-f paired trace

First list the matched objects and queries for an R1 validation frame:

```bash
SCENE_NAME=val_10 SCENE_FRAME_INDEX=3 \
  bash experiments/query_trace/run_pccr_r1_r1f.sh
```

Inspect
`val_10_frame_003/_precheck/R1_val/query_candidates.csv`, choose one or more
query IDs, and run the paired trace:

```bash
SCENE_NAME=val_10 SCENE_FRAME_INDEX=3 QUERY_INDICES="735" \
  bash experiments/query_trace/run_pccr_r1_r1f.sh
```

The R1-f run does **not** reuse the R1 query indices. Instead, it reads
`R1_val/trace_summary.json`, matches each target by class and 3D GT center,
and traces the decoder query assigned to that same physical object. Matching
fails if the nearest center is more than 0.25 m away, preventing an accidental
comparison of different frames or objects.

Results are grouped first by the selected R1 query, making each cross-rig
comparison self-contained:

```text
pccr_r1_vs_r1f/
└── val_10_frame_003/
    └── query_0735/
    │   ├── R1_val/
    │   └── R1-f_val/
```

Each rig directory contains the full camera attention grid, layerwise tensor
trace, and BEV ray-attention plots for decoder layers 1 through 6. Override
`CHECKPOINT`, `DEVICE`, `OUTPUT_DIR`, or `PYTHON_BIN` through environment
variables when needed.

The equivalent direct command for listing R1 candidates is:

```bash
python experiments/query_trace/trace_query.py \
  --config projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py \
  --checkpoint results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth \
  --data-root data/pccr/R1/ \
  --ann-file data/pccr/R1/R1_infos_val.pkl \
  --dataset-name R1 --sample-index 0 --list-queries
```

The CSV reports the query ID, matched GT ID and class, classification result,
confidence, and BEV localization error. This supports choosing a correctly
localized object, a failed object, or both before performing the expensive
layerwise trace.
