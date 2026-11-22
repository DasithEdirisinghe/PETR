# Paired PCC-R cross-rig attention autopsy

This experiment compares an R1-trained PETR model on the same synchronized
GT object in R1 and R1-f. It reuses the validated query-attention tracer and
does not modify PETR inference.

It traces three conditions:

1. the R1 query assigned to the selected GT;
2. the query independently assigned to that physical GT in R1-f;
3. the original R1 query index retained in R1-f, even if Hungarian matching
   assigns a different query.

The third condition fixes the learned query reference and positional
embedding. Its first cross-attention layer is therefore the cleanest view of
changes caused by `X`, `G3D`, and `GMV` on the target rig.

## 1. List source-rig candidates

```bash
python experiments/cross_rig_attention/run_paired_trace.py \
  --scene-name val_03 \
  --scene-frame-index 17 \
  --list-source-queries
```

This writes the R1 Hungarian assignments to the pair's
`R1_object_matched/query_candidates.csv`. Selection does not require a score
threshold or a correct class once an explicit GT/query index is supplied.

## 2. Run the paired experiment

Select one or more zero-based GT indices:

```bash
python experiments/cross_rig_attention/run_paired_trace.py \
  --scene-name val_03 \
  --scene-frame-index 17 \
  --gt-indices 4 9 \
  --classes car truck bus \
  --device cuda:0
```

An R1 query index from the precheck can be used instead:

```text
--query-indices 163
```

Use `--skip-existing` to reuse complete GPU traces and regenerate only the
comparison report.

## Outputs

Each pair is written below
`experiments/cross_rig_attention/outputs/<scene>_frame_<index>_R1_vs_R1-f/`.
The three trace directories retain the existing attention heatmaps, BEV ray
maps, tensors, and summaries. `comparison/` contains:

- `paired_layer_metrics.csv`: machine-readable metrics for every condition,
  target, layer, and attention term;
- `paired_diagnostics.json`: the same results grouped by physical target;
- `diagnostic_summary.md`: compact final-outcome and first-layer term tables;
- `outcomes_over_layers.png`: GT-ray mass, normalized attention entropy,
  GT-class confidence, and BEV localization error;
- `term_diagnostics.png`: GT discrimination and directional utility for
  `H-X`, `H-G3D`, `H-GMV`, `E-X`, `E-G3D`, and `E-GMV`;
- `paired_attention_all_layers.png`: side-by-side R1/R1-f attention canvases;
- `paired_bev_rays_L06.png`: side-by-side final-layer BEV ray maps.

Feature-cell rays are marked GT-consistent when they intersect the oriented
GT 3D box expanded by 0.5 m by default. Change this tolerance with
`--box-expansion`.

For term `t`, GT discrimination is

```text
(mean logit on GT-consistent rays - mean logit on other valid rays)
------------------------------------------------------------------
                    standard deviation over valid rays
```

Directional utility is

```text
GT mass under full attention - GT mass after removing term t
```

Positive values mean the term helps direct attention toward the GT; negative
values mean it directs attention away. This is different from the existing
JS impact, which measures the magnitude of a change but not whether the change
is useful.

## PBS execution

```bash
qsub -v SCENE_NAME=val_03,FRAME_INDEX=17,GT_INDICES="4 9" \
  experiments/cross_rig_attention/run_paired_autopsy_a100.pbs
```

Use `QUERY_INDICES` instead of `GT_INDICES` when selecting source queries.

## Whole-validation car localization check

To test whether BEV center error explains the car AP collapse across the full
PCC-R evaluation set without running inference again, use the saved formatted
predictions:

```bash
python experiments/cross_rig_attention/plot_car_bev_localization.py
```

This creates one focused figure,
`outputs/car_validation_localization/car_bev_localization_vs_ap.png`. Solid
curves show the fraction of validation GT cars matched by a class-correct
prediction as the permitted BEV center error increases. Dashed points are the
official car AP values at the PCC-R 0.5, 1, 2, and 4 m thresholds. Localization
recall uses predictions with confidence at least 0.35 by default; change it
with `--score-threshold`.
