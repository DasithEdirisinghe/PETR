# Frozen R1 PETR: paired-R1/R1-f diagnostic report

8 matched GT cars; 6 changed independently assigned query.
The model, weights, and input images are unchanged. GT was used only for pairing and scoring.

## Observed pathway

| Measure | R1 frames | R1-f frames, fixed R1 query |
| --- | ---: | ---: |
| Final median BEV center error (m) | 1.021 | 3.708 |
| Final median signed radial error (m) | -0.400 | 2.432 |
| Final GT-ray attention enrichment | 1.000 | 0.745 |
| Final G3D directional utility | 0.014 | -0.007 |
| Final X directional utility | 0.004 | -0.000 |

R1-car prototype discrimination (AUROC): R1 0.944, R1-f 0.934. Cross-rig car-feature cosine: 0.863.

## Controlled key-component probes

Each row shows the best tested non-1.0 scale on R1-f. Negative Δ means lower final center error than the original model.

| Branch | Scale | R1-f Δerror (m) | Cars improved | R1 Δerror (m) at same scale |
| --- | ---: | ---: | ---: | ---: |
| x_key | 1.25 | -0.040 | 50.0% | -0.003 |
| g3d_key | 0.75 | -0.170 | 75.0% | 0.157 |
| all_key | 0.75 | -0.067 | 75.0% | -0.031 |

## Reading the result

A drop in feature AUROC suggests lost image evidence. Preserved feature evidence with early GT-ray attention loss suggests a key-query routing problem. A selective branch intervention that improves R1-f without material R1 damage strengthens that hypothesis; similar benefit from `all_key` may instead reflect attention temperature. Later-layer error growth with preserved early attention points toward value aggregation or box refinement. These are hypotheses to validate per car and with full detection AP, not unique causal identifications.

Inspect `per_car_per_layer.csv` in both directories and the PNG plots before choosing an architectural intervention.
