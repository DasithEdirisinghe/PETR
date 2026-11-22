# Four-run PETR pipeline comparison

8 paired physical GT cars; the same two checkpoints were run on both rigs.
R1-f/R1 is the failure run; R1-f/R1-f is the successful reference. GT is used for pairing and scoring only.

## Upstream car-feature evidence

Held-out GT-ray versus same-camera background AUROC. The feature vectors themselves are **not** compared across separately trained models.

| Stage | R1 / R1 model | R1 / R1-f model | R1-f / R1 model | R1-f / R1-f model | Target advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| backbone_0 | 0.843 | 0.831 | 0.909 | 0.888 | -0.021 |
| backbone_1 | 0.937 | 0.952 | 0.919 | 0.957 | 0.037 |
| fpn_0 | 0.922 | 0.920 | 0.953 | 0.918 | -0.035 |
| fpn_1 | 0.888 | 0.901 | 0.917 | 0.877 | -0.040 |
| input_proj | 0.948 | 0.916 | 0.967 | 0.928 | -0.039 |

## Query and decoder

| Measure | R1 / R1 model | R1 / R1-f model | R1-f / R1 model | R1-f / R1-f model |
| --- | ---: | ---: | ---: | ---: |
| Selected query reference-to-GT error (m) | 5.670 | 6.975 | 7.078 | 7.518 |
| Final selected-query car confidence | 0.984 | 0.972 | 0.947 | 0.859 |

| Decoder layer | R1 / R1 model center (m) | R1 / R1-f model center (m) | R1-f / R1 model center (m) | R1-f / R1-f model center (m) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.176 | 2.143 | 4.185 | 1.269 |
| 2 | 1.239 | 2.136 | 3.181 | 0.823 |
| 3 | 1.291 | 2.261 | 3.455 | 0.547 |
| 4 | 1.035 | 2.441 | 3.822 | 0.557 |
| 5 | 1.034 | 2.580 | 3.556 | 0.372 |
| 6 | 1.021 | 2.497 | 3.532 | 0.412 |

Final decoder GT-ray enrichment and G3D attention utility:

| Measure | R1 / R1 | R1 / R1-f | R1-f / R1 | R1-f / R1-f |
| --- | ---: | ---: | ---: | ---: |
| GT-ray enrichment | 1.000 | 1.621 | 1.439 | 1.930 |
| G3D leave-term-out utility | 0.014 | -0.018 | -0.005 | 0.021 |
| X leave-term-out utility | 0.004 | 0.003 | -0.000 | -0.001 |

## First measured gap

At the tested thresholds (target-model AUROC advantage ≥0.05, or target-model center-error advantage ≥0.5 m), the first upstream feature stage is **none**; the first decoder output is **layer 1**.

These thresholds are descriptive, not significance tests. A feature AUROC gap does not establish that the backbone caused the box error; a decoder gap may have originated upstream. The selected GT-associated queries can have different IDs and reference points across checkpoints. The R1-f checkpoint is a successful behavioral reference, **not tensor ground truth**. Use `feature_evidence.csv`, `decoder_trajectory.csv`, and the saved tensors to inspect car-level exceptions before a causal intervention or full-validation AP test.
