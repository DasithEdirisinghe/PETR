# Experiment 2: apparent area and localization for PETR

GT boxes are ranked by projected pixel area **within each class, 10 m GT-range bin, and annotation visibility category**. Separate `all` visibility rows pool visibility categories. Quartiles use all eligible GT with a camera footprint; center errors use the original 4 m matches only. 0 GT outside the camera images are excluded.

The outcome is mean BEV center error in metres. Each row also reports 4 m GT coverage, mean radial/tangential error, exact mean GT range, viewing angle, and physical GT length/width/height. The Q4−Q1 interval uses scene-cluster standard errors and is shown only as an uncertainty measure for this association; repeated scenes, occlusion, physical size, orientation, and selection into 4 m TPs still limit causal interpretation. Projected GT box area is a geometric footprint, not a count of truly visible object pixels. Intervals are exploratory and are not corrected for the many class/range/visibility comparisons.

## All visibility: Q4 versus Q1

| Class | Range | Q1 / Q4 TPs | Q1 error | Q4 error | Q4−Q1 | 95% scene CI |
|---|---|---:|---:|---:|---:|---:|
| car | 0-10 m | 1,384 / 1,372 | 0.35 m | 0.34 m | -0.02 m | [-0.09, +0.06] |
| car | 10-20 m | 3,025 / 3,017 | 0.82 m | 0.52 m | -0.30 m | [-0.42, -0.18] |
| car | 20-30 m | 3,457 / 3,432 | 1.29 m | 1.16 m | -0.13 m | [-0.25, -0.02] |
| car | 30-40 m | 3,028 / 3,027 | 1.72 m | 1.61 m | -0.11 m | [-0.22, -0.00] |
| car | 40-50 m | 2,170 / 2,154 | 1.89 m | 1.92 m | +0.04 m | [-0.09, +0.16] |
| truck | 0-10 m | 167 / 160 | 0.48 m | 1.11 m | +0.63 m | [+0.28, +0.99] |
| truck | 10-20 m | 451 / 445 | 0.93 m | 1.04 m | +0.11 m | [-0.15, +0.36] |
| truck | 20-30 m | 531 / 544 | 1.40 m | 1.64 m | +0.24 m | [-0.01, +0.48] |
| truck | 30-40 m | 565 / 535 | 1.85 m | 1.96 m | +0.11 m | [-0.06, +0.28] |
| truck | 40-50 m | 447 / 421 | 1.93 m | 2.07 m | +0.13 m | [-0.17, +0.44] |
| bus | 0-10 m | 30 / 30 | 1.88 m | 0.57 m | -1.32 m | [-2.60, -0.03] |
| bus | 10-20 m | 103 / 103 | 0.70 m | 0.95 m | +0.25 m | [-0.15, +0.65] |
| bus | 20-30 m | 119 / 116 | 1.34 m | 1.51 m | +0.16 m | [-0.17, +0.50] |
| bus | 30-40 m | 103 / 123 | 1.58 m | 1.90 m | +0.32 m | [-0.01, +0.65] |
| bus | 40-50 m | 96 / 96 | 1.84 m | 2.12 m | +0.28 m | [-0.09, +0.64] |
| trailer | 10-20 m | 81 / 75 | 1.45 m | 1.89 m | +0.44 m | [-0.06, +0.94] |
| trailer | 20-30 m | 136 / 152 | 1.62 m | 2.27 m | +0.65 m | [+0.29, +1.00] |
| trailer | 30-40 m | 143 / 191 | 2.13 m | 2.68 m | +0.55 m | [+0.10, +1.00] |
| trailer | 40-50 m | 95 / 115 | 1.97 m | 2.05 m | +0.08 m | [-0.26, +0.42] |
| construction vehicle | 10-20 m | 53 / 82 | 1.78 m | 1.66 m | -0.12 m | [-0.54, +0.29] |
| construction vehicle | 20-30 m | 65 / 50 | 2.06 m | 1.81 m | -0.25 m | [-0.76, +0.27] |
| construction vehicle | 30-40 m | 50 / 48 | 2.01 m | 2.34 m | +0.32 m | [-0.24, +0.89] |
| construction vehicle | 40-50 m | 47 / 59 | 2.31 m | 2.98 m | +0.67 m | [+0.18, +1.16] |
| pedestrian | 0-10 m | 722 / 703 | 0.97 m | 0.70 m | -0.28 m | [-0.47, -0.08] |
| pedestrian | 10-20 m | 2,013 / 2,034 | 1.35 m | 0.88 m | -0.47 m | [-0.58, -0.37] |
| pedestrian | 20-30 m | 1,627 / 1,662 | 1.73 m | 1.41 m | -0.32 m | [-0.43, -0.22] |
| pedestrian | 30-40 m | 1,264 / 1,279 | 1.87 m | 1.72 m | -0.14 m | [-0.25, -0.04] |
| motorcycle | 0-10 m | 51 / 53 | 0.64 m | 0.50 m | -0.14 m | [-0.36, +0.08] |
| motorcycle | 10-20 m | 113 / 121 | 1.53 m | 0.90 m | -0.63 m | [-0.99, -0.27] |
| motorcycle | 20-30 m | 108 / 144 | 1.70 m | 1.40 m | -0.30 m | [-0.59, -0.02] |
| motorcycle | 30-40 m | 80 / 84 | 1.77 m | 2.26 m | +0.49 m | [+0.06, +0.92] |
| bicycle | 0-10 m | 54 / 54 | 0.93 m | 0.52 m | -0.40 m | [-0.71, -0.10] |
| bicycle | 10-20 m | 128 / 149 | 1.44 m | 0.88 m | -0.56 m | [-0.98, -0.14] |
| bicycle | 20-30 m | 121 / 122 | 1.26 m | 1.44 m | +0.18 m | [-0.14, +0.50] |
| bicycle | 30-40 m | 57 / 67 | 1.96 m | 1.50 m | -0.47 m | [-0.86, -0.08] |
| traffic cone | 0-10 m | 652 / 684 | 0.90 m | 0.65 m | -0.25 m | [-0.52, +0.03] |
| traffic cone | 10-20 m | 1,235 / 1,269 | 1.15 m | 0.85 m | -0.30 m | [-0.44, -0.16] |
| traffic cone | 20-30 m | 571 / 639 | 1.72 m | 1.41 m | -0.31 m | [-0.47, -0.15] |
| barrier | 0-10 m | 773 / 788 | 1.06 m | 0.94 m | -0.12 m | [-0.28, +0.04] |
| barrier | 10-20 m | 1,601 / 1,626 | 1.18 m | 1.02 m | -0.16 m | [-0.29, -0.03] |
| barrier | 20-30 m | 1,371 / 1,285 | 1.33 m | 1.40 m | +0.07 m | [-0.13, +0.27] |

## Association after measured covariate adjustment

This per-class linear model uses matched 4 m TPs and adjusts for exact GT range (linear and squared), physical dimensions, relative orientation, off-axis angle, visibility category, and camera. The coefficient is the change in mean center error for a doubling of apparent area. A negative value means larger projected area is associated with lower error after these adjustments. Scene-cluster 95% intervals quantify sampling uncertainty, not causal effects.

| Class | 4 m TPs | Change per area doubling | 95% scene CI |
|---|---:|---:|---:|
| car | 52,177 | +0.009 m | [-0.036, +0.054] m |
| truck | 8,565 | +0.059 m | [-0.055, +0.173] m |
| bus | 1,835 | +0.303 m | [+0.101, +0.506] m |
| trailer | 1,912 | +0.205 m | [+0.012, +0.398] m |
| construction vehicle | 1,036 | +0.087 m | [-0.201, +0.375] m |
| pedestrian | 22,615 | +0.052 m | [-0.050, +0.153] m |
| motorcycle | 1,545 | -0.208 m | [-0.454, +0.038] m |
| bicycle | 1,558 | +0.063 m | [-0.108, +0.234] m |
| traffic cone | 10,169 | -0.042 m | [-0.160, +0.075] m |
| barrier | 14,767 | -0.031 m | [-0.108, +0.046] m |

See `area_quartiles.csv` and `q4_vs_q1.csv` for visibility strata and covariate details. `plots/` contains one line chart per class for pooled visibility and each annotation visibility category. A lower Q4 error means the larger-image group has better center localization within that stratum.
