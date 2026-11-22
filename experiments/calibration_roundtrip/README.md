# PCCR paired-rig calibration round-trip

This diagnostic compares the same scene/frame in R1 and R1-f. It projects GT
box gravity centers and corners into every camera using the raw PCCR
calibration, then back-projects each visible center using its true camera-z
depth. It also reproduces PETR's actual test-time image transform, 20x50
feature-token grid, and 64 LID depth candidates. For each visible GT center it
selects the corresponding feature cell and measures:

- the error from the PETR token ray evaluated at the GT camera depth;
- the closest PETR depth candidate and its 3D error to the GT center;
- pixel-to-token quantization and depth-bin quantization separately.

It produces:

- per-rig camera mosaics with projected 3D boxes and centers;
- per-rig BEV plots with camera-to-center rays;
- a paired R1/R1-f BEV comparison;
- per-observation pixel, depth, reconstructed XYZ and round-trip error in CSV;
- `petr_token_geometry.csv` with feature-cell, ray and depth-bin diagnostics;
- per-rig processed-image token overlays and token-ray BEV plots;
- a JSON numerical summary.

Run from the PETR repository in its configured environment:

```bash
python experiments/calibration_roundtrip/validate_pair.py \
  --scene-name val_10 \
  --scene-frame-index 3
```

The default output is
`experiments/calibration_roundtrip/outputs/val_10_frame_003/`.

An approximately zero project/back-project error establishes algebraic matrix
consistency only. A transposed transform, wrong coordinate convention, or
image/calibration synchronization error may remain self-consistent when the
same matrix is used in both directions. The projected box and center overlays
are therefore the decisive visual check.
