# R1-f inference-only PETR pathway diagnosis

This experiment holds **R1-f validation images, camera calibration, and physical GT cars fixed**, then runs each frame through two **frozen** PETR checkpoints: R1-trained and R1-f-trained. It asks where the R1 model's car-center correction diverges, and which cross-attention contributions actually matter to the final center. It trains nothing and never feeds GT into the baseline detector. GT is used afterward to select the same physical car, score center error, define car-intersecting image rays, and evaluate interventions.

## Run

From the PETR repository root, in the `petr` GPU environment:

```bash
bash experiments/cross_rig_attention/run_r1f_inference_pathway.sh val
```

The default cohort is the first eight R1-f cars in `experiments/cross_rig_attention/output/r1f_val/analysis/trace_candidates.json`. Start small if desired:

```bash
bash experiments/cross_rig_attention/run_r1f_inference_pathway.sh val \
  --max-examples 2 --layers 1 3 6 --strengths 0.25 0.5
```

Results go to `experiments/cross_rig_attention/output/r1f_val/inference_pathway/`. The driver reuses case files when the checkpoint and analysis settings match; `--force` reruns them. More cars and intervention strengths make the run substantially longer. `--device cuda:0`, `--output-dir PATH`, `--selection PATH`, `--control-repeats N` (default 3), and `--box-expansion METRES` (default 0.5) are also supported. `--split test` requires the corresponding R1-f test annotations and candidate file.

To regenerate **only the figures and tables** from saved cases (no GPU inference):

```bash
python experiments/cross_rig_attention/plot_r1f_inference_pathway.py \
  --output-dir experiments/cross_rig_attention/output/r1f_val/inference_pathway \
  --selection experiments/cross_rig_attention/output/r1f_val/analysis/trace_candidates.json \
  --max-examples 8 --layers 1 2 3 4 5 6 --strengths 0.5
```

Use the same selection, layer list, and strengths as the driver run.

## What is measured

For each car, the script runs normal PETR inference, associates one final-layer query with the GT car using the checkpoint's own Hungarian assignment, and records its six layer-wise box centers and car scores. Query IDs may differ between checkpoints; comparing raw query vectors directly would be invalid.

The required BEV correction is `GT center − that checkpoint's query reference`. Each predicted correction is `predicted center − reference`. Their cosine measures direction; their length ratio measures whether the model moves far enough. Both are object-aligned geometric quantities, unlike latent-vector distance.

At each cross-attention layer, the script reconstructs the selected query's Q/K/V projections and attention. It checks that the reconstruction matches the module output. The key PE is decomposed into learned 3D position `G3D` and multi-view position `GMV`; the image feature contributes `X`. A GT-car ray mask is formed by back-projecting feature-cell centers with that frame's calibration and intersecting rays with an expanded oriented GT car box. This mask is a *diagnostic grouping*, not ground-truth depth or proof that every selected token depicts the car.

Two intervention families replay the rest of the frozen decoder:

- **Value replacement:** At one layer and one query, blend projected values on GT-car rays toward mean background values from the same camera. Compare its change in final BEV center error against equally many random non-car cells from the same cameras (`value_control`). Positive `car − control` error change suggests the car-ray values are specifically useful there.
- **Key-logit attenuation:** At one layer and one query, subtract a chosen fraction of that branch's Q·K logit contribution for `X`, `G3D`, or `GMV`, then recompute attention and replay the remaining decoder. Positive change in final error means attenuating that branch hurt localization. This is a sensitivity test, **not** a valid replacement model and not a direct measurement of cross-rig causality.

Only the selected query is intervened upon. The other queries follow their native path. The intervention is evaluated on the **same** selected query even if it would no longer win a post-intervention assignment. The script also records selected-query state vectors before/after self-attention, cross-attention, FFN, and normalization. These are diagnostic artifacts; different checkpoints' hidden-vector coordinates are not assumed aligned.

## Outputs and reading order

1. `correction_trajectory.png` (and `.svg`): Does R1 point/move toward the car as effectively as R1-f? Alignment near +1 with a wrong length means a *magnitude* error; low alignment means a *direction* error. The right panel gives actual center error in metres. Medians are across the selected cars.
2. `observed_pathway.png`: Center error, attention enrichment on GT-car rays, and ratio of car-ray contribution norm to all-value contribution norm by layer. The norm ratio is **not a probability** and can exceed 1 due to vector cancellation. Attention enrichment measures routing, **not** useful localization by itself.
3. `query_state_updates.png`: Relative size of self-attention, cross-attention, and FFN updates. This locates large state changes, but a large update is not necessarily a correct one.
4. `causal_interventions.png`: Top row is car-value replacement minus matched control effect; bottom row is the change in final center error after attenuating each key-logit term. Each annotation is a signed number of metres. Positive = worse after intervention; negative = better. Compare R1 and R1-f *within each model* before attributing a mechanism.
5. `diagnostic_report.md` and `summary.json`: Exact cohort medians and interpretation boundaries. `per_car_observations.csv`, `per_car_state_updates.csv`, and `per_car_interventions.csv` expose the individual-car data. `cases/{r1,r1f}/*.json` stores complete per-car traces, including stage vectors and cache signatures.

The experiment first verifies native hooked inference reproduces the unhooked scores/boxes and that the manually reconstructed cross-attention output matches the actual output (tolerance 1e-4). It stops on a mismatch rather than silently plotting an invalid decomposition.

## What it can and cannot conclude

If the models attend to similar car rays but R1's correction is misdirected or undersized, the evidence narrows the problem to *how its decoder turns available features/geometry into a center update*, not simply missed object recognition. If R1-f is more sensitive to removing car-ray values than R1, that supports a difference in how useful car information is integrated. If G3D-term attenuation has a distinctive effect, it implicates that branch's routing in the frozen network, but does not prove G3D alone caused the cross-rig failure.

These selected cars are **not** full-validation AP. The within-model interventions deliberately perturb a trained network and may be out of distribution. A checkpoint comparison is observational; an intervention's differential effect strengthens a functional claim but cannot prove that swapping one component would fix cross-rig generalization. Use the saved full-validation AP results separately for population-level claims.
