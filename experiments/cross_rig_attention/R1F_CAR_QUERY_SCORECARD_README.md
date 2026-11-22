# Full-validation query-level car diagnosis

This is the inference experiment for the question: **on the same R1-f images and calibration, does each checkpoint's GT-associated object query attend toward the car, classify it as car, and place its center accurately?** It does not infer query identity from saved final detections. Both frozen checkpoints perform fresh forward passes on the entire R1-f validation set; GT is used only afterward to associate a query and define diagnostic car rays.

Run in an interactive GPU shell using the `petr` environment:

```bash
bash experiments/cross_rig_attention/run_r1f_car_query_scorecard.sh
```

For a quick instrumentation check before all 1,020 frames:

```bash
bash experiments/cross_rig_attention/run_r1f_car_query_scorecard.sh \
  --max-frames 2 --output-dir experiments/cross_rig_attention/output/r1f_val/query_scorecard_smoke
```

The default output is `experiments/cross_rig_attention/output/r1f_val/query_scorecard/`. Case files are cached per frame/checkpoint with a checkpoint-and-settings signature. `--force` recomputes cases; `--device cuda:0`, `--score-threshold 0.35`, `--box-expansion 0.5`, and `--output-dir PATH` can be changed. Do not mix a smoke test with the full run in one output directory.

## Exactly what the scorecard measures

For every evaluator-valid GT car available to PETR, each checkpoint makes an ordinary head forward pass. We independently select the query assigned to that GT car by PETR's final-layer Hungarian assignment. Query IDs need not match between checkpoints. A query is selected even if its prediction is poor; that is intentional and avoids selecting only successful detections. However, **GT-assisted query selection is diagnostic, not an inference-time operation**.

For that query:

- **Attention score:** reconstruct L1 and L6 decoder cross-attention from their actual Q/K tensors. Calibration back-projects feature-cell centers into 3D rays; positives are **only rays intersecting the selected GT car box** (expanded by 0.5 m), and negatives are other valid rays across all cameras. Cameras where this car has no intersecting rays do not supply positive examples. The AUROC is the probability that a positive ray receives more *mean-over-head attention* than a negative ray. 0.5 is chance; 1.0 is perfect relative ranking. We also save attention enrichment over uniform routing. An AUROC ≥0.6 is the configurable **descriptive attention-pass rule** in the joint score—not proof that the query semantically uses the car. Zero-positive-ray cases have missing AUROC and fail that rule.
- **Per-camera attention:** `per_car_camera_attention.csv` saves every camera's total attention mass, attention mass on car rays, number of GT-car rays, and within-camera attention AUROC at L1 and L6. An invisible camera has **no car-region AUROC** rather than a zero score, but its attention mass is still measured. `camera_attention.svg` visualizes all cameras for the fixed R1-f-detected cohort. The all-camera AUROC and each within-camera AUROC answer different questions; neither should be silently substituted for the other.
- **Classification:** the selected query's *raw final-layer class logits* are converted to sigmoid scores. “Car passes” requires **both** car as the top-scoring class and car score ≥0.35. The CSV saves top class, top score, and car score.
- **Localization:** BEV distance between that query's final predicted center and the GT center. The scorecard reports strict <1 m, while CSV also saves <4 m.
- **All three:** L6 attention AUROC ≥0.6, car classification passes, and center error <1 m **for the same query and same GT car**.

`query_scorecard.svg` has one row per checkpoint for two cohorts: (1) a **fixed reference cohort** where the R1-f checkpoint's GT-associated query passes car classification and lies within 4 m, and (2) all GT cars with assigned queries in both models. It displays median all-camera attention AUROC, class-pass rate, class-and-center-pass rate, and joint rate. `camera_attention.svg` breaks attention down across every camera. `query_scorecard_summary.json` contains exact values and counts. `per_car_query.csv` and `per_car_camera_attention.csv` permit paired per-object analysis. `cases/{r1,r1f}/*.json` permit resumable processing. `run_metadata.json` records any GT cars that could not be assigned a diagnostic query.

**To compare whether the checkpoints actually select the same image cells**, the runner also saves per-car L1/L6 mean-over-head attention vectors in compressed `attention_maps/{r1,r1f}/*.npz`. The feature-cell positions and car-ray masks are checked to be identical across checkpoints. `paired_attention.png/.svg` compares R1 and R1-f GT-car-ray attention mass and spatial-map overlap. `paired_attention.csv` gives for each physical car: R1 and R1-f mass on its GT rays; full-map overlap `sum_i min(a_i,b_i)` (1 = identical distributions); joint mass on GT-car rays; and **within-car overlap** `sum_{i in R} min(a_i/M_R1, b_i/M_R1f)` (1 = the same relative attention pattern *inside* that car's rays). Read within-car overlap together with the two masses: two models can have the same relative pattern while assigning almost no attention to the car. `paired_camera_attention.csv` gives analogous per-camera values, with within-car overlap unavailable if the car has no rays in that camera.

Replot saved CSV without a GPU:

```bash
python experiments/cross_rig_attention/plot_r1f_car_query_scorecard.py \
  --input-dir experiments/cross_rig_attention/output/r1f_val/query_scorecard

python experiments/cross_rig_attention/compare_r1f_query_attention_maps.py \
  --input-dir experiments/cross_rig_attention/output/r1f_val/query_scorecard
```

## Interpretation limits

This is much closer to the intended test than the saved-box figures, but three cautions remain. A 3D-box-intersecting ray is not a segmentation mask: occlusion or background can make the positive set imperfect. High attention AUROC shows **spatial routing**, not that the attended value features improve localization; causal value removal would be a separate test. Finally, query-level percentages are not official AP, which includes ranking, false positives, and post-processing. The strongest evidence for a localization-dominant failure is a large R1 subset where **L6 attention AUROC is high and car classification passes, yet center error exceeds 1 m**, compared with the same R1-f cars.
