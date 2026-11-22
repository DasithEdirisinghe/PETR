# PETR learned-anchor analysis

This experiment studies how PETR's learned 3D reference points relate to its
image-conditioned box predictions. It does not modify model inference.

For every final Hungarian query/GT pair it measures:

- nearest-anchor coverage of each GT object;
- selected anchor to GT distance;
- anchor to prediction correction;
- prediction to GT error;
- matched-class score and class correctness;
- progression through all decoder layers;
- whether each layer independently selects the same query as the final layer.

The primary metrics use BEV distance. Full 3D distances are also saved. Query
success plots are diagnostic results based on Hungarian assignments and are
not official nuScenes AP or recall.

The local nuScenes symlink and official checkpoint must exist before running:

```text
data/nuscenes/nuscenes_infos_temporal_val.pkl
ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth
```

## One-sample sanity check

Run from the PETR repository root in the `petr` environment:

```bash
python experiments/anchor_analysis/sanity_check.py
```

## Local/interactive analysis

```bash
python experiments/anchor_analysis/analyze.py \
  --num-samples 100 \
  --output-dir experiments/anchor_analysis/outputs_100
```

Use `--num-samples 0` for the complete validation set, or `--sample-index N`
for one explicit sample.

## PBS

```bash
qsub experiments/anchor_analysis/run_anchor_analysis_a100.pbs
```

The PBS job runs a one-sample sanity check first. Set `RUN_SANITY=0` to skip
that extra model load after the pipeline has already been verified.

Override defaults when needed:

```bash
qsub -v NUM_SAMPLES=0,SEED=0,OUTPUT_DIR=experiments/anchor_analysis/outputs_full \
  experiments/anchor_analysis/run_anchor_analysis_a100.pbs
```

## Outputs

- `learned_anchors_xyz.npy`: all learned anchors in metric coordinates;
- `query_layer_records.csv`: final matched queries tracked through all layers;
- `anchor_coverage.csv`: nearest learned anchor for every GT object;
- `assignment_stability.csv`: independent layer assignment stability;
- `summary.json`: aggregate diagnostics;
- five PNG figures covering spatial layout, distances, decoder progression,
  coverage/query success, and per-class behavior.
