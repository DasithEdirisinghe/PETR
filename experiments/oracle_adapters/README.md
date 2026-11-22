# PETR cross-rig oracle-adapter experiments

## 1. Purpose

These experiments start from a PETR model trained only on the PCCR **R1**
camera rig and adapt a small part of its localization mechanism using labelled
**R1-f** data.

The experiment answers:

> If the R1 detector is frozen, which small internal representation must be
> changed to recover R1-f localization accuracy?

R1-f ground truth is used to train the adapter, so this is a **supervised
oracle diagnostic**. It is not yet the final GT-free test-time adaptation
method. The successful oracle tells us where a later deployable method should
adapt and provides an upper-bound target for that method.

## 2. Data separation

| Split | Use |
|---|---|
| R1 training data | Used previously to train the source PETR checkpoint |
| R1-f train | Supplies GT for training the oracle adapter |
| R1-f validation | Evaluates progress during adapter training |
| R1-f test | Final held-out target-rig result |
| R1 test | Checks whether adaptation damages source-rig performance |

Do not use the R1-f test result to tune hyperparameters or select a checkpoint.

## 3. What remains frozen

The following source-model components remain frozen:

- image backbone and neck;
- PETR input projection and 3D position encoder;
- transformer decoder;
- classification and regression heads;
- original learned reference points and query embedding network.

Only parameters below `pts_bbox_head.oracle_adapter` are trainable. Before
training, the launcher builds the model and aborts if any other parameter has
`requires_grad=True`.

Every adapter has a zero-initialized final layer. At initialization its
residual is zero, so the adapter model initially reproduces the loaded R1
checkpoint.

## 4. Calibration input

The adapter receives the same augmentation-adjusted `lidar2img` matrices used
by PETR. They already account for the baseline pipeline's image resize, crop,
optional training flip, and global geometric augmentation. The oracle pipeline
does not introduce an additional FOV crop.

For every camera, the calibration encoder constructs:

```text
[fx/W, fy/H, cx/W, cy/H,
 horizontal_FOV/pi, vertical_FOV/pi,
 first two rotation columns,
 camera_center_xyz / translation_scale]
```

This 15-dimensional vector is mapped to a 128-dimensional camera embedding.

- Key adaptation uses the corresponding per-camera embedding.
- Query, reference, and output adaptation use a rig embedding obtained by
  mean/max pooling all camera embeddings and applying an MLP.
- Mean/max set aggregation does not assume a fixed camera ordering.

If separate intrinsics are absent from `img_metas`, they are recovered by RQ
decomposition of the retained `lidar2img` projection matrix.

## 5. Experiment modes

### `output`

The adapter predicts a residual for the final normalized center logits:

```text
delta_center = A_output(decoder_feature, rig_embedding)
```

It modifies predicted `x`, `y`, and `z` centers after the decoder but does not
alter attention or classification.

Question: are R1-f errors mostly a simple systematic correction of otherwise
correct object predictions?

- Strong improvement suggests a correctable output-coordinate bias.
- Weak improvement suggests an earlier error such as association or depth.

### `reference`

The adapter moves each normalized PETR reference point in logit space:

```text
p_adapted = sigmoid(logit(p_R1) + delta_p(query, rig_embedding))
```

PETR regenerates the query positional embedding from the adapted reference.
The adapted point is also used as the regression anchor.

Question: are the fixed R1 spatial hypotheses poorly located for R1-f?

This differs from manually inserting GT centers at inference. The movement is
learned using PETR's detection loss, so the decoder is trained against the
adapted references.

### `query`

The adapter adds a rig-conditioned residual to the query positional embedding:

```text
Q_adapted = Q_R1 + A_query(Q_R1, rig_embedding)
```

The learned reference points remain unchanged.

Question: does Q need explicit knowledge of the new rig to retrieve the right
image tokens and localize objects?

### `key`

The adapter applies per-camera feature-wise modulation to PETR's positional
embedding:

```text
[gamma_c, beta_c] = A_key(camera_embedding_c)
G_adapted_c = G_c + gamma_c * G_c + beta_c
```

Here `G_c` is supplied as `key_pos` and includes PETR 3DPE plus the configured
multiview positional component. Backbone appearance values remain unchanged.

Question: is PETR's interpretation of the new rig's G3D/key distribution the
main failure?

### `query_key`

This enables `query` and `key` together. They share the calibration encoder
but use separate residual heads.

Question: does failure arise from incompatibility between the fixed R1 query
space and the changed R1-f key/G3D space?

If `query_key` clearly beats both single adapters, neither side alone can align
the attention representation. This supports the Q-K compatibility hypothesis.

## 6. Launcher syntax

Activate the environment and enter the repository:

```bash
conda activate petr
cd /mnt/hpccs01/home/n12893251/dasith_ws/dev_ws/PETR
```

General command:

```bash
bash tools/run_pccr_oracle_adapter.sh MODE [NUM_GPUS]
```

### Positional arguments

| Argument | Required | Default | Meaning |
|---|---:|---:|---|
| `MODE` | No | `query` | `output`, `reference`, `query`, `key`, or `query_key` |
| `NUM_GPUS` | No | `1` | GPUs used by distributed training and evaluation |

Examples:

```bash
# Query oracle on one GPU.
bash tools/run_pccr_oracle_adapter.sh query 1

# Joint Q-K oracle on two GPUs.
bash tools/run_pccr_oracle_adapter.sh query_key 2
```

One GPU still uses PETR's distributed launcher with one process. Restrict the
visible interactive GPUs when necessary:

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_pccr_oracle_adapter.sh key 1
```

## 7. Environment-variable arguments

Place environment variables before the `bash` command.

| Variable | Default | Meaning |
|---|---|---|
| `SOURCE_CHECKPOINT` | R1 `latest.pth` | R1 source checkpoint providing all frozen weights |
| `OUTPUT_DIR` | `experiments/oracle_adapters/output/R1_to_R1-f/MODE` | Logs, checkpoints, and predictions directory |
| `PETR_DIR` | Inferred from script | PETR repository root; normally no override is needed |
| `ORACLE_MODE` | `query` | Alternative to positional `MODE`; positional value wins |
| `NUM_GPUS` | `1` | Alternative GPU count; positional value wins |
| `EVAL_ALL_RIGS` | `0` | Set to `1` to evaluate every PCCR rig |
| `DATA_ROOT` | `data/pccr` | PCCR root for optional all-rig evaluation |
| `SAMPLES_PER_GPU` | `4` | All-rig evaluation batch size per GPU |
| `WORKERS_PER_GPU` | `4` | All-rig evaluation loader workers per GPU |
| `SKIP_COMPLETED` | `1` | Reuse completed all-rig outputs |
| `PORT` | PETR launcher default | Distributed-process port; override if another job uses it |
| `BASE_PORT` | All-rig evaluator default | Starting port used only during optional all-rig evaluation |

Use a specific source checkpoint:

```bash
SOURCE_CHECKPOINT=results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/epoch_120.pth \
  bash tools/run_pccr_oracle_adapter.sh query_key 1
```

Use a separate output directory:

```bash
OUTPUT_DIR=experiments/oracle_adapters/output/query_seed0 \
  bash tools/run_pccr_oracle_adapter.sh query 1
```

Run optional all-rig evaluation:

```bash
EVAL_ALL_RIGS=1 \
  bash tools/run_pccr_oracle_adapter.sh query_key 2
```

## 8. What the launcher does

The script executes these stages in order:

1. Resolves the mode, config, checkpoint, output directory, and GPU count.
2. Checks the source checkpoint and three R1-f annotation files.
3. Prints GPU status and the fully resolved configuration.
4. Builds the model and verifies that only the adapter is trainable.
5. Loads the R1 checkpoint and trains the adapter on R1-f train.
6. Evaluates on R1-f validation every five epochs.
7. Saves every five epochs and retains the latest four checkpoints.
8. Evaluates `latest.pth` on held-out R1-f test.
9. Evaluates the same checkpoint on R1 test for source retention.
10. Optionally runs all-rig evaluation when `EVAL_ALL_RIGS=1`.

The launcher stops immediately if any stage fails.

## 9. Training configuration arguments

Common settings are in
`projects/configs/petr/oracle_adapters/petr_r1_to_r1f_oracle_base.py`.

| Setting | Value | Meaning |
|---|---:|---|
| `hidden_dims` | `256` | Width of adapter MLPs |
| `calibration_dims` | `128` | Camera and rig embedding dimension |
| `residual_scale` | `1.0` | Multiplier applied to predicted residuals |
| `translation_scale` | `50.0` | Normalizes camera-center inputs |
| optimizer | AdamW | Optimizes adapter parameters only |
| learning rate | `1e-4` | Initial adapter learning rate |
| weight decay | `1e-4` | Adapter regularization |
| warm-up iterations | `200` | Linear LR warm-up duration |
| warm-up ratio | `0.1` | Starting LR is 10% of configured LR |
| LR schedule | cosine | Learning-rate decay after warm-up |
| minimum LR ratio | `0.01` | Final LR is 1% of initial LR |
| epochs | `20` | Adapter-training duration |
| evaluation interval | `5` | R1-f validation frequency |
| checkpoint interval | `5` | Checkpoint-saving frequency |

The adapter uses PETR's existing Hungarian assignment, classification loss,
and box-regression loss. GT is not directly inserted into a query or reference
point during inference.

## 10. Config files

```text
projects/configs/petr/oracle_adapters/
├── petr_r1_to_r1f_oracle_base.py
├── petr_r1_to_r1f_oracle_output.py
├── petr_r1_to_r1f_oracle_reference.py
├── petr_r1_to_r1f_oracle_query.py
├── petr_r1_to_r1f_oracle_key.py
└── petr_r1_to_r1f_oracle_query_key.py
```

Each mode config inherits the common base and changes only the adapter mode and
output directory, keeping training settings identical across experiments.

## 11. Outputs

For `query_key`, the default directory is:

```text
experiments/oracle_adapters/output/R1_to_R1-f/query_key/
```

Important outputs:

```text
latest.pth                         adapted checkpoint
epoch_5.pth                        periodic checkpoints
epoch_10.pth
epoch_15.pth
epoch_20.pth
evaluation/R1-f_results.pkl        target-rig test predictions
evaluation/R1_results.pkl          source-rig test predictions
cross_rig/                         optional all-rig results
*.log                              training and validation logs
```

The `.pkl` files contain predictions. Metrics are printed in the terminal and
training log by the PCCR evaluator.

The launcher evaluates `latest.pth`, meaning the epoch-20 checkpoint in the
default configuration. Validation runs do not automatically choose a "best"
checkpoint. If an earlier epoch is clearly better on R1-f validation, evaluate
that checkpoint separately and apply the same selection rule to every mode.

## 12. Required comparisons

| Model | Train/adaptation data | Test | Role |
|---|---|---|---|
| R1 PETR | R1 | R1 | Source in-domain baseline |
| R1 PETR | R1 | R1-f | Unadapted cross-rig baseline |
| R1-f PETR | R1-f | R1-f | Target-domain ceiling |
| R1 plus oracle | R1-f adapter data | R1-f | Oracle recovery |
| R1 plus oracle | R1-f adapter data | R1 | Source retention |

For any performance metric `P`, calculate:

```text
recovery = (P_oracle - P_cross_rig) / (P_R1f_ceiling - P_cross_rig)
```

Report recovery independently for overall mAP, AP at 0.5/1/2/4 metres, mATE,
and important individual classes such as car.

## 13. Recommended execution order

Run the central hypothesis tests first:

```bash
bash tools/run_pccr_oracle_adapter.sh query 1
bash tools/run_pccr_oracle_adapter.sh key 1
bash tools/run_pccr_oracle_adapter.sh query_key 1
```

Then run the boundary controls:

```bash
bash tools/run_pccr_oracle_adapter.sh reference 1
bash tools/run_pccr_oracle_adapter.sh output 1
```

Do not run all-rig evaluation until R1-f results identify a useful mode.

## 14. Interpreting the matrix

| Observation | Likely conclusion |
|---|---|
| `output` recovers most of the gap | Correct association with systematic coordinate bias |
| `reference` is strongest | Fixed R1 spatial hypotheses are a major bottleneck |
| `query` is stronger than `key` | Queries need rig-conditioned spatial behavior |
| `key` is stronger than `query` | R1-f G3D/key interpretation is the main mismatch |
| `query_key` beats both | Q-K representation incompatibility is central |
| AP@2m/4m improves but AP@0.5m/1m does not | Association recovers, precise ray/depth localization does not |
| R1-f improves while R1 collapses | Target overfitting; source retention is needed |
| No adapter improves substantially | Adaptation may need image features, decoder, or additional information |

The most useful result would be substantial R1-f recovery from `query`, `key`,
or especially `query_key`, with the R1 detector frozen. The next step would be
to replace the supervised R1-f loss with a GT-free calibration/view-consistency
objective for the successful adapter location.

## 15. Limitations

- The oracle uses R1-f GT during adapter training.
- It demonstrates recoverability, not deployable test-time adaptation.
- R1-f alone does not prove generalization to arbitrary camera translations,
  rotations, camera counts, or appearance domains.
- All modes must use the same checkpoint, split, optimizer, epochs, and seed.
- The target ceiling is the separately trained vanilla R1-f model, not the
  best oracle result.

## 16. Evaluate and compare completed modes

After the desired modes finish, run the comparison pipeline from an interactive
GPU shell. The first argument is the GPU count; any following arguments limit
evaluation to those modes:

```bash
# Evaluate every completed mode. Missing checkpoints are skipped.
bash tools/evaluate_compare_oracle_adapters.sh 1

# Evaluate only the central Q/K ablations.
bash tools/evaluate_compare_oracle_adapters.sh 1 query key query_key
```

Existing metric logs are reused. If only prediction `.pkl` files exist, the
script calculates metrics directly from those files on CPU and does **not** run
model inference again. Force metric recalculation from saved predictions with:

```bash
FORCE_EVAL=1 \
  bash tools/evaluate_compare_oracle_adapters.sh 1 query key query_key
```

Only use `FORCE_INFERENCE=1` when predictions themselves must be regenerated:

```bash
FORCE_INFERENCE=1 \
  bash tools/evaluate_compare_oracle_adapters.sh 1 query key query_key
```

The comparison is written to:

```text
experiments/oracle_adapters/output/R1_to_R1-f/comparison/
├── oracle_comparison.csv
├── oracle_comparison.json
├── oracle_comparison.md
├── oracle_overview.png
├── oracle_overview.svg
├── oracle_strict_ap.png
├── oracle_strict_ap.svg
├── oracle_car_strict_ap.png
├── oracle_car_strict_ap.svg
├── oracle_validation_curves.png
└── oracle_validation_curves.svg
```

The evaluator runs every completed checkpoint on both R1-f test and R1 test,
so the figures compare target recovery and source retention using identical
test splits. It automatically includes both bounds:

- frozen R1 PETR evaluated on R1-f, the unadapted lower baseline;
- vanilla PETR trained from scratch on R1-f, the target-domain ceiling.

Every bar is annotated with its exact numerical value. The strict-threshold
figure reports class-averaged mAP independently at 0.5, 1, 2, and 4 metres;
the car-specific figure reports car AP at the same thresholds.
