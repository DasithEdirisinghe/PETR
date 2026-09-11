#!/usr/bin/env bash

set -eo pipefail

PETR_DIR="${PETR_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PCCR_REPO="${PCCR_REPO:-$PETR_DIR/../plentiful-carla-camera-rigs}"
MODEL_NAME="${MODEL_NAME:-PETR}"
TRAIN_RIG="${TRAIN_RIG:-R1}"
CONFIG="${CONFIG:-projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py}"
CHECKPOINT="${CHECKPOINT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth}"
DATA_ROOT="${DATA_ROOT:-data/pccr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/cross_rig}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
BASE_PORT="${BASE_PORT:-29600}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

DEFAULT_RIGS=(
    R1
    R1-c10
    R1-c6
    R1-f
    R1-r
    R1-t
    R2
    R3
    R4
    R5
    R6
    R7
    R8
    R9
)

if [[ -n "${RIGS:-}" ]]; then
    read -r -a RIG_LIST <<< "$RIGS"
else
    RIG_LIST=("${DEFAULT_RIGS[@]}")
fi

cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

if [[ ! "$MODEL_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "MODEL_NAME must contain only letters, numbers, dot, underscore, or hyphen" >&2
    exit 1
fi
if [[ ! "$TRAIN_RIG" =~ ^R[0-9]+$ ]]; then
    echo "TRAIN_RIG must look like R1, R2, etc.: $TRAIN_RIG" >&2
    exit 1
fi
if [[ "$SKIP_COMPLETED" != "0" && "$SKIP_COMPLETED" != "1" ]]; then
    echo "SKIP_COMPLETED must be 0 or 1" >&2
    exit 1
fi
if [[ ${#RIG_LIST[@]} -eq 0 ]]; then
    echo "No test rigs selected" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Missing config: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$PCCR_REPO/metrics/utils/standardize_results.py" ]]; then
    echo "Missing PCCR standardizer under: $PCCR_REPO" >&2
    exit 1
fi

for rig in "${RIG_LIST[@]}"; do
    if [[ ! -f "$DATA_ROOT/$rig/${rig}_infos_test.pkl" ]]; then
        echo "Missing test annotations: $DATA_ROOT/$rig/${rig}_infos_test.pkl" >&2
        exit 1
    fi
done

RAW_ROOT="$OUTPUT_ROOT/raw"
LOG_DIR="$RAW_ROOT/$MODEL_NAME/$TRAIN_RIG/test_logs"
STANDARDIZED_ROOT="$OUTPUT_ROOT/standardized"
CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
mkdir -p "$LOG_DIR"

echo "Cross-rig evaluation configuration:"
echo "  Model label: $MODEL_NAME"
echo "  Training rig: $TRAIN_RIG"
echo "  Config: $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data root: $DATA_ROOT"
echo "  Output root: $OUTPUT_ROOT"
echo "  Test rigs: ${RIG_LIST[*]}"
echo "  Skip completed: $SKIP_COMPLETED"

for index in "${!RIG_LIST[@]}"; do
    rig="${RIG_LIST[$index]}"
    dataset_path="$DATA_ROOT/$rig"
    eval_dir="$OUTPUT_ROOT/evaluations/$rig"
    formatted_dir="$eval_dir/formatted"
    metrics_path="$formatted_dir/metrics_summary.json"
    log_path="$LOG_DIR/${CHECKPOINT_NAME}_tested_on_${rig}.log"
    port=$((BASE_PORT + index))

    mkdir -p "$eval_dir"

    if [[ "$SKIP_COMPLETED" == "1" && -s "$metrics_path" && -s "$log_path" ]] && \
            grep -Eq "'object/map'|'pts_bbox_NuScenes/mAP'|^mAP:" "$log_path"; then
        echo "Skipping completed evaluation: $TRAIN_RIG -> $rig"
        continue
    fi

    echo "================================================================"
    echo "Checkpoint: $CHECKPOINT"
    echo "Model label: $MODEL_NAME"
    echo "Training rig: $TRAIN_RIG"
    echo "Test rig: $rig"
    echo "Dataset: $dataset_path"
    echo "Output: $eval_dir"
    echo "================================================================"

    PORT="$port" bash tools/dist_test.sh \
        "$CONFIG" \
        "$CHECKPOINT" \
        1 \
        --cfg-options \
            dataset_name="$rig" \
            data_root="$dataset_path/" \
            test_ann_file="$dataset_path/${rig}_infos_test.pkl" \
            data.test.data_root="$dataset_path/" \
            data.test.ann_file="$dataset_path/${rig}_infos_test.pkl" \
            data.samples_per_gpu="$SAMPLES_PER_GPU" \
            data.workers_per_gpu="$WORKERS_PER_GPU" \
        --out "$eval_dir/predictions.pkl" \
        --eval bbox \
        --eval-options jsonfile_prefix="$formatted_dir" \
        2>&1 | tee "$log_path"
done

python "$PCCR_REPO/metrics/utils/standardize_results.py" \
    --raw-root "$RAW_ROOT" \
    --output-root "$STANDARDIZED_ROOT" \
    --write-per-test

echo "Cross-rig evaluation complete."
echo "Standardized result: $STANDARDIZED_ROOT/$MODEL_NAME/trained_on_$TRAIN_RIG.json"
