#!/usr/bin/env bash

set -eo pipefail

PETR_DIR="${PETR_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PCCR_REPO="${PCCR_REPO:-$PETR_DIR/../plentiful-carla-camera-rigs}"
CONFIG="${CONFIG:-projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py}"
CHECKPOINT="${CHECKPOINT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth}"
DATA_ROOT="${DATA_ROOT:-data/pccr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/cross_rig}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
BASE_PORT="${BASE_PORT:-29600}"

RIGS=(
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

cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

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

for rig in "${RIGS[@]}"; do
    if [[ ! -f "$DATA_ROOT/$rig/${rig}_infos_test.pkl" ]]; then
        echo "Missing test annotations: $DATA_ROOT/$rig/${rig}_infos_test.pkl" >&2
        exit 1
    fi
done

RAW_ROOT="$OUTPUT_ROOT/raw"
LOG_DIR="$RAW_ROOT/PETR/R1/test_logs"
STANDARDIZED_ROOT="$OUTPUT_ROOT/standardized"
CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
mkdir -p "$LOG_DIR"

for index in "${!RIGS[@]}"; do
    rig="${RIGS[$index]}"
    dataset_path="$DATA_ROOT/$rig"
    eval_dir="$OUTPUT_ROOT/evaluations/$rig"
    formatted_dir="$eval_dir/formatted"
    metrics_path="$formatted_dir/metrics_summary.json"
    log_path="$LOG_DIR/${CHECKPOINT_NAME}_tested_on_${rig}.log"
    port=$((BASE_PORT + index))

    mkdir -p "$eval_dir"

    if [[ -s "$metrics_path" && -s "$log_path" ]] && grep -q "'object/map'" "$log_path"; then
        echo "Skipping completed evaluation: R1 -> $rig"
        continue
    fi

    echo "================================================================"
    echo "Checkpoint: $CHECKPOINT"
    echo "Training rig: R1"
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
echo "Standardized result: $STANDARDIZED_ROOT/PETR/trained_on_R1.json"
