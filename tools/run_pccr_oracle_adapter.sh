#!/usr/bin/env bash

# Run a supervised PETR oracle adapter from an interactive GPU shell.
# Usage: bash tools/run_pccr_oracle_adapter.sh MODE [NUM_GPUS]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="${PETR_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

ORACLE_MODE="${1:-${ORACLE_MODE:-query}}"
NUM_GPUS="${2:-${NUM_GPUS:-1}}"
case "$ORACLE_MODE" in
    query|key|query_key|reference|output) ;;
    *)
        echo "Mode must be query, key, query_key, reference, or output" >&2
        exit 2
        ;;
esac

CONFIG="projects/configs/petr/oracle_adapters/petr_r1_to_r1f_oracle_${ORACLE_MODE}.py"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/oracle_adapters/output/R1_to_R1-f/$ORACLE_MODE}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth}"
EVAL_ALL_RIGS="${EVAL_ALL_RIGS:-0}"

required_files=(
    "$CONFIG"
    "$SOURCE_CHECKPOINT"
    "data/pccr/R1-f/R1-f_infos_train.pkl"
    "data/pccr/R1-f/R1-f_infos_val.pkl"
    "data/pccr/R1-f/R1-f_infos_test.pkl"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file is missing: $required_file" >&2
        exit 1
    fi
done

echo "Oracle mode: $ORACLE_MODE"
echo "GPUs: $NUM_GPUS"
echo "Source checkpoint: $SOURCE_CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
nvidia-smi

python tools/misc/print_config.py "$CONFIG" \
    --options load_from="$SOURCE_CHECKPOINT" work_dir="$OUTPUT_DIR"

python tools/oracle_adapter_sanity.py "$CONFIG" \
    --options load_from="$SOURCE_CHECKPOINT" work_dir="$OUTPUT_DIR"

bash tools/dist_train.sh "$CONFIG" "$NUM_GPUS" \
    --work-dir "$OUTPUT_DIR" \
    --seed 0 \
    --cfg-options load_from="$SOURCE_CHECKPOINT"

ADAPTED_CHECKPOINT="$OUTPUT_DIR/latest.pth"
if [[ ! -f "$ADAPTED_CHECKPOINT" ]]; then
    echo "Expected adapted checkpoint is missing: $ADAPTED_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/evaluation"

# Held-out target-rig test set.
bash tools/dist_test.sh "$CONFIG" "$ADAPTED_CHECKPOINT" "$NUM_GPUS" \
    --eval bbox \
    --out "$OUTPUT_DIR/evaluation/R1-f_results.pkl"

# Source-rig retention test.
bash tools/dist_test.sh "$CONFIG" "$ADAPTED_CHECKPOINT" "$NUM_GPUS" \
    --eval bbox \
    --out "$OUTPUT_DIR/evaluation/R1_results.pkl" \
    --cfg-options \
        data.test.data_root="data/pccr/R1/" \
        data.test.ann_file="data/pccr/R1/R1_infos_test.pkl"

if [[ "$EVAL_ALL_RIGS" == "1" ]]; then
    export MODEL_NAME="PETR-oracle-$ORACLE_MODE"
    export TRAIN_RIG="R1-f"
    export CONFIG
    export CHECKPOINT="$ADAPTED_CHECKPOINT"
    export DATA_ROOT="${DATA_ROOT:-data/pccr}"
    export OUTPUT_ROOT="$OUTPUT_DIR/cross_rig"
    export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
    export WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
    export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
    bash tools/eval_pccr_r1_all_rigs.sh
fi

echo "Completed $ORACLE_MODE oracle adapter."
echo "Checkpoint: $ADAPTED_CHECKPOINT"
echo "Target result: $OUTPUT_DIR/evaluation/R1-f_results.pkl"
echo "Source result: $OUTPUT_DIR/evaluation/R1_results.pkl"
