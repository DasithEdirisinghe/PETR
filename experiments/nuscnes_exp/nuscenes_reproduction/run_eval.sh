#!/bin/bash
set -euo pipefail

CONFIG=${1:-projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_local.py}
CHECKPOINT=${2:-ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth}
OUTPUT_DIR=${3:-experiments/nuscenes_reproduction/outputs}
NUM_GPUS=${NUM_GPUS:-1}
PORT=${PORT:-29511}

test -f "$CONFIG"
test -f "$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"

PORT="$PORT" bash tools/dist_test.sh \
  "$CONFIG" \
  "$CHECKPOINT" \
  "$NUM_GPUS" \
  --out "$OUTPUT_DIR/predictions.pkl" \
  --eval bbox \
  --eval-options jsonfile_prefix="$OUTPUT_DIR/formatted" \
  2>&1 | tee "$OUTPUT_DIR/eval.log"

python3 experiments/nuscenes_reproduction/plot_metrics.py \
  --metrics "$OUTPUT_DIR/formatted" \
  --output-dir "$OUTPUT_DIR/plots"
