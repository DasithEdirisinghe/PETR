#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
CONFIG="${CONFIG:-projects/configs/petr/petr_r50dcn_gridmask_p4_nuscenes_train_eval.py}"
CHECKPOINT="${CHECKPOINT:-ckpts/petr_r50dcn_gridmask_p4_epoch_24.pth}"
GPUS="${GPUS:-1}"
PORT="${PORT:-29541}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/petr_r50/train}"
INFERENCE_DIR="$OUTPUT_ROOT/inference"
PICKLE="$INFERENCE_DIR/raw_results.pkl"
JSON_PREFIX="$INFERENCE_DIR/formatted"
RESULTS_JSON="$JSON_PREFIX/pts_bbox/results_nusc.json"

cd "$PETR_DIR"
[[ -f "$CONFIG" ]] || { echo "Missing config: $PETR_DIR/$CONFIG" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $PETR_DIR/$CHECKPOINT" >&2; exit 1; }
[[ -f data/nuscenes/nuscenes_infos_train.pkl ]] || {
  echo "Missing complete PETR train info file: $PETR_DIR/data/nuscenes/nuscenes_infos_train.pkl" >&2
  exit 1
}
mkdir -p "$INFERENCE_DIR"

if [[ "${FORCE:-0}" == "1" || ! -s "$RESULTS_JSON" ]]; then
  echo "Running PETR-R50 inference on the nuScenes TRAIN split"
  PORT="$PORT" conda run --no-capture-output -n "$CONDA_ENV" \
    bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
    --out "$PICKLE" --format-only \
    --eval-options "jsonfile_prefix=$JSON_PREFIX"
else
  echo "Reusing train predictions: $RESULTS_JSON"
fi

[[ -s "$RESULTS_JSON" ]] || {
  echo "Inference finished but expected JSON was not created: $RESULTS_JSON" >&2
  exit 1
}
echo "Train prediction JSON: $RESULTS_JSON"

