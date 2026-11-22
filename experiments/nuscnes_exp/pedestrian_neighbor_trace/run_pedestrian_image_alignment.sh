#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-streampetr}"

RESULTS_JSON="${RESULTS_JSON:-$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/pedestrian_image_alignment_4m}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is unavailable. Load Miniconda/Anaconda first." >&2
    exit 1
fi
if [[ ! -s "$RESULTS_JSON" ]]; then
    echo "Missing PETR results: $RESULTS_JSON" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/v1.0-trainval" ]]; then
    echo "Missing nuScenes metadata: $DATA_ROOT/v1.0-trainval" >&2
    exit 1
fi

cd "$PETR_DIR"
echo "Conda environment: $CONDA_ENV"
echo "PETR predictions: $RESULTS_JSON"
echo "Output: $OUTPUT_DIR"
conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/pedestrian_image_alignment.py" \
    --results-json "$RESULTS_JSON" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --threshold 4 \
    "$@"

conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/build_hypothesis_pair_gallery.py" \
    --cases-csv "$OUTPUT_DIR/all_gt.csv" \
    --data-root "$DATA_ROOT" \
    --gallery-dir "$SCRIPT_DIR/outputs/visualizations/hypothesis_pairs" \
    --max-cases 1000

python3 "$WORKSPACE/StreamPETR/experiments/eva02_gallery/build_fast_gallery.py" \
    --refresh-page-only
