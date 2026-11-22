#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-streampetr}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/pedestrian_image_alignment_4m}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"

cd "$PETR_DIR"
conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/build_hypothesis_pair_gallery.py" \
    --cases-csv "$OUTPUT_DIR/all_gt.csv" \
    --summary-json "$OUTPUT_DIR/summary.json" \
    --data-root "$DATA_ROOT" \
    --gallery-dir "$SCRIPT_DIR/outputs/visualizations/hypothesis_pairs" \
    --max-cases 1000 \
    --sample-seed 0

echo "Hypothesis-pair gallery rebuilt with a deterministic 1,000-case sample."
