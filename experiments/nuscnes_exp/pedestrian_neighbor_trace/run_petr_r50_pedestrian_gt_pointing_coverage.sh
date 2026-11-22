#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
RESULTS_JSON="${RESULTS_JSON:-$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json}"
CACHED_PROJECTIONS="${CACHED_PROJECTIONS:-$SCRIPT_DIR/outputs/remaining_fp_projection_audit_4m/petr_r50/pedestrian/remaining_fp_records.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/gt_pointing_coverage/petr_r50/pedestrian}"

cd "$PETR_DIR"
conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/gt_pointing_coverage.py" \
    --results-json "$RESULTS_JSON" \
    --data-root "$DATA_ROOT" \
    --cached-projections "$CACHED_PROJECTIONS" \
    --output-dir "$OUTPUT_DIR" \
    --class-name pedestrian \
    --projection-iou-threshold 0.01

echo "GT pointing coverage: $OUTPUT_DIR/visibility_summary.csv"
