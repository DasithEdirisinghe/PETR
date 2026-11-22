#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
RESULTS_JSON="${RESULTS_JSON:-$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json}"
RULE4_CASES="${RULE4_CASES:-$SCRIPT_DIR/outputs/multimodel_rule4_analysis_4m_margin_0p05/petr_r50/pedestrian/rule4_cases.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/remaining_fp_projection_audit_4m/petr_r50/pedestrian}"
PROJECTION_IOU_THRESHOLD="${PROJECTION_IOU_THRESHOLD:-0.01}"

cd "$PETR_DIR"
conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/remaining_fp_projection_audit.py" \
    --results-json "$RESULTS_JSON" \
    --data-root "$DATA_ROOT" \
    --rule4-cases "$RULE4_CASES" \
    --output-dir "$OUTPUT_DIR" \
    --class-name pedestrian \
    --threshold 4 \
    --projection-iou-threshold "$PROJECTION_IOU_THRESHOLD"

echo "Remaining-FP audit: $OUTPUT_DIR/summary.csv"
