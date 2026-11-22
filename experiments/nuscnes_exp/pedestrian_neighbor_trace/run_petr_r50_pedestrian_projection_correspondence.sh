#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"

cd "$PETR_DIR"
conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/projection_correspondence_audit.py" \
    --data-root "$WORKSPACE/RoboBEV/data/sets/nuscenes" \
    --all-gt-csv "$SCRIPT_DIR/outputs/multimodel_hypothesis_analysis_4m/petr_r50/pedestrian/hypothesis/all_gt.csv" \
    --output-dir "$SCRIPT_DIR/outputs/projection_correspondence_audit_4m/petr_r50/pedestrian" \
    --background-iou-threshold "${BACKGROUND_IOU_THRESHOLD:-0.000001}"
