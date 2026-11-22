#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
CLASSES="${CLASSES:-pedestrian car bicycle}"
MAX_PER_CATEGORY="${MAX_PER_CATEGORY:-500}"

cd "$PETR_DIR"
for class_name in $CLASSES; do
    echo "Preparing common multi-model cases for $class_name"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/prepare_common_multimodel_gallery.py" \
        --class-name "$class_name" \
        --data-root "$DATA_ROOT" \
        --max-per-category "$MAX_PER_CATEGORY" \
        --seed 0
done

echo "Prepared synchronized case data for: $CLASSES"
