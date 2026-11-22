#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-streampetr}"
RESULTS_JSON="${RESULTS_JSON:-$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
CLASSES="${CLASSES:-pedestrian car bicycle}"
THRESHOLD="${THRESHOLD:-4}"

cd "$PETR_DIR"
for class_name in $CLASSES; do
    input_dir="$SCRIPT_DIR/outputs/${class_name}_image_alignment_4m"
    if [[ "$class_name" == "pedestrian" ]]; then
        input_dir="$SCRIPT_DIR/outputs/pedestrian_image_alignment_4m"
    fi
    output_dir="$SCRIPT_DIR/outputs/cross_assignment_audit_4m/$class_name"
    echo "Auditing cross-assigned official TPs for $class_name"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/cross_assignment_audit.py" \
        --class-name "$class_name" \
        --all-gt-csv "$input_dir/all_gt.csv" \
        --results-json "$RESULTS_JSON" \
        --data-root "$DATA_ROOT" \
        --output-dir "$output_dir" \
        --threshold "$THRESHOLD"
done

echo "Cross-assignment audit completed for: $CLASSES"
