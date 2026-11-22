#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-streampetr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
CLASSES="${CLASSES:-car bicycle}"
MAX_GALLERY_CASES="${MAX_GALLERY_CASES:-1000}"

cd "$PETR_DIR"
for class_name in $CLASSES; do
    output_dir="$SCRIPT_DIR/outputs/${class_name}_image_alignment_4m"
    gallery_dir="$SCRIPT_DIR/outputs/visualizations/hypothesis_pairs_${class_name}"
    if [[ ! -s "$output_dir/all_gt.csv" || ! -s "$output_dir/summary.json" ]]; then
        echo "Missing analysis outputs for $class_name under $output_dir" >&2
        exit 1
    fi
    echo "Rebuilding $class_name gallery with ego/camera BEV context"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/build_hypothesis_pair_gallery.py" \
        --class-name "$class_name" \
        --cases-csv "$output_dir/all_gt.csv" \
        --summary-json "$output_dir/summary.json" \
        --data-root "$DATA_ROOT" \
        --gallery-dir "$gallery_dir" \
        --max-cases "$MAX_GALLERY_CASES" \
        --sample-seed 0
done

echo "Rebuilt galleries: $CLASSES"
