#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-streampetr}"
RESULTS_JSON="${RESULTS_JSON:-$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
CLASSES="${CLASSES:-car bicycle}"
THRESHOLD="${THRESHOLD:-4}"
MAX_GALLERY_CASES="${MAX_GALLERY_CASES:-1000}"

cd "$PETR_DIR"
for class_name in $CLASSES; do
    output_dir="$SCRIPT_DIR/outputs/${class_name}_image_alignment_4m"
    gallery_dir="$SCRIPT_DIR/outputs/visualizations/hypothesis_pairs_${class_name}"

    echo "Analyzing PETR class: $class_name"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/pedestrian_image_alignment.py" \
        --class-name "$class_name" \
        --results-json "$RESULTS_JSON" \
        --data-root "$DATA_ROOT" \
        --output-dir "$output_dir" \
        --threshold "$THRESHOLD"

    echo "Building $class_name hypothesis gallery"
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

echo "Completed classes: $CLASSES"
echo "Galleries are under: $SCRIPT_DIR/outputs/visualizations/hypothesis_pairs_<class>/"
