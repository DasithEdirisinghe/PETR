#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
TRACE_DIR="$PETR_DIR/experiments/nuscnes_exp/pedestrian_neighbor_trace"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
PREDICTIONS="${PREDICTIONS:-$SCRIPT_DIR/outputs/petr_r50/train/inference/formatted/pts_bbox/results_nusc.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/petr_r50/train_info_subset_25433}"
CLASSES="${CLASSES:-car truck bus trailer construction_vehicle pedestrian motorcycle bicycle traffic_cone barrier}"
THRESHOLD="${THRESHOLD:-4}"
TARGET_MARGIN_IOU2D="${TARGET_MARGIN_IOU2D:-0.05}"
MAX_GT="${MAX_GT:-0}"
FORCE="${FORCE:-0}"

cd "$PETR_DIR"
[[ -s "$PREDICTIONS" ]] || { echo "Missing predictions: $PREDICTIONS" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

echo "Analyzing the existing PETR train-info subset predictions"
echo "Predictions: $PREDICTIONS"
echo "Outputs: $OUTPUT_ROOT"

for class_name in $CLASSES; do
  output_dir="$OUTPUT_ROOT/rule4_4m_margin_0p05/$class_name"
  if [[ "$FORCE" != "1" && -s "$output_dir/summary.json" ]]; then
    echo "Reusing $output_dir/summary.json"
    continue
  fi
  echo "Rule 4 subset: $class_name"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$TRACE_DIR/target_corresponding_alternatives.py" \
    --model-name "PETR R50 train-info subset" --class-name "$class_name" \
    --results-json "$PREDICTIONS" --data-root "$DATA_ROOT" \
    --output-dir "$output_dir" --eval-set train \
    --allow-prediction-subset --threshold "$THRESHOLD" \
    --target-margin-iou2d "$TARGET_MARGIN_IOU2D" --max-gt "$MAX_GT"
done

oracle_args=()
for class_name in $CLASSES; do oracle_args+=("$class_name"); done
echo "Fixed-box oracle subset AP"
conda run --no-capture-output -n "$CONDA_ENV" \
  python "$TRACE_DIR/all_class_reranking_oracle.py" \
  --results-json "$PREDICTIONS" --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/oracle_ap" --eval-set train \
  --restrict-to-result-tokens --classes "${oracle_args[@]}"

conda run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/summarize_subset_diagnostics.py" --root "$OUTPUT_ROOT"
echo "Completed subset analysis: $OUTPUT_ROOT/README.md"

