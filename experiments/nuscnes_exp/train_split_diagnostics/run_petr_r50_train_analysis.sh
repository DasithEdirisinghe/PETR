#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
TRACE_DIR="$PETR_DIR/experiments/nuscnes_exp/pedestrian_neighbor_trace"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/petr_r50/train}"
RESULTS_JSON="${RESULTS_JSON:-$OUTPUT_ROOT/inference/formatted/pts_bbox/results_nusc.json}"
CLASSES="${CLASSES:-car truck bus trailer construction_vehicle pedestrian motorcycle bicycle traffic_cone barrier}"
THRESHOLD="${THRESHOLD:-4}"
TARGET_MARGIN_IOU2D="${TARGET_MARGIN_IOU2D:-0.05}"
MAX_GT="${MAX_GT:-0}"
FORCE="${FORCE:-0}"

cd "$PETR_DIR"
[[ -s "$RESULTS_JSON" ]] || {
  echo "Missing train predictions: $RESULTS_JSON" >&2
  echo "Run run_petr_r50_train_inference.sh first." >&2
  exit 1
}

echo "=== Official nuScenes TRAIN diagnostic AP ==="
eval_args=()
[[ "$FORCE" == "1" ]] && eval_args+=(--force)
conda run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/evaluate_train_ap.py" \
  --results-json "$RESULTS_JSON" --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/official_eval" "${eval_args[@]}"

ANALYSIS_RESULTS_JSON="$(tr -d '\r\n' < \
  "$OUTPUT_ROOT/official_eval/evaluation_results_path.txt")"
[[ -s "$ANALYSIS_RESULTS_JSON" ]] || {
  echo "Normalized train submission is missing: $ANALYSIS_RESULTS_JSON" >&2
  exit 1
}

echo "=== Rule-4 TRAIN analysis ==="
for class_name in $CLASSES; do
  output_dir="$OUTPUT_ROOT/rule4_4m_margin_0p05/$class_name"
  if [[ "$FORCE" != "1" && -s "$output_dir/summary.json" ]]; then
    echo "Reusing $output_dir/summary.json"
    continue
  fi
  echo "Rule 4: $class_name"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$TRACE_DIR/target_corresponding_alternatives.py" \
    --model-name "PETR R50 train split" --class-name "$class_name" \
    --results-json "$ANALYSIS_RESULTS_JSON" --data-root "$DATA_ROOT" \
    --output-dir "$output_dir" --eval-set train \
    --threshold "$THRESHOLD" \
    --target-margin-iou2d "$TARGET_MARGIN_IOU2D" --max-gt "$MAX_GT"
done

echo "=== Fixed-box full-ranking oracle on TRAIN ==="
oracle_args=()
for class_name in $CLASSES; do oracle_args+=("$class_name"); done
conda run --no-capture-output -n "$CONDA_ENV" \
  python "$TRACE_DIR/all_class_reranking_oracle.py" \
  --results-json "$ANALYSIS_RESULTS_JSON" --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/oracle_ap" --eval-set train \
  --source-metrics "$OUTPUT_ROOT/official_eval/metrics_summary.json" \
  --classes "${oracle_args[@]}"

conda run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/summarize_train_diagnostics.py" --root "$OUTPUT_ROOT"

echo "Completed PETR-R50 train diagnostics: $OUTPUT_ROOT"
