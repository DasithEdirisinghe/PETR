#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
MODELS="${MODELS:-petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16}"
CLASSES="${CLASSES:-car truck bus trailer construction_vehicle pedestrian motorcycle bicycle traffic_cone barrier}"
THRESHOLD="${THRESHOLD:-4}"
TARGET_MARGIN_IOU2D="${TARGET_MARGIN_IOU2D:-0.05}"
MARGIN_TAG="${TARGET_MARGIN_IOU2D//./p}"
THRESHOLD_TAG="${THRESHOLD//./p}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/multimodel_rule4_analysis_${THRESHOLD_TAG}m_margin_${MARGIN_TAG}}"
MAX_GT="${MAX_GT:-0}"
FORCE="${FORCE:-0}"

PETR_EVA_RESULTS="${PETR_EVA_RESULTS:-$PETR_DIR/results/petr_eva02_l_p4_800x320_nuscenes_global_batch_8/epoch_16_formatted/pts_bbox/results_nusc.json}"
MODEL_SLUGS=(petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16)
MODEL_NAMES=("PETR R50" "StreamPETR R50" "RepDETR3D EVA02-L" "PETR EVA02-L epoch 16")
RESULT_FILES=(
  "$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json"
  "$WORKSPACE/StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json"
  "$WORKSPACE/StreamPETR/experiments/eva02_gallery/outputs/inference/formatted/pts_bbox/results_nusc.json"
  "$PETR_EVA_RESULTS"
)

selected() {
    case " $1 " in
        *" $2 "*) return 0 ;;
        *) return 1 ;;
    esac
}

cd "$PETR_DIR"
for i in "${!MODEL_SLUGS[@]}"; do
    model_slug="${MODEL_SLUGS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    results_json="${RESULT_FILES[$i]}"
    selected "$MODELS" "$model_slug" || continue
    if [[ ! -s "$results_json" ]]; then
        echo "Missing predictions for $model_name: $results_json" >&2
        exit 1
    fi
    for class_name in $CLASSES; do
        output_dir="$OUTPUT_ROOT/$model_slug/$class_name"
        if [[ "$FORCE" != "1" && -s "$output_dir/summary.json" ]]; then
            echo "Reusing $output_dir/summary.json"
            continue
        fi
        echo "=== Rule 4 | $model_name | $class_name ==="
        conda run --no-capture-output -n "$CONDA_ENV" \
            python "$SCRIPT_DIR/target_corresponding_alternatives.py" \
            --model-name "$model_name" \
            --class-name "$class_name" \
            --results-json "$results_json" \
            --data-root "$DATA_ROOT" \
            --output-dir "$output_dir" \
            --threshold "$THRESHOLD" \
            --target-margin-iou2d "$TARGET_MARGIN_IOU2D" \
            --max-gt "$MAX_GT"
    done
done

python "$SCRIPT_DIR/summarize_rule4_multimodel.py" --root "$OUTPUT_ROOT"
echo "Completed Rule-4 analysis: $OUTPUT_ROOT/comparison.csv"
