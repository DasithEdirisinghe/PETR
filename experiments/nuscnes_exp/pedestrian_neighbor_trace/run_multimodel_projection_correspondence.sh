#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
INPUT_ROOT="${INPUT_ROOT:-$SCRIPT_DIR/outputs/multimodel_hypothesis_analysis_4m}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/projection_correspondence_audit_4m}"
MODELS="${MODELS:-petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16}"
CLASSES="${CLASSES:-pedestrian car bicycle}"
BACKGROUND_IOU_THRESHOLD="${BACKGROUND_IOU_THRESHOLD:-0.000001}"
FORCE="${FORCE:-0}"

MODEL_SLUGS=(petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16)
MODEL_NAMES=("PETR R50" "StreamPETR R50" "RepDETR3D EVA02-L" "PETR EVA02-L epoch 16")

selected() {
    case " $1 " in
        *" $2 "*) return 0 ;;
        *) return 1 ;;
    esac
}

for i in "${!MODEL_SLUGS[@]}"; do
    model_slug="${MODEL_SLUGS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    selected "$MODELS" "$model_slug" || continue

    for class_name in $CLASSES; do
        input_csv="$INPUT_ROOT/$model_slug/$class_name/hypothesis/all_gt.csv"
        output_dir="$OUTPUT_ROOT/$model_slug/$class_name"
        summary="$output_dir/summary.json"

        if [[ ! -s "$input_csv" ]]; then
            echo "Missing hypothesis input: $input_csv" >&2
            exit 1
        fi
        if [[ "$FORCE" != "1" && -s "$summary" ]]; then
            echo "Reusing $summary"
            continue
        fi

        echo "=== $model_name | $class_name ==="
        conda run --no-capture-output -n "$CONDA_ENV" \
            python "$SCRIPT_DIR/projection_correspondence_audit.py" \
            --model-name "$model_name" \
            --class-name "$class_name" \
            --data-root "$DATA_ROOT" \
            --all-gt-csv "$input_csv" \
            --output-dir "$output_dir" \
            --background-iou-threshold "$BACKGROUND_IOU_THRESHOLD"
    done
done

echo "Completed projection-correspondence audits: $OUTPUT_ROOT"
