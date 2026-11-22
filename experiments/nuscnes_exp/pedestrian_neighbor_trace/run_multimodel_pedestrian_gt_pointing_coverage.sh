#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/gt_pointing_coverage}"
MODELS="${MODELS:-petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16}"
FORCE="${FORCE:-0}"

MODEL_SLUGS=(petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16)
MODEL_NAMES=("PETR R50" "StreamPETR R50" "RepDETR3D EVA02-L" "PETR EVA02-L epoch 16")
RESULT_FILES=(
  "$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json"
  "$WORKSPACE/StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json"
  "$WORKSPACE/StreamPETR/experiments/eva02_gallery/outputs/inference/formatted/pts_bbox/results_nusc.json"
  "$PETR_DIR/results/petr_eva02_l_p4_800x320_nuscenes_global_batch_8/epoch_16_formatted/pts_bbox/results_nusc.json"
)

selected() {
    case " $1 " in
        *" $2 "*) return 0 ;;
        *) return 1 ;;
    esac
}

cd "$PETR_DIR"
mkdir -p "$OUTPUT_ROOT"
for i in "${!MODEL_SLUGS[@]}"; do
    slug="${MODEL_SLUGS[$i]}"
    name="${MODEL_NAMES[$i]}"
    results="${RESULT_FILES[$i]}"
    selected "$MODELS" "$slug" || continue
    output_dir="$OUTPUT_ROOT/$slug/pedestrian"
    if [[ "$FORCE" != "1" && -s "$output_dir/summary.json" && \
          -s "$output_dir/per_gt.csv" ]]; then
        echo "Reusing completed coverage: $name"
        continue
    fi
    if [[ ! -s "$results" ]]; then
        echo "Missing predictions for $name: $results" >&2
        exit 1
    fi
    cache="$SCRIPT_DIR/outputs/remaining_fp_projection_audit_4m/$slug/pedestrian/remaining_fp_records.csv"
    cache_args=()
    if [[ -s "$cache" ]]; then
        echo "Using cached projections for $name: $cache"
        cache_args+=(--cached-projections "$cache")
    else
        echo "No projection cache for $name; projecting every pedestrian prediction"
    fi
    echo "=== GT-pointing coverage | $name ==="
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/gt_pointing_coverage.py" \
        --results-json "$results" \
        --data-root "$DATA_ROOT" \
        --output-dir "$output_dir" \
        --class-name pedestrian \
        --projection-iou-threshold 0.01 \
        "${cache_args[@]}"
done

python "$SCRIPT_DIR/summarize_multimodel_gt_pointing_coverage.py" \
    --root "$OUTPUT_ROOT"
echo "Comparison: $OUTPUT_ROOT/comparison.csv"
