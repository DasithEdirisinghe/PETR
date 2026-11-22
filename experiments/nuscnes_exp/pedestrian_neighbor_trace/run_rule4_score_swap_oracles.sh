#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
RULE4_ROOT="${RULE4_ROOT:-$SCRIPT_DIR/outputs/multimodel_rule4_analysis_4m_margin_0p05}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/rule4_score_swap_oracle}"
MODELS="${MODELS:-petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16}"
BUILD_ONLY="${BUILD_ONLY:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

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
    source_results="${RESULT_FILES[$i]}"
    selected "$MODELS" "$slug" || continue
    model_dir="$OUTPUT_ROOT/$slug"
    build_valid=0
    built_now=0
    if [[ -s "$model_dir/build_summary.json" ]] && \
       grep -q '"build_version": 3' "$model_dir/build_summary.json" && \
       [[ -s "$model_dir/case1/results_nusc.json" ]] && \
       [[ -s "$model_dir/case2/results_nusc.json" ]] && \
       [[ -s "$model_dir/combined/results_nusc.json" ]]; then
        build_valid=1
    fi
    if [[ "$FORCE_BUILD" == "1" || "$build_valid" != "1" ]]; then
        echo "=== Build score-swap submissions | $name ==="
        conda run --no-capture-output -n "$CONDA_ENV" \
            python "$SCRIPT_DIR/build_rule4_score_swap_oracles.py" \
            --model-name "$name" \
            --source-results "$source_results" \
            --rule4-model-dir "$RULE4_ROOT/$slug" \
            --output-dir "$model_dir"
        built_now=1
    else
        echo "Reusing $model_dir/build_summary.json"
    fi
    if [[ "$BUILD_ONLY" != "1" ]]; then
        echo "=== Official nuScenes evaluation | $name ==="
        force_args=()
        if [[ "$FORCE_EVAL" == "1" || "$built_now" == "1" ]]; then
            force_args+=(--force)
        fi
        conda run --no-capture-output -n "$CONDA_ENV" \
            python "$SCRIPT_DIR/evaluate_rule4_score_swap_oracles.py" \
            --model-name "$name" \
            --model-dir "$model_dir" \
            --data-root "$DATA_ROOT" \
            "${force_args[@]}"
    fi
done

if [[ "$BUILD_ONLY" != "1" ]]; then
    python "$SCRIPT_DIR/summarize_rule4_score_swap_oracles.py" \
        --root "$OUTPUT_ROOT"
fi
echo "Rule-4 score-swap oracle outputs: $OUTPUT_ROOT"
