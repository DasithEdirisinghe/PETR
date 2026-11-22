#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$PETR_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/RoboBEV/data/sets/nuscenes}"
CLASSES="${CLASSES:-pedestrian car bicycle}"
MODELS="${MODELS:-petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16}"
THRESHOLD="${THRESHOLD:-4}"
MAX_GALLERY_CASES="${MAX_GALLERY_CASES:-1000}"
BUILD_GALLERIES="${BUILD_GALLERIES:-1}"
RUN_EVA_INFERENCE="${RUN_EVA_INFERENCE:-1}"
FORCE="${FORCE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/multimodel_hypothesis_analysis_4m}"

PETR_EVA_CONFIG="${PETR_EVA_CONFIG:-$PETR_DIR/projects/configs/petr/petr_eva02_l_p4_800x320_nuscenes_global_batch_8.py}"
PETR_EVA_CHECKPOINT="${PETR_EVA_CHECKPOINT:-$PETR_DIR/results/petr_eva02_l_p4_800x320_nuscenes_global_batch_8/epoch_16.pth}"
PETR_EVA_PREFIX="${PETR_EVA_PREFIX:-$PETR_DIR/results/petr_eva02_l_p4_800x320_nuscenes_global_batch_8/epoch_16_formatted}"
PETR_EVA_RESULTS="${PETR_EVA_RESULTS:-$PETR_EVA_PREFIX/pts_bbox/results_nusc.json}"

if [[ ! -s "$PETR_EVA_RESULTS" ]]; then
    if [[ "$RUN_EVA_INFERENCE" != "1" ]]; then
        echo "Missing PETR EVA02 epoch-16 predictions: $PETR_EVA_RESULTS" >&2
        echo "Rerun with RUN_EVA_INFERENCE=1 on a GPU node." >&2
        exit 1
    fi
    echo "Formatting PETR EVA02-L epoch-16 predictions"
    mkdir -p "$PETR_EVA_PREFIX"
    cd "$PETR_DIR"
    conda run --no-capture-output -n "$CONDA_ENV" \
        bash tools/dist_test.sh "$PETR_EVA_CONFIG" "$PETR_EVA_CHECKPOINT" 1 \
        --format-only --eval-options "jsonfile_prefix=$PETR_EVA_PREFIX"
fi

MODEL_SLUGS=(petr_r50 streampetr_r50 repdetr3d_eva02_l petr_eva02_l_epoch16)
MODEL_NAMES=("PETR R50" "StreamPETR R50" "RepDETR3D EVA02-L" "PETR EVA02-L epoch 16")
RESULT_FILES=(
  "$PETR_DIR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json"
  "$WORKSPACE/StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json"
  "$WORKSPACE/StreamPETR/experiments/eva02_gallery/outputs/inference/formatted/pts_bbox/results_nusc.json"
  "$PETR_EVA_RESULTS"
)

for i in "${!MODEL_SLUGS[@]}"; do
    model_slug="${MODEL_SLUGS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    results_json="${RESULT_FILES[$i]}"
    case " $MODELS " in
        *" $model_slug "*) ;;
        *) echo "Skipping $model_name"; continue ;;
    esac
    if [[ ! -s "$results_json" ]]; then
        echo "Missing predictions for $model_name: $results_json" >&2
        exit 1
    fi
    for class_name in $CLASSES; do
        combo="$OUTPUT_ROOT/$model_slug/$class_name"
        hypothesis="$combo/hypothesis"
        cross="$combo/cross_assignment"
        echo "=== $model_name | $class_name ==="
        if [[ "$FORCE" == "1" || ! -s "$hypothesis/summary.json" ]] || \
           ! head -n 1 "$hypothesis/all_gt.csv" 2>/dev/null | grep -q 'preferred_alt_rule_version_3'; then
            conda run --no-capture-output -n "$CONDA_ENV" \
                python "$SCRIPT_DIR/pedestrian_image_alignment.py" \
                --model-name "$model_name" --class-name "$class_name" \
                --results-json "$results_json" --data-root "$DATA_ROOT" \
                --output-dir "$hypothesis" --threshold "$THRESHOLD"
        else
            echo "Reusing $hypothesis/summary.json"
        fi
        if [[ "$FORCE" == "1" || ! -s "$cross/summary.json" ]]; then
            conda run --no-capture-output -n "$CONDA_ENV" \
                python "$SCRIPT_DIR/cross_assignment_audit.py" \
                --model-name "$model_name" --class-name "$class_name" \
                --all-gt-csv "$hypothesis/all_gt.csv" \
                --results-json "$results_json" --data-root "$DATA_ROOT" \
                --output-dir "$cross" --threshold "$THRESHOLD"
        else
            echo "Reusing $cross/summary.json"
        fi
        if [[ "$BUILD_GALLERIES" == "1" ]]; then
            if [[ "$class_name" == "pedestrian" ]]; then gallery_name=hypothesis_pairs; else gallery_name="hypothesis_pairs_$class_name"; fi
            gallery="$OUTPUT_ROOT/$model_slug/galleries/$gallery_name"
            if [[ "$FORCE" == "1" || ! -s "$gallery/cases.js" ]]; then
                conda run --no-capture-output -n "$CONDA_ENV" \
                    python "$SCRIPT_DIR/build_hypothesis_pair_gallery.py" \
                    --model-name "$model_name" --class-name "$class_name" \
                    --cases-csv "$hypothesis/all_gt.csv" \
                    --summary-json "$hypothesis/summary.json" \
                    --data-root "$DATA_ROOT" --gallery-dir "$gallery" \
                    --max-cases "$MAX_GALLERY_CASES" --sample-seed 0
            else
                echo "Reusing $gallery/cases.js"
            fi
        fi
    done
done

python3 "$SCRIPT_DIR/summarize_multimodel_hypothesis.py" --root "$OUTPUT_ROOT"
echo "Completed multi-model hypothesis analysis: $OUTPUT_ROOT/comparison.csv"
