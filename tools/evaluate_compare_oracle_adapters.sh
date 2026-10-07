#!/usr/bin/env bash

# Evaluate completed oracle modes on R1-f and R1, then build comparison plots.
# Usage: bash tools/evaluate_compare_oracle_adapters.sh [NUM_GPUS] [MODE ...]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="${PETR_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

NUM_GPUS="${1:-1}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
if [[ "$#" -gt 0 ]]; then
    MODES=("$@")
else
    MODES=(query key query_key reference output)
fi

ROOT="${ORACLE_ROOT:-experiments/oracle_adapters/output/R1_to_R1-f}"
PLOT_DIR="${PLOT_DIR:-$ROOT/comparison}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_INFERENCE="${FORCE_INFERENCE:-0}"
mkdir -p "$PLOT_DIR"

for mode in "${MODES[@]}"; do
    case "$mode" in
        query|key|query_key|reference|output) ;;
        *)
            echo "Unknown mode '$mode'; skipping." >&2
            continue
            ;;
    esac

    config="projects/configs/petr/oracle_adapters/petr_r1_to_r1f_oracle_${mode}.py"
    mode_dir="$ROOT/$mode"
    checkpoint="$mode_dir/latest.pth"
    eval_dir="$mode_dir/evaluation"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Skipping unfinished mode '$mode': $checkpoint is absent."
        continue
    fi
    mkdir -p "$eval_dir"

    target_log="$eval_dir/R1-f_metrics.log"
    target_predictions="$eval_dir/R1-f_results.pkl"
    if [[ -f "$target_log" && "$FORCE_EVAL" != "1" ]]; then
        echo "Reusing $target_log"
    elif [[ -f "$target_predictions" && "$FORCE_INFERENCE" != "1" ]]; then
        echo "Computing $mode R1-f metrics from saved predictions (no inference)."
        python tools/evaluate_saved_pccr_results.py \
            "$config" "$target_predictions" 2>&1 | tee "$target_log"
    else
        echo "Evaluating $mode on target rig R1-f."
        PORT="${PORT:-29500}" bash tools/dist_test.sh \
            "$config" "$checkpoint" "$NUM_GPUS" \
            --eval bbox \
            --out "$target_predictions" \
            2>&1 | tee "$target_log"
    fi

    source_log="$eval_dir/R1_metrics.log"
    source_predictions="$eval_dir/R1_results.pkl"
    if [[ -f "$source_log" && "$FORCE_EVAL" != "1" ]]; then
        echo "Reusing $source_log"
    elif [[ -f "$source_predictions" && "$FORCE_INFERENCE" != "1" ]]; then
        echo "Computing $mode R1 metrics from saved predictions (no inference)."
        python tools/evaluate_saved_pccr_results.py \
            "$config" "$source_predictions" \
            --cfg-options \
                data.test.data_root="data/pccr/R1/" \
                data.test.ann_file="data/pccr/R1/R1_infos_test.pkl" \
            2>&1 | tee "$source_log"
    else
        echo "Evaluating $mode on source rig R1."
        PORT="${PORT:-29500}" bash tools/dist_test.sh \
            "$config" "$checkpoint" "$NUM_GPUS" \
            --eval bbox \
            --out "$eval_dir/R1_results.pkl" \
            --cfg-options \
                data.test.data_root="data/pccr/R1/" \
                data.test.ann_file="data/pccr/R1/R1_infos_test.pkl" \
            2>&1 | tee "$source_log"
    fi
done

python tools/analysis_tools/compare_oracle_adapters.py \
    --root "$ROOT" \
    --output-dir "$PLOT_DIR" \
    --baseline-json \
      results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/cross_rig/standardized/PETR/trained_on_R1.json \
    --target-json \
      results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/cross_rig/standardized/PETR/trained_on_R1-f.json

echo "Oracle comparison complete: $PLOT_DIR"
