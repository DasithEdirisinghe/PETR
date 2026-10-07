#!/usr/bin/env bash
# Compare both PETR checkpoints on one target rig's frames.
# Usage: bash experiments/cross_rig_attention/run_car_model_comparison.sh [val|test] [R1-f|R1]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

SPLIT="${1:-val}"
TARGET_RIG="${2:-R1-f}"
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "Split must be val or test, got: $SPLIT" >&2
    exit 2
fi
if [[ "$TARGET_RIG" != "R1" && "$TARGET_RIG" != "R1-f" ]]; then
    echo "Target rig must be R1 or R1-f, got: $TARGET_RIG" >&2
    exit 2
fi

if [[ "$TARGET_RIG" == "R1" ]]; then
    RUN_NAME="r1_${SPLIT}"
else
    RUN_NAME="r1f_${SPLIT}"
fi
OUT="${CAR_COMPARE_OUTPUT:-experiments/cross_rig_attention/output/$RUN_NAME}"
ANN="data/pccr/$TARGET_RIG/${TARGET_RIG}_infos_${SPLIT}.pkl"
CONFIG="projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py"
R1_CKPT="results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth"
R1F_CKPT="results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$OUT"

if [[ ! -f "$ANN" ]]; then
    echo "Missing annotation file: $ANN" >&2
    exit 1
fi

resolve_result() {
    local label="$1"
    local checkpoint="$2"
    local cached_dir="$OUT/$label/formatted"
    local cached_json="$cached_dir/results_pccr.json"
    local cached_metrics="$cached_dir/metrics_summary.json"
    local saved_dir="results/petr_r50dcn_gridmask_p4_800x320_pccr/$label/cross_rig/evaluations/$TARGET_RIG/formatted"

    if [[ -f "$cached_json" && -f "$cached_metrics" ]]; then
        echo "Reusing experiment cache for $label $SPLIT" >&2
        RES_JSON="$cached_json"
        RES_METRICS="$cached_metrics"
        return
    fi
    if [[ "$SPLIT" == "test" && -f "$saved_dir/results_pccr.json" && -f "$saved_dir/metrics_summary.json" ]]; then
        echo "Reusing saved $label $TARGET_RIG test detections; token coverage will be checked" >&2
        RES_JSON="$saved_dir/results_pccr.json"
        RES_METRICS="$saved_dir/metrics_summary.json"
        return
    fi
    if [[ "${CAR_COMPARE_ALLOW_INFERENCE:-1}" != "1" ]]; then
        echo "No cached $label $TARGET_RIG $SPLIT detections. Set CAR_COMPARE_ALLOW_INFERENCE=1 to run normal PETR inference." >&2
        exit 1
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint" >&2
        exit 1
    fi
    mkdir -p "$cached_dir"
    echo "Running normal PETR inference once for $label on $TARGET_RIG $SPLIT" >&2
    "$PYTHON_BIN" tools/test.py "$CONFIG" "$checkpoint" \
        --out "$OUT/$label/predictions.pkl" \
        --eval bbox \
        --eval-options "jsonfile_prefix=$cached_dir" \
        --cfg-options \
            "data.test.data_root=data/pccr/$TARGET_RIG/" \
            "data.test.ann_file=$ANN" \
        > "$OUT/$label/inference.log" 2>&1
    if [[ ! -f "$cached_json" || ! -f "$cached_metrics" ]]; then
        echo "Inference finished but formatted results are missing; see $OUT/$label/inference.log" >&2
        exit 1
    fi
    RES_JSON="$cached_json"
    RES_METRICS="$cached_metrics"
}

resolve_result R1 "$R1_CKPT"
R1_JSON="$RES_JSON"
R1_METRICS="$RES_METRICS"
resolve_result R1-f "$R1F_CKPT"
R1F_JSON="$RES_JSON"
R1F_METRICS="$RES_METRICS"

"$PYTHON_BIN" experiments/cross_rig_attention/compare_r1_r1f_cars.py \
    --ann-file "$ANN" \
    --r1-results "$R1_JSON" \
    --r1-metrics "$R1_METRICS" \
    --r1f-results "$R1F_JSON" \
    --r1f-metrics "$R1F_METRICS" \
    --target-rig "$TARGET_RIG" \
    --output-dir "$OUT/analysis" \
    --score-threshold "${CAR_COMPARE_SCORE_THRESHOLD:-0.35}" \
    --association-distance "${CAR_COMPARE_ASSOCIATION_DISTANCE:-4.0}"

echo "Car comparison: $OUT/analysis"
