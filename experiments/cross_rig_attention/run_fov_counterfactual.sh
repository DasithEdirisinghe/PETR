#!/usr/bin/env bash
# Usage: bash experiments/cross_rig_attention/run_fov_counterfactual.sh smoke|full|analyze|traces
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
OUT="experiments/cross_rig_attention/output/fov_counterfactual/r1f_to_r1_val"

case "$MODE" in
    smoke)
        "$PYTHON_BIN" experiments/cross_rig_attention/run_fov_counterfactual.py \
            --source-rig R1 --target-rig R1-f --split val --max-frames 8 \
            --output-dir experiments/cross_rig_attention/output/fov_counterfactual/smoke
        ;;
    full)
        "$PYTHON_BIN" experiments/cross_rig_attention/run_fov_counterfactual.py \
            --source-rig R1 --target-rig R1-f --split val
        "$PYTHON_BIN" experiments/cross_rig_attention/analyze_fov_counterfactual.py \
            --experiment-dir "$OUT" \
            --ann-file data/pccr/R1-f/R1-f_infos_val.pkl
        ;;
    analyze)
        "$PYTHON_BIN" experiments/cross_rig_attention/analyze_fov_counterfactual.py \
            --experiment-dir "$OUT" \
            --ann-file data/pccr/R1-f/R1-f_infos_val.pkl
        ;;
    traces)
        "$PYTHON_BIN" experiments/cross_rig_attention/analyze_fov_traces.py \
            --split val
        ;;
    *)
        echo 'Usage: bash experiments/cross_rig_attention/run_fov_counterfactual.sh smoke|full|analyze|traces' >&2
        exit 2
        ;;
esac
