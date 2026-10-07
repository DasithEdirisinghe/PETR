#!/usr/bin/env bash
# Frozen R1 PETR on unchanged paired R1/R1-f frames: trace + branch probes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${1:-eight}"
if [[ $# -gt 0 ]]; then shift; fi

case "$MODE" in
    eight)
        MAX_EXAMPLES="${MAX_EXAMPLES:-8}"
        SELECTION_ARGS=()
        TRACE_ARGS=()
        TRACE_OUTPUT="experiments/cross_rig_attention/output/r1f_val/traces/r1_unwarped_internal"
        PROBE_OUTPUT="experiments/cross_rig_attention/output/r1f_val/traces/r1_component_sensitivity"
        ;;
    balanced)
        PER_STRATUM="${PER_STRATUM:-5}"
        MAX_EXAMPLES="${MAX_EXAMPLES:-10000}"
        COHORT="experiments/cross_rig_attention/output/r1f_val/traces/cohorts/balanced_${PER_STRATUM}"
        "$PYTHON_BIN" experiments/cross_rig_attention/build_r1_component_cohort.py \
            --split val --per-stratum "$PER_STRATUM"
        SELECTION_ARGS=(--paired-selection "$COHORT/paired_selection.json")
        TRACE_ARGS=(--trace-missing)
        TRACE_OUTPUT="$COHORT/internal"
        PROBE_OUTPUT="$COHORT/sensitivity"
        ;;
    *)
        echo "Usage: $0 [eight|balanced]" >&2
        exit 2
        ;;
esac

"$PYTHON_BIN" experiments/cross_rig_attention/r1_paired_rig_component_trace.py \
    --split val --max-examples "$MAX_EXAMPLES" --device "$DEVICE" \
    "${SELECTION_ARGS[@]}" "${TRACE_ARGS[@]}" "$@"
"$PYTHON_BIN" experiments/cross_rig_attention/probe_r1_attention_components.py \
    --split val --max-examples "$MAX_EXAMPLES" --device "$DEVICE" \
    "${SELECTION_ARGS[@]}"
"$PYTHON_BIN" experiments/cross_rig_attention/summarize_r1_component_diagnosis.py \
    --trace-dir "$TRACE_OUTPUT" --probe-dir "$PROBE_OUTPUT"

echo "Trace and report: $TRACE_OUTPUT/"
echo "Probe: $PROBE_OUTPUT/"
