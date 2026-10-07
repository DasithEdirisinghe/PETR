#!/usr/bin/env bash
# Trace one R1 checkpoint on paired R1/R1-f frames, including a fixed-query control.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"${PYTHON_BIN:-python}" experiments/cross_rig_attention/r1_paired_rig_component_trace.py \
    --split val --max-examples "${MAX_EXAMPLES:-8}" --device "${DEVICE:-cuda:0}" "$@"
