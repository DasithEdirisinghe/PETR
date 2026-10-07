#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"${PYTHON_BIN:-python}" experiments/cross_rig_attention/run_four_run_pipeline.py "$@"
