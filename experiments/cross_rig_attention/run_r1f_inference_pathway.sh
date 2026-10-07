#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Pass any Python options after the split, e.g. --max-examples 8 --strengths 0.25 0.5.
SPLIT="${1:-val}"
if [[ $# -gt 0 ]]; then shift; fi
python experiments/cross_rig_attention/diagnose_r1f_inference_pathway.py \
  --split "$SPLIT" "$@"
