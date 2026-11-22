#!/usr/bin/env bash
set -euo pipefail

# Run in an interactive node with the PETR Python environment activated.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:-"$PETR_DIR/../RoboBEV/data/sets/nuscenes"}
RESULTS_JSON=${RESULTS_JSON:-"$PETR_DIR/experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/train/inference/formatted/pts_bbox/results_nusc.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PETR_DIR/experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/train/reranking_decomposition"}

test -s "$RESULTS_JSON" || { echo "Missing train predictions: $RESULTS_JSON" >&2; exit 1; }
test -d "$DATA_ROOT/v1.0-trainval" || { echo "Missing nuScenes metadata: $DATA_ROOT" >&2; exit 1; }
command -v python >/dev/null || { echo "Activate the PETR Python environment first" >&2; exit 1; }

# Optional: pass class names to run a smaller pilot, e.g. pedestrian car.
CLASS_ARGS=()
if (( $# > 0 )); then
    CLASS_ARGS=(--classes "$@")
fi
python experiments/nuscnes_exp/pedestrian_neighbor_trace/decompose_reranking_headroom.py \
    --results-json "$RESULTS_JSON" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --eval-set train \
    --restrict-to-result-tokens \
    "${CLASS_ARGS[@]}"

python - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    result = json.load(handle)
print('Baseline mean AP: {:.3f}%'.format(result['baseline_mean_ap_percent']))
print('GT-informed oracle mean AP: {:.3f}%'.format(result['oracle_mean_ap_percent']))
print('Combined category action mean AP: {:.3f}%'.format(
    result['combined_action_mean_ap_percent']))
print('Independent category gains (pp; do not add):')
for name, gain in result['category_counterfactual_mean_gain_pp'].items():
    print('  {}: {:+.3f}'.format(name, gain))
print('Full report: {}'.format(sys.argv[1]))
PY
