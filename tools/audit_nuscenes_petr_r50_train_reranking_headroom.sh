#!/usr/bin/env bash
set -euo pipefail

# Run from an interactive shell with the PETR conda environment activated.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:-"$PETR_DIR/../RoboBEV/data/sets/nuscenes"}
RESULTS_JSON=${RESULTS_JSON:-"$PETR_DIR/experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/train/inference/formatted/pts_bbox/results_nusc.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PETR_DIR/experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/train/oracle_ap"}
COVERAGE_AUDIT="$PETR_DIR/experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/train/official_eval/sample_token_audit.json"

test -s "$RESULTS_JSON" || { echo "Missing train predictions: $RESULTS_JSON" >&2; exit 1; }
test -d "$DATA_ROOT/v1.0-trainval" || { echo "Missing nuScenes metadata: $DATA_ROOT" >&2; exit 1; }
command -v python >/dev/null || { echo "Activate the PETR Python environment first" >&2; exit 1; }

echo "Train predictions: $RESULTS_JSON"
echo "nuScenes root: $DATA_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "The oracle will report how many train frames the prediction file covers."
if [[ -s "$COVERAGE_AUDIT" ]]; then
    python - "$COVERAGE_AUDIT" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    coverage = json.load(handle)
print('Saved prediction coverage: {:,} of {:,} train frames'.format(
    coverage['prediction_sample_count'], coverage['expected_sample_count']))
PY
fi

python experiments/nuscnes_exp/pedestrian_neighbor_trace/all_class_reranking_oracle.py \
    --results-json "$RESULTS_JSON" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --eval-set train \
    --restrict-to-result-tokens

test -s "$OUTPUT_DIR/summary.json"
python - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
print('Fixed-box train ranking headroom (on analyzed frames):')
print('  Baseline mean AP: {:.3f}%'.format(result['baseline_mean_ap_percent']))
print('  GT-informed rank mean AP: {:.3f}%'.format(result['oracle_rank_mean_ap_percent']))
print('  Gain: {:.3f} percentage points'.format(result['gain_percentage_points']))
print('Full report: {}'.format(sys.argv[1]))
PY
