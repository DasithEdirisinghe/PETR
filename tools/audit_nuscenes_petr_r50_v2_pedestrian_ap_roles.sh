#!/usr/bin/env bash
set -euo pipefail

# Run in an interactive A100 shell with the PETR environment activated.
# Usage: bash tools/audit_nuscenes_petr_r50_v2_pedestrian_ap_roles.sh [max_frames]
# Uses the official PETR-R50 epoch-24 checkpoint. V6 supplies the opt-in V2
# rule settings for analysis; this script does not fine-tune the checkpoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

MAX_FRAMES=${1:-500}

CONFIG="$PETR_DIR/projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v6.py"
CHECKPOINT=${CHECKPOINT:-"$PETR_DIR/results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/epoch_24.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PETR_DIR/experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/train/baseline_v2_pedestrian_ap_roles"}
REFERENCE_RESULTS_JSON=${REFERENCE_RESULTS_JSON:-"$PETR_DIR/experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/train/inference/formatted/pts_bbox/results_nusc.json"}
DATA_ROOT=${DATA_ROOT:-"$PETR_DIR/../RoboBEV/data/sets/nuscenes"}

test -s "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
test -s "$REFERENCE_RESULTS_JSON" || { echo "Missing reference train results: $REFERENCE_RESULTS_JSON" >&2; exit 1; }
test -d "$DATA_ROOT/v1.0-trainval" || { echo "Missing nuScenes metadata: $DATA_ROOT" >&2; exit 1; }
command -v python >/dev/null || { echo "Activate PETR Python first" >&2; exit 1; }

python experiments/nuscnes_exp/pedestrian_neighbor_trace/audit_v2_pedestrian_ap_roles.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --reference-results-json "$REFERENCE_RESULTS_JSON" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES"

echo "Report: $OUTPUT_DIR/summary.json"
echo "Role/AP table: $OUTPUT_DIR/pedestrian_role_ap_decomposition.csv"
echo "Retained query trace: $OUTPUT_DIR/retained_pedestrian_query_roles.csv"
