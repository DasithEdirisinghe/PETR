#!/usr/bin/env bash
# Run inside an allocated interactive A100 shell (not on a login node).
set -euo pipefail

PETR_DIR=${PETR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PETR_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "petr" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate petr
fi

export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

CONFIG=projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v6.py
CHECKPOINT=results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/epoch_24.pth
OUTPUT_DIR=${OUTPUT_DIR:-results/petr_r50dcn_gridmask_p4_rule_v2_assignment_audit}
MAX_BATCHES=${MAX_BATCHES:-20}

test -f "$CONFIG"
test -f "$CHECKPOINT"
test -f data/nuscenes/nuscenes_infos_temporal_train.pkl
test -d data/nuscenes/samples
test -d data/nuscenes/v1.0-trainval
test -f ../mmdetection3d/configs/_base_/datasets/nus-3d.py

python -c 'import torch; assert torch.cuda.is_available(), "Run this script inside an interactive GPU allocation"; print("CUDA device:", torch.cuda.get_device_name(0), flush=True)'
echo "PETR directory: $PETR_DIR"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Train batches to audit: $MAX_BATCHES"

python -u experiments/nuscnes_exp/pedestrian_neighbor_trace/test_rule_v2_smoke.py
python -u experiments/nuscnes_exp/pedestrian_neighbor_trace/audit_rule_v2_assignments.py \
    --config "$CONFIG" --max-batches "$MAX_BATCHES" --output-dir "$OUTPUT_DIR"
