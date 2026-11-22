#!/usr/bin/env bash
# Run inside an allocated interactive A100 shell; no checkpoint is written.
set -euo pipefail

PETR_DIR=${PETR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PETR_DIR"
if [[ "${CONDA_DEFAULT_ENV:-}" != "petr" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate petr
fi
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

test -f results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/epoch_24.pth
test -f data/nuscenes/nuscenes_infos_temporal_train.pkl
test -d data/nuscenes/samples
python -c 'import torch; assert torch.cuda.is_available(), "Interactive CUDA allocation required"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'

python -u experiments/nuscnes_exp/pedestrian_neighbor_trace/test_rule_v2_smoke.py
for version in v6 v7; do
    echo "Checking $version on one real training batch"
    python -u experiments/nuscnes_exp/pedestrian_neighbor_trace/test_rule_v2_real_train_step.py \
        --config "projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_${version}.py"
done
