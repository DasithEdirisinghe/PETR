#!/usr/bin/env bash
# Run inside an allocated interactive A100; original epoch-24 checkpoint only.
set -euo pipefail
PETR_DIR=${PETR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PETR_DIR"
if [[ "${CONDA_DEFAULT_ENV:-}" != "petr" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate petr
fi
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
python -c 'import torch; assert torch.cuda.is_available(), "Run inside an interactive GPU allocation"'
python experiments/nuscnes_exp/pedestrian_neighbor_trace/install_rule_assignment_gallery.py
python -u experiments/nuscnes_exp/pedestrian_neighbor_trace/export_rule_assignment_gallery.py \
    --max-batches "${MAX_BATCHES:-20}" \
    --max-cases-per-category "${MAX_CASES_PER_CATEGORY:-60}"
