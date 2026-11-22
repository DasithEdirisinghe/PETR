#!/usr/bin/env bash
# Train PETR with an EVA02-L backbone on nuScenes trainval/val.
set -euo pipefail

PETR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PETR_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-petr}"
fi
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CONFIG="${CONFIG:-projects/configs/petr/petr_eva02_l_p4_800x320_nuscenes.py}"
SOURCE=../StreamPETR/ckpts/eva02_L_coco_det_sys_o365_remapped.pth
INIT=ckpts/petr_eva02_l_p4_backbone_init.pth
DATA=../RoboBEV/data/sets/nuscenes
WORK_DIR="${WORK_DIR:-results/petr_eva02_l_p4_800x320_nuscenes}"
GPUS="${GPUS:-2}"
PORT="${PORT:-29539}"
SEED="${SEED:-0}"

if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer" >&2
    exit 1
fi
LR="${LR:-$(awk -v g="$GPUS" 'BEGIN {printf "%.8g", 2e-4 * g / 8}')}"

for file in "$CONFIG" "$SOURCE" "$DATA/nuscenes_infos_temporal_train.pkl" \
            "$DATA/nuscenes_infos_temporal_val.pkl"; do
    if [[ ! -s "$file" ]]; then
        echo "Missing or empty input: $PETR_DIR/$file" >&2
        exit 1
    fi
done
for directory in "$DATA/samples" "$DATA/v1.0-trainval"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing nuScenes directory: $PETR_DIR/$directory" >&2
        exit 1
    fi
done
python -c 'import torch, mmcv, mmdet, einops, scipy; print("EVA dependencies OK")'

echo "Preparing EVA backbone-only initialization checkpoint..."
python -u tools/prepare_petr_eva02_backbone.py \
    --config "$CONFIG" --source "$SOURCE" --output "$INIT"

if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
    echo "Backbone preparation passed; PREPARE_ONLY=1 skips training."
    exit 0
fi

mkdir -p "$WORK_DIR"
echo "Config: $CONFIG"
echo "Work directory: $WORK_DIR"
echo "GPUs: $GPUS; learning rate: $LR; seed: $SEED"
echo "Initialization: $INIT (EVA backbone only; PETR head is random)"

PORT="$PORT" bash tools/dist_train.sh \
    "$CONFIG" "$GPUS" \
    --work-dir "$WORK_DIR" \
    --seed "$SEED" \
    "$@" \
    --cfg-options "optimizer.lr=$LR"
