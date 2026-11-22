#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../../../.." && pwd)"

python "$HERE/threshold_switch_duplicates.py" \
  --model "PETR-R50=$WORKSPACE/PETR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json" \
  --model "PETR-EVA02-L=$HERE/outputs/eva02_epoch16_iou3d/formatted/pts_bbox/results_nusc.json" \
  --model "StreamPETR-R50=$WORKSPACE/StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/Fri_Oct__9_14_02_06_2026/pts_bbox/results_nusc.json" \
  --model "RepDETR3D-EVA02-L=$WORKSPACE/StreamPETR/experiments/eva02_gallery/outputs/inference/formatted/pts_bbox/results_nusc.json" \
  "$@"
