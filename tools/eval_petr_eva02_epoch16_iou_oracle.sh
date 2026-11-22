#!/usr/bin/env bash
# Format PETR EVA02 epoch-16 nuScenes predictions and run fixed-box 3D-IoU oracle AP.
set -euo pipefail

PETR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PETR_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-petr}"
fi
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CONFIG=results/petr_eva02_l_p4_800x320_nuscenes/petr_eva02_l_p4_800x320_nuscenes.py
CHECKPOINT=results/petr_eva02_l_p4_800x320_nuscenes/epoch_16.pth
DATA_ROOT=../RoboBEV/data/sets/nuscenes
OUTPUT_DIR=experiments/nuscnes_exp/pedestrian_neighbor_trace/outputs/eva02_epoch16_iou3d
PREFIX="$OUTPUT_DIR/formatted"
RESULTS_JSON="$PREFIX/pts_bbox/results_nusc.json"
GPUS="${GPUS:-1}"
PORT="${PORT:-29541}"

for file in "$CONFIG" "$CHECKPOINT" "$DATA_ROOT/nuscenes_infos_temporal_val.pkl"; do
    if [[ ! -s "$file" ]]; then
        echo "Missing or empty input: $file" >&2
        exit 1
    fi
done
if [[ ! -d "$DATA_ROOT/v1.0-trainval" ]]; then
    echo "Missing nuScenes metadata: $DATA_ROOT/v1.0-trainval" >&2
    exit 1
fi
if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [[ ! -s "$RESULTS_JSON" ]]; then
    PORT="$PORT" bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
        --format-only \
        --eval-options "jsonfile_prefix=$PREFIX" \
        2>&1 | tee "$OUTPUT_DIR/inference.log"
fi
if [[ ! -s "$RESULTS_JSON" ]]; then
    echo "Inference did not write $RESULTS_JSON" >&2
    exit 1
fi

if [[ ! -s "$OUTPUT_DIR/oracle/summary.json" ]]; then
    python -u experiments/nuscnes_exp/pedestrian_neighbor_trace/all_class_reranking_oracle.py \
        --results-json "$RESULTS_JSON" \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR/oracle" \
        2>&1 | tee "$OUTPUT_DIR/oracle.log"
fi

if [[ ! -s "$OUTPUT_DIR/pr_curves/pr_summary.csv" ]]; then
    python -u experiments/nuscnes_exp/pedestrian_neighbor_trace/plot_pr_curves.py \
        --results-json "$RESULTS_JSON" \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR/pr_curves" \
        --model-name 'PETR EVA02-L (epoch 16)' \
        2>&1 | tee "$OUTPUT_DIR/pr_curves.log"
fi

python experiments/nuscnes_exp/cross_model_ap_comparison/plot_ap_comparison.py
python ../StreamPETR/experiments/eva02_gallery/summarize_pr_oracle.py
python ../StreamPETR/experiments/eva02_gallery/plot_pr_oracle_comparison.py

bash ../StreamPETR/experiments/eva02_gallery/run_ray_frustum_oracle.sh \
    custom 'PETR EVA02-L (epoch 16)' "$PETR_DIR/$RESULTS_JSON"

python ../StreamPETR/experiments/eva02_gallery/build_fast_gallery.py --refresh-page-only
echo "Oracle summary: $OUTPUT_DIR/oracle/summary.json"
echo "Per-class AP: $OUTPUT_DIR/oracle/class_mean_ap_comparison.csv"
echo "Gallery: experiments/nuscnes_exp/pedestrian_neighbor_trace/outputs/visualizations/index.html"
