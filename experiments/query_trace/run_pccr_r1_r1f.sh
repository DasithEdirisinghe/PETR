#!/usr/bin/env bash

set -eo pipefail

PETR_DIR="${PETR_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py}"
CHECKPOINT="${CHECKPOINT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth}"
SCENE_NAME="${SCENE_NAME:-}"
SCENE_FRAME_INDEX="${SCENE_FRAME_INDEX:-}"
QUERY_INDICES="${QUERY_INDICES:-}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/query_trace/outputs/pccr_r1_vs_r1f}"

cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

if [[ -z "$SCENE_NAME" || -z "$SCENE_FRAME_INDEX" ]]; then
    echo "SCENE_NAME and zero-based SCENE_FRAME_INDEX are required." >&2
    echo "Example: SCENE_NAME=val_03 SCENE_FRAME_INDEX=17 $0" >&2
    exit 1
fi
if [[ ! "$SCENE_FRAME_INDEX" =~ ^[0-9]+$ ]]; then
    echo "SCENE_FRAME_INDEX must be a non-negative integer." >&2
    exit 1
fi
printf -v frame_directory '%s_frame_%03d' "$SCENE_NAME" "$SCENE_FRAME_INDEX"
frame_output="$OUTPUT_DIR/$frame_directory"

if [[ -z "$QUERY_INDICES" ]]; then
    "$PYTHON_BIN" experiments/query_trace/trace_query.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --data-root data/pccr/R1/ \
        --ann-file data/pccr/R1/R1_infos_val.pkl \
        --dataset-name R1 \
        --scene-name "$SCENE_NAME" \
        --scene-frame-index "$SCENE_FRAME_INDEX" \
        --run-label R1_val \
        --output-dir "$frame_output/_precheck" \
        --device "$DEVICE" \
        --list-queries
    echo "Query candidates written to: $frame_output/_precheck/R1_val/query_candidates.csv"
    echo "Choose query IDs, then rerun with QUERY_INDICES=\"<id> [<id> ...]\"."
    exit 0
fi

read -r -a SELECTED_QUERIES <<< "$QUERY_INDICES"

for query_index in "${SELECTED_QUERIES[@]}"; do
    if [[ ! "$query_index" =~ ^[0-9]+$ ]]; then
        echo "Invalid query index: $query_index" >&2
        exit 1
    fi
    printf -v query_directory 'query_%04d' "$query_index"
    query_output="$frame_output/$query_directory"

    "$PYTHON_BIN" experiments/query_trace/trace_query.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --data-root data/pccr/R1/ \
        --ann-file data/pccr/R1/R1_infos_val.pkl \
        --dataset-name R1 \
        --scene-name "$SCENE_NAME" \
        --scene-frame-index "$SCENE_FRAME_INDEX" \
        --run-label R1_val \
        --output-dir "$query_output" \
        --device "$DEVICE" \
        --query-indices "$query_index" \
        --plot-bev-rays \
        --bev-ray-layers 1 2 3 4 5 6

    "$PYTHON_BIN" experiments/query_trace/trace_query.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --data-root data/pccr/R1-f/ \
        --ann-file data/pccr/R1-f/R1-f_infos_val.pkl \
        --dataset-name R1-f \
        --scene-name "$SCENE_NAME" \
        --scene-frame-index "$SCENE_FRAME_INDEX" \
        --run-label R1-f_val \
        --output-dir "$query_output" \
        --device "$DEVICE" \
        --match-targets-from "$query_output/R1_val/trace_summary.json" \
        --plot-bev-rays \
        --bev-ray-layers 1 2 3 4 5 6

    echo "Paired trace written to: $query_output"
done
