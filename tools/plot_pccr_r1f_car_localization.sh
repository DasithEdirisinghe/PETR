#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="${PETR_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PETR_DIR"

INPUT="${INPUT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/cross_rig/standardized/PETR/trained_on_R1-f.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/cross_rig/comparison/car_localization}"

python tools/analysis_tools/plot_cross_rig_class_localization.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --class-name car \
    --train-rig R1-f

echo "R1-f-trained PETR car-localization plots: $OUTPUT_DIR"
