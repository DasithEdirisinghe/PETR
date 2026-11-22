#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUTS="$SCRIPT_DIR/outputs"
ALL_GT_CSV="${ALL_GT_CSV:-$OUTPUTS/multimodel_rule4_analysis_4m_margin_0p05/petr_r50/pedestrian/all_gt.csv}"
FP_RECORDS_CSV="${FP_RECORDS_CSV:-$OUTPUTS/remaining_fp_projection_audit_4m/petr_r50/pedestrian/remaining_fp_records.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUTS/remaining_fp_projection_audit_4m/petr_r50/pedestrian/rank_analysis}"

python "$SCRIPT_DIR/summarize_fp_before_final_tp.py" \
    --all-gt-csv "$ALL_GT_CSV" \
    --fp-records-csv "$FP_RECORDS_CSV" \
    --output-dir "$OUTPUT_DIR"

echo "Rank-aware FP summary: $OUTPUT_DIR/before_final_tp_summary.csv"
