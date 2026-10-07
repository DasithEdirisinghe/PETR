#!/bin/bash -l

set -euo pipefail

conda activate petr

PETR_DIR="${PETR_DIR:-/mnt/hpccs01/home/n12893251/dasith_ws/dev_ws/PETR}"
cd "$PETR_DIR"
export PYTHONPATH="$PETR_DIR:${PYTHONPATH:-}"

CONFIG="${CONFIG:-projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr_r1_r1f.py}"
CHECKPOINT="${CHECKPOINT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1_R1f/latest.pth}"
DATA_ROOT="${DATA_ROOT:-data/pccr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1_R1f/cross_rig_test}"
R1_RESULT="${R1_RESULT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/cross_rig/standardized/PETR/trained_on_R1.json}"
R1F_RESULT="${R1F_RESULT:-results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/cross_rig/standardized/PETR/trained_on_R1-f.json}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
BASE_PORT="${BASE_PORT:-29600}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

RIGS=(R1 R1-c10 R1-c6 R1-f R1-r R1-t R2 R3 R4 R5 R6 R7 R8 R9)

echo "PBS job: ${PBS_JOBID:-unknown}"
echo "Host: $(hostname)"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output root: $OUTPUT_ROOT"
nvidia-smi

for required_file in "$CONFIG" "$CHECKPOINT" "$R1_RESULT" "$R1F_RESULT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file is missing: $required_file" >&2
        exit 1
    fi
done
for rig in "${RIGS[@]}"; do
    ann_file="$DATA_ROOT/$rig/${rig}_infos_test.pkl"
    if [[ ! -f "$ann_file" ]]; then
        echo "Test annotations are missing: $ann_file" >&2
        exit 1
    fi
done

for index in "${!RIGS[@]}"; do
    rig="${RIGS[$index]}"
    ann_file="$DATA_ROOT/$rig/${rig}_infos_test.pkl"
    eval_dir="$OUTPUT_ROOT/evaluations/$rig"
    formatted_dir="$eval_dir/formatted"
    metrics_file="$formatted_dir/metrics_summary.json"
    predictions_file="$eval_dir/predictions.pkl"
    log_file="$eval_dir/inference.log"
    mkdir -p "$eval_dir"

    if [[ "$SKIP_COMPLETED" == "1" && -s "$metrics_file" && -s "$predictions_file" ]]; then
        echo "Reusing completed test inference for $rig"
        continue
    fi

    echo "Evaluating $CHECKPOINT on $rig test split"
    PORT="$((BASE_PORT + index))" bash tools/dist_test.sh \
        "$CONFIG" "$CHECKPOINT" 1 \
        --cfg-options \
            "dataset_name=$rig" \
            "data_root=$DATA_ROOT/$rig/" \
            "test_ann_file=$ann_file" \
            "data.test.data_root=$DATA_ROOT/$rig/" \
            "data.test.ann_file=$ann_file" \
            "data.samples_per_gpu=$SAMPLES_PER_GPU" \
            "data.workers_per_gpu=$WORKERS_PER_GPU" \
        --out "$predictions_file" \
        --eval bbox \
        --eval-options "jsonfile_prefix=$formatted_dir" \
        2>&1 | tee "$log_file"

    if [[ ! -s "$metrics_file" || ! -s "$predictions_file" ]]; then
        echo "Test inference did not produce predictions and metrics for $rig" >&2
        exit 1
    fi
done

RESULT="$OUTPUT_ROOT/standardized/PETR/trained_on_R1_R1f.json"
python tools/analysis_tools/collect_pccr_val_results.py \
    --evaluations "$OUTPUT_ROOT/evaluations" \
    --output "$RESULT" \
    --model PETR --trained-on R1_R1f --split test

PLOT_DIR="$OUTPUT_ROOT/comparison"
PROFILE="$PLOT_DIR/petr_r1_r1f_test_profile"
COMPARISON="$PLOT_DIR/petr_r1_vs_r1f_vs_r1_r1f_test"
python tools/analysis_tools/plot_cross_rig_profile.py \
    "$RESULT" "$PROFILE.svg" \
    --label "Vanilla PETR trained on R1 + R1-f" \
    --title "PETR test performance after training on R1 + R1-f" \
    --train-rigs R1 R1-f --split test
python tools/analysis_tools/plot_cross_rig_comparison.py \
    "$R1_RESULT" "$R1F_RESULT" "$COMPARISON.svg" \
    --reference-label "Vanilla PETR trained on R1" \
    --new-label "Vanilla PETR trained on R1-f" \
    --third "$RESULT" \
    --third-label "Vanilla PETR trained on R1 + R1-f" \
    --reference-components "petr3dpe,multiview,query3d" \
    --new-components "petr3dpe,multiview,query3d" \
    --third-components "petr3dpe,multiview,query3d" \
    --train-rig "R1 + R1-f"

if command -v rsvg-convert >/dev/null 2>&1; then
    rsvg-convert -w 1600 -h 780 "$PROFILE.svg" -o "$PROFILE.png"
    rsvg-convert -w 1600 -h 1180 "$COMPARISON.svg" -o "$COMPARISON.png"
fi

echo "All-rig test inference and plots complete."
echo "Results: $RESULT"
echo "Profile: $PROFILE.svg"
echo "Comparison: $COMPARISON.svg"
