#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_petr_r50_train_inference.sh"
bash "$SCRIPT_DIR/run_petr_r50_train_analysis.sh"

