#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-petr}"

cd "$PETR_DIR"

# Recompute only hypothesis CSVs that do not yet contain the visualization-
# specific unmatched, 2D-first preferred alternative. Existing cross-audits
# are reused, and no galleries are built in this phase.
CONDA_ENV="$CONDA_ENV" RUN_EVA_INFERENCE=0 BUILD_GALLERIES=0 \
    bash "$SCRIPT_DIR/run_multimodel_hypothesis_analysis.sh"

# Rebuild synchronized common cases using:
# unmatched first; projected 2D IoU; then 3D IoU; then center distance when
# both 3D IoUs are zero.
CONDA_ENV="$CONDA_ENV" MAX_PER_CATEGORY="${MAX_PER_CATEGORY:-500}" \
    bash "$SCRIPT_DIR/prepare_common_multimodel_galleries.sh"

# Lightweight HTML and per-class count tables.
python "$SCRIPT_DIR/build_common_multimodel_gallery_page.py"
python "$SCRIPT_DIR/build_projection_correspondence_report.py"

echo "Preferred common multi-model gallery is ready."
