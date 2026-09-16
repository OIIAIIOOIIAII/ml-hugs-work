#!/bin/bash
# Stage 3b: Scene from Scratch — freeze human + attention, re-initialize scene
# from point cloud, train 8000 steps, then evaluate temporal quality.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../../miniconda3/envs/hugs/bin/python"

WORK_DIR="$(dirname "$SCRIPT_DIR")"
CFG="cfg_files/release/neuman/hugs_depth_sup12k_scene_scratch_8000_lab.yaml"
EXP_BASE="output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_scene_scratch_8000_lab_20260531"
LOG_DIR="run_logs"
GROUND_NPZ="eval_contact/lab/ground_plane.npz"

cd "$WORK_DIR"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "[STEP 1] Stage 3b Training: scene from scratch (8000 steps)"
echo "Config: $CFG"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

$PYTHON main.py --cfg_file "$CFG" \
    2>&1 | tee "$LOG_DIR/scene_scratch_train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "[STEP 1] Training done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

RUN_DIR=$(ls -dt "$EXP_BASE"/2026-* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: no run directory found under $EXP_BASE" >&2
    exit 1
fi
echo "Run dir: $RUN_DIR"

echo "============================================================"
echo "[STEP 2] Temporal quality evaluation"
echo "============================================================"

$PYTHON scripts/evaluate_temporal_quality.py \
    --run-dir "$RUN_DIR" \
    --seq lab \
    --ground-npz "$GROUND_NPZ" \
    2>&1 | tee "$LOG_DIR/temporal_eval_scene_scratch_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "[STEP 2] Evaluation done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Report: $RUN_DIR/temporal_eval/report.txt"
echo "============================================================"

if [ -f "$RUN_DIR/temporal_eval/report.txt" ]; then
    echo ""
    cat "$RUN_DIR/temporal_eval/report.txt"
fi
