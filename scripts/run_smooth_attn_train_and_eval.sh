#!/bin/bash
# Train temporal attention with delta_correction_smooth_w=0.1, then run temporal eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../../miniconda3/envs/hugs/bin/python"

WORK_DIR="$(dirname "$SCRIPT_DIR")"
CFG="cfg_files/release/neuman/hugs_depth_sup12k_temporal_attention_6000_lab.yaml"
EXP_BASE="output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_temporal_smooth_attn_6000_lab_20260531"
LOG_DIR="run_logs"
GROUND_NPZ="eval_contact/lab/ground_plane.npz"

cd "$WORK_DIR"

echo "============================================================"
echo "[STEP 1] Training: temporal attention + smooth loss"
echo "Config: $CFG"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

$PYTHON main.py --cfg_file "$CFG" \
    2>&1 | tee "$LOG_DIR/smooth_attn_train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "[STEP 1] Training done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Find the newest run directory
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
    2>&1 | tee "$LOG_DIR/temporal_eval_smooth_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "[STEP 2] Evaluation done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Report: $RUN_DIR/temporal_eval/report.txt"
echo "============================================================"

# Print the final report
if [ -f "$RUN_DIR/temporal_eval/report.txt" ]; then
    echo ""
    cat "$RUN_DIR/temporal_eval/report.txt"
fi
