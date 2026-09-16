#!/bin/bash
# 串行评估三个实验的时序接触质量
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

LOG=run_logs/temporal_eval_20260531.log
mkdir -p run_logs

run_eval() {
    local name="$1"; local run_dir="$2"
    echo ""
    echo "============================================================"
    echo "[$name] $(date '+%H:%M:%S')"
    echo "============================================================"
    python scripts/evaluate_temporal_quality.py \
        --run-dir "$run_dir" \
        --seq lab \
        --contact-thresh 0.43 \
        --foot-topk 300
}

run_eval "HUGS_baseline" \
    "output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260527/2026-05-27_18-27-43"

run_eval "depth_sup12k_attn6k" \
    "output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_lab_20260528/2026-05-28_23-44-41"

run_eval "depth_sup12k_temporal_attn6k" \
    "output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_temporal_attn_6000_lab_20260531/2026-05-31_17-16-35"

echo ""
echo "All evaluations done at $(date '+%H:%M:%S')"
