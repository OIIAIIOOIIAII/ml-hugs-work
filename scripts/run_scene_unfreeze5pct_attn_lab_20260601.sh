#!/bin/bash
# 实验：attention phase 解冻场景高斯（5% Phase1 LR）
# 基线：depth_sup12k + transl_xyz attn 6k（HUMAN PSNR 19.7831）
# 目标：场景GS在attention阶段自适应微调，改善接触边界渲染质量
#
# 用法：
#   nohup bash scripts/run_scene_unfreeze5pct_attn_lab_20260601.sh \
#     > run_logs/scene_unfreeze5pct_attn_lab_20260601.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hugs

LOG_FILE="run_logs/scene_unfreeze5pct_attn_lab_20260601.log"
mkdir -p run_logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=========================================="
log "=== scene_unfreeze5pct attention 6000 ==="
log "=========================================="
log "实验：attention phase 解冻场景GS（5% LR）"
log "基线 HUMAN PSNR: 19.7831"
log "cfg: hugs_depth_sup12k_scene_unfreeze5pct_attn_6000_lab.yaml"

python main.py \
    --cfg_file cfg_files/release/neuman/hugs_depth_sup12k_scene_unfreeze5pct_attn_6000_lab.yaml \
    2>&1 | tee -a "$LOG_FILE"

log "=== 实验完成 ==="
