#!/bin/bash
# 粗对齐初始化实验：ExpA（HUGS baseline 15000）和 ExpB（anchor attention 15000）
# 顺序执行：A 完成后再跑 B
#
# 用法：
#   nohup bash scripts/run_coarse_align_exps_20260603.sh \
#       > run_logs/coarse_align_exps_20260603_nohup.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hugs

LOG="run_logs/coarse_align_exps_20260603.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "===== 粗对齐初始化对比实验 (lab_coarse) ====="

# ─── Experiment A：HUGS baseline 15000 ─────────────────────────────────────
log ""
log "========================================"
log ">>> Exp A: HUGS baseline 15000 + coarse align init"
log "========================================"

python main.py \
    --cfg_file cfg_files/release/neuman/hugs_coarsealign_lab_expA.yaml \
    2>&1 | tee -a "$LOG"

log ">>> Exp A 完成！"

# ─── Experiment B：anchor attention 15000 ──────────────────────────────────
log ""
log "========================================"
log ">>> Exp B: anchor attention 15000 + coarse align init"
log "========================================"

python main.py \
    --cfg_file cfg_files/release/neuman/hugs_coarsealign_lab_expB.yaml \
    2>&1 | tee -a "$LOG"

log ">>> Exp B 完成！"
log ""
log "===== 全部实验完成！====="
log "日志：$LOG"
log "Exp A 输出：output/human_scene/neuman/lab_coarse/hugs_trimlp/coarse_align_hugs_baseline_lab_20260603/"
log "Exp B 输出：output/human_scene/neuman/lab_coarse/hugs_trimlp/coarse_align_anchor_attention_lab_20260603/"
