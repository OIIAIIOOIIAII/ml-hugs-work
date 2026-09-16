#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../miniconda3/etc/profile.d/conda.sh"
conda activate hugs

cd "$SCRIPT_DIR/.."

DATE_TAG=20260528
LOG_FILE=run_logs/depth_supervised_lab_${DATE_TAG}.log
mkdir -p run_logs

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== Depth Supervised HUGS lab 实验 =====" | tee -a "$LOG_FILE"

# =============================================================
# Exp 1: 12000 步（depth_w=0.05）
# =============================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Exp1: 12000 steps with depth supervision..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file cfg_files/release/neuman/hugs_depth_supervised_12000_lab.yaml \
  --exp-name "hugs_depth_sup_12000_lab_${DATE_TAG}" \
  train.save_progress_images=true \
  train.progress_save_interval=1000 \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Exp1 完成." | tee -a "$LOG_FILE"

# =============================================================
# Exp 2: 15000 步（depth_w=0.05）
# =============================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Exp2: 15000 steps with depth supervision..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file cfg_files/release/neuman/hugs_depth_supervised_15000_lab.yaml \
  --exp-name "hugs_depth_sup_15000_lab_${DATE_TAG}" \
  train.save_progress_images=true \
  train.progress_save_interval=1000 \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Exp2 完成." | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== 全部完成！=====" | tee -a "$LOG_FILE"
