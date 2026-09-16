#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../miniconda3/etc/profile.d/conda.sh"
conda activate hugs

cd "$SCRIPT_DIR/.."

DATE_TAG=20260528
LOG_FILE=run_logs/depth_sup12k_attention_lab_${DATE_TAG}.log
mkdir -p run_logs

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== depth_sup12k + transl_xyz attention 6000步 =====" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file cfg_files/release/neuman/hugs_depth_sup12k_transl_xyz_attention_6000_lab.yaml \
  --exp-name "depth_sup12k_transl_xyz_attn_6000_lab_${DATE_TAG}" \
  train.save_progress_images=true \
  train.progress_save_interval=1000 \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== 完成！=====" | tee -a "$LOG_FILE"
