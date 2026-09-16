#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../miniconda3/etc/profile.d/conda.sh"
conda activate hugs

cd "$SCRIPT_DIR/.."

DATE_TAG=20260527
BASE_DIR=output/human_scene/neuman/lab/hugs_trimlp
LOG_FILE=run_logs/adc12000_transl_xyz_attention_lab_${DATE_TAG}.log
STAGE_CFG=cfg_files/release/neuman/hugs_adc12000_transl_xyz_attention_plus3000_stage_lab.yaml

mkdir -p run_logs

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== 开始 ADC12000 + transl+xyz attention 实验 =====" | tee -a "$LOG_FILE"

# =============================================================
# Phase 1: ADC 12000 standalone
# =============================================================
EXP_ADC=adc12000_standalone_lab_${DATE_TAG}
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 1: ADC12000 standalone training..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file cfg_files/release/neuman/hugs_global_adc_only_12000_lab.yaml \
  --exp-name "${EXP_ADC}" \
  train.save_progress_images=true \
  train.progress_save_interval=1000 \
  2>&1 | tee -a "$LOG_FILE"

ADC_DIR=$(ls -td "${BASE_DIR}/${EXP_ADC}/"*/ 2>/dev/null | head -1)
if [ -z "$ADC_DIR" ]; then
  echo "ERROR: ADC12000 output directory not found" | tee -a "$LOG_FILE"
  exit 1
fi
ADC_CKPT="${ADC_DIR}ckpt"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ADC12000 完成，ckpt: ${ADC_CKPT}" | tee -a "$LOG_FILE"

# =============================================================
# Phase 2: transl+xyz attention correction Stage1 (3000 steps)
# =============================================================
EXP_S1=adc12000_transl_xyz_attention_plus3000_stage1_lab_${DATE_TAG}
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 2: transl+xyz correction Stage1..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file "${STAGE_CFG}" \
  --exp-name "${EXP_S1}" \
  "human.ckpt=${ADC_CKPT}/human_final.pth" \
  "scene.ckpt=${ADC_CKPT}/scene_final.pth" \
  2>&1 | tee -a "$LOG_FILE"

S1_DIR=$(ls -td "${BASE_DIR}/${EXP_S1}/"*/ 2>/dev/null | head -1)
if [ -z "$S1_DIR" ]; then
  echo "ERROR: Stage1 output directory not found" | tee -a "$LOG_FILE"
  exit 1
fi
S1_CKPT="${S1_DIR}ckpt"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Stage1 完成，ckpt: ${S1_CKPT}" | tee -a "$LOG_FILE"

# =============================================================
# Phase 3: transl+xyz attention correction Stage2 (3000 steps)
# =============================================================
EXP_S2=adc12000_transl_xyz_attention_plus3000_stage2_lab_${DATE_TAG}
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 3: transl+xyz correction Stage2..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file "${STAGE_CFG}" \
  --exp-name "${EXP_S2}" \
  "human.ckpt=${S1_CKPT}/human_final.pth" \
  "scene.ckpt=${S1_CKPT}/scene_final.pth" \
  "anchor_attention.ckpt=${S1_CKPT}/anchor_attention_final.pth" \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== 全部完成！=====" | tee -a "$LOG_FILE"
