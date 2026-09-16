#!/usr/bin/env bash
# ADC12000 + Mask-aware Floater Suppression + 6000-step transl_xyz Attention
# 3-phase pipeline: [ADC+maskaware 12000] -> [attention stage1 3000] -> [attention stage2 3000]
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../miniconda3/etc/profile.d/conda.sh"
conda activate hugs

cd "$SCRIPT_DIR/.."

DATE_TAG=20260528
BASE_DIR=output/human_scene/neuman/lab/hugs_trimlp
LOG_FILE=run_logs/adc12000_maskaware_transl_xyz_attention_lab_${DATE_TAG}.log
STAGE_CFG=cfg_files/release/neuman/hugs_adc12000_transl_xyz_attention_plus3000_stage_lab.yaml

mkdir -p run_logs

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== 开始 ADC12000 + MaskAware + transl+xyz attention 实验 =====" | tee -a "$LOG_FILE"

# =============================================================
# Phase 1: ADC 12000 + mask-aware floater suppression
# =============================================================
EXP_ADC=adc12000_maskaware_standalone_lab_${DATE_TAG}
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 1: ADC12000 + mask-aware training..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file cfg_files/release/neuman/hugs_global_adc_only_12000_lab.yaml \
  --exp-name "${EXP_ADC}" \
  scene.mask_aware_enabled=true \
  scene.mask_aware_loss_w=0.1 \
  scene.mask_aware_densify=true \
  train.save_progress_images=true \
  train.progress_save_interval=1000 \
  2>&1 | tee -a "$LOG_FILE"

ADC_DIR=$(ls -td "${BASE_DIR}/${EXP_ADC}/"*/ 2>/dev/null | head -1)
if [ -z "$ADC_DIR" ]; then
  echo "ERROR: ADC12000+maskaware output directory not found" | tee -a "$LOG_FILE"
  exit 1
fi
ADC_CKPT="${ADC_DIR}ckpt"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 1 完成，ckpt: ${ADC_CKPT}" | tee -a "$LOG_FILE"

# =============================================================
# Phase 2: transl+xyz attention correction Stage1 (3000 steps)
# =============================================================
EXP_S1=adc12000_maskaware_transl_xyz_stage1_lab_${DATE_TAG}
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 2: transl+xyz correction Stage1..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file "${STAGE_CFG}" \
  --exp-name "${EXP_S1}" \
  scene.mask_aware_enabled=true \
  scene.mask_aware_loss_w=0.1 \
  scene.mask_aware_densify=true \
  "human.ckpt=${ADC_CKPT}/human_final.pth" \
  "scene.ckpt=${ADC_CKPT}/scene_final.pth" \
  2>&1 | tee -a "$LOG_FILE"

S1_DIR=$(ls -td "${BASE_DIR}/${EXP_S1}/"*/ 2>/dev/null | head -1)
if [ -z "$S1_DIR" ]; then
  echo "ERROR: Stage1 output directory not found" | tee -a "$LOG_FILE"
  exit 1
fi
S1_CKPT="${S1_DIR}ckpt"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 2 Stage1 完成，ckpt: ${S1_CKPT}" | tee -a "$LOG_FILE"

# =============================================================
# Phase 3: transl+xyz attention correction Stage2 (3000 steps)
# =============================================================
EXP_S2=adc12000_maskaware_transl_xyz_stage2_lab_${DATE_TAG}
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Phase 3: transl+xyz correction Stage2..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file "${STAGE_CFG}" \
  --exp-name "${EXP_S2}" \
  scene.mask_aware_enabled=true \
  scene.mask_aware_loss_w=0.1 \
  scene.mask_aware_densify=true \
  "human.ckpt=${S1_CKPT}/human_final.pth" \
  "scene.ckpt=${S1_CKPT}/scene_final.pth" \
  "anchor_attention.ckpt=${S1_CKPT}/anchor_attention_final.pth" \
  2>&1 | tee -a "$LOG_FILE"

S2_DIR=$(ls -td "${BASE_DIR}/${EXP_S2}/"*/ 2>/dev/null | head -1)
if [ -z "$S2_DIR" ]; then
  echo "ERROR: Stage2 output directory not found" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ===== 全部完成！=====" | tee -a "$LOG_FILE"
echo "Stage2 结果目录: ${S2_DIR}" | tee -a "$LOG_FILE"

# 打印最终指标
if [ -f "${S2_DIR}results_train.json" ]; then
  echo "Stage2 final metrics:" | tee -a "$LOG_FILE"
  python -c "import json; d=json.load(open('${S2_DIR}results_train.json')); f=d.get('final',{}); print(f'  HUGS PSNR={f.get(\"hugs_psnr\",\"N/A\"):.4f} SSIM={f.get(\"hugs_ssim\",\"N/A\"):.4f} LPIPS={f.get(\"hugs_lpips\",\"N/A\"):.4f}'); print(f'  HUMAN PSNR={f.get(\"hugs_human_psnr\",\"N/A\"):.4f} SSIM={f.get(\"hugs_human_ssim\",\"N/A\"):.4f} LPIPS={f.get(\"hugs_human_lpips\",\"N/A\"):.4f}')" 2>/dev/null | tee -a "$LOG_FILE" || true
fi
