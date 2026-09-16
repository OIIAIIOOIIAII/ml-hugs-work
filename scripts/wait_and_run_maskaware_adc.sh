#!/usr/bin/env bash
# 等待当前 mask-aware standalone 训练结束，再启动 ADC+mask-aware+attention 实验
# 用法: bash scripts/wait_and_run_maskaware_adc.sh <PID>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_PID=${1:-369640}
RESULTS_DIR="output/human_scene/neuman/lab/hugs_trimlp/full_noamass/2026-05-28_00-14-27"
NEXT_SCRIPT="scripts/run_adc12000_maskaware_transl_xyz_attention_lab_20260528.sh"

cd "$SCRIPT_DIR/.."

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] 等待 PID ${WATCH_PID} 结束..."

# 等待进程结束
while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 30
done

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] PID ${WATCH_PID} 已结束"

# 检查结果文件
if [ -f "${RESULTS_DIR}/results_train.json" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] 当前实验结果："
  python -c "
import json
d = json.load(open('${RESULTS_DIR}/results_train.json'))
f = d.get('final', {})
print(f'  HUGS  PSNR={f.get(\"hugs_psnr\",\"N/A\"):.4f} SSIM={f.get(\"hugs_ssim\",\"N/A\"):.4f} LPIPS={f.get(\"hugs_lpips\",\"N/A\"):.4f}')
print(f'  HUMAN PSNR={f.get(\"hugs_human_psnr\",\"N/A\"):.4f} SSIM={f.get(\"hugs_human_ssim\",\"N/A\"):.4f} LPIPS={f.get(\"hugs_human_lpips\",\"N/A\"):.4f}')
" 2>/dev/null || echo "  (无法解析指标)"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] 警告：未找到 results_train.json，可能训练未正常完成"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] 启动 ADC12000+maskaware+transl_xyz 实验..."
bash "${NEXT_SCRIPT}"
