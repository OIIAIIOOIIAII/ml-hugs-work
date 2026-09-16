#!/usr/bin/env bash
# 用 v4 粗对齐跑 3 个崩坏场景（bike/jogging/seattle），与 STM 结果比较
# STM 基准：bike=18.6133  jogging=17.3776  seattle=16.7278
set -e
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work

PYTHON=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
LOG_DIR=run_logs
mkdir -p "$LOG_DIR"

run_scene() {
  local seq=$1
  local cfg="cfg_files/release/neuman/hugs_vimo_v4_${seq}_inline_attn_18k.yaml"
  local log="${LOG_DIR}/vimo_v4_inline_attn_18k_${seq}_20260613.log"
  echo "[$(date '+%H:%M:%S')] >>> 开始 ${seq} (v4 粗对齐)"
  $PYTHON scripts/run_neuman_human_scene_noamass.py --cfg-file "$cfg" --seq "${seq}_vimo_v4" > "$log" 2>&1
  # 打印最终 HUMAN_PSNR
  echo "[$(date '+%H:%M:%S')] <<< ${seq} 完成"
  grep "HUGS_HUMAN_PSNR" "$log" | tail -1
  echo ""
}

run_scene bike
run_scene jogging
run_scene seattle

echo "=== 全部完成 ==="
echo "STM 基准: bike=18.6133  jogging=17.3776  seattle=16.7278"
for seq in bike jogging seattle; do
  log="${LOG_DIR}/vimo_v4_inline_attn_18k_${seq}_20260613.log"
  best=$(grep "HUGS_HUMAN_PSNR" "$log" | awk '{print $NF}' | sort -n | tail -1)
  final=$(grep "HUGS_HUMAN_PSNR" "$log" | tail -1 | awk '{print $NF}')
  echo "  ${seq}: best=${best}  final=${final}"
done
