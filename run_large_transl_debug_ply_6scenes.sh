#!/bin/bash
# 6场景串行：v4 + large_transl + inline_attn + debug_ply_interval=1000
# 正确代码（full viewspace_points），18k steps

set -e
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
PYTHON=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
mkdir -p run_logs

SCENES=(bike jogging seattle lab parkinglot citron)

for scene in "${SCENES[@]}"; do
    cfg="cfg_files/release/neuman/hugs_vimo_v4_${scene}_large_transl_debug_ply_18k.yaml"
    log="run_logs/large_transl_debug_ply_${scene}.log"
    echo "========================================" | tee -a run_logs/large_transl_debug_ply_6scenes_progress.log
    echo "[$(date)] 启动: $scene" | tee -a run_logs/large_transl_debug_ply_6scenes_progress.log
    $PYTHON main.py --cfg_file "$cfg" > "$log" 2>&1
    echo "[$(date)] 完成: $scene  (PSNR final: $(grep 'final - HUGS_HUMAN_PSNR' $log | tail -1 | awk '{print $NF}'))" | tee -a run_logs/large_transl_debug_ply_6scenes_progress.log
done

echo "========================================" | tee -a run_logs/large_transl_debug_ply_6scenes_progress.log
echo "[$(date)] 全部6场景完成" | tee -a run_logs/large_transl_debug_ply_6scenes_progress.log
