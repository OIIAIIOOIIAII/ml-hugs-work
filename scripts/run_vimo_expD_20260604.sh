#!/bin/bash
# Exp D: VIMO 粗对齐初始化 + depth_sup 12k + transl_xyz attn 6k
# 对比基准: GT transl → HUMAN PSNR 19.78 (同 pipeline)
# 粗对齐 v3 (ROMP) → HUMAN PSNR 16.16 (-3.62)
# VIMO transl 误差 ~0.088m (vs ROMP 0.19m)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

nohup conda run -n hugs --no-capture-output bash -c "

LOG=run_logs/vimo_expD_20260604.log
mkdir -p run_logs

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ===== Exp D 开始 =====\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] VIMO transl 初始化 (mean err ~0.088m)\" | tee -a \$LOG

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] >>> Phase 1: depth_sup 12k\" | tee -a \$LOG
python main.py \
    --cfg_file cfg_files/release/neuman/hugs_vimo_lab_expD_phase1.yaml \
    2>&1 | tee -a \$LOG

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] >>> Phase 1 完成！\" | tee -a \$LOG

PHASE1_DIR=output/human_scene/neuman/lab_vimo/hugs_trimlp/vimo_depth_sup12k_lab_20260604
HUMAN_CKPT=\$(find \$PHASE1_DIR -name 'human_final.pth' | sort | tail -1)
SCENE_CKPT=\$(find \$PHASE1_DIR -name 'scene_final.pth' | sort | tail -1)
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] human_ckpt: \$HUMAN_CKPT\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] scene_ckpt: \$SCENE_CKPT\" | tee -a \$LOG

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] >>> Phase 2: transl_xyz attn 6k\" | tee -a \$LOG
python main.py \
    --cfg_file cfg_files/release/neuman/hugs_vimo_lab_expD_phase2.yaml \
    human.ckpt=\"\$HUMAN_CKPT\" \
    scene.ckpt=\"\$SCENE_CKPT\" \
    2>&1 | tee -a \$LOG

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ===== Exp D 全部完成！ =====\" | tee -a \$LOG

" > run_logs/vimo_expD_20260604_nohup.log 2>&1 &

echo "Exp D 已在后台启动 (PID $!)"
echo "nohup 日志: run_logs/vimo_expD_20260604_nohup.log"
echo "详细日志:   run_logs/vimo_expD_20260604.log"
echo ""
echo "进度检查:"
echo "  grep -E 'Phase|HUMAN_PSNR' run_logs/vimo_expD_20260604.log | tail -10"
