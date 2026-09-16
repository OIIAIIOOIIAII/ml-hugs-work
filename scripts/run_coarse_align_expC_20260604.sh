#!/bin/bash
# Exp C：粗对齐初始化 + 最优 pipeline（depth_sup 12k + transl_xyz attn 6k）
# 对比基准：GT transl depth_sup12k+attn6k → HUMAN PSNR 19.78
# 启动方式：nohup bash scripts/run_coarse_align_expC_20260604.sh \
#             > run_logs/coarse_align_expC_20260604_nohup.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
conda run -n hugs --no-capture-output bash -c "

LOG=run_logs/coarse_align_expC_20260604.log
mkdir -p run_logs

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ======================================\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] >>> Phase 1: depth_sup 12k + coarse align init\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ======================================\" | tee -a \$LOG

python main.py \
    --cfg_file cfg_files/release/neuman/hugs_coarsealign_lab_expC_phase1.yaml \
    2>&1 | tee -a \$LOG

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] >>> Phase 1 完成！\" | tee -a \$LOG

# 找 phase 1 输出的 ckpt
PHASE1_DIR=output/human_scene/neuman/lab_coarse/hugs_trimlp/coarse_align_depth_sup_12k_lab_20260604
HUMAN_CKPT=\$(find \$PHASE1_DIR -name 'human_final.pth' | sort | tail -1)
SCENE_CKPT=\$(find \$PHASE1_DIR -name 'scene_final.pth' | sort | tail -1)

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Phase 1 human ckpt: \$HUMAN_CKPT\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Phase 1 scene ckpt: \$SCENE_CKPT\" | tee -a \$LOG

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ======================================\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] >>> Phase 2: transl_xyz attn 6k\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ======================================\" | tee -a \$LOG

python main.py \
    --cfg_file cfg_files/release/neuman/hugs_coarsealign_lab_expC_phase2.yaml \
    human.ckpt=\"\$HUMAN_CKPT\" \
    scene.ckpt=\"\$SCENE_CKPT\" \
    2>&1 | tee -a \$LOG

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] >>> Phase 2 完成！\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ===== Exp C 全部完成！=====\" | tee -a \$LOG
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 输出：output/human_scene/neuman/lab_coarse/hugs_trimlp/coarse_align_depth_sup12k_attn6k_lab_20260604/\" | tee -a \$LOG
"
