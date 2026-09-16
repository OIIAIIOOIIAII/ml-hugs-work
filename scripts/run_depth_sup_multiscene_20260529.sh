#!/bin/bash
# depth_sup 12000 + transl_xyz attention 6000，顺序跑 bike / seattle / parkinglot
# 用法：bash scripts/run_depth_sup_multiscene_20260529.sh
# 建议后台运行：nohup bash scripts/run_depth_sup_multiscene_20260529.sh > run_logs/depth_sup_multiscene_20260529_nohup.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hugs

DATE_TAG=20260529
SCENES=("bike" "seattle" "parkinglot")
LOG_FILE="run_logs/depth_sup_multiscene_${DATE_TAG}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S CST')] $*" | tee -a "$LOG_FILE"
}

log "===== depth_sup 12000 + transl_xyz attention 6000 多场景实验 ====="
log "场景顺序: ${SCENES[*]}"

for SCENE in "${SCENES[@]}"; do
    log ""
    log "=========================================="
    log ">>> 场景: $SCENE"
    log "=========================================="

    # ---- Phase 1: depth_sup 12000 ----
    log "--- Phase 1: depth_sup 12000 - $SCENE ---"
    python main.py \
        --cfg_file "cfg_files/release/neuman/hugs_depth_supervised_12000_${SCENE}.yaml" \
        2>&1 | tee -a "$LOG_FILE"

    # 找 phase 1 输出目录
    PHASE1_OUT=$(ls -dt "output/human_scene/neuman/${SCENE}/hugs_trimlp/hugs_depth_sup_12000_${SCENE}_${DATE_TAG}/"*/ 2>/dev/null | head -1)
    if [ -z "$PHASE1_OUT" ]; then
        log "ERROR: Phase 1 输出目录未找到，跳过 $SCENE"
        continue
    fi
    HUMAN_CKPT="${PHASE1_OUT}ckpt/human_final.pth"
    SCENE_CKPT="${PHASE1_OUT}ckpt/scene_final.pth"
    log "Phase 1 ckpt: $PHASE1_OUT"

    # ---- Phase 2: transl_xyz attention 6000 ----
    log "--- Phase 2: transl_xyz attention 6000 - $SCENE ---"
    python main.py \
        --cfg_file "cfg_files/release/neuman/hugs_depth_sup12k_transl_xyz_attention_6000_${SCENE}.yaml" \
        "human.ckpt=${HUMAN_CKPT}" \
        "scene.ckpt=${SCENE_CKPT}" \
        2>&1 | tee -a "$LOG_FILE"

    log ">>> $SCENE 完成！"
done

log ""
log "===== 全部完成！====="
