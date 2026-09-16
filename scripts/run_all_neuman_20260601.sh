#!/bin/bash
# 全量 NeuMan 实验：A1-A5 HUGS 15K baseline + B1-B2 depth_sup 12K + C1-C2 attn 6K
# 最后生成 5 场景 × 2 mode 对比图
#
# 用法：
#   nohup bash scripts/run_all_neuman_20260601.sh > run_logs/all_neuman_20260601.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hugs

DATE_TAG=20260601
LOG_FILE="run_logs/all_neuman_20260601.log"
mkdir -p run_logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=========================================="
log "===== NeuMan 全量实验 20260601 开始 ====="
log "=========================================="

# ==========================================================
# Group A: HUGS 15K baseline（5 个场景）
# ==========================================================
log ""
log "===== Group A: HUGS 15K baseline ====="

for SCENE in seattle parkinglot bike citron jogging; do
    log "--- A: HUGS 15K - ${SCENE} ---"
    python main.py \
        --cfg_file "cfg_files/release/neuman/hugs_human_scene_${SCENE}_original_15000.yaml" \
        2>&1 | tee -a "$LOG_FILE"
    log "--- A: ${SCENE} 完成 ---"
done

# ==========================================================
# Group B+C: depth_sup 12K -> attn 6K（仅 citron / jogging）
# ==========================================================
log ""
log "===== Group B+C: depth_sup 12K + attn 6K（citron / jogging）====="

for SCENE in citron jogging; do
    log ""
    log "=========================================="
    log ">>> 场景: ${SCENE}"
    log "=========================================="

    # ---- Phase 1: depth_sup 12K ----
    log "--- B: depth_sup 12000 - ${SCENE} ---"
    python main.py \
        --cfg_file "cfg_files/release/neuman/hugs_depth_supervised_12000_${SCENE}.yaml" \
        2>&1 | tee -a "$LOG_FILE"

    # 找 Phase 1 输出目录
    PHASE1_OUT=$(ls -dt "output/human_scene/neuman/${SCENE}/hugs_trimlp/hugs_depth_sup_12000_${SCENE}_${DATE_TAG}/"*/ 2>/dev/null | head -1)
    if [ -z "$PHASE1_OUT" ]; then
        log "ERROR: Phase 1 输出目录未找到，跳过 ${SCENE}"
        continue
    fi
    HUMAN_CKPT="${PHASE1_OUT}ckpt/human_final.pth"
    SCENE_CKPT="${PHASE1_OUT}ckpt/scene_final.pth"
    log "Phase 1 ckpt dir: ${PHASE1_OUT}"

    # ---- Phase 2: transl_xyz attention 6K ----
    log "--- C: transl_xyz attention 6000 - ${SCENE} ---"
    python main.py \
        --cfg_file "cfg_files/release/neuman/hugs_depth_sup12k_transl_xyz_attention_6000_${SCENE}.yaml" \
        "human.ckpt=${HUMAN_CKPT}" \
        "scene.ckpt=${SCENE_CKPT}" \
        2>&1 | tee -a "$LOG_FILE"

    log ">>> ${SCENE} 全流程完成！"
done

# ==========================================================
# 生成对比图
# ==========================================================
log ""
log "===== 生成对比图 ====="

# seattle / parkinglot / bike：our run 来自 20260529
for SCENE in seattle parkinglot bike; do
    BASELINE_DIR="output/human_scene/neuman/${SCENE}/hugs_trimlp/exp0_hugs_original_15000_${SCENE}_${DATE_TAG}"
    OUR_DIR="output/human_scene/neuman/${SCENE}/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_${SCENE}_20260529"

    if [ ! -d "$BASELINE_DIR" ] || [ ! -d "$OUR_DIR" ]; then
        log "WARNING: ${SCENE} 目录不存在，跳过对比图生成"
        log "  baseline: $BASELINE_DIR"
        log "  our:      $OUR_DIR"
        continue
    fi

    for MODE in full human; do
        log "生成对比图: ${SCENE} mode=${MODE}"
        python scripts/make_comparison_figure.py \
            --run-dir      "$OUR_DIR" \
            --baseline-dir "$BASELINE_DIR" \
            --n-frames 4 \
            --step final \
            --mode "$MODE" \
            --col-labels "GT,HUGS (15K),depth-sup+attn (Ours)" \
            2>&1 | tee -a "$LOG_FILE"
    done
done

# citron / jogging：our run 来自 20260601
for SCENE in citron jogging; do
    BASELINE_DIR="output/human_scene/neuman/${SCENE}/hugs_trimlp/exp0_hugs_original_15000_${SCENE}_${DATE_TAG}"
    OUR_DIR="output/human_scene/neuman/${SCENE}/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_${SCENE}_${DATE_TAG}"

    if [ ! -d "$BASELINE_DIR" ] || [ ! -d "$OUR_DIR" ]; then
        log "WARNING: ${SCENE} 目录不存在，跳过对比图生成"
        log "  baseline: $BASELINE_DIR"
        log "  our:      $OUR_DIR"
        continue
    fi

    for MODE in full human; do
        log "生成对比图: ${SCENE} mode=${MODE}"
        python scripts/make_comparison_figure.py \
            --run-dir      "$OUR_DIR" \
            --baseline-dir "$BASELINE_DIR" \
            --n-frames 4 \
            --step final \
            --mode "$MODE" \
            --col-labels "GT,HUGS (15K),depth-sup+attn (Ours)" \
            2>&1 | tee -a "$LOG_FILE"
    done
done

log ""
log "=========================================="
log "===== 全部完成！对比图已生成 ====="
log "=========================================="
