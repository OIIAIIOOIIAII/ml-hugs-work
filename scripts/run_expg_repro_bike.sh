#!/bin/bash
# ExpG 两阶段复现：Stage1 12k（无attention）+ Stage2 6k（有attention）
# 使用原始 NeuMan bike 数据集（GT 对齐）

set -e
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work

source /workspace/nas_auto_backup/yuzilang/miniconda3/etc/profile.d/conda.sh
conda activate hugs

LOG_DIR="run_logs"
mkdir -p "$LOG_DIR"

# ============ Stage 1 ============
echo "========== Stage 1: 12k steps, no attention =========="
python main.py --cfg_file cfg_files/debug/expg_repro_bike_s1_12k.yaml --seq bike \
    2>&1 | tee "$LOG_DIR/expg_repro_bike_s1_12k.log"

# 找到 Stage 1 最新的输出目录
S1_LOGDIR=$(ls -td output/human_scene/neuman/bike/hugs_trimlp/expg_repro_bike_s1_12k/*/ 2>/dev/null | head -1)
if [ -z "$S1_LOGDIR" ]; then
    echo "ERROR: Stage 1 output directory not found!"
    exit 1
fi
S1_CKPT_DIR="${S1_LOGDIR}ckpt"
echo "Stage 1 checkpoint dir: $S1_CKPT_DIR"

HUMAN_CKPT=$(ls "$S1_CKPT_DIR"/human_final.pth 2>/dev/null || ls "$S1_CKPT_DIR"/human_*.pth 2>/dev/null | sort | tail -1)
SCENE_CKPT=$(ls "$S1_CKPT_DIR"/scene_final.pth 2>/dev/null || ls "$S1_CKPT_DIR"/scene_*.pth 2>/dev/null | sort | tail -1)

if [ -z "$HUMAN_CKPT" ] || [ -z "$SCENE_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoints not found in $S1_CKPT_DIR"
    exit 1
fi
echo "Human ckpt: $HUMAN_CKPT"
echo "Scene ckpt: $SCENE_CKPT"

# ============ Stage 2 ============
echo "========== Stage 2: 6k steps, with attention =========="

# 将 Stage 1 的 checkpoint 路径写入 Stage 2 配置
S2_CFG="cfg_files/debug/expg_repro_bike_s2_6k.yaml"
S2_CFG_TMP="cfg_files/debug/expg_repro_bike_s2_6k_run.yaml"

sed "s|PLACEHOLDER_HUMAN_CKPT|$HUMAN_CKPT|g; s|PLACEHOLDER_SCENE_CKPT|$SCENE_CKPT|g" \
    "$S2_CFG" > "$S2_CFG_TMP"

python main.py --cfg_file "$S2_CFG_TMP" --seq bike \
    2>&1 | tee "$LOG_DIR/expg_repro_bike_s2_6k.log"

echo "========== Done! =========="
