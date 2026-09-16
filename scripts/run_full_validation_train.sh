#!/bin/bash
# 完整验证训练排队脚本（在 VIMO 预处理完成后运行）
# 顺序：每个场景按 OUR → STM → HUGS 依次启动，等上一个结束再启动下一个
# 用法:
#   cd ml-hugs-work
#   nohup bash scripts/run_full_validation_train.sh > run_logs/full_validation_train.log 2>&1 &

set -e

HUGS_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
STM_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/StM/bin/python
HUGS_WORK=/workspace/nas_auto_backup/yuzilang/ml-hugs-work
STM_WORK=$HUGS_WORK/Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context

PREP_LOG=$HUGS_WORK/run_logs/full_validation_prep.log

echo "===== 完整验证训练排队启动 $(date) ====="

# ── 等待预处理完成 ─────────────────────────────────────────────────────────────
echo "等待 VIMO 预处理完成..."
while true; do
    if grep -q "预处理全部完成" "$PREP_LOG" 2>/dev/null; then
        echo "预处理已完成，开始训练排队 $(date)"
        break
    fi
    if grep -q "set -e" "$PREP_LOG" 2>/dev/null && grep -q "Traceback\|Error\|error" "$PREP_LOG" 2>/dev/null; then
        echo "WARNING: 预处理日志中检测到错误，继续检查各场景目录..."
        break
    fi
    sleep 60
done

# ── 再次确认各场景 _vimo 目录 ─────────────────────────────────────────────────
for seq in bike citron jogging parkinglot seattle; do
    if [ ! -d "$HUGS_WORK/data/neuman/dataset/${seq}_vimo" ]; then
        echo "ERROR: ${seq}_vimo 目录不存在，预处理可能未完成。退出。"
        exit 1
    fi
done
echo "所有场景 _vimo 目录确认存在"

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
run_and_wait() {
    local role=$1
    local exp_name=$2
    local log_file=$3
    local cmd=$4
    local work_dir=$5

    echo ""
    echo "---------- [$role] $exp_name ----------"
    echo "  启动时间: $(date)"
    echo "  日志: $log_file"

    cd "$work_dir"
    eval "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    echo "  PID: $pid"

    wait $pid
    local exit_code=$?
    echo "  结束时间: $(date), exit_code=$exit_code"
    cd "$HUGS_WORK"
    return $exit_code
}

# ── 场景顺序：bike citron jogging parkinglot seattle ─────────────────────────
# 每个场景：OUR(18k) → STM(20k) → HUGS(15k)

SCENES=(bike citron jogging parkinglot seattle)

for seq in "${SCENES[@]}"; do
    echo ""
    echo "========== [$seq] =========="

    # ── OUR (HUGS + anchor attention, 18k) ──────────────────────────────────
    OUR_CFG="cfg_files/release/neuman/hugs_vimo_${seq}_inline_attn_18k.yaml"
    OUR_EXP="vimo_inline_attn_18k_${seq}_20260607"
    OUR_LOG="$HUGS_WORK/run_logs/${OUR_EXP}.log"

    if ls "$HUGS_WORK/output/human_scene/neuman/$seq/hugs_trimlp/${OUR_EXP}/"*/ 2>/dev/null | grep -q .; then
        echo "[$seq][OUR] 已有结果，跳过"
    else
        run_and_wait "OUR" "$OUR_EXP" "$OUR_LOG" \
            "$HUGS_PY main.py --cfg_file $OUR_CFG" \
            "$HUGS_WORK"
    fi

    # ── STM (StM + VIMO, 20k) ───────────────────────────────────────────────
    STM_CFG="cfg_files/stm_vimo_${seq}_20k.yaml"
    STM_EXP="stm_vimo_${seq}_20k_20260607"
    STM_LOG="$HUGS_WORK/run_logs/${STM_EXP}.log"

    if ls "$STM_WORK/output_stm/human_scene/neuman/$seq/hugs_trimlp/${STM_EXP}/"*/ 2>/dev/null | grep -q .; then
        echo "[$seq][STM] 已有结果，跳过"
    else
        run_and_wait "STM" "$STM_EXP" "$STM_LOG" \
            "$STM_PY main.py --cfg_file $STM_CFG" \
            "$STM_WORK"
    fi

    # ── HUGS baseline (HUGS + VIMO, 15k, no attention) ─────────────────────
    HUGS_CFG="cfg_files/release/neuman/hugs_vimo_${seq}_baseline_15k.yaml"
    HUGS_EXP="vimo_hugs_baseline_${seq}_20260607"
    HUGS_LOG="$HUGS_WORK/run_logs/${HUGS_EXP}.log"

    if ls "$HUGS_WORK/output/human_scene/neuman/$seq/hugs_trimlp/${HUGS_EXP}/"*/ 2>/dev/null | grep -q .; then
        echo "[$seq][HUGS] 已有结果，跳过"
    else
        run_and_wait "HUGS" "$HUGS_EXP" "$HUGS_LOG" \
            "$HUGS_PY main.py --cfg_file $HUGS_CFG" \
            "$HUGS_WORK"
    fi

    echo "[$seq] 全部完成"
done

echo ""
echo "===== 全部训练完成 $(date) ====="
echo "现在可运行对比图生成脚本："
echo "  python scripts/make_validation_comparison.py --tag vimo_inline_attn_20260607 \\"
echo "    --scenes lab bike citron jogging parkinglot seattle \\"
echo "    --our   'output/human_scene/neuman/{seq}/hugs_trimlp/vimo_inline_attn_18k_{seq}_20260607' \\"
echo "    --stm   'Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/output_stm/human_scene/neuman/{seq}/hugs_trimlp/stm_vimo_{seq}_20k_20260607' \\"
echo "    --hugs  'output/human_scene/neuman/{seq}/hugs_trimlp/vimo_hugs_baseline_{seq}_20260607'"
