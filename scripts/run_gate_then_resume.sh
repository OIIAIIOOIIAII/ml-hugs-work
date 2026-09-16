#!/bin/bash
# 在 parkinglot STM 结束后：
#   1. 停止原训练排队脚本
#   2. 运行 gate 实验（bike + jogging）
#   3. 继续剩余验证实验（parkinglot HUGS + seattle OUR/STM/HUGS）
#
# 用法（在 ml-hugs-work 目录下）：
#   nohup bash scripts/run_gate_then_resume.sh > run_logs/gate_then_resume.log 2>&1 &

set -e

HUGS_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
STM_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/StM/bin/python
HUGS_WORK=/workspace/nas_auto_backup/yuzilang/ml-hugs-work
STM_WORK=$HUGS_WORK/Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context
DATE=20260607

# 原训练脚本的根进程 PID（手动确认：ps aux | grep run_full_validation）
OLD_SCRIPT_PID=983995

STM_PARKINGLOT_LOG="$HUGS_WORK/run_logs/stm_vimo_parkinglot_20k_${DATE}.log"

echo "===== gate_then_resume 启动 $(date) ====="
echo "  监视 parkinglot STM 日志：$STM_PARKINGLOT_LOG"
echo "  原脚本 PID：$OLD_SCRIPT_PID"

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
run_exp() {
    local role=$1 exp=$2 log=$3 cmd=$4 work=${5:-$HUGS_WORK}
    echo ""
    echo "---------- [$role] $exp ----------"
    echo "  启动时间: $(date)"
    echo "  日志: $log"
    cd "$work"
    eval "$cmd" > "$log" 2>&1 &
    local pid=$!
    echo "  PID: $pid"
    wait $pid
    local code=$?
    echo "  结束时间: $(date), exit_code=$code"
    cd "$HUGS_WORK"
    [ $code -eq 0 ] || { echo "[$exp] FAILED (exit $code)"; exit 1; }
}

skip_if_done() {
    # 返回 0 = 跳过（已有结果），1 = 需要运行
    local out_dir=$1
    ls "$out_dir/"*/ 2>/dev/null | grep -q . && return 0 || return 1
}

# ── 1. 等待 parkinglot STM 完成 ────────────────────────────────────────────
echo ""
echo "[1/4] 等待 parkinglot STM 完成..."
while true; do
    if grep -q "final - HUGS_HUMAN_PSNR" "$STM_PARKINGLOT_LOG" 2>/dev/null; then
        echo "  ✓ parkinglot STM 已完成（检测到 final PSNR）"
        break
    fi
    sleep 15
done

# STM 结束后还有 video 生成，等待 90 秒让 STM 完全退出，防止 parkinglot HUGS 已启动
echo "  等待 90 秒，确保 STM 进程完全退出..."
sleep 90

# ── 2. 停止原训练脚本（阻止 parkinglot HUGS 自动启动）───────────────────────
echo ""
echo "[2/4] 停止原训练脚本 (PID $OLD_SCRIPT_PID)..."
# 先 SIGTERM，给 bash 机会清理；再 SIGKILL 确保停止
kill -TERM $OLD_SCRIPT_PID 2>/dev/null && echo "  SIGTERM 已发送" || echo "  原脚本已不存在（可能已自然结束）"
sleep 5
kill -KILL $OLD_SCRIPT_PID 2>/dev/null || true
# 同时清理所有子 bash（parkinglot HUGS 若刚启动则也要停）
pkill -P $OLD_SCRIPT_PID 2>/dev/null || true
# 如果 parkinglot HUGS Python 已经启动，优雅地等它跑几步后再考虑 kill
# （此处不 kill Python，因为 parkinglot HUGS 若已启动则让它跑完也可；
#  如果需要强制 kill 可取消下行注释）
# pkill -f "hugs_vimo_parkinglot_baseline_15k" 2>/dev/null || true
echo "  原脚本已停止"

# ── 3. Gate 实验 ────────────────────────────────────────────────────────────
echo ""
echo "[3/4] 运行 Global Scene Gate 实验..."

# bike gate
GATE_BIKE_EXP="vimo_gate_18k_bike_${DATE}"
GATE_BIKE_OUT="$HUGS_WORK/output/human_scene/neuman/bike_vimo/hugs_trimlp/${GATE_BIKE_EXP}"
if skip_if_done "$GATE_BIKE_OUT"; then
    echo "[bike gate] 已有结果，跳过"
else
    run_exp "GATE" "$GATE_BIKE_EXP" \
        "$HUGS_WORK/run_logs/${GATE_BIKE_EXP}.log" \
        "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_bike_gate_18k.yaml" \
        "$HUGS_WORK"
    echo "[bike gate] 完成"
fi

# jogging gate
GATE_JOG_EXP="vimo_gate_18k_jogging_${DATE}"
GATE_JOG_OUT="$HUGS_WORK/output/human_scene/neuman/jogging_vimo/hugs_trimlp/${GATE_JOG_EXP}"
if skip_if_done "$GATE_JOG_OUT"; then
    echo "[jogging gate] 已有结果，跳过"
else
    run_exp "GATE" "$GATE_JOG_EXP" \
        "$HUGS_WORK/run_logs/${GATE_JOG_EXP}.log" \
        "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_jogging_gate_18k.yaml" \
        "$HUGS_WORK"
    echo "[jogging gate] 完成"
fi

# ── 4. 继续剩余验证实验 ─────────────────────────────────────────────────────
echo ""
echo "[4/4] 继续剩余验证实验（parkinglot HUGS + seattle）..."

# parkinglot HUGS baseline
PK_HUGS_EXP="vimo_hugs_baseline_parkinglot_${DATE}"
PK_HUGS_OUT="$HUGS_WORK/output/human_scene/neuman/parkinglot_vimo/hugs_trimlp/${PK_HUGS_EXP}"
if skip_if_done "$PK_HUGS_OUT"; then
    echo "[parkinglot HUGS] 已有结果，跳过"
else
    run_exp "HUGS" "$PK_HUGS_EXP" \
        "$HUGS_WORK/run_logs/${PK_HUGS_EXP}.log" \
        "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_parkinglot_baseline_15k.yaml"
    echo "[parkinglot HUGS] 完成"
fi

# seattle OUR
SEA_OUR_EXP="vimo_inline_attn_18k_seattle_${DATE}"
SEA_OUR_OUT="$HUGS_WORK/output/human_scene/neuman/seattle_vimo/hugs_trimlp/${SEA_OUR_EXP}"
if skip_if_done "$SEA_OUR_OUT"; then
    echo "[seattle OUR] 已有结果，跳过"
else
    run_exp "OUR" "$SEA_OUR_EXP" \
        "$HUGS_WORK/run_logs/${SEA_OUR_EXP}.log" \
        "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_seattle_inline_attn_18k.yaml"
    echo "[seattle OUR] 完成"
fi

# seattle STM
SEA_STM_EXP="stm_vimo_seattle_20k_${DATE}"
SEA_STM_OUT="$STM_WORK/output_stm/human_scene/neuman/seattle_vimo/hugs_trimlp/${SEA_STM_EXP}"
if skip_if_done "$SEA_STM_OUT"; then
    echo "[seattle STM] 已有结果，跳过"
else
    run_exp "STM" "$SEA_STM_EXP" \
        "$HUGS_WORK/run_logs/${SEA_STM_EXP}.log" \
        "$STM_PY main.py --cfg_file cfg_files/stm_vimo_seattle_20k.yaml" \
        "$STM_WORK"
    echo "[seattle STM] 完成"
fi

# seattle HUGS baseline
SEA_HUGS_EXP="vimo_hugs_baseline_seattle_${DATE}"
SEA_HUGS_OUT="$HUGS_WORK/output/human_scene/neuman/seattle_vimo/hugs_trimlp/${SEA_HUGS_EXP}"
if skip_if_done "$SEA_HUGS_OUT"; then
    echo "[seattle HUGS] 已有结果，跳过"
else
    run_exp "HUGS" "$SEA_HUGS_EXP" \
        "$HUGS_WORK/run_logs/${SEA_HUGS_EXP}.log" \
        "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_seattle_baseline_15k.yaml"
    echo "[seattle HUGS] 完成"
fi

# ── 5. 截帧实验（往返轨迹假说验证）────────────────────────────────────────────
echo ""
echo "[5/5] 截帧实验：bike 前60帧 / jogging 前50帧，OUR + STM..."

# bike f60 OUR
EXP="vimo_inline_attn_18k_bike_f60_${DATE}"
OUT="$HUGS_WORK/output/human_scene/neuman/bike/hugs_trimlp/${EXP}"
if skip_if_done "$OUT"; then
    echo "[bike f60 OUR] 已有结果，跳过"
else
    run_exp "OUR f60" "$EXP" \
        "$HUGS_WORK/run_logs/${EXP}.log" \
        "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_bike_f60_inline_attn_18k.yaml"
    echo "[bike f60 OUR] 完成"
fi

# bike f60 STM
EXP="stm_vimo_bike_f60_20k_${DATE}"
OUT="$STM_WORK/output_stm/human_scene/neuman/bike/hugs_trimlp/${EXP}"
if skip_if_done "$OUT"; then
    echo "[bike f60 STM] 已有结果，跳过"
else
    run_exp "STM f60" "$EXP" \
        "$HUGS_WORK/run_logs/${EXP}.log" \
        "$STM_PY main.py --cfg_file cfg_files/stm_vimo_bike_f60_20k.yaml" \
        "$STM_WORK"
    echo "[bike f60 STM] 完成"
fi

# jogging f50 OUR
EXP="vimo_inline_attn_18k_jogging_f50_${DATE}"
OUT="$HUGS_WORK/output/human_scene/neuman/jogging/hugs_trimlp/${EXP}"
if skip_if_done "$OUT"; then
    echo "[jogging f50 OUR] 已有结果，跳过"
else
    run_exp "OUR f50" "$EXP" \
        "$HUGS_WORK/run_logs/${EXP}.log" \
        "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_jogging_f50_inline_attn_18k.yaml"
    echo "[jogging f50 OUR] 完成"
fi

# jogging f50 STM
EXP="stm_vimo_jogging_f50_20k_${DATE}"
OUT="$STM_WORK/output_stm/human_scene/neuman/jogging/hugs_trimlp/${EXP}"
if skip_if_done "$OUT"; then
    echo "[jogging f50 STM] 已有结果，跳过"
else
    run_exp "STM f50" "$EXP" \
        "$HUGS_WORK/run_logs/${EXP}.log" \
        "$STM_PY main.py --cfg_file cfg_files/stm_vimo_jogging_f50_20k.yaml" \
        "$STM_WORK"
    echo "[jogging f50 STM] 完成"
fi

echo ""
echo "===== 全部完成 $(date) ====="
