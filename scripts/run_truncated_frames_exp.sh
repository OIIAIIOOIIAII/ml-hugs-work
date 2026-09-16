#!/bin/bash
# 截帧实验：bike 前60帧 / jogging 前50帧，各跑 OUR 和 STM
# 假设：场景 _vimo 目录已存在
# 用法:
#   cd ml-hugs-work
#   nohup bash scripts/run_truncated_frames_exp.sh > run_logs/truncated_frames_exp.log 2>&1 &

HUGS_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
STM_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/StM/bin/python
HUGS_WORK=/workspace/nas_auto_backup/yuzilang/ml-hugs-work
STM_WORK=$HUGS_WORK/Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context

echo "===== truncated_frames_exp 启动 $(date) ====="

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

# ── bike 前60帧 OUR ───────────────────────────────────────────────────────────
EXP="vimo_inline_attn_18k_bike_f60_20260607"
CFG="cfg_files/release/neuman/hugs_vimo_bike_f60_inline_attn_18k.yaml"
LOG="$HUGS_WORK/run_logs/${EXP}.log"
if ls "$HUGS_WORK/output/human_scene/neuman/bike/hugs_trimlp/${EXP}/"*/ 2>/dev/null | grep -q .; then
    echo "[bike f60][OUR] 已有结果，跳过"
else
    run_and_wait "OUR f60" "$EXP" "$LOG" "$HUGS_PY main.py --cfg_file $CFG" "$HUGS_WORK"
fi

# ── bike 前60帧 STM ───────────────────────────────────────────────────────────
EXP="stm_vimo_bike_f60_20k_20260607"
CFG="cfg_files/stm_vimo_bike_f60_20k.yaml"
LOG="$HUGS_WORK/run_logs/${EXP}.log"
if ls "$STM_WORK/output_stm/human_scene/neuman/bike/hugs_trimlp/${EXP}/"*/ 2>/dev/null | grep -q .; then
    echo "[bike f60][STM] 已有结果，跳过"
else
    run_and_wait "STM f60" "$EXP" "$LOG" "$STM_PY main.py --cfg_file $CFG" "$STM_WORK"
fi

# ── jogging 前50帧 OUR ───────────────────────────────────────────────────────
EXP="vimo_inline_attn_18k_jogging_f50_20260607"
CFG="cfg_files/release/neuman/hugs_vimo_jogging_f50_inline_attn_18k.yaml"
LOG="$HUGS_WORK/run_logs/${EXP}.log"
if ls "$HUGS_WORK/output/human_scene/neuman/jogging/hugs_trimlp/${EXP}/"*/ 2>/dev/null | grep -q .; then
    echo "[jogging f50][OUR] 已有结果，跳过"
else
    run_and_wait "OUR f50" "$EXP" "$LOG" "$HUGS_PY main.py --cfg_file $CFG" "$HUGS_WORK"
fi

# ── jogging 前50帧 STM ───────────────────────────────────────────────────────
EXP="stm_vimo_jogging_f50_20k_20260607"
CFG="cfg_files/stm_vimo_jogging_f50_20k.yaml"
LOG="$HUGS_WORK/run_logs/${EXP}.log"
if ls "$STM_WORK/output_stm/human_scene/neuman/jogging/hugs_trimlp/${EXP}/"*/ 2>/dev/null | grep -q .; then
    echo "[jogging f50][STM] 已有结果，跳过"
else
    run_and_wait "STM f50" "$EXP" "$LOG" "$STM_PY main.py --cfg_file $CFG" "$STM_WORK"
fi

echo ""
echo "===== truncated_frames_exp 全部完成 $(date) ====="
