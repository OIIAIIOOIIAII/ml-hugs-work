#!/bin/bash
# 截帧实验：往返轨迹假说验证
#   bike 前60帧 (OUR + STM)
#   jogging 前50帧 (OUR + STM)

set -e

HUGS_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
STM_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/StM/bin/python
HUGS_WORK=/workspace/nas_auto_backup/yuzilang/ml-hugs-work
STM_WORK=$HUGS_WORK/Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context
DATE=20260607

skip_if_done() {
    local out=$1
    [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null)" ]
}

run_exp() {
    local role=$1 exp=$2 log=$3 cmd=$4 work=${5:-$HUGS_WORK}
    echo ""
    echo "---------- [$role] $exp ----------"
    echo "  启动时间: $(date)"
    echo "  日志: $log"
    cd "$work"
    eval "$cmd" > "$log" 2>&1
    local code=$?
    echo "  结束时间: $(date), exit_code=$code"
    [ $code -eq 0 ] || { echo "[$exp] FAILED (exit $code)"; exit 1; }
}

echo "===== 截帧实验开始 $(date) ====="

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
echo "===== 截帧实验全部完成 $(date) ====="
