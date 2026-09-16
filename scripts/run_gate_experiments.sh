#!/bin/bash
# Global Scene Gate 对比实验：bike + jogging
# 等 GPU 空闲后依次启动，用法：
#   nohup bash scripts/run_gate_experiments.sh > run_logs/gate_experiments.log 2>&1 &

set -e
HUGS_PY=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
HUGS_WORK=/workspace/nas_auto_backup/yuzilang/ml-hugs-work
DATE=20260607

cd "$HUGS_WORK"

run_and_wait() {
    local label=$1 exp=$2 log=$3; shift 3
    echo ""
    echo "---------- [$label] $exp ----------"
    echo "  启动时间: $(date)"
    echo "  日志: $log"
    "$@" > "$log" 2>&1
    local code=$?
    echo "  结束时间: $(date), exit_code=$code"
    [ $code -eq 0 ] || { echo "[$exp] FAILED"; exit 1; }
}

# ── bike with global gate ──────────────────────────────────────────────────
run_and_wait "GATE" "vimo_gate_18k_bike_${DATE}" \
    "$HUGS_WORK/run_logs/vimo_gate_18k_bike_${DATE}.log" \
    "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_bike_gate_18k.yaml"
echo "[bike gate] 完成"

# ── jogging with global gate ───────────────────────────────────────────────
run_and_wait "GATE" "vimo_gate_18k_jogging_${DATE}" \
    "$HUGS_WORK/run_logs/vimo_gate_18k_jogging_${DATE}.log" \
    "$HUGS_PY main.py --cfg_file cfg_files/release/neuman/hugs_vimo_jogging_gate_18k.yaml"
echo "[jogging gate] 完成"

echo ""
echo "=== 全部 gate 实验完成 ==="
