#!/bin/bash
# 等待 Exp F3 (PID 880011) 结束后运行 Exp G + vitpose_kp
F3_PID=880011
LOG_FILE="run_logs/expG_vitpose_kp_20260606.log"

echo "Waiting for Exp F3 (PID $F3_PID) to finish..."
while kill -0 $F3_PID 2>/dev/null; do
    sleep 60
done

echo "Exp F3 finished. Launching Exp G + vitpose_kp at $(date)"
mkdir -p run_logs
nohup /workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python main.py \
    --cfg_file cfg_files/release/neuman/hugs_vimo_lab_inline_attn_vitpose_kp_18k.yaml \
    > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "Started Exp G+vitpose_kp with PID: $NEW_PID"
echo "Log: $LOG_FILE"
