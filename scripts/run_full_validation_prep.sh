#!/bin/bash
# 完整验证预处理：为 5 个场景跑 coarse_align_v3 + VIMO + 创建 {seq}_vimo 目录
# 用法: cd ml-hugs-work && bash scripts/run_full_validation_prep.sh > run_logs/full_validation_prep.log 2>&1

set -e
PYTHON=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
DATA_ROOT=data/neuman/dataset
SCENES=(bike citron jogging parkinglot seattle)

echo "===== 开始完整验证预处理 $(date) ====="

for seq in "${SCENES[@]}"; do
    echo ""
    echo "========== [$seq] =========="
    seq_dir="$DATA_ROOT/$seq"
    vimo_seq_dir="$DATA_ROOT/${seq}_vimo"

    # ── Step 1: coarse_align_v3（foot-contact scale 估计）──────────────────
    if [ -f "output/coarse_align_v3/$seq/coarse_align_v3.json" ]; then
        echo "[$seq] coarse_align_v3 已存在，跳过"
    else
        echo "[$seq] 运行 coarse_align_v3 ..."
        $PYTHON scripts/coarse_align_smpl_neuman_v3.py --seq $seq
        echo "[$seq] coarse_align_v3 完成"
    fi

    # ── Step 2: VIMO 推断（视频 HMR）─────────────────────────────────────
    if [ -f "output/vimo_hps/$seq/vimo_results.npz" ]; then
        echo "[$seq] vimo_results 已存在，跳过"
    else
        echo "[$seq] 运行 VIMO ..."
        $PYTHON scripts/run_vimo_neuman.py --seq $seq
        echo "[$seq] VIMO 完成"
    fi

    # ── Step 3: 生成 VIMO 对齐的 SMPL npz ──────────────────────────────
    vimo_npz="$seq_dir/4d_humans/smpl_optimized_aligned_scale_vimo.npz"
    if [ -f "$vimo_npz" ]; then
        echo "[$seq] gen_coarse_npz_vimo 已存在，跳过"
    else
        echo "[$seq] 生成 VIMO 对齐 SMPL npz ..."
        $PYTHON scripts/gen_coarse_npz_vimo.py --seq $seq
        echo "[$seq] gen_coarse_npz_vimo 完成"
    fi

    # ── Step 4: 创建 {seq}_vimo 目录结构 ────────────────────────────────
    if [ -d "$vimo_seq_dir" ]; then
        echo "[$seq] ${seq}_vimo 目录已存在，跳过"
        continue
    fi

    echo "[$seq] 创建 ${seq}_vimo 目录结构 ..."
    mkdir -p "$vimo_seq_dir"

    # 顶层符号链接
    for item in alignments.npy densepose depth_maps images keypoints \
                masked_images mono_depth segmentations sparse vitpose_out \
                smpl_output_optimized.pkl smpl_output_romp.pkl smpl_pred; do
        src="$(pwd)/$seq_dir/$item"
        dst="$vimo_seq_dir/$item"
        if [ -e "$src" ] && [ ! -e "$dst" ]; then
            ln -sf "$src" "$dst"
        fi
    done

    # 4d_humans 子目录
    mkdir -p "$vimo_seq_dir/4d_humans"
    orig_4d="$(pwd)/$seq_dir/4d_humans"
    vimo_4d="$(pwd)/$vimo_seq_dir/4d_humans"

    # 链接原始文件
    for item in cameras.npz det_track_results.pkl hmr_results.pkl \
                images masks poses.npz poses_optimized.npz \
                sam_segmentations smpl_opt_render; do
        src="$orig_4d/$item"
        dst="$vimo_4d/$item"
        if [ -e "$src" ] && [ ! -e "$dst" ]; then
            ln -sf "$src" "$dst"
        fi
    done

    # 原始 GT 链接
    ln -sf "$orig_4d/smpl_optimized_aligned_scale.npz" \
           "$vimo_4d/smpl_optimized_aligned_scale_gt.npz"

    # VIMO 版本作为主 SMPL 文件
    cp "$orig_4d/smpl_optimized_aligned_scale_vimo.npz" \
       "$vimo_4d/smpl_optimized_aligned_scale.npz"

    echo "[$seq] ${seq}_vimo 目录创建完成"
done

echo ""
echo "===== 预处理全部完成 $(date) ====="
