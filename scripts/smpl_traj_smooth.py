"""
方案 B：SMPL 轨迹平滑
对 VIMO 输出的 transl 序列用 Savitzky-Golay 滤波去除高频抖动。
不修正系统性偏差，只减少逐帧噪声，使相邻帧间的 SMPL 位置更连贯。

效果：
  - 减少逐帧高频噪声（避免多帧信号方向矛盾导致高斯球摊平/模糊）
  - 不改变低频运动轨迹（window 越大平滑越强但会丢失真实运动细节）
  - 不修正 VIMO 的系统性误差（~90mm 均值偏差）

用法：
    cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
    conda run -n hugs python scripts/smpl_traj_smooth.py \
        --seq lab_vimo [--window 11] [--polyorder 3] [--output_suffix _smoothed]
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter


def smooth(seq, window=11, polyorder=3, output_suffix='_smoothed',
           dataset_root='data/neuman/dataset'):
    smpl_path = f'{dataset_root}/{seq}/4d_humans/smpl_optimized_aligned_scale.npz'
    output_path = smpl_path.replace('.npz', f'{output_suffix}.npz')

    smpl_data = np.load(smpl_path)
    smpl_dict = {k: smpl_data[k].copy() for k in smpl_data.files}
    transl = smpl_dict['transl']  # (N, 3)

    n = transl.shape[0]
    print(f'[smooth] seq={seq}, n_frames={n}, window={window}, polyorder={polyorder}')
    print(f'  input:  {smpl_path}')
    print(f'  output: {output_path}')

    # 平滑前统计
    diffs_before = np.linalg.norm(np.diff(transl, axis=0), axis=1)
    print(f'  frame-to-frame delta (before): mean={diffs_before.mean():.4f}, '
          f'max={diffs_before.max():.4f} COLMAP units')

    # Savitzky-Golay 滤波（对 x/y/z 分量分别滤波）
    transl_smooth = savgol_filter(transl, window_length=window, polyorder=polyorder, axis=0)

    diffs_after = np.linalg.norm(np.diff(transl_smooth, axis=0), axis=1)
    print(f'  frame-to-frame delta (after):  mean={diffs_after.mean():.4f}, '
          f'max={diffs_after.max():.4f} COLMAP units')

    correction = transl_smooth - transl
    print(f'  correction norm: mean={np.linalg.norm(correction, axis=1).mean():.4f}, '
          f'max={np.linalg.norm(correction, axis=1).max():.4f} COLMAP units')

    smpl_dict['transl'] = transl_smooth.astype(np.float32)
    np.savez(output_path, **smpl_dict)
    print(f'[smooth] saved to {output_path}')
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='lab_vimo')
    parser.add_argument('--window', type=int, default=11,
                        help='Savitzky-Golay window length (odd integer)')
    parser.add_argument('--polyorder', type=int, default=3)
    parser.add_argument('--output_suffix', default='_smoothed')
    parser.add_argument('--dataset_root', default='data/neuman/dataset')
    args = parser.parse_args()

    os.chdir(str(Path(__file__).parent.parent))
    smooth(args.seq, args.window, args.polyorder, args.output_suffix, args.dataset_root)
