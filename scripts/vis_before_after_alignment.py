#!/usr/bin/env python3
"""
粗对齐 before/after 点云对比图（答辩用）。

before = 完全没对齐：SMPL 在相机空间（smpl_pred的 verts + trans，米制），
         直接丢进 COLMAP场景坐标系 —— 尺度/位置都不匹配。
after  = 对齐后：aligned npz 经 SMPL forward + scale + transl，人体正确嵌入场景。

输出一张 PNG：2 行（Top X-Z / Side Z-Y）× 2 列（before / after）。
每一行 before/after 共享同一坐标范围（=场景范围），直观看出对齐差异。

Usage:
  HP=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python
  $HP scripts/vis_before_after_alignment.py --seq bike --step 8
"""
import argparse, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '/workspace/nas_auto_backup/yuzilang/ml-hugs-work')
from hugs.models.modules.smpl_layer import SMPL

SMPL_MODEL = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/smpl/SMPL_NEUTRAL.pkl'
DATA_ROOT  = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/neuman/dataset'


def load_colmap_points(txt_path):
    xyz, rgb = [], []
    with open(txt_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8)


def smpl_verts_world(smpl, global_orient, body_pose, betas, transl, scale):
    """对齐后：canonical SMPL -> scale -> +transl (world/COLMAP 单位)。"""
    go = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
    bp = torch.tensor(body_pose[:69], dtype=torch.float32).unsqueeze(0)
    b  = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)
    t  = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        out = smpl(global_orient=go, body_pose=bp, betas=b, transl=t)
    v = out.vertices[0].numpy()
    return (v * scale + transl).astype(np.float32)


def get_before_verts(seq, frames):
    """完全没对齐：相机空间 verts(含姿态) + trans（米制），不做任何场景对齐。"""
    files = sorted(glob.glob(f'{DATA_ROOT}/{seq}/smpl_pred/*.npz'))
    out = []
    for fi in frames:
        if fi >= len(files):
            continue
        r = np.load(files[fi], allow_pickle=True)['results'].item()
        v = np.asarray(r['verts'], dtype=np.float32)      # (6890,3) 原点中心
        t = np.asarray(r['trans'], dtype=np.float32)       # (3,)相机空间平移(米)
        out.append(v + t[None, :])
    return out


def get_after_verts(smpl, seq, frames, npz_name):
    npz = np.load(f'{DATA_ROOT}/{seq}/4d_humans/{npz_name}', allow_pickle=True)
    betas = npz['betas'][0] if npz['betas'].ndim == 2 else npz['betas']
    out = []
    for fi in frames:
        sc = float(npz['scale'][fi] if npz['scale'].ndim == 1 else npz['scale'][fi, 0])
        bp = npz['body_pose'][fi]; bp = bp[3:] if len(bp) == 72 else bp
        out.append(smpl_verts_world(smpl, npz['global_orient'][fi], bp,
                                    betas, npz['transl'][fi], sc))
    return out


def draw_panel(ax, scene_xyz, scene_col, human_list, hcolor, xi, yi, xl, yl, title):
    ax.scatter(scene_xyz[:, xi], scene_xyz[:, yi],
               c=scene_col, s=1.5, alpha=0.35, rasterized=True)
    for v in human_list:
        ax.scatter(v[:, xi], v[:, yi], c=hcolor, s=1.0, alpha=0.5, rasterized=True)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='bike')
    ap.add_argument('--step', type=int, default=8)
    ap.add_argument('--after-npz', default='smpl_optimized_aligned_scale.npz',
                    help='对齐后 npz 文件名（默认 GT 对齐）')
    ap.add_argument('--pad', type=float, default=1.15, help='坐标范围外扩倍数')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    out_path = args.out or f'paper_figures/before_after_{args.seq}.png'
    seq_dir = f'{DATA_ROOT}/{args.seq}'

    print('[smpl] loading...')
    smpl = SMPL(model_path=SMPL_MODEL).eval()

    print('[scene] loading COLMAP...')
    scene_xyz, scene_rgb = load_colmap_points(f'{seq_dir}/sparse/points3D.txt')
    scene_col = (scene_rgb.astype(float) / 255.0) * 0.5 + 0.35

    npz = np.load(f'{seq_dir}/4d_humans/{args.after_npz}', allow_pickle=True)
    N = len(npz['transl'])
    frames = list(range(0, N, args.step))
    print(f'[frames] N={N}, step={args.step} -> {len(frames)} frames')

    before = get_before_verts(args.seq, frames)
    after  = get_after_verts(smpl, args.seq, frames, args.after_npz)
    print(f'[before] {len(before)} frames (相机空间米制)')
    print(f'[after ] {len(after)} frames (world/COLMAP)')

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    fig.suptitle(f'Coarse Alignment — {args.seq}   '
                 f'(gray=COLMAP scene, red=NOT aligned, green=aligned)',
                 fontsize=15, fontweight='bold')

    views = [(0, 2, 'X', 'Z (depth)', 'Top view'),
             (2, 1, 'Z (depth)', 'Y (height)', 'Side view')]

    for row, (xi, yi, xl, yl, vname) in enumerate(views):
        af = np.concatenate(after, axis=0)
        cx = np.concatenate([scene_xyz[:, xi], af[:, xi]])
        cy = np.concatenate([scene_xyz[:, yi], af[:, yi]])
        x0, x1 = np.percentile(cx, [1, 99]); y0, y1 = np.percentile(cy, [1, 99])
        mx = (x1 - x0) * (args.pad - 1) / 2; my = (y1 - y0) * (args.pad - 1) / 2
        xlim = (x0 - mx, x1 + mx); ylim = (y0 - my, y1 + my)

        draw_panel(axes[row, 0], scene_xyz, scene_col, before, 'red',
                   xi, yi, xl, yl, f'{vname} — BEFORE (not aligned)')
        draw_panel(axes[row, 1], scene_xyz, scene_col, after, 'green',
                   xi, yi, xl, yl, f'{vname} — AFTER (aligned)')
        for c in (0, 1):
            axes[row, c].set_xlim(*xlim); axes[row, c].set_ylim(*ylim)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n[done] -> {out_path}')


if __name__ == '__main__':
    main()
