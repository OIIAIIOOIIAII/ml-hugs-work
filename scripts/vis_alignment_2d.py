#!/usr/bin/env python3
"""
2D 投影可视化：场景点云 + GT/v4 人体 bbox + 轨迹
输出 PNG，直接用 matplotlib，无需图形界面。

Usage:
  python scripts/vis_alignment_2d.py --seq seattle
  python scripts/vis_alignment_2d.py --seq bike
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, '/workspace/nas_auto_backup/yuzilang/ml-hugs-work')
from hugs.models.modules.smpl_layer import SMPL

SMPL_MODEL = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/smpl/SMPL_NEUTRAL.pkl'
DATA_ROOT   = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/neuman/dataset'


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


def get_smpl_verts(smpl, global_orient, body_pose, betas, transl, scale):
    go = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
    bp = torch.tensor(body_pose[:69],  dtype=torch.float32).unsqueeze(0)
    b  = torch.tensor(betas[:10],      dtype=torch.float32).unsqueeze(0)
    t  = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        out = smpl(global_orient=go, body_pose=bp, betas=b, transl=t)
    v = out.vertices[0].numpy()
    return (v * scale + transl).astype(np.float32)


def bbox_from_verts(verts):
    return verts.min(0), verts.max(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq',    default='seattle')
    parser.add_argument('--suffix', default='vimo_v4')
    parser.add_argument('--step',   type=int, default=5)
    parser.add_argument('--out',    default='')
    args = parser.parse_args()

    out_path = args.out or f'/tmp/vis2d_{args.seq}_{args.suffix}.png'
    gt_dir = f'{DATA_ROOT}/{args.seq}'
    v4_dir = f'{DATA_ROOT}/{args.seq}_{args.suffix}'

    print('[smpl] loading...')
    smpl = SMPL(model_path=SMPL_MODEL).eval()

    print('[scene] loading...')
    scene_xyz, scene_rgb = load_colmap_points(f'{gt_dir}/sparse/points3D.txt')
    scene_col = scene_rgb.astype(float) / 255.0

    gt = np.load(f'{gt_dir}/4d_humans/smpl_optimized_aligned_scale.npz', allow_pickle=True)
    for cand in [
        f'{v4_dir}/4d_humans/smpl_optimized_aligned_scale.npz',
        f'{gt_dir}/4d_humans/smpl_optimized_aligned_scale_{args.suffix}.npz',
    ]:
        if os.path.exists(cand):
            v4 = np.load(cand, allow_pickle=True); break

    N = len(gt['transl'])
    frames = list(range(0, N, args.step))
    print(f'[frames] N={N}, sample every {args.step} → {len(frames)} frames')

    gt_betas = gt['betas'][0] if gt['betas'].ndim == 2 else gt['betas']
    v4_betas = v4['betas'][0] if v4['betas'].ndim == 2 else v4['betas']

    gt_boxes, v4_boxes, gt_trans, v4_trans = [], [], [], []
    for fi in frames:
        sc_gt = float(gt['scale'][fi] if gt['scale'].ndim == 1 else gt['scale'][fi, 0])
        sc_v4 = float(v4['scale'][fi] if v4['scale'].ndim == 1 else v4['scale'][fi, 0])
        bp_gt = gt['body_pose'][fi]; bp_gt = bp_gt[3:] if len(bp_gt) == 72 else bp_gt
        bp_v4 = v4['body_pose'][fi]; bp_v4 = bp_v4[3:] if len(bp_v4) == 72 else bp_v4

        vg = get_smpl_verts(smpl, gt['global_orient'][fi], bp_gt, gt_betas, gt['transl'][fi], sc_gt)
        vv = get_smpl_verts(smpl, v4['global_orient'][fi], bp_v4, v4_betas, v4['transl'][fi], sc_v4)
        gt_boxes.append(bbox_from_verts(vg))
        v4_boxes.append(bbox_from_verts(vv))
        gt_trans.append(gt['transl'][fi])
        v4_trans.append(v4['transl'][fi])

    gt_trans = np.array(gt_trans)
    v4_trans = np.array(v4_trans)

    # ── Compute frame-averaged bboxes for visual overlap check ────────────────
    mean_gt_min = np.mean([b[0] for b in gt_boxes], axis=0)
    mean_gt_max = np.mean([b[1] for b in gt_boxes], axis=0)
    mean_v4_min = np.mean([b[0] for b in v4_boxes], axis=0)
    mean_v4_max = np.mean([b[1] for b in v4_boxes], axis=0)

    # Count scene points inside each mean bbox
    def count_in_box(pts, bmin, bmax):
        return ((pts >= bmin) & (pts <= bmax)).all(1).sum()

    n_gt = count_in_box(scene_xyz, mean_gt_min, mean_gt_max)
    n_v4 = count_in_box(scene_xyz, mean_v4_min, mean_v4_max)
    print(f'[bbox] COLMAP points inside GT bbox: {n_gt}')
    print(f'[bbox] COLMAP points inside v4 bbox: {n_v4}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Alignment: {args.seq} | GT(green) vs {args.suffix}(red)  '
                 f'[COLMAP pts in bbox: GT={n_gt}  v4={n_v4}]',
                 fontsize=13, fontweight='bold')

    views = [
        ('Top view (X–Z)', 0, 2, 'X', 'Z (depth)'),
        ('Side view (Z–Y)', 2, 1, 'Z (depth)', 'Y'),
    ]

    for ax, (title, xi, yi, xl, yl) in zip(axes, views):
        # Scene point cloud
        sc = scene_col * 0.4 + 0.35   # lighten
        ax.scatter(scene_xyz[:, xi], scene_xyz[:, yi],
                   c=sc, s=1, alpha=0.3, rasterized=True, label='scene')

        # Per-frame bboxes (transparent)
        for bmin, bmax in gt_boxes:
            w = bmax[xi] - bmin[xi]
            h = bmax[yi] - bmin[yi]
            rect = mpatches.FancyBboxPatch(
                (bmin[xi], bmin[yi]), w, h,
                linewidth=0.8, edgecolor='green', facecolor='green', alpha=0.08)
            ax.add_patch(rect)
        for bmin, bmax in v4_boxes:
            w = bmax[xi] - bmin[xi]
            h = bmax[yi] - bmin[yi]
            rect = mpatches.FancyBboxPatch(
                (bmin[xi], bmin[yi]), w, h,
                linewidth=0.8, edgecolor='red', facecolor='red', alpha=0.08)
            ax.add_patch(rect)

        # Mean bboxes (solid outline)
        for (bmin, bmax), col, lbl in [
            (gt_boxes[len(gt_boxes)//2], 'green', 'GT bbox (mid frame)'),
            (v4_boxes[len(v4_boxes)//2], 'red',   f'{args.suffix} bbox (mid frame)'),
        ]:
            w = bmax[xi] - bmin[xi]
            h = bmax[yi] - bmin[yi]
            rect = mpatches.FancyBboxPatch(
                (bmin[xi], bmin[yi]), w, h,
                linewidth=2.0, edgecolor=col, facecolor='none', label=lbl)
            ax.add_patch(rect)

        # Trajectories
        ax.plot(gt_trans[:, xi], gt_trans[:, yi],
                'o-', color='green', ms=4, lw=1.2, label='GT traj', zorder=5)
        ax.plot(v4_trans[:, xi], v4_trans[:, yi],
                's--', color='red', ms=4, lw=1.2, label=f'{args.suffix} traj', zorder=5)

        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n[done] saved → {out_path}')

    transl_err = np.linalg.norm(v4['transl'] - gt['transl'], axis=1)
    sc_mean = float(np.mean(gt['scale'] if gt['scale'].ndim == 1 else gt['scale'][:, 0]))
    print(f'[err]  transl mean={transl_err.mean():.3f}  max={transl_err.max():.3f} COLMAP units')
    print(f'[err]  transl_norm={transl_err.mean()/sc_mean:.3f} body heights')
    print(f'\nLegend:  green=GT  red={args.suffix}  gray=COLMAP scene')


if __name__ == '__main__':
    main()
