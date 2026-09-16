#!/usr/bin/env python3
"""
深度监督(depth_w) on/off 渲染效果对比图（答辩用）。

对照实验（均无 AnchorAttention，唯一变量 depth_w）：
  无 depth = exp0_hugs_original_15000_{scene}
  有 depth = hugs_depth_sup_12000_{scene}   (depth_w=0.05)

val 图 full_final_NNN.png：左半=GT，右半=渲染，单幅宽 = W//2。
输出：每行一帧，3 列 GT | 无depth | 有depth。
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = 'output/human_scene/neuman'


def val_dir(scene, key):
    pat = (f'{ROOT}/{scene}/hugs_trimlp/exp0_hugs_original_15000*/*/val' if key == 'off'
           else f'{ROOT}/{scene}/hugs_trimlp/hugs_depth_sup_12000*/*/val')
    ds = glob.glob(pat)
    return ds[0] if ds else None


def half(img_path, side):
    im = np.asarray(Image.open(img_path))
    w = im.shape[1] // 2
    return im[:, :w] if side == 'left' else im[:, w:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='citron')
    ap.add_argument('--frames', type=int, nargs='+', default=[0, 2, 4])
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    off = val_dir(args.scene, 'off'); on = val_dir(args.scene, 'on')
    assert off and on, f'val dir missing off={off} on={on}'
    out = args.out or f'paper_figures/depth_ablation_{args.scene}.png'

    n = len(args.frames)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.2 * n))
    if n == 1:
        axes = axes[None, :]
    cols = ['GT', 'w/o depth loss', 'w/ depth loss (depth_w=0.05)']

    for r, fi in enumerate(args.frames):
        foff = f'{off}/full_final_{fi:03d}.png'
        fon = f'{on}/full_final_{fi:03d}.png'
        imgs = [half(foff, 'left'), half(foff, 'right'), half(fon, 'right')]
        for c in range(3):
            axes[r, c].imshow(imgs[c]); axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=14, fontweight='bold')
        axes[r, 0].set_ylabel(f'frame {fi}', fontsize=11)

    fig.suptitle(f'Depth Supervision Ablation — {args.scene}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
