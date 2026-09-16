#!/usr/bin/env python3
"""
Run VIMO (from TRAM) on NeuMan data — bypasses DROID-SLAM.

NeuMan already has COLMAP cameras (better than DROID-SLAM), so we:
  1. Extract per-frame human bounding boxes from NeuMan segmentation masks
  2. Feed COLMAP intrinsics + image paths directly to VIMO
  3. Save temporally-consistent SMPL estimates to output dir

NeuMan segmentation convention: black(0)=human, white(255)=background.

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  conda activate hugs
  python scripts/run_vimo_neuman.py --seq lab
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch

warnings.filterwarnings('ignore')

TRAM_DIR = os.path.join(os.path.dirname(__file__), '..', 'JOSH', 'third_party', 'tram')
sys.path.insert(0, TRAM_DIR)

from lib.models import get_hmr_vimo  # noqa: E402

VIMO_CKPT = os.path.join(TRAM_DIR, '..', '..', 'data', 'checkpoints', 'vimo_checkpoint.pth.tar')


# ─── COLMAP helpers ───────────────────────────────────────────────────────────

def read_cameras_txt(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            cam_id, model = int(p[0]), p[1]
            W, H = int(p[2]), int(p[3])
            params = list(map(float, p[4:]))
            if model == 'PINHOLE':
                fx, fy, cx, cy = params
            elif model == 'SIMPLE_PINHOLE':
                fx = fy = params[0]; cx, cy = params[1], params[2]
            else:
                raise ValueError(f'Unsupported camera model: {model}')
            cams[cam_id] = dict(focal=fx, cx=cx, cy=cy, W=W, H=H)
    return cams


# ─── bbox from segmentation mask ─────────────────────────────────────────────

def mask_to_bbox(seg_path):
    """NeuMan: black(0)=human. Returns [x1,y1,x2,y2] or None."""
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        return None
    human = seg < 128
    ys, xs = np.where(human)
    if len(xs) < 100:
        return None
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/neuman/dataset')
    parser.add_argument('--seq', default='lab')
    parser.add_argument('--out_dir', default='output/vimo_hps')
    args = parser.parse_args()

    seq_dir  = os.path.join(args.data_root, args.seq)
    img_dir  = os.path.join(seq_dir, 'images')
    seg_dir  = os.path.join(seq_dir, 'segmentations')
    cam_path = os.path.join(seq_dir, 'sparse', 'cameras.txt')
    out_dir  = os.path.join(args.out_dir, args.seq)
    os.makedirs(out_dir, exist_ok=True)

    # ── camera intrinsics ─────────────────────────────────────────────────────
    cams = read_cameras_txt(cam_path)
    cam  = list(cams.values())[0]
    img_focal  = float(cam['focal'])
    img_center = np.array([cam['cx'], cam['cy']], dtype=np.float32)
    print(f'[camera] focal={img_focal:.2f}  center=({img_center[0]:.1f}, {img_center[1]:.1f})')

    # ── collect frames ────────────────────────────────────────────────────────
    from glob import glob
    imgfiles = sorted(glob(os.path.join(img_dir, '*.png')))
    N = len(imgfiles)
    print(f'[frames] {N} images found')

    # ── extract bboxes ────────────────────────────────────────────────────────
    stems  = [os.path.splitext(os.path.basename(f))[0] for f in imgfiles]
    bboxes = []
    valid  = []
    for stem in stems:
        seg_path = os.path.join(seg_dir, stem + '.png')
        bb = mask_to_bbox(seg_path)
        if bb is not None:
            bboxes.append(bb)
            valid.append(True)
        else:
            bboxes.append(np.array([0, 0, 1, 1], dtype=np.float32))
            valid.append(False)
    bboxes = np.stack(bboxes)
    valid  = np.array(valid)
    print(f'[bbox] {valid.sum()}/{N} frames have valid human bbox')

    # ── load VIMO ─────────────────────────────────────────────────────────────
    ckpt = os.path.abspath(VIMO_CKPT)
    print(f'[vimo] loading checkpoint: {ckpt}')
    model = get_hmr_vimo(checkpoint=ckpt, device='cuda')
    print(f'[vimo] model ready ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)')

    # ── inference ─────────────────────────────────────────────────────────────
    frame_idx = np.arange(N)
    results = model.inference(
        imgfiles=imgfiles,
        boxes=bboxes,
        img_focal=img_focal,
        img_center=img_center,
        valid=valid,
        frame=frame_idx,
    )

    if results is None:
        print('[ERROR] VIMO returned None — not enough valid frames (need ≥16)')
        return

    # pred_trans: (M, 1, 3) camera-space root translation in meters
    # pred_rotmat: (M, 24, 3, 3)  pred_pose: (M, 144)  pred_shape: (M, 10)
    out = {k: v.cpu().numpy() for k, v in results.items()}
    out['img_focal']  = img_focal
    out['img_center'] = img_center
    out['imgfiles']   = np.array(imgfiles)
    out['valid_mask'] = valid

    save_path = os.path.join(out_dir, 'vimo_results.npz')
    np.savez(save_path, **out)
    print(f'[done] saved → {save_path}')

    M     = out['pred_trans'].shape[0]
    trans = out['pred_trans'].squeeze(1)
    print(f'  frames processed : {M}')
    print(f'  pred_trans mean  : {trans.mean(0).round(3)}')
    print(f'  pred_trans std   : {trans.std(0).round(3)}  (lower = smoother)')


if __name__ == '__main__':
    main()
