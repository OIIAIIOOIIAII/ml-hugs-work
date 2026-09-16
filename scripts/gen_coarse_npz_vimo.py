#!/usr/bin/env python3
"""
Generate smpl_optimized_aligned_scale.npz from VIMO output.

Strategy:
  - VIMO pred_trans: camera-space root translation in meters (temporally smooth)
  - scale: from coarse_align JSON (v3 or v4)
  - Convert cam → world: verts_world = R_w2c.T @ (scale * pred_trans_cam - t_w2c)
  - pose/betas: from GT npz (isolate translation quality, same as Exp A-C)

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/gen_coarse_npz_vimo.py [--seq lab]
  # use v4 scale for all scenes:
  python scripts/gen_coarse_npz_vimo.py --seq lab --v3_dir output/coarse_align_v4 --json_version v4 --out_suffix vimo_v4
"""

import argparse
import json
import os
from glob import glob

import numpy as np


# ─── COLMAP helpers ───────────────────────────────────────────────────────────

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def read_images_txt(path):
    """Return dict: image_name → {R_w2c (3,3), t_w2c (3,)}"""
    cams = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        img_id = int(p[0])
        qvec   = list(map(float, p[1:5]))
        tvec   = list(map(float, p[5:8]))
        name   = p[9]
        R = qvec2rotmat(qvec)
        t = np.array(tvec, dtype=np.float64)
        cams[name] = {'R_w2c': R, 't_w2c': t}
        i += 2  # skip point2D line
    return cams


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',    default='data/neuman/dataset')
    parser.add_argument('--seq',          default='lab')
    parser.add_argument('--vimo_dir',     default='output/vimo_hps')
    parser.add_argument('--v3_dir',       default='output/coarse_align_v3')
    parser.add_argument('--json_version', default='v3', help='v3 or v4, controls which JSON filename to read')
    parser.add_argument('--out_suffix',   default='vimo')
    args = parser.parse_args()

    seq_dir  = os.path.join(args.data_root, args.seq)
    img_txt  = os.path.join(seq_dir, 'sparse', 'images.txt')
    gt_npz   = os.path.join(seq_dir, '4d_humans', 'smpl_optimized_aligned_scale.npz')
    vimo_npz = os.path.join(args.vimo_dir, args.seq, 'vimo_results.npz')
    json_fname = f'coarse_align_{args.json_version}.json'
    align_json = os.path.join(args.v3_dir, args.seq, json_fname)

    out_npz  = os.path.join(seq_dir, '4d_humans',
                            f'smpl_optimized_aligned_scale_{args.out_suffix}.npz')

    # ── load scale from coarse_align JSON ────────────────────────────────────
    with open(align_json) as f:
        align = json.load(f)
    scale = float(align['global_scale'])
    t_world_global = np.array(align.get('global_t_world', [0.0, 0.0, 0.0]), dtype=np.float64)
    method = align.get('scale_method', 'unknown')
    print(f'[{args.json_version}] scale={scale:.4f}  t_world={t_world_global.round(4).tolist()}  method={method}')

    # ── load COLMAP extrinsics ────────────────────────────────────────────────
    colmap_cams = read_images_txt(img_txt)
    # build name→index map (NeuMan images are 00000.png, 00001.png …)
    sorted_names = sorted(colmap_cams.keys())
    name2idx = {n: i for i, n in enumerate(sorted_names)}
    N = len(sorted_names)
    print(f'[colmap] {N} images')

    # ── load VIMO results ─────────────────────────────────────────────────────
    vimo = np.load(vimo_npz, allow_pickle=True)
    pred_trans  = vimo['pred_trans'].squeeze(1)   # (M, 3) cam-space meters
    pred_pose   = vimo['pred_pose']                # (M, 144)
    pred_shape  = vimo['pred_shape']               # (M, 10)
    pred_rotmat = vimo['pred_rotmat']              # (M, 24, 3, 3)
    vimo_frames = vimo['frame'].astype(int)        # (M,) frame indices
    M = len(vimo_frames)
    print(f'[vimo] {M} frames with SMPL output')

    # ── load GT npz for pose/betas/bbox/vertex_colors ────────────────────────
    gt = np.load(gt_npz, allow_pickle=True)
    print(f'[gt] keys: {list(gt.keys())}')

    # ── build per-frame world-space transl ────────────────────────────────────
    # VIMO trans is camera-space in meters, same unit/convention as ROMP.
    # ROMP focal correction NOT needed here: VIMO already uses COLMAP focal.
    out_transl = np.zeros((N, 3), dtype=np.float32)

    for mi, fi in enumerate(vimo_frames):
        name = sorted_names[fi]
        cam  = colmap_cams[name]
        R_w2c = cam['R_w2c']   # (3,3)
        t_w2c = cam['t_w2c']   # (3,)

        trans_cam_m    = pred_trans[mi]                                    # meters
        trans_cam_cu   = scale * trans_cam_m                               # COLMAP units
        trans_world    = R_w2c.T @ (trans_cam_cu - t_w2c) + t_world_global  # world COLMAP units

        out_transl[fi] = trans_world.astype(np.float32)

    # Fill frames without VIMO output by interpolating from GT
    vimo_set = set(vimo_frames.tolist())
    missing = [i for i in range(N) if i not in vimo_set]
    if missing:
        print(f'[warn] {len(missing)} frames missing VIMO output → using GT transl')
        gt_transl = gt['transl']  # (N, 3)
        for i in missing:
            out_transl[i] = gt_transl[i]

    # ── assemble full-sequence npz ─────────────────────────────────────────────
    # Build arrays indexed [0..N-1]
    # Pose: use VIMO rotmat for vimo frames, GT for rest
    gt_rotmat = gt['global_orient'] if 'global_orient' in gt else gt['transl'] * 0
    # Actually use GT pose for all (same as Exp A-C: isolate transl quality)
    # Comment out next block to use VIMO pose instead
    out_data = dict(gt)  # copy all GT fields
    out_data['transl'] = out_transl
    out_data['scale']  = np.full(N, scale, dtype=np.float32)  # (N,) per-frame

    np.savez(out_npz, **{k: v for k, v in out_data.items()})
    print(f'[done] saved → {out_npz}')

    # ── sanity check ──────────────────────────────────────────────────────────
    gt_transl  = gt['transl']
    diff = np.linalg.norm(out_transl[list(vimo_set)] - gt_transl[list(vimo_set)], axis=1)
    print(f'[check] vs GT transl (VIMO frames):  mean={diff.mean():.3f}  max={diff.max():.3f} COLMAP units')

    # Compare std (temporal smoothness)
    vimo_std = out_transl[list(vimo_set)].std(0)
    gt_std   = gt_transl[list(vimo_set)].std(0)
    print(f'[smooth] VIMO transl std: {vimo_std.round(3)}')
    print(f'[smooth]   GT  transl std: {gt_std.round(3)}')


if __name__ == '__main__':
    main()
