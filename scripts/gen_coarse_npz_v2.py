#!/usr/bin/env python3
"""
Generate smpl_optimized_aligned_scale.npz for lab_coarse using v2 coarse alignment.

Strategy:
  - Copy global_orient / body_pose / betas / bbox / vertex_colors from GT NPZ
    (these are well-optimized; only global position needs coarse estimation)
  - Recompute transl per frame:
      trans_romp → ROMP focal correction → scale_v2 * trans_cam → world → + t_world_v2
  - Set uniform scale = global_scale from v2 JSON

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/gen_coarse_npz_v2.py [--seq lab] [--dry_run]
"""

import argparse
import json
import os
from glob import glob

import numpy as np


# ─── geometry helpers ─────────────────────────────────────────────────────────

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def cam_to_world(pts_cam, R_w2c, t_w2c):
    return (R_w2c.T @ (pts_cam - t_w2c).T).T


def read_colmap_cameras(path):
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
                f, cx, cy = params; fx = fy = f
            else:
                raise NotImplementedError(model)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            cams[cam_id] = {'K': K, 'W': W, 'H': H}
    return cams


def read_colmap_images(path):
    frames = {}
    with open(path) as f:
        lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 9:
            i += 1
            continue
        img_name = p[-1]
        cam_id = int(p[-2])
        qvec = list(map(float, p[1:5]))
        tvec = np.array(list(map(float, p[5:8])), dtype=np.float64)
        R_w2c = qvec2rotmat(qvec).astype(np.float64)
        frames[img_name] = {'cam_id': cam_id, 'R_w2c': R_w2c, 't_w2c': tvec}
        i += 2  # skip the points2D line
    return frames


def load_romp_trans(npz_path, K):
    """Load ROMP trans with focal correction."""
    data = np.load(npz_path, allow_pickle=True)
    r = data['results'].item()
    trans = np.array(r['trans'], dtype=np.float64)
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    # ROMP assumes focal = cx; correct for actual focal
    trans[0] = trans[0] * (cx / fx)
    trans[1] = trans[1] * (cy / fy)
    return trans


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='lab')
    ap.add_argument('--data_root', default='data/neuman/dataset')
    ap.add_argument('--v2_json', default=None,
                    help='Path to coarse_align JSON (default: output/coarse_align_v2/<seq>/coarse_align_v2.json)')
    ap.add_argument('--out_seq', default=None,
                    help='Output sequence name (default: <seq>_coarse)')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print results without writing NPZ')
    args = ap.parse_args()

    seq_dir = os.path.join(args.data_root, args.seq)
    out_seq = args.out_seq or f'{args.seq}_coarse'
    coarse_dir = os.path.join(args.data_root, out_seq)
    sparse_dir = os.path.join(seq_dir, 'sparse')
    smpl_pred_dir = os.path.join(seq_dir, 'smpl_pred')
    gt_npz_path = os.path.join(seq_dir, '4d_humans', 'smpl_optimized_aligned_scale.npz')
    out_npz_path = os.path.join(coarse_dir, '4d_humans', 'smpl_optimized_aligned_scale.npz')

    v2_json = args.v2_json or os.path.join(
        'output', 'coarse_align_v2', args.seq, 'coarse_align_v2.json')

    # Load v2 result
    print(f'[1/4] Loading v2 coarse align result: {v2_json}')
    with open(v2_json) as f:
        v2 = json.load(f)
    global_scale = float(v2['global_scale'])
    global_t_world = np.array(v2['global_t_world'], dtype=np.float64)
    print(f'  global_scale = {global_scale:.4f}')
    print(f'  global_t_world = {global_t_world.tolist()}')

    # Load COLMAP
    print(f'\n[2/4] Loading COLMAP from {sparse_dir}')
    cams = read_colmap_cameras(os.path.join(sparse_dir, 'cameras.txt'))
    colmap_frames = read_colmap_images(os.path.join(sparse_dir, 'images.txt'))
    print(f'  {len(colmap_frames)} frames, {len(cams)} cameras')

    # Load GT NPZ (for poses / betas / bbox / vertex_colors)
    print(f'\n[3/4] Loading GT NPZ: {gt_npz_path}')
    gt = np.load(gt_npz_path)
    n_frames = gt['transl'].shape[0]
    print(f'  n_frames = {n_frames}')
    print(f'  GT scale (mean): {gt["scale"].mean():.4f}')

    # Compute per-frame transl using v2 coarse alignment
    print(f'\n[4/4] Computing per-frame transl ...')
    smpl_files = sorted(glob(os.path.join(smpl_pred_dir, '*.npz')))
    # Map frame_name → smpl_file
    frame2smpl = {}
    for sf in smpl_files:
        base = os.path.basename(sf)
        frame_name = base.replace('_png.npz', '.png')
        frame2smpl[frame_name] = sf

    all_transl = np.zeros((n_frames, 3), dtype=np.float32)
    success = 0

    for idx in range(n_frames):
        frame_name = f'{idx:05d}.png'

        if frame_name not in colmap_frames:
            # fallback: nearest frame transl (shouldn't happen for NeuMan)
            print(f'  [warn] frame {frame_name} not in COLMAP, using GT transl')
            all_transl[idx] = gt['transl'][idx]
            continue

        cf = colmap_frames[frame_name]
        K = cams[cf['cam_id']]['K']
        R_w2c = cf['R_w2c']
        t_w2c = cf['t_w2c']

        if frame_name not in frame2smpl:
            print(f'  [warn] frame {frame_name} no ROMP pred, using GT transl')
            all_transl[idx] = gt['transl'][idx]
            continue

        trans_cam = load_romp_trans(frame2smpl[frame_name], K)  # (3,) meters
        trans_cam_colmap = global_scale * trans_cam              # (3,) COLMAP units (cam)
        trans_world = cam_to_world(
            trans_cam_colmap[None, :], R_w2c, t_w2c)[0] + global_t_world
        all_transl[idx] = trans_world.astype(np.float32)
        success += 1

    print(f'  Computed transl for {success}/{n_frames} frames')
    print(f'  transl mean: {all_transl.mean(0).tolist()}')
    print(f'  GT transl mean: {gt["transl"].mean(0).tolist()}')

    # Compare with GT for per-frame error
    err_per_frame = np.linalg.norm(all_transl - gt['transl'], axis=1)
    print(f'\n[Transl error vs GT]')
    print(f'  mean: {err_per_frame.mean():.3f}  median: {np.median(err_per_frame):.3f}')
    print(f'  min: {err_per_frame.min():.3f}  max: {err_per_frame.max():.3f}')

    if args.dry_run:
        print('\n[dry_run] Not writing NPZ.')
        return

    # Write NPZ
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez(
        out_npz_path,
        global_orient=gt['global_orient'],
        body_pose=gt['body_pose'],
        betas=gt['betas'],
        transl=all_transl,
        scale=np.full(n_frames, global_scale, dtype=np.float32),
        bbox=gt['bbox'],
        vertex_colors=gt['vertex_colors'],
    )
    print(f'\nSaved: {out_npz_path}')
    print(f'  scale (uniform): {global_scale:.4f}')
    print(f'  transl sample (first 2 frames):\n{all_transl[:2]}')


if __name__ == '__main__':
    main()
