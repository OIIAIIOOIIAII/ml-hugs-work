#!/usr/bin/env python3
"""
给已有的 debug_ply 快照追加 SMPL 人体顶点（红色），并把 scene GS
按距人体中心的距离重新上色（橙色=近/侵入区，灰色=远）。

Usage:
  python scripts/patch_debug_ply.py \
    --ply_dir output/human_scene/neuman/bike_vimo_v4/hugs_trimlp/full_noamass/2026-06-13_19-43-07/debug_ply \
    --seq bike_vimo_v4
"""
import argparse, os, sys, struct
import numpy as np
import torch

sys.path.insert(0, '/workspace/nas_auto_backup/yuzilang/ml-hugs-work')
from hugs.models.modules.smpl_layer import SMPL

SMPL_MODEL = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/smpl/SMPL_NEUTRAL.pkl'
DATA_ROOT   = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/neuman/dataset'


def read_binary_ply(path):
    """Read binary PLY → (N,3) float32 xyz, (N,3) uint8 rgb."""
    with open(path, 'rb') as f:
        # parse header
        n_verts = 0
        props = []
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                props.append((parts[1], parts[2]))  # (type, name)
            elif line == 'end_header':
                break
        dt = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ])
        data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
    rgb = np.stack([data['red'], data['green'], data['blue']], axis=1)
    return xyz, rgb


def write_binary_ply(path, xyz, rgb):
    n = len(xyz)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    ).encode('ascii')
    dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                   ('red','u1'),('green','u1'),('blue','u1')])
    rec = np.empty(n, dtype=dt)
    rec['x']=xyz[:,0]; rec['y']=xyz[:,1]; rec['z']=xyz[:,2]
    rec['red']=rgb[:,0]; rec['green']=rgb[:,1]; rec['blue']=rgb[:,2]
    with open(path, 'wb') as f:
        f.write(header)
        rec.tofile(f)


def get_smpl_verts(smpl, global_orient, body_pose, betas, transl, scale):
    go = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
    bp = body_pose[:69] if len(body_pose) == 72 else body_pose
    bp = torch.tensor(bp, dtype=torch.float32).unsqueeze(0)
    b  = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = smpl(global_orient=go, body_pose=bp, betas=b, transl=torch.zeros(1,3))
    return (out.vertices[0].numpy() * scale + transl).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_dir', required=True)
    parser.add_argument('--seq',     default='bike_vimo_v4')
    parser.add_argument('--frame',   type=int, default=-1,
                        help='frame index in NPZ; -1 = mid-sequence')
    parser.add_argument('--radius',  type=float, default=1.5,
                        help='COLMAP units radius for orange near-human zone')
    args = parser.parse_args()

    # Load SMPL
    print('[smpl] loading...')
    smpl = SMPL(model_path=SMPL_MODEL).eval()

    # Load aligned NPZ
    npz_path = f'{DATA_ROOT}/{args.seq}/4d_humans/smpl_optimized_aligned_scale.npz'
    print(f'[npz] loading {npz_path}')
    npz = np.load(npz_path, allow_pickle=True)
    N = len(npz['transl'])
    fi = N // 2 if args.frame < 0 else args.frame
    print(f'[npz] using frame {fi}/{N}')

    sc = float(npz['scale'][fi] if npz['scale'].ndim == 1 else npz['scale'][fi, 0])
    human_center = npz['transl'][fi]
    go  = npz['global_orient'][fi]
    bp  = npz['body_pose'][fi]
    bet = npz['betas'][0] if npz['betas'].ndim == 2 else npz['betas']

    print(f'[human] center z={human_center[2]:.3f}  scale={sc:.3f}')
    smpl_verts = get_smpl_verts(smpl, go, bp, bet, human_center, sc)
    h_rgb = np.full((len(smpl_verts), 3), [220, 30, 30], dtype=np.uint8)
    print(f'[human] {len(smpl_verts)} SMPL vertices (red)')

    # Process each PLY in dir
    plys = sorted([f for f in os.listdir(args.ply_dir) if f.endswith('.ply')])
    print(f'[PLY] found {len(plys)} files in {args.ply_dir}')

    for fname in plys:
        fpath = os.path.join(args.ply_dir, fname)
        out_path = fpath  # overwrite in-place

        xyz, rgb = read_binary_ply(fpath)

        # Re-color scene GS based on distance to human center
        # Identify which points are scene GS: those that are NOT white (COLMAP)
        # COLMAP points have roughly equal R≈G≈B; let's identify by original gray color
        # Actually original gray = [140-150, 140-150, 140-150], COLMAP = original scene RGB
        # Safest: re-color ALL non-COLMAP points based on distance
        # We don't know which are which → recolor all by distance from human center
        dist = np.linalg.norm(xyz - human_center, axis=1)
        near_mask = dist < args.radius

        # Orange for near-human scene pts, keep others as-is
        new_rgb = rgb.copy()
        new_rgb[near_mask] = [255, 140, 0]   # orange = invasion zone

        # Append SMPL human verts
        all_xyz = np.concatenate([xyz, smpl_verts], axis=0)
        all_rgb = np.concatenate([new_rgb, h_rgb], axis=0)

        write_binary_ply(out_path, all_xyz, all_rgb)
        n_near = near_mask.sum()
        print(f'  {fname}: near={n_near:,} orange  +{len(smpl_verts)} red human verts')

    print('\n[done] all PLY files patched.')
    print(f'Colors: orange=scene GS within {args.radius} COLMAP units of human  gray/other=far scene  red=SMPL body  white=COLMAP')


if __name__ == '__main__':
    main()
