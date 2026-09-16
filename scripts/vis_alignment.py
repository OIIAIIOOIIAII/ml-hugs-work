#!/usr/bin/env python3
"""
可视化人体对齐结果 vs 场景点云。
输出一个 colored PLY 文件，可在 MeshLab / CloudCompare 中打开：
  - 灰色：COLMAP 场景点云
  - 绿色：GT 对齐的人体 mesh（半透明轮廓）
  - 红色：v4 (VIMO) 对齐的人体 mesh
  - 蓝色：所有帧的人体中心轨迹（transl）

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/vis_alignment.py --seq seattle [--step 4] [--out /tmp/vis.ply]
  python scripts/vis_alignment.py --seq bike [--step 4]
"""
import argparse
import sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/nas_auto_backup/yuzilang/ml-hugs-work')
from hugs.models.modules.smpl_layer import SMPL


SMPL_MODEL = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/smpl/SMPL_NEUTRAL.pkl'
DATA_ROOT   = '/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/neuman/dataset'


# ─── PLY helpers ──────────────────────────────────────────────────────────────

def write_ply(path, verts, colors):
    """verts: (N,3) float, colors: (N,3) uint8"""
    n = len(verts)
    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(verts.tolist(), colors.tolist()):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")
    print(f"[PLY] saved {n:,} points → {path}")


def write_ply_with_faces(path, verts, colors, faces=None, extra_verts=None, extra_colors=None):
    """Write PLY with optional triangle faces for SMPL meshes."""
    all_v = [verts]
    all_c = [colors]
    if extra_verts is not None:
        all_v.append(extra_verts)
        all_c.append(extra_colors)
    V = np.concatenate(all_v, axis=0)
    C = np.concatenate(all_c, axis=0)

    nv = len(V)
    nf = len(faces) if faces is not None else 0

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if nf > 0:
            f.write(f"element face {nf}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(V.tolist(), C.tolist()):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")
        if faces is not None:
            for tri in faces.tolist():
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    print(f"[PLY] {nv:,} verts, {nf:,} faces → {path}")


# ─── COLMAP points3D.txt parser ───────────────────────────────────────────────

def load_colmap_points_txt(txt_path):
    """Return (N,3) xyz and (N,3) rgb arrays from points3D.txt."""
    xyz_list, rgb_list = [], []
    with open(txt_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            # POINT3D_ID X Y Z R G B ERROR TRACK[]
            xyz_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb_list.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.array(xyz_list, dtype=np.float32), np.array(rgb_list, dtype=np.uint8)


def load_colmap_points_ply(ply_path):
    """Simple ASCII PLY reader for points3D.ply."""
    header_done = False
    n_verts = 0
    has_rgb = False
    props = []
    xyz_list, rgb_list = [], []
    with open(ply_path) as f:
        for line in f:
            line = line.strip()
            if not header_done:
                if line.startswith('element vertex'):
                    n_verts = int(line.split()[-1])
                elif line.startswith('property'):
                    props.append(line.split()[-1])
                elif line == 'end_header':
                    header_done = True
                    has_rgb = 'red' in props
                continue
            parts = line.split()
            xi = props.index('x')
            yi = props.index('y')
            zi = props.index('z')
            xyz_list.append([float(parts[xi]), float(parts[yi]), float(parts[zi])])
            if has_rgb:
                ri = props.index('red')
                gi = props.index('green')
                bi = props.index('blue')
                rgb_list.append([int(parts[ri]), int(parts[gi]), int(parts[bi])])
            else:
                rgb_list.append([180, 180, 180])
            if len(xyz_list) >= n_verts:
                break
    return np.array(xyz_list, dtype=np.float32), np.array(rgb_list, dtype=np.uint8)


# ─── SMPL forward pass (numpy, no grad) ───────────────────────────────────────

def get_smpl_verts(smpl_model, global_orient, body_pose, betas, transl, scale):
    """
    global_orient: (3,) axis-angle
    body_pose:     (69,) axis-angle  (23 joints × 3)
    betas:         (10,)
    transl:        (3,) world-space translation (COLMAP units)
    scale:         scalar, COLMAP units / meter
    Returns (6890, 3) world-space vertex positions.
    """
    go  = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)   # (1,3)
    bp  = torch.tensor(body_pose,     dtype=torch.float32).unsqueeze(0)   # (1,69)
    b   = torch.tensor(betas[:10],    dtype=torch.float32).unsqueeze(0)   # (1,10)
    t   = torch.zeros(1, 3, dtype=torch.float32)  # zero transl for now

    with torch.no_grad():
        out = smpl_model(global_orient=go, body_pose=bp, betas=b, transl=t)
    verts = out.vertices[0].numpy()          # (6890, 3) metric, zero-transl

    # scale from meters to COLMAP units, then add world transl
    verts_world = verts * scale + transl
    return verts_world.astype(np.float32)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq',    default='seattle')
    parser.add_argument('--suffix', default='vimo_v4', help='npz suffix for "other" alignment (e.g. vimo_v4)')
    parser.add_argument('--step',   type=int, default=4, help='sample every N frames')
    parser.add_argument('--out',    default='', help='output PLY path (default: /tmp/vis_<seq>.ply)')
    args = parser.parse_args()

    out_path = args.out or f'/tmp/vis_{args.seq}_{args.suffix}.ply'

    gt_seq_dir = f'{DATA_ROOT}/{args.seq}'
    v4_seq_dir = f'{DATA_ROOT}/{args.seq}_{args.suffix}'

    # ── Load SMPL ─────────────────────────────────────────────────────────────
    print('[smpl] loading model...')
    smpl = SMPL(model_path=SMPL_MODEL)
    smpl.eval()
    faces = smpl.faces.numpy() if hasattr(smpl.faces, 'numpy') else np.array(smpl.faces)

    # ── Load scene point cloud ────────────────────────────────────────────────
    txt_path = f'{gt_seq_dir}/sparse/points3D.txt'
    print('[scene] loading COLMAP point cloud...')
    scene_xyz, scene_rgb = load_colmap_points_txt(txt_path)
    print(f'[scene] {len(scene_xyz):,} points, xyz range:')
    print(f'        x=[{scene_xyz[:,0].min():.2f}, {scene_xyz[:,0].max():.2f}]')
    print(f'        y=[{scene_xyz[:,1].min():.2f}, {scene_xyz[:,1].max():.2f}]')
    print(f'        z=[{scene_xyz[:,2].min():.2f}, {scene_xyz[:,2].max():.2f}]')
    # Darken scene to gray so human stands out
    scene_gray = (scene_rgb.astype(float) * 0.5 + 80).clip(0, 255).astype(np.uint8)

    # ── Load GT npz ───────────────────────────────────────────────────────────
    gt_npz_path = f'{gt_seq_dir}/4d_humans/smpl_optimized_aligned_scale.npz'
    print('[gt] loading', gt_npz_path)
    gt = np.load(gt_npz_path, allow_pickle=True)
    gt_transl  = gt['transl']   # (N, 3)
    gt_scale   = gt['scale']    # (N,) or (N,3)
    gt_go      = gt['global_orient']  # (N, 3)
    gt_bp      = gt['body_pose']      # (N, 69) or (N, 72)
    gt_betas   = gt['betas'][0] if gt['betas'].ndim == 2 else gt['betas']

    # ── Load v4 npz ───────────────────────────────────────────────────────────
    # Try both naming conventions
    import os
    for cand in [
        f'{v4_seq_dir}/4d_humans/smpl_optimized_aligned_scale.npz',
        f'{gt_seq_dir}/4d_humans/smpl_optimized_aligned_scale_{args.suffix}.npz',
    ]:
        if os.path.exists(cand):
            v4_npz_path = cand
            break
    else:
        raise FileNotFoundError(f'Cannot find v4 npz for seq={args.seq} suffix={args.suffix}')
    print('[v4] loading', v4_npz_path)
    v4 = np.load(v4_npz_path, allow_pickle=True)
    v4_transl  = v4['transl']
    v4_scale   = v4['scale']
    v4_go      = v4['global_orient']
    v4_bp      = v4['body_pose']
    v4_betas   = v4['betas'][0] if v4['betas'].ndim == 2 else v4['betas']

    N = len(gt_transl)
    frames = list(range(0, N, args.step))
    print(f'[frames] total={N}, sampling every {args.step} → {len(frames)} frames')

    # ── Compute SMPL vertices for sampled frames ───────────────────────────────
    gt_verts_all, v4_verts_all = [], []
    traj_gt_pts, traj_v4_pts = [], []

    for fi in frames:
        sc_gt = float(gt_scale[fi]) if gt_scale.ndim == 1 else float(gt_scale[fi, 0])
        sc_v4 = float(v4_scale[fi]) if v4_scale.ndim == 1 else float(v4_scale[fi, 0])

        bp_gt = gt_bp[fi]
        if bp_gt.shape[0] == 72:   # includes global_orient
            bp_gt = bp_gt[3:]
        bp_v4 = v4_bp[fi]
        if bp_v4.shape[0] == 72:
            bp_v4 = bp_v4[3:]

        v_gt = get_smpl_verts(smpl, gt_go[fi], bp_gt, gt_betas, gt_transl[fi], sc_gt)
        v_v4 = get_smpl_verts(smpl, v4_go[fi], bp_v4, v4_betas, v4_transl[fi], sc_v4)

        gt_verts_all.append(v_gt)
        v4_verts_all.append(v_v4)
        traj_gt_pts.append(gt_transl[fi])
        traj_v4_pts.append(v4_transl[fi])

    # ── Build colored point cloud arrays ─────────────────────────────────────
    all_xyz, all_rgb = [scene_xyz], [scene_gray]

    # GT human: green vertices
    for v in gt_verts_all:
        all_xyz.append(v)
        all_rgb.append(np.tile([30, 220, 30], (len(v), 1)).astype(np.uint8))

    # v4 human: red vertices
    for v in v4_verts_all:
        all_xyz.append(v)
        all_rgb.append(np.tile([220, 30, 30], (len(v), 1)).astype(np.uint8))

    # Trajectory dots: blue (GT) and orange (v4)
    traj_gt  = np.array(traj_gt_pts, dtype=np.float32)
    traj_v4  = np.array(traj_v4_pts, dtype=np.float32)
    all_xyz.append(traj_gt)
    all_rgb.append(np.tile([0, 100, 255], (len(traj_gt), 1)).astype(np.uint8))
    all_xyz.append(traj_v4)
    all_rgb.append(np.tile([255, 140, 0], (len(traj_v4), 1)).astype(np.uint8))

    combined_xyz = np.concatenate(all_xyz, axis=0)
    combined_rgb = np.concatenate(all_rgb, axis=0)

    write_ply(out_path, combined_xyz, combined_rgb)

    # ── Summary ───────────────────────────────────────────────────────────────
    transl_err = np.linalg.norm(v4_transl - gt_transl, axis=1)
    sc_mean = float(np.mean(gt_scale[:, 0] if gt_scale.ndim == 2 else gt_scale))
    print(f'\n[summary] GT mean_z={gt_transl[:,2].mean():.3f}  v4 mean_z={v4_transl[:,2].mean():.3f}')
    print(f'[summary] transl err: mean={transl_err.mean():.3f}  max={transl_err.max():.3f} COLMAP units')
    print(f'[summary] transl_norm (÷scale): mean={transl_err.mean()/sc_mean:.3f}')
    print(f'\nColors:  gray=scene  green=GT-human  red=v4-human  blue=GT-traj  orange=v4-traj')
    print(f'Open in: meshlab {out_path}')
    print(f'      or: cloudcompare {out_path}')


if __name__ == '__main__':
    main()
