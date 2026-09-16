#!/usr/bin/env python3
"""
Coarse SMPL-to-scene alignment for NeuMan sequences.

Given ROMP per-frame SMPL predictions (camera space) and COLMAP reconstruction,
optimizes a single global (scale, t_world) so that:
    scale * SMPL_verts_world + t_world  ≈  scene / human mask

Losses:
  1. Projection IoU  – SMPL silhouette vs human segmentation mask
  2. Scene Chamfer   – one-sided: SMPL verts → nearest COLMAP 3D point
                       pulls mesh into the scene bounding region

Output:
  <out_dir>/coarse_align.json  – {global_scale, global_t_world}
  <out_dir>/vis/               – camera-view overlays + 3D scatter plots

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/coarse_align_smpl_neuman.py \
      --data_root data/neuman/dataset --seq lab \
      --out_dir output/coarse_align
"""

import argparse
import json
import os
import struct
import warnings
from glob import glob

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP I/O
# ─────────────────────────────────────────────────────────────────────────────

def qvec2rotmat(q):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix (NumPy)."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)


def read_colmap_cameras(path):
    """Return dict: cam_id → {K:(3,3), W:int, H:int}."""
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
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            cams[cam_id] = {'K': K, 'W': W, 'H': H}
    return cams


def read_colmap_images(path):
    """Return dict: img_name → {R_w2c:(3,3), t_w2c:(3,), cam_id:int}."""
    frames = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 10:
            i += 1
            continue
        q = list(map(float, p[1:5]))
        t = np.array(list(map(float, p[5:8])), dtype=np.float32)
        cam_id = int(p[8])
        name = p[9]
        frames[name] = {'R_w2c': qvec2rotmat(q), 't_w2c': t, 'cam_id': cam_id}
        i += 2
    return frames


def read_colmap_points3d(path, max_pts=100_000):
    """Return (N,3) world-space float32 array."""
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
    pts = np.array(pts, dtype=np.float32)
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# ROMP predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_romp(npz_path, K=None):
    """
    Return dict with verts(6890,3), trans(3,), poses(72,), betas(10,).

    ROMP assumes focal = image_W/2 (= cx) when computing trans X/Y.
    If actual focal (K[0,0]) differs, we rescale trans_x/y so that
    the projected verts match the actual camera.
    """
    data = np.load(npz_path, allow_pickle=True)
    r = data['results'].item()
    trans = np.array(r['trans'], dtype=np.float32)
    if K is not None:
        # ROMP focal assumption: cx = K[0,2]
        focal_romp = K[0, 2]   # e.g. 638
        focal_actual_x = K[0, 0]  # e.g. 1099.98
        focal_actual_y = K[1, 1]
        trans[0] = trans[0] * (focal_romp / focal_actual_x)
        trans[1] = trans[1] * (focal_romp / focal_actual_y)
    return {
        'verts': np.array(r['verts'], dtype=np.float32),
        'trans': trans,
        'poses': np.array(r['poses'], dtype=np.float32),
        'betas': np.array(r['betas'], dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cam_to_world(pts_cam, R_w2c, t_w2c):
    """(N,3) cam → world: p_world = R.T @ (p_cam - t)."""
    return (R_w2c.T @ (pts_cam - t_w2c).T).T


def world_to_cam(pts_world, R_w2c, t_w2c):
    return (R_w2c @ pts_world.T).T + t_w2c


def project(pts_world, R_w2c, t_w2c, K):
    """Project (N,3) world pts → (N,2) pixel coords + valid mask."""
    pc = world_to_cam(pts_world, R_w2c, t_w2c)
    valid = pc[:, 2] > 0.01
    uv = np.zeros((len(pc), 2), dtype=np.float32)
    z = pc[valid, 2]
    uv[valid, 0] = pc[valid, 0] / z * K[0, 0] + K[0, 2]
    uv[valid, 1] = pc[valid, 1] / z * K[1, 1] + K[1, 2]
    return uv, valid


# ─────────────────────────────────────────────────────────────────────────────
# Depth-map calibration (mono_depth uint16 → COLMAP metric depth)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_mono_depth(mono_uint16, colmap_pts, R_w2c, t_w2c, K, W, H, mask_bg):
    """
    Calibrate mono depth map against COLMAP metric depths.

    NeuMan mono_depth convention: uint16, higher value = farther (direct depth).
    Returns (a, b) such that: colmap_metric_z ≈ a * mono_uint16 + b

    Uses background pixels only (mask_bg = True for background).
    """
    # Project COLMAP points to get metric z in camera space
    pc = world_to_cam(colmap_pts, R_w2c, t_w2c)
    valid_z = pc[:, 2] > 0.01
    u = (pc[valid_z, 0] / pc[valid_z, 2] * K[0, 0] + K[0, 2]).astype(np.int32)
    v = (pc[valid_z, 1] / pc[valid_z, 2] * K[1, 1] + K[1, 2]).astype(np.int32)
    z_colmap = pc[valid_z, 2]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z_colmap = u[in_img], v[in_img], z_colmap[in_img]

    # Filter to background pixels
    in_bg = mask_bg[v, u]
    u, v, z_colmap = u[in_bg], v[in_bg], z_colmap[in_bg]

    if len(u) < 10:
        return None

    # mono depth value (uint16, 0 = invalid, higher = farther)
    mono_vals = mono_uint16[v, u].astype(np.float32)
    valid_mono = mono_vals > 0
    if valid_mono.sum() < 10:
        return None
    mono_vals = mono_vals[valid_mono]
    z_colmap = z_colmap[valid_mono]

    # Linear fit: z_colmap = a * mono_uint16 + b
    A = np.stack([mono_vals, np.ones_like(mono_vals)], axis=1)
    result = np.linalg.lstsq(A, z_colmap, rcond=None)
    a, b = result[0]
    return float(a), float(b)


def get_human_pts_from_depth(mono_uint16, calib, R_w2c, t_w2c, K, mask_human,
                              max_pts=4000, seed=0):
    """
    Backproject calibrated human-mask depth pixels to world space.
    calib = (a, b) from calibrate_mono_depth.
    Returns (M, 3) world-space points.
    """
    H, W = mono_uint16.shape
    ys, xs = np.where(mask_human)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # subsample
    rng = np.random.default_rng(seed)
    if len(ys) > max_pts:
        idx = rng.choice(len(ys), max_pts, replace=False)
        ys, xs = ys[idx], xs[idx]

    mono_vals = mono_uint16[ys, xs].astype(np.float32)
    a, b = calib
    z = a * mono_vals + b  # linear: colmap_z = a * mono_uint16 + b

    valid = (mono_vals > 0) & (z > 0.01)
    xs, ys, z = xs[valid], ys[valid], z[valid]

    # Backproject to camera space
    x_cam = (xs - K[0, 2]) / K[0, 0] * z
    y_cam = (ys - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)

    # Camera → world
    pts_world = cam_to_world(pts_cam, R_w2c, t_w2c)
    return pts_world


# ─────────────────────────────────────────────────────────────────────────────
# Losses (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_one_sided(src, tgt, chunk=1024):
    """min distance from each src point to tgt. Returns mean min-dist²."""
    dists = []
    for i in range(0, len(src), chunk):
        d = (src[i:i+chunk, None, :] - tgt[None, :, :]).pow(2).sum(-1)
        dists.append(d.min(-1).values)
    return torch.cat(dists).mean()


def soft_silhouette_loss(verts_cam_t, K, H, W, mask_t, sigma=3.0, n_sample=800, device='cuda'):
    """
    Soft projection IoU between SMPL silhouette and human mask.

    verts_cam_t: (N,3) camera-space vertices (torch)
    mask_t: (1,1,H,W) float tensor
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    z = verts_cam_t[:, 2].clamp(min=0.1)
    u = verts_cam_t[:, 0] / z * fx + cx  # (N,)
    v = verts_cam_t[:, 1] / z * fy + cy

    # subsample for efficiency
    perm = torch.randperm(len(u), device=device)[:n_sample]
    u_s = u[perm]
    v_s = v[perm]

    # Build soft silhouette on a downsampled grid
    scale = 4
    h2, w2 = H // scale, W // scale
    yy = torch.arange(h2, device=device, dtype=torch.float32).view(h2, 1) * scale
    xx = torch.arange(w2, device=device, dtype=torch.float32).view(1, w2) * scale

    d2 = ((xx.unsqueeze(2) - u_s) ** 2 + (yy.unsqueeze(2) - v_s) ** 2)  # (h2,w2,n)
    soft = torch.exp(-d2 / (2 * sigma**2)).sum(-1)  # (h2,w2)
    soft = soft / (soft.max() + 1e-8)
    soft = soft.unsqueeze(0).unsqueeze(0)  # (1,1,h2,w2)

    mask_ds = F.interpolate(mask_t, size=(h2, w2), mode='bilinear', align_corners=False)

    inter = (soft * mask_ds).sum()
    union = (soft + mask_ds - soft * mask_ds).sum()
    iou = inter / (union + 1e-8)
    return 1.0 - iou, float(iou.item())


# ─────────────────────────────────────────────────────────────────────────────
# Main optimization
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(key_records, n_iters, lr, w_proj, w_chamfer_scene,
                     w_chamfer_human, device, scale_init=5.0, w_t_reg=0.5):
    """
    Optimize (scale, t_world) where scale converts ROMP camera-space meters
    to COLMAP world units (expected ~8.4 for NeuMan lab), applied BEFORE
    cam_to_world so that Chamfer vs COLMAP scene pts correctly drives scale.

    scale_init: estimated from calibration (background COLMAP z / ROMP z).
    w_t_reg:    L2 regularization on t_world to prevent it from absorbing scale.

    Forward:
        verts_cam_colmap  = scale * verts_cam_meters          (N,3)
        verts_world       = R.T @ (verts_cam_colmap - t_w2c)  (N,3)  COLMAP units
        verts_world_align = verts_world + t_world              (N,3)  small correction
        verts_cam_proj    = R @ verts_world_align + t_w2c      (N,3)  for IoU
                          = scale * verts_cam + R @ t_world    (scale cancels in proj)
    """
    init = float(np.log(max(scale_init, 0.1)))
    log_scale = torch.nn.Parameter(torch.tensor([init], dtype=torch.float32, device=device))
    t_world   = torch.nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam([log_scale, t_world], lr=lr)

    best = {'loss': float('inf'), 'iter': -1, 'scale': scale_init,
            't_world': np.zeros(3, dtype=np.float32)}
    log = []

    for it in trange(n_iters, desc='Optimizing'):
        optimizer.zero_grad()
        scale = torch.exp(log_scale)
        loss_total = torch.zeros(1, device=device)
        iou_avg = 0.0

        for rec in key_records:
            R = rec['R_w2c_t']   # (3,3)
            t = rec['t_w2c_t']   # (3,)

            # Scale ROMP meters → COLMAP units in camera space, then to world
            verts_cam_colmap = scale * rec['verts_cam_t']        # (N,3)
            verts_world = (R.T @ (verts_cam_colmap - t).T).T     # (N,3)
            verts_world_align = verts_world + t_world             # (N,3)

            # Project back to this camera for IoU (scale cancels, only t_world shifts)
            verts_cam_proj = (R @ verts_world_align.T).T + t     # (N,3)

            # Loss 1: projection IoU
            if w_proj > 0:
                l_proj, iou = soft_silhouette_loss(
                    verts_cam_proj, rec['K'], rec['H'], rec['W'],
                    rec['mask_t'], device=device)
                loss_total = loss_total + w_proj * l_proj
                iou_avg += iou

            # Loss 2: one-sided Chamfer → COLMAP scene points (in world COLMAP units)
            if w_chamfer_scene > 0 and rec['scene_pts_t'] is not None:
                l_cs = chamfer_one_sided(verts_world_align, rec['scene_pts_t'])
                loss_total = loss_total + w_chamfer_scene * l_cs

        loss_total = loss_total / len(key_records)
        iou_avg /= len(key_records)

        # t_world regularization: keep small to prevent it absorbing scale error
        if w_t_reg > 0:
            loss_total = loss_total + w_t_reg * (t_world ** 2).sum()

        cur = float(loss_total.item())
        sc  = float(scale.item())
        tw  = t_world.detach().cpu().numpy().tolist()

        if cur < best['loss']:
            best['loss']    = cur
            best['iter']    = it
            best['scale']   = sc
            best['t_world'] = t_world.detach().cpu().numpy().copy()

        loss_total.backward()
        optimizer.step()

        log.append({'iter': it, 'loss': cur, 'iou': iou_avg, 'scale': sc, 't_world': tw})

        if (it == 0) or ((it + 1) % 50 == 0) or (it + 1 == n_iters):
            print(f'  iter {it+1:4d}/{n_iters}  loss={cur:.5f}  iou={iou_avg:.3f}'
                  f'  scale={sc:.3f}  t={[round(x, 3) for x in tw]}')

    return best, log


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_pts_on_img(img, pts_cam, K, W, H, color, radius=2, max_draw=1000):
    """Draw projected 3D camera-space points onto image copy."""
    out = img.copy()
    if len(pts_cam) == 0:
        return out
    idx = np.random.choice(len(pts_cam), min(max_draw, len(pts_cam)), replace=False)
    pc = pts_cam[idx]
    valid = pc[:, 2] > 0.01
    u = (pc[valid, 0] / pc[valid, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = (pc[valid, 1] / pc[valid, 2] * K[1, 1] + K[1, 2]).astype(int)
    for ui, vi in zip(u, v):
        if 0 <= ui < W and 0 <= vi < H:
            cv2.circle(out, (ui, vi), radius, color, -1)
    return out


def make_camera_view_comparison(rec, verts_cam_before, verts_cam_after,
                                 scale, t_world_str):
    """Return side-by-side before/after overlay image."""
    img = rec['img']
    if img is None:
        img = np.ones((rec['H'], rec['W'], 3), dtype=np.uint8) * 30

    K, W, H = rec['K'], rec['W'], rec['H']

    # Project COLMAP background points (white) for scene context
    scene_cam = rec['scene_pts_cam']
    img_before = draw_pts_on_img(img, scene_cam, K, W, H, (200, 200, 200), 1, 3000)
    img_after = img_before.copy()

    # Human depth points (blue) if available
    if rec.get('human_pts_cam') is not None:
        img_before = draw_pts_on_img(img_before, rec['human_pts_cam'], K, W, H,
                                     (255, 160, 30), 2, 1500)
        img_after = draw_pts_on_img(img_after, rec['human_pts_cam'], K, W, H,
                                     (255, 160, 30), 2, 1500)

    # SMPL verts: before (red), after (green)
    img_before = draw_pts_on_img(img_before, verts_cam_before, K, W, H,
                                  (0, 0, 220), 2, 800)
    img_after = draw_pts_on_img(img_after, verts_cam_after, K, W, H,
                                 (0, 200, 0), 2, 800)

    # Labels
    def label(im, txt):
        cv2.putText(im, txt, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return im

    label(img_before, 'Before  [SMPL=red  scene=gray  human=orange]')
    label(img_after, f'After   scale={scale:.3f}  t={t_world_str}  [SMPL=green]')

    return np.concatenate([img_before, img_after], axis=1)


def make_3d_scatter(rec, verts_world_before, verts_world_after, scale, out_path):
    """3D scatter: COLMAP points (gray) + human depth pts (orange) + SMPL (red/green)."""
    fig = plt.figure(figsize=(14, 6))
    for ci, (title, vw, color) in enumerate([
        ('Before (cam→world, scale=1, t=0)', verts_world_before, 'red'),
        (f'After (scale={scale:.3f})', verts_world_after, 'green'),
    ]):
        ax = fig.add_subplot(1, 2, ci + 1, projection='3d')

        # Scene context (COLMAP)
        sp = rec['scene_pts_np']
        if len(sp) > 0:
            sidx = np.random.choice(len(sp), min(3000, len(sp)), replace=False)
            ax.scatter(sp[sidx, 0], sp[sidx, 1], sp[sidx, 2],
                       s=0.5, c='gray', alpha=0.3, label='COLMAP')

        # Human depth pts
        hp = rec.get('human_pts_np')
        if hp is not None and len(hp) > 0:
            ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2],
                       s=1.5, c='orange', alpha=0.6, label='human depth')

        # SMPL verts
        vidx = np.random.choice(len(vw), min(600, len(vw)), replace=False)
        ax.scatter(vw[vidx, 0], vw[vidx, 1], vw[vidx, 2],
                   s=3, c=color, alpha=0.8, label='SMPL verts')

        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc='upper left')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def make_loss_curve(log, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    iters = [e['iter'] for e in log]
    axes[0].plot(iters, [e['loss'] for e in log]); axes[0].set_title('Total loss')
    axes[1].plot(iters, [e['iou'] for e in log]);  axes[1].set_title('Proj IoU')
    axes[2].plot(iters, [e['scale'] for e in log]); axes[2].set_title('Scale')
    for ax in axes:
        ax.set_xlabel('iter'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/neuman/dataset')
    p.add_argument('--seq', default='lab')
    p.add_argument('--out_dir', default='output/coarse_align')
    p.add_argument('--n_keyframes', type=int, default=8,
                   help='Number of keyframes to use for optimization')
    p.add_argument('--n_iters', type=int, default=400)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--w_proj', type=float, default=2.0,
                   help='Weight for projection IoU loss')
    p.add_argument('--w_chamfer_scene', type=float, default=0.1,
                   help='Weight for one-sided Chamfer to COLMAP scene pts')
    p.add_argument('--w_chamfer_human', type=float, default=0.5,
                   help='Weight for Chamfer to calibrated human depth pts')
    p.add_argument('--max_scene_pts', type=int, default=50_000)
    p.add_argument('--max_human_pts', type=int, default=3_000)
    p.add_argument('--verts_add_trans', type=int, default=1, choices=[0, 1],
                   help='1: verts_cam = romp_verts + romp_trans (default). '
                        '0: verts_cam = romp_verts (ROMP already applied trans)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seq_dir = os.path.join(args.data_root, args.seq)
    sparse_dir = os.path.join(seq_dir, 'sparse')
    smpl_dir = os.path.join(seq_dir, 'smpl_pred')
    seg_dir = os.path.join(seq_dir, 'segmentations')
    img_dir = os.path.join(seq_dir, 'images')
    mono_dir = os.path.join(seq_dir, 'mono_depth')
    out_dir = os.path.join(args.out_dir, args.seq)
    vis_dir = os.path.join(out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # ── 1. COLMAP data ────────────────────────────────────────────────────────
    print('\n[1/5] Loading COLMAP ...')
    colmap_cams = read_colmap_cameras(os.path.join(sparse_dir, 'cameras.txt'))
    colmap_frames = read_colmap_images(os.path.join(sparse_dir, 'images.txt'))
    scene_pts_np = read_colmap_points3d(
        os.path.join(sparse_dir, 'points3D.txt'), args.max_scene_pts)
    print(f'  {len(colmap_frames)} frames, {len(scene_pts_np)} COLMAP 3D pts')

    scene_pts_t = torch.from_numpy(scene_pts_np).to(device)

    # ── 2. Enumerate available frames ────────────────────────────────────────
    smpl_files = sorted(glob(os.path.join(smpl_dir, '*.npz')))
    frame_names = [os.path.basename(f).replace('_png.npz', '.png') for f in smpl_files]
    valid = [(n, sf) for n, sf in zip(frame_names, smpl_files) if n in colmap_frames]
    print(f'  {len(valid)} frames with both SMPL pred and COLMAP pose')

    # ── 3. Select keyframes uniformly ────────────────────────────────────────
    n_kf = min(args.n_keyframes, len(valid))
    kf_idx = np.round(np.linspace(0, len(valid) - 1, n_kf)).astype(int)
    keyframes = [valid[i] for i in kf_idx]
    print(f'  Keyframes ({n_kf}): {[kf[0] for kf in keyframes]}')

    # ── 4. Build per-keyframe records ────────────────────────────────────────
    print('\n[2/5] Building keyframe records ...')
    key_records = []
    for frame_name, smpl_path in keyframes:
        cf = colmap_frames[frame_name]
        cam_id = cf['cam_id']
        K  = colmap_cams[cam_id]['K']
        W  = colmap_cams[cam_id]['W']
        H  = colmap_cams[cam_id]['H']
        R_w2c = cf['R_w2c']
        t_w2c = cf['t_w2c']

        # SMPL in camera space (pass K to correct ROMP focal assumption)
        romp = load_romp(smpl_path, K=K)
        if args.verts_add_trans:
            verts_cam = romp['verts'] + romp['trans'][None, :]
        else:
            verts_cam = romp['verts']

        # Camera → world (ROMP metric scale, no global scale yet)
        verts_world = cam_to_world(verts_cam, R_w2c, t_w2c)

        # Load mask (segmentation)
        # NeuMan convention: white(255)=background, black(0)=human (opposite of normal)
        seg_path = os.path.join(seg_dir, frame_name)
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if seg is None:
            seg = np.zeros((H, W), dtype=np.uint8)
        mask_human = seg < 128   # black=0=human
        mask_bg = seg > 127      # white=255=background

        # Load mono_depth for depth calibration
        mono_path = os.path.join(mono_dir, frame_name)
        mono = cv2.imread(mono_path, cv2.IMREAD_UNCHANGED)  # uint16

        human_pts_np = None
        human_pts_cam = None
        calib = None
        if mono is not None:
            calib = calibrate_mono_depth(
                mono, scene_pts_np, R_w2c, t_w2c, K, W, H, mask_bg)
            if calib is not None:
                a, b = calib
                human_pts_np = get_human_pts_from_depth(
                    mono, calib, R_w2c, t_w2c, K, mask_human,
                    args.max_human_pts, args.seed)
                # Also get camera-space version for visualization
                if len(human_pts_np) > 0:
                    human_pts_cam = world_to_cam(human_pts_np, R_w2c, t_w2c)
                print(f'  {frame_name}: calib a={a:.4f} b={b:.4f}'
                      f'  human_pts={len(human_pts_np)}')
            else:
                print(f'  {frame_name}: calib failed (too few BG pts)')
        else:
            print(f'  {frame_name}: no mono_depth found')

        # Load image for visualization
        img_path = os.path.join(img_dir, frame_name)
        img = cv2.imread(img_path)

        # COLMAP scene points projected to this camera (for viz)
        scene_cam = world_to_cam(scene_pts_np, R_w2c, t_w2c)

        key_records.append({
            'frame_name': frame_name,
            'K': K, 'W': W, 'H': H,
            'R_w2c': R_w2c, 't_w2c': t_w2c,
            'R_w2c_t': torch.from_numpy(R_w2c).to(device),
            't_w2c_t': torch.from_numpy(t_w2c).to(device),
            'verts_cam': verts_cam,                                          # numpy, ROMP meters
            'verts_cam_t': torch.from_numpy(verts_cam).to(device),          # torch, ROMP meters
            'verts_world': verts_world,                                      # numpy (only for viz before-state)
            'mask_t': torch.from_numpy(mask_human.astype(np.float32)).to(device
                       ).unsqueeze(0).unsqueeze(0),
            'scene_pts_t': scene_pts_t,
            'scene_pts_np': scene_pts_np,
            'scene_pts_cam': scene_cam,
            'human_pts_np': human_pts_np,
            'human_pts_cam': human_pts_cam,
            'img': img,
        })

    # Human depth chamfer disabled: mono_depth in human region is unreliable
    # (monocular depth models estimate background depth for human pixels).
    args.w_chamfer_human = 0.0

    # ── Estimate scale_init via foot-contact COLMAP points ───────────────────
    # The person stands on the floor. COLMAP reconstructs floor points.
    # For each keyframe:
    #   1. Project SMPL foot verts (lowest-Y verts) to image.
    #   2. Find COLMAP pts that project near the foot region.
    #   3. Their camera-space z (COLMAP units) / foot ROMP camera-z (meters)
    #      gives colmap_per_meter estimate.
    print('\n[scale_init] Estimating COLMAP/meter ratio from foot-contact COLMAP pts ...')
    scale_estimates = []
    for rec in key_records:
        verts_cam = rec['verts_cam']                      # (N,3) ROMP meters
        K  = rec['K']
        W  = rec['W']
        H  = rec['H']
        R_w2c = rec['R_w2c']
        t_w2c = rec['t_w2c']

        # Foot verts: bottom 5% by Y (highest Y value in SMPL = lowest in scene for upright person)
        # ROMP body space: Y axis points DOWN from head → feet at max Y
        foot_thresh = np.percentile(verts_cam[:, 1], 95)
        foot_mask = verts_cam[:, 1] >= foot_thresh
        foot_cam = verts_cam[foot_mask]                   # (M,3) foot verts, ROMP meters
        if len(foot_cam) == 0:
            continue

        # Project foot verts to image using ACTUAL focal (no unit conversion yet)
        z_f = foot_cam[:, 2].clip(0.01)
        u_f = (foot_cam[:, 0] / z_f * K[0, 0] + K[0, 2]).astype(int)
        v_f = (foot_cam[:, 1] / z_f * K[1, 1] + K[1, 2]).astype(int)
        # foot centroid pixel
        uf_c = int(np.median(u_f.clip(0, W-1)))
        vf_c = int(np.median(v_f.clip(0, H-1)))
        foot_romp_z = float(np.median(z_f))              # ROMP z of foot, in meters

        # Find COLMAP pts projecting near foot pixel (±50px radius)
        pc_all = world_to_cam(scene_pts_np, R_w2c, t_w2c)
        valid  = pc_all[:, 2] > 0.01
        pc_v   = pc_all[valid]
        u_s = (pc_v[:, 0] / pc_v[:, 2] * K[0, 0] + K[0, 2]).astype(int)
        v_s = (pc_v[:, 1] / pc_v[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        z_s = pc_v[:, 2]
        in_img = (u_s >= 0) & (u_s < W) & (v_s >= 0) & (v_s < H)
        u_s, v_s, z_s = u_s[in_img], v_s[in_img], z_s[in_img]

        radius = 80
        near_foot = ((u_s >= uf_c - radius) & (u_s <= uf_c + radius) &
                     (v_s >= vf_c - radius) & (v_s <= vf_c + radius))
        if near_foot.sum() < 3:
            continue

        # Use the NEAREST (smallest z) COLMAP pts near foot – likely floor points
        z_near = z_s[near_foot]
        z_floor_colmap = float(np.percentile(z_near, 10))  # near floor pts

        s_est = z_floor_colmap / foot_romp_z
        scale_estimates.append(s_est)
        print(f'  {rec["frame_name"]}: foot_pixel=({uf_c},{vf_c})  '
              f'z_foot_romp={foot_romp_z:.2f}m  z_floor_colmap={z_floor_colmap:.2f}  '
              f'scale_est={s_est:.2f}')

    if scale_estimates:
        scale_init = float(np.median(scale_estimates))
        print(f'[scale_init] median = {scale_init:.3f}  (GT mean ≈ 8.6 for NeuMan lab)')
    else:
        scale_init = 5.0
        print('[scale_init] fallback to 5.0 (no foot-contact COLMAP pts found)')

    # ── 5. Optimization ───────────────────────────────────────────────────────
    print(f'\n[3/5] Optimizing: {n_kf} keyframes, {args.n_iters} iters ...')
    print(f'  losses: proj={args.w_proj}  scene_chamfer={args.w_chamfer_scene}'
          f'  human_chamfer={args.w_chamfer_human}  scale_init={scale_init:.2f}')

    best, log = run_optimization(
        key_records, args.n_iters, args.lr,
        args.w_proj, args.w_chamfer_scene, args.w_chamfer_human, device,
        scale_init=scale_init, w_t_reg=0.5)

    global_scale = best['scale']
    global_t = best['t_world']

    print(f'\n[Result]')
    print(f'  global_scale = {global_scale:.4f}')
    print(f'  global_t_world = {global_t.tolist()}')
    print(f'  best_iter = {best["iter"]}, best_loss = {best["loss"]:.5f}')

    # ── 6. Compare with GT (if available) ────────────────────────────────────
    gt_path = os.path.join(seq_dir, '4d_humans', 'smpl_optimized_aligned_scale.npz')
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        gt_scale = float(gt['scale'].mean())
        gt_transl = gt['transl'].mean(0)
        print(f'\n[GT comparison]')
        print(f'  GT scale (mean): {gt_scale:.4f}   ours: {global_scale:.4f}')
        print(f'  GT transl (mean): {gt_transl.tolist()}   ours: {global_t.tolist()}')

    # ── 7. Save JSON ─────────────────────────────────────────────────────────
    result = {
        'seq': args.seq,
        'global_scale': float(global_scale),
        'global_t_world': [float(x) for x in global_t.tolist()],
        'best_iter': int(best['iter']),
        'best_loss': float(best['loss']),
        'verts_add_trans': bool(args.verts_add_trans),
    }
    result_path = os.path.join(out_dir, 'coarse_align.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {result_path}')

    # ── 8. Visualizations ────────────────────────────────────────────────────
    print('\n[4/5] Generating visualizations ...')

    t_str = '[' + ', '.join(f'{x:.2f}' for x in global_t.tolist()) + ']'

    for rec in key_records:
        fid = rec['frame_name'].replace('.png', '')
        fvis = os.path.join(vis_dir, fid)
        os.makedirs(fvis, exist_ok=True)

        R_w2c = rec['R_w2c']
        t_w2c = rec['t_w2c']

        # Compute before/after in world and camera space.
        # "before": scale=1 (unit ROMP meters), no t_world offset.
        # "after":  apply global_scale in camera space, then cam_to_world + t_world.
        vc_before = rec['verts_cam']
        vw_before = cam_to_world(vc_before, R_w2c, t_w2c)     # meters, wrong scale

        vc_scaled = global_scale * vc_before                    # COLMAP units (cam space)
        vw_after  = cam_to_world(vc_scaled, R_w2c, t_w2c) + global_t  # world COLMAP
        vc_after  = world_to_cam(vw_after, R_w2c, t_w2c)

        # Camera-view comparison
        cmp = make_camera_view_comparison(rec, vc_before, vc_after,
                                          global_scale, t_str)
        cv2.imwrite(os.path.join(fvis, 'camera_view.jpg'), cmp)

        # 3D scatter
        make_3d_scatter(rec, vw_before, vw_after, global_scale,
                        os.path.join(fvis, '3d_scatter.png'))

        print(f'  Saved vis for {fid}')

    # Loss curve
    make_loss_curve(log, os.path.join(out_dir, 'loss_curve.png'))

    # Summary panel: all keyframes stacked
    panels = []
    for rec in key_records:
        fid = rec['frame_name'].replace('.png', '')
        p = cv2.imread(os.path.join(vis_dir, fid, 'camera_view.jpg'))
        if p is not None:
            # Scale down to fit
            panels.append(cv2.resize(p, (p.shape[1]//2, p.shape[0]//2)))
    if panels:
        grid = np.concatenate(panels, axis=0)
        cv2.imwrite(os.path.join(out_dir, 'summary_all_keyframes.jpg'), grid)
        print(f'\nSaved summary: {os.path.join(out_dir, "summary_all_keyframes.jpg")}')

    print(f'\n[DONE] All outputs in: {out_dir}')


if __name__ == '__main__':
    main()
