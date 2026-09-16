#!/usr/bin/env python3
"""
Coarse SMPL-to-scene alignment v2 — DepthPro edition.

Core improvements over v1:
  • DepthPro metric depth (float32 .npy) replaces unreliable mono_depth uint16
  • Background pixels calibrate DepthPro (meters) → COLMAP units via linear fit
  • Human region depth is now reliable: enables bidirectional trimmed Chamfer
    between SMPL mesh and the DepthPro human point cloud (key loss)
  • scale_init from calibration slope a (colmap_per_meter) — more accurate
  • Outlier removal on depth pts (camera-depth percentile + MAD filter)
  • Multi-keyframe joint optimization: one shared (scale, t_world) across all KFs

Optimization losses:
  1. Bidirectional trimmed Chamfer — SMPL verts ↔ DepthPro human pts (COLMAP world)
  2. Soft projection IoU           — SMPL silhouette vs human segmentation mask
  3. Scale regularization          — keep scale near init, prevent collapse
  4. Translation regularization    — keep t_world small

Output:
  <out_dir>/<seq>/coarse_align_v2.json
  <out_dir>/<seq>/vis/

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/coarse_align_smpl_neuman_v2.py \
      --data_root data/neuman/dataset --seq lab \
      --out_dir output/coarse_align_v2
"""

import argparse
import json
import os
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
# COLMAP I/O  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)


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
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            cams[cam_id] = {'K': K, 'W': W, 'H': H}
    return cams


def read_colmap_images(path):
    frames = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 10:
            i += 1; continue
        q = list(map(float, p[1:5]))
        t = np.array(list(map(float, p[5:8])), dtype=np.float32)
        cam_id = int(p[8])
        name = p[9]
        frames[name] = {'R_w2c': qvec2rotmat(q), 't_w2c': t, 'cam_id': cam_id}
        i += 2
    return frames


def read_colmap_points3d(path, max_pts=80_000):
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

def load_romp(npz_path, K):
    """Load ROMP prediction and correct ROMP's focal assumption."""
    data = np.load(npz_path, allow_pickle=True)
    r = data['results'].item()
    trans = np.array(r['trans'], dtype=np.float32)
    # ROMP uses focal = cx when projecting; correct X/Y for actual focal
    focal_romp = K[0, 2]
    trans[0] *= focal_romp / K[0, 0]
    trans[1] *= focal_romp / K[1, 1]
    return {
        'verts': np.array(r['verts'], dtype=np.float32),
        'trans': trans,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cam_to_world(pts_cam, R_w2c, t_w2c):
    return (R_w2c.T @ (pts_cam - t_w2c).T).T

def world_to_cam(pts_world, R_w2c, t_w2c):
    return (R_w2c @ pts_world.T).T + t_w2c


# ─────────────────────────────────────────────────────────────────────────────
# DepthPro depth handling
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_depthpro(depth_m, colmap_pts, R_w2c, t_w2c, K, W, H, mask_bg,
                       min_samples=30):
    """
    Calibrate DepthPro metric depth (meters) to COLMAP units via background pixels.

    DepthPro is already metric but COLMAP has its own unit (colmap_units/meter ≈ 8.4 for NeuMan lab).
    We fit: z_colmap = a * z_depthpro_m + b  (expect b ≈ 0, a ≈ colmap_per_meter)

    Returns (a, b) or None on failure.
    """
    pc = world_to_cam(colmap_pts, R_w2c, t_w2c)
    valid_z = pc[:, 2] > 0.01
    pc_v = pc[valid_z]
    u = (pc_v[:, 0] / pc_v[:, 2] * K[0, 0] + K[0, 2]).astype(np.int32)
    v = (pc_v[:, 1] / pc_v[:, 2] * K[1, 1] + K[1, 2]).astype(np.int32)
    z_colmap = pc_v[:, 2]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z_colmap = u[in_img], v[in_img], z_colmap[in_img]

    in_bg = mask_bg[v, u]
    u, v, z_colmap = u[in_bg], v[in_bg], z_colmap[in_bg]
    if len(u) < min_samples:
        return None

    z_dp = depth_m[v, u].astype(np.float64)
    valid = (z_dp > 0.01) & np.isfinite(z_dp) & (z_colmap > 0.01)
    if valid.sum() < min_samples:
        return None

    z_dp, z_colmap = z_dp[valid], z_colmap[valid].astype(np.float64)

    # Robust fit via median slope (Theil-Sen lite: use median of z_colmap/z_dp as slope)
    # Then refine with least squares on the inliers
    slope_est = np.median(z_colmap / z_dp)

    # Residual-based inlier selection (keep within 2× MAD)
    resid = z_colmap - slope_est * z_dp
    mad = np.median(np.abs(resid - np.median(resid)))
    inlier = np.abs(resid - np.median(resid)) < 3.0 * max(mad, 1e-6)
    if inlier.sum() < min_samples:
        inlier = np.ones(len(z_dp), dtype=bool)

    A = np.stack([z_dp[inlier], np.ones(inlier.sum())], axis=1)
    result = np.linalg.lstsq(A, z_colmap[inlier], rcond=None)
    a, b = float(result[0][0]), float(result[0][1])
    return a, b


def extract_human_pts_depthpro(depth_m, calib, R_w2c, t_w2c, K,
                                mask_human, max_pts=4000,
                                far_percentile=97.0, mad_k=3.5,
                                seed=0):
    """
    Backproject DepthPro human-region depth to COLMAP world-space 3D points.

    Applies outlier removal:
      - Far outlier removal via camera-space z percentile + MAD filter
    Returns (M, 3) world-space float32 array.
    """
    a, b = calib
    ys, xs = np.where(mask_human)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Subsample for speed (we'll remove outliers after)
    rng = np.random.default_rng(seed)
    if len(ys) > max_pts * 3:
        idx = rng.choice(len(ys), max_pts * 3, replace=False)
        ys, xs = ys[idx], xs[idx]

    z_dp = depth_m[ys, xs].astype(np.float32)
    z_colmap = a * z_dp + b

    valid = (z_dp > 0.05) & (z_colmap > 0.01) & np.isfinite(z_colmap)
    xs, ys, z_colmap = xs[valid], ys[valid], z_colmap[valid]
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Backproject to camera space (COLMAP units)
    x_cam = (xs - K[0, 2]) / K[0, 0] * z_colmap
    y_cam = (ys - K[1, 2]) / K[1, 1] * z_colmap
    pts_cam = np.stack([x_cam, y_cam, z_colmap], axis=1).astype(np.float32)

    # Outlier removal: far depth percentile
    z_vals = pts_cam[:, 2]
    z_thresh = np.percentile(z_vals, far_percentile)
    pts_cam = pts_cam[z_vals <= z_thresh]
    if len(pts_cam) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # MAD-based outlier removal on z
    z_vals = pts_cam[:, 2]
    med_z = np.median(z_vals)
    mad = np.median(np.abs(z_vals - med_z))
    keep = np.abs(z_vals - med_z) <= mad_k * max(mad, 1e-4)
    pts_cam = pts_cam[keep]
    if len(pts_cam) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Camera → world (COLMAP world units)
    pts_world = cam_to_world(pts_cam, R_w2c, t_w2c)

    # Final subsample
    if len(pts_world) > max_pts:
        idx = rng.choice(len(pts_world), max_pts, replace=False)
        pts_world = pts_world[idx]

    return pts_world.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Losses (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_trimmed(src, tgt, trim_ratio=0.10, chunk=2048):
    """
    Bidirectional trimmed Chamfer distance.

    Trims the top trim_ratio fraction of distances (outlier pairs) before averaging.
    Returns scalar loss (mean of sqrt distances after trim, numerically sqrt makes
    it less sensitive to large errors).
    """
    # src → tgt: min dist² for each src point
    d_st_list = []
    for i in range(0, len(src), chunk):
        d = (src[i:i+chunk, None, :] - tgt[None, :, :]).pow(2).sum(-1)
        d_st_list.append(d.min(-1).values)
    d_st = torch.cat(d_st_list)  # (|src|,)

    # tgt → src: min dist² for each tgt point
    d_ts_list = []
    for i in range(0, len(tgt), chunk):
        d = (tgt[i:i+chunk, None, :] - src[None, :, :]).pow(2).sum(-1)
        d_ts_list.append(d.min(-1).values)
    d_ts = torch.cat(d_ts_list)  # (|tgt|,)

    # Trim largest distances (outliers)
    k_st = max(1, int(len(d_st) * (1.0 - trim_ratio)))
    k_ts = max(1, int(len(d_ts) * (1.0 - trim_ratio)))
    d_st_trimmed = torch.topk(d_st, k_st, largest=False).values
    d_ts_trimmed = torch.topk(d_ts, k_ts, largest=False).values

    return 0.5 * (d_st_trimmed.mean() + d_ts_trimmed.mean())


def soft_silhouette_loss(verts_cam_t, K, H, W, mask_t, sigma=3.0,
                          n_sample=800, device='cuda'):
    """
    Soft projection IoU between SMPL silhouette and human mask.
    Returns (loss, iou_float).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    z = verts_cam_t[:, 2].clamp(min=0.1)
    u = verts_cam_t[:, 0] / z * fx + cx
    v = verts_cam_t[:, 1] / z * fy + cy

    perm = torch.randperm(len(u), device=device)[:n_sample]
    u_s = u[perm]
    v_s = v[perm]

    scale = 4
    h2, w2 = H // scale, W // scale
    yy = torch.arange(h2, device=device, dtype=torch.float32).view(h2, 1) * scale
    xx = torch.arange(w2, device=device, dtype=torch.float32).view(1, w2) * scale

    d2 = (xx.unsqueeze(2) - u_s) ** 2 + (yy.unsqueeze(2) - v_s) ** 2
    soft = torch.exp(-d2 / (2 * sigma ** 2)).sum(-1)
    soft = soft / (soft.max() + 1e-8)
    soft = soft.unsqueeze(0).unsqueeze(0)

    mask_ds = F.interpolate(mask_t, size=(h2, w2), mode='bilinear', align_corners=False)
    inter = (soft * mask_ds).sum()
    union = (soft + mask_ds - soft * mask_ds).sum()
    iou = inter / (union + 1e-8)
    return 1.0 - iou, float(iou.item())


# ─────────────────────────────────────────────────────────────────────────────
# Main optimization
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(key_records, n_iters, lr,
                     w_chamfer, w_proj, w_reg_scale, w_reg_trans,
                     trim_ratio, device, scale_init=8.0):
    """
    Joint optimization over all keyframes.

    Forward (per keyframe):
        verts_cam_colmap  = scale * verts_cam_romp         (COLMAP camera units)
        verts_world       = R.T @ (verts_cam_colmap - t)   (COLMAP world units)
        verts_world_align = verts_world + t_world           (small additive correction)
        verts_cam_proj    = R @ verts_world_align + t       (for IoU projection)
    """
    log_scale = torch.nn.Parameter(
        torch.tensor([np.log(max(scale_init, 0.1))], dtype=torch.float32, device=device))
    t_world = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([log_scale, t_world], lr=lr)

    best = {'loss': float('inf'), 'iter': -1,
            'scale': scale_init, 't_world': np.zeros(3, dtype=np.float32)}
    log = []

    for it in trange(n_iters, desc='Optimizing', ncols=80):
        optimizer.zero_grad()
        scale = torch.exp(log_scale)

        loss_chamfer_sum = torch.zeros(1, dtype=torch.float32, device=device)
        loss_proj_sum    = torch.zeros(1, dtype=torch.float32, device=device)
        iou_sum = 0.0
        n_kf_used = 0

        for rec in key_records:
            R = rec['R_t']  # (3,3)
            t = rec['t_t']  # (3,)

            verts_cam_colmap = scale * rec['verts_cam_t']        # (N,3)
            verts_world      = (R.T @ (verts_cam_colmap - t).T).T  # (N,3)
            verts_align      = verts_world + t_world              # (N,3)

            # Bidirectional trimmed Chamfer: SMPL verts ↔ human depth pts
            if w_chamfer > 0 and rec['human_pts_t'] is not None:
                hp = rec['human_pts_t']
                if hp.shape[0] >= 32:
                    loss_chamfer_sum = loss_chamfer_sum + chamfer_trimmed(
                        verts_align, hp, trim_ratio=trim_ratio)

            # Soft projection IoU
            if w_proj > 0:
                verts_cam_proj = (R @ verts_align.T).T + t
                l_proj, iou = soft_silhouette_loss(
                    verts_cam_proj, rec['K'], rec['H'], rec['W'],
                    rec['mask_t'], device=device)
                loss_proj_sum = loss_proj_sum + l_proj
                iou_sum += iou

            n_kf_used += 1

        if n_kf_used == 0:
            continue
        nf = float(n_kf_used)

        loss_chamfer = loss_chamfer_sum / nf
        loss_proj    = loss_proj_sum    / nf

        # Regularizers
        reg_scale = (scale - torch.tensor(scale_init, device=device)) ** 2
        reg_trans = (t_world ** 2).sum()

        total = (w_chamfer * loss_chamfer
                 + w_proj  * loss_proj
                 + w_reg_scale * reg_scale
                 + w_reg_trans * reg_trans)

        cur = float(total.item())
        sc  = float(scale.item())
        tw  = t_world.detach().cpu().numpy().tolist()

        if cur < best['loss']:
            best.update({'loss': cur, 'iter': it, 'scale': sc,
                         't_world': t_world.detach().cpu().numpy().copy()})

        total.backward()
        optimizer.step()

        log.append({'iter': it, 'loss': cur,
                    'chamfer': float(loss_chamfer.item()),
                    'proj_iou': iou_sum / nf,
                    'scale': sc, 't_world': tw})

        if (it == 0) or ((it + 1) % 50 == 0) or (it + 1 == n_iters):
            print(f'  iter {it+1:4d}/{n_iters}  loss={cur:.5f}'
                  f'  chamfer={float(loss_chamfer.item()):.5f}'
                  f'  iou={iou_sum/nf:.3f}'
                  f'  scale={sc:.3f}  t={[round(x, 3) for x in tw]}')

    return best, log


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers  (camera-view overlay)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pts_on_img(img, pts_cam, K, W, H, color, radius=2, max_draw=1500):
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


def save_camera_overlay(rec, verts_cam_before, verts_cam_after, global_scale,
                        global_t, K, out_path):
    img = rec['img']
    if img is None:
        img = np.ones((rec['H'], rec['W'], 3), dtype=np.uint8) * 30

    # COLMAP scene pts: gray background context
    scene_cam = world_to_cam(rec['scene_pts_np'], rec['R_w2c'], rec['t_w2c'])
    b = draw_pts_on_img(img, scene_cam, K, rec['W'], rec['H'], (200, 200, 200), 1, 3000)
    a = b.copy()

    # Human depth pts: orange
    hp_np = rec.get('human_pts_np')
    if hp_np is not None and len(hp_np) > 0:
        hp_cam = world_to_cam(hp_np, rec['R_w2c'], rec['t_w2c'])
        b = draw_pts_on_img(b, hp_cam, K, rec['W'], rec['H'], (30, 160, 255), 2, 1500)
        a = draw_pts_on_img(a, hp_cam, K, rec['W'], rec['H'], (30, 160, 255), 2, 1500)

    b = draw_pts_on_img(b, verts_cam_before, K, rec['W'], rec['H'], (0, 0, 220), 2, 600)
    a = draw_pts_on_img(a, verts_cam_after,  K, rec['W'], rec['H'], (0, 200, 0), 2, 600)

    t_str = '[' + ', '.join(f'{x:.2f}' for x in global_t.tolist()) + ']'
    cv2.putText(b, 'Before  [SMPL=red  scene=gray  human=orange]',
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(a, f'After   scale={global_scale:.3f}  t={t_str}',
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, np.concatenate([b, a], axis=1))


def save_3d_scatter(rec, verts_world_before, verts_world_after, scale, out_path):
    fig = plt.figure(figsize=(14, 6))
    for ci, (title, vw, color) in enumerate([
        ('Before (no align)', verts_world_before, 'red'),
        (f'After  (scale={scale:.3f})', verts_world_after, 'green'),
    ]):
        ax = fig.add_subplot(1, 2, ci + 1, projection='3d')

        sp = rec['scene_pts_np']
        if len(sp) > 0:
            sidx = np.random.choice(len(sp), min(2000, len(sp)), replace=False)
            ax.scatter(sp[sidx,0], sp[sidx,1], sp[sidx,2], s=0.5, c='gray', alpha=0.3)

        hp = rec.get('human_pts_np')
        if hp is not None and len(hp) > 0:
            ax.scatter(hp[:,0], hp[:,1], hp[:,2], s=2, c='orange', alpha=0.6)

        vidx = np.random.choice(len(vw), min(500, len(vw)), replace=False)
        ax.scatter(vw[vidx,0], vw[vidx,1], vw[vidx,2], s=3, c=color, alpha=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def save_loss_curve(log, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    iters = [e['iter'] for e in log]
    for ax, key, title in zip(axes,
        ['loss', 'chamfer', 'proj_iou', 'scale'],
        ['Total loss', 'Chamfer', 'Proj IoU', 'Scale']):
        ax.plot(iters, [e[key] for e in log])
        ax.set_title(title); ax.set_xlabel('iter'); ax.grid(True, alpha=0.3)
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
    p.add_argument('--out_dir', default='output/coarse_align_v2')
    p.add_argument('--n_keyframes', type=int, default=10)
    p.add_argument('--n_iters', type=int, default=500)
    p.add_argument('--lr', type=float, default=3e-3)

    # Loss weights
    p.add_argument('--w_chamfer', type=float, default=1.0,
                   help='Weight for bidirectional trimmed Chamfer (human pts ↔ SMPL)')
    p.add_argument('--w_proj', type=float, default=5.0,
                   help='Weight for soft projection IoU')
    p.add_argument('--w_reg_scale', type=float, default=0.01,
                   help='Scale regularization weight')
    p.add_argument('--w_reg_trans', type=float, default=0.001,
                   help='Translation regularization weight')
    p.add_argument('--trim_ratio', type=float, default=0.10,
                   help='Fraction of worst Chamfer pairs to trim')

    # Point cloud settings
    p.add_argument('--max_scene_pts', type=int, default=80_000)
    p.add_argument('--max_human_pts', type=int, default=4_000,
                   help='Max DepthPro human pts per keyframe')
    p.add_argument('--far_percentile', type=float, default=97.0)
    p.add_argument('--mad_k', type=float, default=3.5)

    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seq_dir  = os.path.join(args.data_root, args.seq)
    sparse_dir  = os.path.join(seq_dir, 'sparse')
    smpl_dir    = os.path.join(seq_dir, 'smpl_pred')
    seg_dir     = os.path.join(seq_dir, 'segmentations')
    img_dir     = os.path.join(seq_dir, 'images')
    depth_dir   = os.path.join(seq_dir, 'depth_pro')   # ← DepthPro outputs
    out_dir     = os.path.join(args.out_dir, args.seq)
    vis_dir     = os.path.join(out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(
            f'DepthPro depth directory not found: {depth_dir}\n'
            f'Run scripts/run_depthpro_neuman.py first.')

    # ── 1. COLMAP ─────────────────────────────────────────────────────────────
    print('\n[1/5] Loading COLMAP ...')
    colmap_cams   = read_colmap_cameras(os.path.join(sparse_dir, 'cameras.txt'))
    colmap_frames = read_colmap_images(os.path.join(sparse_dir, 'images.txt'))
    scene_pts_np  = read_colmap_points3d(
        os.path.join(sparse_dir, 'points3D.txt'), args.max_scene_pts)
    print(f'  {len(colmap_frames)} COLMAP frames, {len(scene_pts_np)} 3D pts')

    # ── 2. Enumerate & select keyframes ───────────────────────────────────────
    smpl_files = sorted(glob(os.path.join(smpl_dir, '*.npz')))
    frame_names = [os.path.basename(f).replace('_png.npz', '.png') for f in smpl_files]
    valid = [(n, sf) for n, sf in zip(frame_names, smpl_files) if n in colmap_frames]
    print(f'  {len(valid)} frames with SMPL pred + COLMAP pose')

    n_kf = min(args.n_keyframes, len(valid))
    kf_idx = np.round(np.linspace(0, len(valid) - 1, n_kf)).astype(int)
    keyframes = [valid[i] for i in kf_idx]
    print(f'  Keyframes ({n_kf}): {[kf[0] for kf in keyframes]}')

    # ── 3. Build per-keyframe records ─────────────────────────────────────────
    print('\n[2/5] Building keyframe records (DepthPro calibration + human pts) ...')
    key_records = []
    calib_slopes = []

    for frame_name, smpl_path in keyframes:
        cf     = colmap_frames[frame_name]
        cam_id = cf['cam_id']
        K      = colmap_cams[cam_id]['K']
        W      = colmap_cams[cam_id]['W']
        H      = colmap_cams[cam_id]['H']
        R_w2c  = cf['R_w2c']
        t_w2c  = cf['t_w2c']

        # SMPL (camera space, meters, ROMP focal corrected)
        romp = load_romp(smpl_path, K)
        verts_cam = romp['verts'] + romp['trans'][None, :]  # (6890,3)

        # Segmentation mask: NeuMan convention black=human, white=background
        seg_path = os.path.join(seg_dir, frame_name)
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if seg is None:
            seg = np.zeros((H, W), dtype=np.uint8)
        mask_human = seg < 128
        mask_bg    = seg > 127

        # DepthPro depth
        stem = os.path.splitext(frame_name)[0]
        dp_path = os.path.join(depth_dir, stem + '.npy')
        if not os.path.exists(dp_path):
            print(f'  [skip] {frame_name}: no DepthPro depth at {dp_path}')
            continue
        depth_m = np.load(dp_path).astype(np.float32)

        # Calibrate DepthPro → COLMAP units using background pixels
        calib = calibrate_depthpro(
            depth_m, scene_pts_np, R_w2c, t_w2c, K, W, H, mask_bg)
        if calib is None:
            print(f'  [warn] {frame_name}: calibration failed, skipping')
            continue
        a, b = calib
        calib_slopes.append(a)
        print(f'  {frame_name}: calib a={a:.3f} b={b:.3f}  '
              f'(colmap_per_meter≈{a:.2f})')

        # Human point cloud from DepthPro
        human_pts_np = extract_human_pts_depthpro(
            depth_m, calib, R_w2c, t_w2c, K, mask_human,
            max_pts=args.max_human_pts,
            far_percentile=args.far_percentile,
            mad_k=args.mad_k, seed=args.seed)
        print(f'    human pts: {len(human_pts_np)}')

        human_pts_t = (torch.from_numpy(human_pts_np).to(device)
                       if len(human_pts_np) >= 32 else None)

        img = cv2.imread(os.path.join(img_dir, frame_name))

        key_records.append({
            'frame_name': frame_name,
            'K': K, 'W': W, 'H': H,
            'R_w2c': R_w2c, 't_w2c': t_w2c,
            'R_t': torch.from_numpy(R_w2c).to(device),
            't_t': torch.from_numpy(t_w2c).to(device),
            'verts_cam': verts_cam,
            'verts_cam_t': torch.from_numpy(verts_cam).to(device),
            'mask_t': torch.from_numpy(mask_human.astype(np.float32)).to(device
                       ).unsqueeze(0).unsqueeze(0),
            'human_pts_np': human_pts_np,
            'human_pts_t': human_pts_t,
            'scene_pts_np': scene_pts_np,
            'img': img,
        })

    if len(key_records) == 0:
        raise RuntimeError('No valid keyframes. Check DepthPro depth directory and SMPL predictions.')

    # scale_init: median of calibration slopes (colmap_per_meter)
    if calib_slopes:
        scale_init = float(np.median(calib_slopes))
        print(f'\n[scale_init] from DepthPro calibration: {scale_init:.3f}'
              f'  (GT for NeuMan lab ≈ 8.6)')
    else:
        scale_init = 8.0
        print('[scale_init] fallback to 8.0')

    n_kf_used = len(key_records)
    n_with_human = sum(1 for r in key_records if r['human_pts_t'] is not None)
    print(f'\n[3/5] Optimizing: {n_kf_used} keyframes ({n_with_human} with human pts), '
          f'{args.n_iters} iters')
    print(f'  w_chamfer={args.w_chamfer}  w_proj={args.w_proj}  '
          f'trim_ratio={args.trim_ratio}  scale_init={scale_init:.3f}')

    # ── 4. Optimization ───────────────────────────────────────────────────────
    best, log = run_optimization(
        key_records, args.n_iters, args.lr,
        args.w_chamfer, args.w_proj,
        args.w_reg_scale, args.w_reg_trans,
        args.trim_ratio, device, scale_init=scale_init)

    global_scale = best['scale']
    global_t     = best['t_world']

    print(f'\n[Result]')
    print(f'  global_scale  = {global_scale:.4f}')
    print(f'  global_t_world = {global_t.tolist()}')
    print(f'  best_iter = {best["iter"]}, best_loss = {best["loss"]:.5f}')

    # GT comparison
    gt_path = os.path.join(seq_dir, '4d_humans', 'smpl_optimized_aligned_scale.npz')
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        gt_scale = float(gt['scale'].mean())
        gt_transl = gt['transl'].mean(0)
        print(f'\n[GT comparison]')
        print(f'  scale: ours={global_scale:.4f}  GT={gt_scale:.4f}  '
              f'err={abs(global_scale-gt_scale)/gt_scale*100:.1f}%')

        # Per-frame transl error vs GT
        errs = []
        for rec in key_records:
            fname = rec['frame_name']
            frame_idx = int(os.path.splitext(fname)[0])
            R_w2c = rec['R_w2c']
            t_w2c = rec['t_w2c']
            romp = load_romp(
                os.path.join(seq_dir, 'smpl_pred',
                             fname.replace('.png', '_png.npz')), rec['K'])
            trans_cam_colmap = global_scale * romp['trans']
            root_world = cam_to_world(
                trans_cam_colmap[None, :], R_w2c, t_w2c)[0] + global_t
            if frame_idx < len(gt['transl']):
                gt_t = gt['transl'][frame_idx]
                err = float(np.linalg.norm(root_world - gt_t))
                errs.append(err)
                print(f'  {fname}: our_transl={root_world.tolist()}  '
                      f'gt={gt_t.tolist()}  dist={err:.3f} COLMAP units')
        if errs:
            print(f'  Mean transl error: {np.mean(errs):.3f} COLMAP units '
                  f'(vs GT, best prev was ~1.67)')

    # ── 5. Save JSON ─────────────────────────────────────────────────────────
    result = {
        'seq': args.seq,
        'global_scale': float(global_scale),
        'global_t_world': [float(x) for x in global_t.tolist()],
        'scale_init_from_depthpro': float(scale_init),
        'best_iter': int(best['iter']),
        'best_loss': float(best['loss']),
        'n_keyframes': n_kf_used,
        'n_keyframes_with_human_pts': n_with_human,
    }
    result_path = os.path.join(out_dir, 'coarse_align_v2.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {result_path}')

    # ── 6. Visualizations ────────────────────────────────────────────────────
    print('\n[4/5] Generating visualizations ...')
    for rec in key_records:
        fid  = rec['frame_name'].replace('.png', '')
        fvis = os.path.join(vis_dir, fid)
        os.makedirs(fvis, exist_ok=True)

        R_w2c = rec['R_w2c']
        t_w2c = rec['t_w2c']
        K     = rec['K']

        vc_before = rec['verts_cam']
        vw_before = cam_to_world(vc_before, R_w2c, t_w2c)

        vc_scaled = global_scale * vc_before
        vw_after  = cam_to_world(vc_scaled, R_w2c, t_w2c) + global_t
        vc_after  = world_to_cam(vw_after, R_w2c, t_w2c)

        save_camera_overlay(rec, vc_before, vc_after, global_scale, global_t, K,
                            os.path.join(fvis, 'camera_view.jpg'))
        save_3d_scatter(rec, vw_before, vw_after, global_scale,
                        os.path.join(fvis, '3d_scatter.png'))
        print(f'  vis: {fid}')

    save_loss_curve(log, os.path.join(out_dir, 'loss_curve.png'))

    panels = []
    for rec in key_records:
        fid = rec['frame_name'].replace('.png', '')
        p = cv2.imread(os.path.join(vis_dir, fid, 'camera_view.jpg'))
        if p is not None:
            panels.append(cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2)))
    if panels:
        cv2.imwrite(os.path.join(out_dir, 'summary.jpg'), np.concatenate(panels, axis=0))

    print(f'\n[DONE] All outputs in: {out_dir}')


if __name__ == '__main__':
    main()
