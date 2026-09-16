#!/usr/bin/env python3
"""
Coarse SMPL-to-scene alignment v3 — foot-contact COLMAP depth edition.

Core insight (borrowed from JOSH):
  SMPL foot depth (camera space, meters × scale)
      ≈ COLMAP scene depth at foot pixel (COLMAP units)
  → scale = colmap_depth_at_foot / smpl_foot_z_meters

This avoids the noisy background DepthPro calibration used in v2.
The COLMAP sparse points near the foot projection (floor texture)
give a direct, robust depth reference in COLMAP units.

Pipeline:
  1. Per keyframe: project SMPL foot region to image
     → find nearest COLMAP 3D points → get COLMAP depth at foot
     → scale_estimate = colmap_z_foot / smpl_foot_z_m
  2. scale_init = median(per-KF scale estimates)
  3. Joint optimization (shared scale + t_world across all KFs):
       loss_contact  — (scale * foot_z_m - colmap_z_foot)^2   [primary scale constraint]
       loss_chamfer  — bidirectional trimmed Chamfer, SMPL ↔ DepthPro human pts  [t_world]
       loss_proj     — soft projection IoU                      [t_world]
       loss_reg_*    — regularization

Output:
  <out_dir>/<seq>/coarse_align_v3.json
  <out_dir>/<seq>/vis/

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/coarse_align_smpl_neuman_v3.py \
      --data_root data/neuman/dataset --seq lab \
      --out_dir output/coarse_align_v3
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
# COLMAP I/O
# ─────────────────────────────────────────────────────────────────────────────

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float32)


def read_colmap_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
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
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
            cams[cam_id] = {'K': K, 'W': W, 'H': H}
    return cams


def read_colmap_images(path):
    frames = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 10: i += 1; continue
        q = list(map(float, p[1:5]))
        t = np.array(list(map(float, p[5:8])), dtype=np.float32)
        cam_id = int(p[8]); name = p[9]
        frames[name] = {'R_w2c': qvec2rotmat(q), 't_w2c': t, 'cam_id': cam_id}
        i += 2
    return frames


def read_colmap_points3d(path, max_pts=100_000):
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
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
    """Load ROMP, correct ROMP focal assumption (ROMP uses focal=cx)."""
    data = np.load(npz_path, allow_pickle=True)
    r = data['results'].item()
    trans = np.array(r['trans'], dtype=np.float32)
    trans[0] *= K[0, 2] / K[0, 0]   # x: cx/fx
    trans[1] *= K[1, 2] / K[1, 1]   # y: cy/fy
    return {
        'verts': np.array(r['verts'], dtype=np.float32),   # (6890,3) without trans
        'trans': trans,                                      # (3,) camera space meters
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cam_to_world(pts_cam, R_w2c, t_w2c):
    return (R_w2c.T @ (pts_cam - t_w2c).T).T

def world_to_cam(pts_world, R_w2c, t_w2c):
    return (R_w2c @ pts_world.T).T + t_w2c

def project_pts(pts_cam, K):
    """(N,3) camera-space → (N,2) pixel coords + valid mask."""
    valid = pts_cam[:, 2] > 0.01
    u = np.zeros(len(pts_cam)); v = np.zeros(len(pts_cam))
    z = pts_cam[valid, 2]
    u[valid] = pts_cam[valid, 0] / z * K[0, 0] + K[0, 2]
    v[valid] = pts_cam[valid, 1] / z * K[1, 1] + K[1, 2]
    return u, v, valid


# ─────────────────────────────────────────────────────────────────────────────
# ★ NEW — Foot-contact COLMAP depth
# ─────────────────────────────────────────────────────────────────────────────

def get_foot_region(verts_cam_m, K, mask_human, W, H,
                    bottom_frac=0.04, min_verts=20):
    """
    Find SMPL foot vertices by projecting to image and taking bottom N%.

    verts_cam_m: (6890, 3) SMPL verts in camera space, meters (verts + trans).

    Returns:
        foot_uv   — (2,) mean image pixel (u, v) of foot region
        foot_z_m  — scalar mean z in camera space (meters)
        n_verts   — number of foot verts found
    or None if not enough verts.
    """
    # Project all verts to image
    z = verts_cam_m[:, 2]
    valid_z = z > 0.1
    u_all = verts_cam_m[:, 0] / z * K[0, 0] + K[0, 2]
    v_all = verts_cam_m[:, 1] / z * K[1, 1] + K[1, 2]

    # Keep verts inside image bounds
    in_img = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H) & valid_z
    if in_img.sum() < min_verts:
        return None

    u_in = u_all[in_img]; v_in = v_all[in_img]; z_in = z[in_img]

    # Optional: further filter to human mask region
    if mask_human is not None:
        ui = u_in.astype(int).clip(0, W-1)
        vi = v_in.astype(int).clip(0, H-1)
        in_mask = mask_human[vi, ui]
        if in_mask.sum() >= min_verts:
            u_in = u_in[in_mask]; v_in = v_in[in_mask]; z_in = z_in[in_mask]

    if len(u_in) < min_verts:
        return None

    # Bottom fraction by image-v (large v = low on image = feet for upright person)
    n_foot = max(min_verts, int(len(v_in) * bottom_frac))
    foot_idx = np.argpartition(v_in, -n_foot)[-n_foot:]

    foot_u = u_in[foot_idx].mean()
    foot_v = v_in[foot_idx].mean()
    foot_z = z_in[foot_idx].mean()

    return np.array([foot_u, foot_v]), foot_z, len(foot_idx)


def get_colmap_depth_at_pixel(foot_uv, colmap_pts_world, R_w2c, t_w2c, K,
                               W, H, radius_px=40, min_pts=5,
                               depth_pct_lo=10, depth_pct_hi=90):
    """
    Find COLMAP 3D points projecting near foot_uv; return median camera-space z.

    Returns scalar (COLMAP units) or None.
    """
    # Project COLMAP world pts to camera
    colmap_cam = world_to_cam(colmap_pts_world, R_w2c, t_w2c)
    u_c, v_c, valid = project_pts(colmap_cam, K)

    in_img = valid & (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
    u_c = u_c[in_img]; v_c = v_c[in_img]
    z_c = colmap_cam[in_img, 2]

    dist2d = np.sqrt((u_c - foot_uv[0])**2 + (v_c - foot_uv[1])**2)
    near = dist2d < radius_px

    if near.sum() < min_pts:
        return None, near.sum()

    z_near = z_c[near]
    # Use percentile range to remove outliers (walls vs floor)
    lo, hi = np.percentile(z_near, [depth_pct_lo, depth_pct_hi])
    # Keep points closer than median (floor contact, not walls behind)
    z_floor = z_near[z_near <= np.median(z_near)]
    if len(z_floor) < 3:
        z_floor = z_near

    return float(np.median(z_floor)), near.sum()


def estimate_scale_from_foot_contact(verts_cam_m, colmap_pts_world,
                                      R_w2c, t_w2c, K, mask_human, W, H,
                                      radius_px=40, verbose=True):
    """
    Estimate scale (COLMAP units / meter) from foot-contact depth constraint.

    For a frame where the foot is on the ground:
        scale × smpl_foot_z_m  ≈  colmap_z_at_foot_pixel
    → scale_estimate = colmap_z_at_foot / smpl_foot_z_m

    Returns (scale_estimate, foot_uv, foot_z_m, colmap_z) or None.
    """
    result = get_foot_region(verts_cam_m, K, mask_human, W, H)
    if result is None:
        if verbose: print('    foot: not enough verts in image')
        return None

    foot_uv, foot_z_m, n_verts = result

    colmap_z, n_near = get_colmap_depth_at_pixel(
        foot_uv, colmap_pts_world, R_w2c, t_w2c, K, W, H, radius_px)

    if colmap_z is None:
        if verbose:
            print(f'    foot: foot_z={foot_z_m:.3f}m  '
                  f'foot_pixel=({foot_uv[0]:.0f},{foot_uv[1]:.0f})  '
                  f'colmap_near={n_near} (< min 5, skip)')
        return None

    scale_est = colmap_z / foot_z_m
    if verbose:
        print(f'    foot: foot_z={foot_z_m:.3f}m  '
              f'pixel=({foot_uv[0]:.0f},{foot_uv[1]:.0f})  '
              f'colmap_z={colmap_z:.3f}cu  n_near={n_near}  '
              f'→ scale_est={scale_est:.3f}')

    return scale_est, foot_uv, foot_z_m, colmap_z


# ─────────────────────────────────────────────────────────────────────────────
# DepthPro human pts (unchanged from v2, used for Chamfer / t_world)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_depthpro(depth_m, colmap_pts, R_w2c, t_w2c, K, W, H, mask_bg,
                       min_samples=30):
    """Background calibration: z_colmap = a * z_depthpro + b."""
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
    if len(u) < min_samples: return None
    z_dp = depth_m[v, u].astype(np.float64)
    valid = (z_dp > 0.01) & np.isfinite(z_dp) & (z_colmap > 0.01)
    if valid.sum() < min_samples: return None
    z_dp, z_colmap = z_dp[valid], z_colmap[valid].astype(np.float64)
    slope_est = np.median(z_colmap / z_dp)
    resid = z_colmap - slope_est * z_dp
    mad = np.median(np.abs(resid - np.median(resid)))
    inlier = np.abs(resid - np.median(resid)) < 3.0 * max(mad, 1e-6)
    if inlier.sum() < min_samples: inlier = np.ones(len(z_dp), dtype=bool)
    A = np.stack([z_dp[inlier], np.ones(inlier.sum())], axis=1)
    result = np.linalg.lstsq(A, z_colmap[inlier], rcond=None)
    return float(result[0][0]), float(result[0][1])


def extract_human_pts_depthpro(depth_m, calib, R_w2c, t_w2c, K, mask_human,
                                max_pts=4000, far_percentile=97.0, mad_k=3.5, seed=0):
    """Backproject DepthPro human-region depth → COLMAP world pts."""
    a, b = calib
    ys, xs = np.where(mask_human)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)
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
    x_cam = (xs - K[0, 2]) / K[0, 0] * z_colmap
    y_cam = (ys - K[1, 2]) / K[1, 1] * z_colmap
    pts_cam = np.stack([x_cam, y_cam, z_colmap], axis=1).astype(np.float32)
    z_vals = pts_cam[:, 2]
    z_thresh = np.percentile(z_vals, far_percentile)
    pts_cam = pts_cam[z_vals <= z_thresh]
    if len(pts_cam) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z_vals = pts_cam[:, 2]
    med_z = np.median(z_vals)
    mad = np.median(np.abs(z_vals - med_z))
    keep = np.abs(z_vals - med_z) <= mad_k * max(mad, 1e-4)
    pts_cam = pts_cam[keep]
    if len(pts_cam) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_world = cam_to_world(pts_cam, R_w2c, t_w2c)
    if len(pts_world) > max_pts:
        idx = rng.choice(len(pts_world), max_pts, replace=False)
        pts_world = pts_world[idx]
    return pts_world.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_trimmed(src, tgt, trim_ratio=0.10, chunk=2048):
    d_st_list = []
    for i in range(0, len(src), chunk):
        d = (src[i:i+chunk, None, :] - tgt[None, :, :]).pow(2).sum(-1)
        d_st_list.append(d.min(-1).values)
    d_st = torch.cat(d_st_list)
    d_ts_list = []
    for i in range(0, len(tgt), chunk):
        d = (tgt[i:i+chunk, None, :] - src[None, :, :]).pow(2).sum(-1)
        d_ts_list.append(d.min(-1).values)
    d_ts = torch.cat(d_ts_list)
    k_st = max(1, int(len(d_st) * (1.0 - trim_ratio)))
    k_ts = max(1, int(len(d_ts) * (1.0 - trim_ratio)))
    d_st_t = torch.topk(d_st, k_st, largest=False).values
    d_ts_t = torch.topk(d_ts, k_ts, largest=False).values
    return 0.5 * (d_st_t.mean() + d_ts_t.mean())


def soft_silhouette_loss(verts_cam_t, K, H, W, mask_t, sigma=3.0,
                          n_sample=800, device='cuda'):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = verts_cam_t[:, 2].clamp(min=0.1)
    u = verts_cam_t[:, 0] / z * fx + cx
    v = verts_cam_t[:, 1] / z * fy + cy
    perm = torch.randperm(len(u), device=device)[:n_sample]
    u_s = u[perm]; v_s = v[perm]
    scale_ds = 4
    h2, w2 = H // scale_ds, W // scale_ds
    yy = torch.arange(h2, device=device, dtype=torch.float32).view(h2, 1) * scale_ds
    xx = torch.arange(w2, device=device, dtype=torch.float32).view(1, w2) * scale_ds
    d2 = (xx.unsqueeze(2) - u_s)**2 + (yy.unsqueeze(2) - v_s)**2
    soft = torch.exp(-d2 / (2 * sigma**2)).sum(-1)
    soft = soft / (soft.max() + 1e-8)
    soft = soft.unsqueeze(0).unsqueeze(0)
    mask_ds = F.interpolate(mask_t, size=(h2, w2), mode='bilinear', align_corners=False)
    inter = (soft * mask_ds).sum()
    union = (soft + mask_ds - soft * mask_ds).sum()
    iou = inter / (union + 1e-8)
    return 1.0 - iou, float(iou.item())


# ─────────────────────────────────────────────────────────────────────────────
# Main optimization (v3: adds contact loss for scale)
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(key_records, n_iters, lr,
                     w_contact, w_chamfer, w_proj,
                     w_reg_scale, w_reg_trans,
                     trim_ratio, device, scale_init=8.0):
    """
    Joint optimization over all keyframes.

    key_records[i] now carries:
        foot_z_m          — SMPL foot z in cam space (meters), precomputed
        colmap_z_foot_cu  — COLMAP scene depth at foot pixel (COLMAP units)
        (plus all v2 fields: verts_cam_t, human_pts_t, mask_t, ...)

    Losses:
        contact: (scale * foot_z_m - colmap_z_foot)^2    ← primary scale driver
        chamfer: SMPL verts ↔ DepthPro human pts          ← t_world driver
        proj:    soft silhouette IoU                       ← t_world driver
    """
    log_scale = torch.nn.Parameter(
        torch.tensor([np.log(max(scale_init, 0.1))],
                     dtype=torch.float32, device=device))
    t_world = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([log_scale, t_world], lr=lr)

    best = {'loss': float('inf'), 'iter': -1,
            'scale': scale_init, 't_world': np.zeros(3, dtype=np.float32)}
    log = []

    for it in trange(n_iters, desc='Optimizing', ncols=80):
        optimizer.zero_grad()
        scale = torch.exp(log_scale)

        loss_contact_sum = torch.zeros(1, dtype=torch.float32, device=device)
        loss_chamfer_sum = torch.zeros(1, dtype=torch.float32, device=device)
        loss_proj_sum    = torch.zeros(1, dtype=torch.float32, device=device)
        n_contact = 0; iou_sum = 0.0; n_kf = 0

        for rec in key_records:
            R = rec['R_t']; t = rec['t_t']

            verts_cam_cu = scale * rec['verts_cam_t']            # (N,3) COLMAP cam
            verts_world  = (R.T @ (verts_cam_cu - t).T).T        # (N,3) COLMAP world
            verts_align  = verts_world + t_world                  # (N,3)

            # ── contact loss: foot z in COLMAP cam = target ──────────────
            if w_contact > 0 and rec.get('foot_z_m') is not None:
                foot_z_pred_cu = scale * rec['foot_z_m_t']       # scalar tensor
                loss_contact_sum = loss_contact_sum + (
                    foot_z_pred_cu - rec['colmap_z_foot_t']) ** 2
                n_contact += 1

            # ── chamfer: SMPL verts ↔ DepthPro human pts ─────────────────
            if w_chamfer > 0 and rec.get('human_pts_t') is not None:
                hp = rec['human_pts_t']
                if hp.shape[0] >= 32:
                    loss_chamfer_sum = loss_chamfer_sum + chamfer_trimmed(
                        verts_align, hp, trim_ratio=trim_ratio)

            # ── soft projection IoU ───────────────────────────────────────
            if w_proj > 0:
                verts_cam_proj = (R @ verts_align.T).T + t
                l_proj, iou = soft_silhouette_loss(
                    verts_cam_proj, rec['K'], rec['H'], rec['W'],
                    rec['mask_t'], device=device)
                loss_proj_sum = loss_proj_sum + l_proj
                iou_sum += iou

            n_kf += 1

        if n_kf == 0: continue
        nf = float(n_kf)
        nc = float(max(n_contact, 1))

        loss_contact = loss_contact_sum / nc
        loss_chamfer = loss_chamfer_sum / nf
        loss_proj    = loss_proj_sum    / nf

        reg_scale = (scale - torch.tensor(scale_init, device=device)) ** 2
        reg_trans = (t_world ** 2).sum()

        total = (w_contact  * loss_contact
               + w_chamfer  * loss_chamfer
               + w_proj     * loss_proj
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

        if (it == 0) or ((it+1) % 50 == 0) or (it+1 == n_iters):
            print(f'  iter {it+1:4d}/{n_iters}  loss={cur:.5f}'
                  f'  contact={float(loss_contact.item()):.5f}'
                  f'  chamfer={float(loss_chamfer.item()):.5f}'
                  f'  iou={iou_sum/nf:.3f}'
                  f'  scale={sc:.4f}  t={[round(x,3) for x in tw]}')

        log.append({'iter': it, 'loss': cur,
                    'contact': float(loss_contact.item()),
                    'chamfer': float(loss_chamfer.item()),
                    'proj_iou': iou_sum / nf,
                    'scale': sc, 't_world': tw})

    return best, log


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_pts_on_img(img, pts_cam, K, W, H, color, radius=2, max_draw=1500):
    out = img.copy()
    if len(pts_cam) == 0: return out
    idx = np.random.choice(len(pts_cam), min(max_draw, len(pts_cam)), replace=False)
    pc = pts_cam[idx]
    valid = pc[:, 2] > 0.01
    u = (pc[valid, 0] / pc[valid, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = (pc[valid, 1] / pc[valid, 2] * K[1, 1] + K[1, 2]).astype(int)
    for ui, vi in zip(u, v):
        if 0 <= ui < W and 0 <= vi < H:
            cv2.circle(out, (ui, vi), radius, color, -1)
    return out


def save_camera_overlay(rec, vc_before, vc_after, global_scale, global_t, K, out_path,
                         foot_uv=None):
    img = rec['img']
    if img is None:
        img = np.ones((rec['H'], rec['W'], 3), dtype=np.uint8) * 30
    R_w2c, t_w2c = rec['R_w2c'], rec['t_w2c']
    scene_cam = world_to_cam(rec['scene_pts_np'], R_w2c, t_w2c)
    b = draw_pts_on_img(img, scene_cam, K, rec['W'], rec['H'], (200,200,200), 1, 3000)
    a = b.copy()
    hp = rec.get('human_pts_np')
    if hp is not None and len(hp) > 0:
        hp_cam = world_to_cam(hp, R_w2c, t_w2c)
        b = draw_pts_on_img(b, hp_cam, K, rec['W'], rec['H'], (30,160,255), 2, 1500)
        a = draw_pts_on_img(a, hp_cam, K, rec['W'], rec['H'], (30,160,255), 2, 1500)
    b = draw_pts_on_img(b, vc_before, K, rec['W'], rec['H'], (0,0,220), 2, 600)
    a = draw_pts_on_img(a, vc_after,  K, rec['W'], rec['H'], (0,200,0),  2, 600)
    # Mark foot pixel
    if foot_uv is not None:
        fu, fv = int(foot_uv[0]), int(foot_uv[1])
        cv2.circle(b, (fu, fv), 8, (0, 255, 255), 2)
        cv2.circle(a, (fu, fv), 8, (0, 255, 255), 2)
    t_str = '[' + ', '.join(f'{x:.2f}' for x in global_t.tolist()) + ']'
    cv2.putText(b, 'Before  [SMPL=blue  scene=gray  human=orange  foot=cyan]',
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(a, f'After   scale={global_scale:.3f}  t={t_str}',
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imwrite(out_path, np.concatenate([b, a], axis=1))


def save_loss_curve(log, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    iters = [e['iter'] for e in log]
    for ax, key, title in zip(axes,
        ['loss', 'contact', 'chamfer', 'proj_iou', 'scale'],
        ['Total', 'Contact', 'Chamfer', 'ProjIoU', 'Scale']):
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
    p.add_argument('--out_dir', default='output/coarse_align_v3')
    p.add_argument('--n_keyframes', type=int, default=15,
                   help='More KFs → more contact constraints')
    p.add_argument('--n_iters', type=int, default=500)
    p.add_argument('--lr', type=float, default=3e-3)
    # Loss weights
    p.add_argument('--w_contact',   type=float, default=10.0,
                   help='Contact depth loss weight (primary scale driver)')
    p.add_argument('--w_chamfer',   type=float, default=1.0)
    p.add_argument('--w_proj',      type=float, default=3.0)
    p.add_argument('--w_reg_scale', type=float, default=0.02)
    p.add_argument('--w_reg_trans', type=float, default=0.001)
    p.add_argument('--trim_ratio',  type=float, default=0.10)
    # Contact detection
    p.add_argument('--contact_radius_px', type=int, default=50,
                   help='Search radius (pixels) for COLMAP pts near foot')
    p.add_argument('--contact_min_colmap', type=int, default=5,
                   help='Min COLMAP pts required to use a frame for contact loss')
    p.add_argument('--bottom_frac', type=float, default=0.04,
                   help='Bottom fraction of projected verts used as foot region')
    # DepthPro (for Chamfer human pts)
    p.add_argument('--max_human_pts', type=int, default=4000)
    p.add_argument('--max_scene_pts', type=int, default=100_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seq_dir   = os.path.join(args.data_root, args.seq)
    sparse_dir  = os.path.join(seq_dir, 'sparse')
    smpl_dir    = os.path.join(seq_dir, 'smpl_pred')
    seg_dir     = os.path.join(seq_dir, 'segmentations')
    img_dir     = os.path.join(seq_dir, 'images')
    depth_dir   = os.path.join(seq_dir, 'depth_pro')
    out_dir     = os.path.join(args.out_dir, args.seq)
    vis_dir     = os.path.join(out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # ── 1. COLMAP ─────────────────────────────────────────────────────────────
    print('\n[1/5] Loading COLMAP ...')
    colmap_cams   = read_colmap_cameras(os.path.join(sparse_dir, 'cameras.txt'))
    colmap_frames = read_colmap_images(os.path.join(sparse_dir, 'images.txt'))
    scene_pts_np  = read_colmap_points3d(
        os.path.join(sparse_dir, 'points3D.txt'), args.max_scene_pts)
    print(f'  {len(colmap_frames)} frames  |  {len(scene_pts_np)} 3D pts')

    # ── 2. Keyframe selection ─────────────────────────────────────────────────
    smpl_files = sorted(glob(os.path.join(smpl_dir, '*.npz')))
    frame_names = [os.path.basename(f).replace('_png.npz', '.png') for f in smpl_files]
    valid = [(n, sf) for n, sf in zip(frame_names, smpl_files) if n in colmap_frames]
    print(f'  {len(valid)} frames with SMPL + COLMAP')

    n_kf = min(args.n_keyframes, len(valid))
    kf_idx = np.round(np.linspace(0, len(valid)-1, n_kf)).astype(int)
    keyframes = [valid[i] for i in kf_idx]
    print(f'  Keyframes ({n_kf}): {[kf[0] for kf in keyframes]}')

    # ── 3. Per-keyframe records ───────────────────────────────────────────────
    print('\n[2/5] Building keyframe records ...')
    key_records = []
    contact_scale_estimates = []

    for frame_name, smpl_path in keyframes:
        cf    = colmap_frames[frame_name]
        K     = colmap_cams[cf['cam_id']]['K']
        W     = colmap_cams[cf['cam_id']]['W']
        H     = colmap_cams[cf['cam_id']]['H']
        R_w2c = cf['R_w2c']
        t_w2c = cf['t_w2c']

        # SMPL in camera space (meters)
        romp = load_romp(smpl_path, K)
        verts_cam_m = romp['verts'] + romp['trans'][None, :]   # (6890, 3)

        # Segmentation mask
        seg = cv2.imread(os.path.join(seg_dir, frame_name), cv2.IMREAD_GRAYSCALE)
        if seg is None: seg = np.zeros((H, W), dtype=np.uint8)
        mask_human = seg < 128
        mask_bg    = seg > 127

        # ── ★ foot contact scale estimate ────────────────────────────────────
        print(f'  {frame_name}:')
        foot_info = estimate_scale_from_foot_contact(
            verts_cam_m, scene_pts_np, R_w2c, t_w2c, K, mask_human, W, H,
            radius_px=args.contact_radius_px, verbose=True)

        foot_z_m = None; colmap_z_foot = None; foot_uv_np = None
        if foot_info is not None:
            scale_est, foot_uv_np, foot_z_m, colmap_z_foot = foot_info
            # Sanity: scale should be in [3, 20] for NeuMan
            if 3.0 < scale_est < 20.0:
                contact_scale_estimates.append(scale_est)
            else:
                print(f'    → scale_est {scale_est:.2f} out of range, skipped')
                foot_z_m = None

        # ── DepthPro human pts (for Chamfer, t_world) ─────────────────────────
        stem = os.path.splitext(frame_name)[0]
        dp_path = os.path.join(depth_dir, stem + '.npy')
        human_pts_np = np.zeros((0, 3), dtype=np.float32)
        if os.path.exists(dp_path):
            depth_m = np.load(dp_path).astype(np.float32)
            calib = calibrate_depthpro(
                depth_m, scene_pts_np, R_w2c, t_w2c, K, W, H, mask_bg)
            if calib is not None:
                human_pts_np = extract_human_pts_depthpro(
                    depth_m, calib, R_w2c, t_w2c, K, mask_human,
                    max_pts=args.max_human_pts, seed=args.seed)
                print(f'    depthpro calib a={calib[0]:.3f}  human_pts={len(human_pts_np)}')

        img = cv2.imread(os.path.join(img_dir, frame_name))

        rec = {
            'frame_name': frame_name,
            'K': K, 'W': W, 'H': H,
            'R_w2c': R_w2c, 't_w2c': t_w2c,
            'R_t': torch.from_numpy(R_w2c).to(device),
            't_t': torch.from_numpy(t_w2c).to(device),
            'verts_cam_t': torch.from_numpy(verts_cam_m).to(device),    # (6890,3) meters
            'mask_t': torch.from_numpy(mask_human.astype(np.float32)).to(device
                       ).unsqueeze(0).unsqueeze(0),
            'human_pts_np': human_pts_np,
            'human_pts_t': (torch.from_numpy(human_pts_np).to(device)
                            if len(human_pts_np) >= 32 else None),
            'scene_pts_np': scene_pts_np,
            'img': img,
            'foot_uv': foot_uv_np,
            # contact tensors
            'foot_z_m': foot_z_m,
            'foot_z_m_t': (torch.tensor(foot_z_m, dtype=torch.float32, device=device)
                           if foot_z_m is not None else None),
            'colmap_z_foot_t': (torch.tensor(colmap_z_foot, dtype=torch.float32, device=device)
                                if colmap_z_foot is not None else None),
        }
        key_records.append(rec)

    if not key_records:
        raise RuntimeError('No valid keyframes.')

    # ── 4. Scale init ─────────────────────────────────────────────────────────
    print(f'\n[3/5] Scale estimation from foot contact:')
    print(f'  Contact estimates: {[f"{s:.3f}" for s in contact_scale_estimates]}')

    if contact_scale_estimates:
        scale_init = float(np.median(contact_scale_estimates))
        scale_std  = float(np.std(contact_scale_estimates))
        print(f'  → scale_init = {scale_init:.4f}  (std={scale_std:.3f})'
              f'  [GT for NeuMan lab ≈ 8.617]')
    else:
        scale_init = 8.0
        print(f'  → No contact estimates, fallback scale_init = {scale_init}')

    n_contact_kf = sum(1 for r in key_records if r['foot_z_m'] is not None)
    n_human_kf   = sum(1 for r in key_records if r['human_pts_t'] is not None)
    print(f'\n[4/5] Optimizing: {len(key_records)} KFs  '
          f'({n_contact_kf} contact, {n_human_kf} with human pts), {args.n_iters} iters')
    print(f'  w_contact={args.w_contact}  w_chamfer={args.w_chamfer}  '
          f'w_proj={args.w_proj}  scale_init={scale_init:.4f}')

    # ── 5. Optimization ───────────────────────────────────────────────────────
    best, log = run_optimization(
        key_records, args.n_iters, args.lr,
        args.w_contact, args.w_chamfer, args.w_proj,
        args.w_reg_scale, args.w_reg_trans,
        args.trim_ratio, device, scale_init=scale_init)

    global_scale = best['scale']
    global_t     = best['t_world']

    print(f'\n[Result]')
    print(f'  global_scale   = {global_scale:.4f}')
    print(f'  global_t_world = {global_t.tolist()}')
    print(f'  best_iter={best["iter"]}  best_loss={best["loss"]:.5f}')

    # ── GT comparison ─────────────────────────────────────────────────────────
    gt_path = os.path.join(seq_dir, '4d_humans', 'smpl_optimized_aligned_scale.npz')
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        gt_scale = float(gt['scale'].mean())
        print(f'\n[GT comparison]')
        print(f'  scale: ours={global_scale:.4f}  GT={gt_scale:.4f}  '
              f'err={abs(global_scale-gt_scale)/gt_scale*100:.1f}%')
        errs = []
        for rec in key_records:
            fname = rec['frame_name']
            frame_idx = int(os.path.splitext(fname)[0])
            romp = load_romp(
                os.path.join(smpl_dir, fname.replace('.png', '_png.npz')), rec['K'])
            trans_cam_cu = global_scale * romp['trans']
            root_world   = cam_to_world(trans_cam_cu[None], rec['R_w2c'], rec['t_w2c'])[0] + global_t
            if frame_idx < len(gt['transl']):
                gt_t = gt['transl'][frame_idx]
                err  = float(np.linalg.norm(root_world - gt_t))
                errs.append(err)
                print(f'  {fname}: dist={err:.3f} cu')
        if errs:
            print(f'  Mean transl error: {np.mean(errs):.3f} cu  '
                  f'median={np.median(errs):.3f}')

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        'seq': args.seq,
        'global_scale': float(global_scale),
        'global_t_world': [float(x) for x in global_t.tolist()],
        'scale_init_from_contact': float(scale_init),
        'n_contact_keyframes': n_contact_kf,
        'contact_scale_estimates': contact_scale_estimates,
        'best_iter': int(best['iter']),
        'best_loss': float(best['loss']),
    }
    result_path = os.path.join(out_dir, 'coarse_align_v3.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {result_path}')

    # ── Visualizations ────────────────────────────────────────────────────────
    print('\n[5/5] Generating visualizations ...')
    panels = []
    for rec in key_records:
        fid  = rec['frame_name'].replace('.png', '')
        fvis = os.path.join(vis_dir, fid)
        os.makedirs(fvis, exist_ok=True)

        vc_before = rec['verts_cam_t'].cpu().numpy()
        vw_before = cam_to_world(vc_before, rec['R_w2c'], rec['t_w2c'])

        vc_scaled = global_scale * vc_before
        vw_after  = cam_to_world(vc_scaled, rec['R_w2c'], rec['t_w2c']) + global_t
        vc_after  = world_to_cam(vw_after, rec['R_w2c'], rec['t_w2c'])

        save_camera_overlay(
            rec, vc_before, vc_after, global_scale, global_t, rec['K'],
            os.path.join(fvis, 'camera_view.jpg'), foot_uv=rec.get('foot_uv'))

        p = cv2.imread(os.path.join(fvis, 'camera_view.jpg'))
        if p is not None:
            panels.append(cv2.resize(p, (p.shape[1]//2, p.shape[0]//2)))

    save_loss_curve(log, os.path.join(out_dir, 'loss_curve.png'))
    if panels:
        cv2.imwrite(os.path.join(out_dir, 'summary.jpg'), np.concatenate(panels, axis=0))

    print(f'\n[DONE] All outputs in: {out_dir}')


if __name__ == '__main__':
    main()
