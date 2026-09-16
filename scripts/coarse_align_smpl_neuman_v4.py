#!/usr/bin/env python3
"""
Coarse SMPL-to-scene alignment v4 — DepthPro fallback for scale.

Difference from v3:
  v3: uses foot-contact COLMAP depth as primary scale constraint.
      When n_contact=0 (bike/jogging/seattle/citron), falls back to scale_init=8.0.
  v4: when n_contact=0, falls back to DepthPro calibrated body-center depth.
      scale_fallback = (a * dp_body_z_m + b) / romp_trans_z_m
      where (a,b) is the per-frame background DepthPro calibration already
      computed for the Chamfer human points.

Pipeline:
  1. Per keyframe: try foot-contact COLMAP scale estimate (same as v3).
  2. If foot contact fails, try DepthPro body-center scale estimate.
  3. scale_init = median(all valid estimates, contact preferred).
  4. Joint optimization with contact loss (if available) or depthpro_body
     loss (fallback), plus Chamfer and soft IoU.

Usage:
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/coarse_align_smpl_neuman_v4.py \\
      --data_root data/neuman/dataset --seq bike \\
      --out_dir output/coarse_align_v4
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
    trans[0] *= K[0, 2] / K[0, 0]
    trans[1] *= K[1, 2] / K[1, 1]
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

def project_pts(pts_cam, K):
    valid = pts_cam[:, 2] > 0.01
    u = np.zeros(len(pts_cam)); v = np.zeros(len(pts_cam))
    z = pts_cam[valid, 2]
    u[valid] = pts_cam[valid, 0] / z * K[0, 0] + K[0, 2]
    v[valid] = pts_cam[valid, 1] / z * K[1, 1] + K[1, 2]
    return u, v, valid


# ─────────────────────────────────────────────────────────────────────────────
# Foot-contact COLMAP depth (same as v3)
# ─────────────────────────────────────────────────────────────────────────────

def get_foot_region(verts_cam_m, K, mask_human, W, H,
                    bottom_frac=0.04, min_verts=20):
    z = verts_cam_m[:, 2]
    valid_z = z > 0.1
    u_all = verts_cam_m[:, 0] / z * K[0, 0] + K[0, 2]
    v_all = verts_cam_m[:, 1] / z * K[1, 1] + K[1, 2]
    in_img = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H) & valid_z
    if in_img.sum() < min_verts:
        return None
    u_in = u_all[in_img]; v_in = v_all[in_img]; z_in = z[in_img]
    if mask_human is not None:
        ui = u_in.astype(int).clip(0, W-1)
        vi = v_in.astype(int).clip(0, H-1)
        in_mask = mask_human[vi, ui]
        if in_mask.sum() >= min_verts:
            u_in = u_in[in_mask]; v_in = v_in[in_mask]; z_in = z_in[in_mask]
    if len(u_in) < min_verts:
        return None
    n_foot = max(min_verts, int(len(v_in) * bottom_frac))
    foot_idx = np.argpartition(v_in, -n_foot)[-n_foot:]
    return np.array([u_in[foot_idx].mean(), v_in[foot_idx].mean()]), z_in[foot_idx].mean(), len(foot_idx)


def get_colmap_depth_at_pixel(foot_uv, colmap_pts_world, R_w2c, t_w2c, K,
                               W, H, radius_px=40, min_pts=5):
    colmap_cam = world_to_cam(colmap_pts_world, R_w2c, t_w2c)
    u_c, v_c, valid = project_pts(colmap_cam, K)
    in_img = valid & (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
    u_c = u_c[in_img]; v_c = v_c[in_img]; z_c = colmap_cam[in_img, 2]
    dist2d = np.sqrt((u_c - foot_uv[0])**2 + (v_c - foot_uv[1])**2)
    near = dist2d < radius_px
    if near.sum() < min_pts:
        return None, near.sum()
    z_near = z_c[near]
    z_floor = z_near[z_near <= np.median(z_near)]
    if len(z_floor) < 3:
        z_floor = z_near
    return float(np.median(z_floor)), near.sum()


def estimate_scale_from_foot_contact(verts_cam_m, colmap_pts_world,
                                      R_w2c, t_w2c, K, mask_human, W, H,
                                      radius_px=40, verbose=True):
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
# ★ NEW v4 — DepthPro body-center scale estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_scale_from_depthpro_body(romp_trans, depth_m, calib, mask_human,
                                       pct_lo=20, pct_hi=80, min_valid=30,
                                       scale_range=(1.0, 15.0), verbose=True):
    """
    Fallback scale when foot contact fails.

    Calibration: z_colmap = a * z_depthpro_m + b  (fit on background pixels)
    Human body depth: robust median of calibrated DepthPro in human mask (IQR).
    Scale: z_body_colmap / romp_trans_z_m

    Returns (scale_est, z_body_colmap, romp_trans_z_m) or None.
    """
    if calib is None:
        return None
    a, b = calib
    romp_trans_z_m = float(romp_trans[2])
    if romp_trans_z_m <= 0.1:
        return None

    ys, xs = np.where(mask_human)
    if len(ys) < min_valid:
        return None

    z_dp = depth_m[ys, xs].astype(np.float32)
    valid = (z_dp > 0.1) & np.isfinite(z_dp)
    z_dp_valid = z_dp[valid]
    if len(z_dp_valid) < min_valid:
        return None

    # IQR to avoid head / feet / boundary contamination
    lo = np.percentile(z_dp_valid, pct_lo)
    hi = np.percentile(z_dp_valid, pct_hi)
    z_dp_iqr = z_dp_valid[(z_dp_valid >= lo) & (z_dp_valid <= hi)]
    if len(z_dp_iqr) < 10:
        z_dp_iqr = z_dp_valid

    z_dp_body = float(np.median(z_dp_iqr))
    z_body_colmap = a * z_dp_body + b

    if z_body_colmap <= 0:
        if verbose:
            print(f'    depthpro: calibrated z={z_body_colmap:.3f} <= 0, skip')
        return None

    scale_est = z_body_colmap / romp_trans_z_m
    if not (scale_range[0] < scale_est < scale_range[1]):
        if verbose:
            print(f'    depthpro: scale_est={scale_est:.3f} out of range '
                  f'{scale_range}, skip')
        return None

    if verbose:
        print(f'    depthpro: a={a:.3f} b={b:.3f}  dp_z={z_dp_body:.3f}m  '
              f'body_z_cu={z_body_colmap:.3f}  romp_z={romp_trans_z_m:.3f}m  '
              f'n_px={len(z_dp_valid)}  → scale_est={scale_est:.3f}')

    return scale_est, z_body_colmap, romp_trans_z_m


# ─────────────────────────────────────────────────────────────────────────────
# DepthPro human pts (unchanged from v3)
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
# Main optimization (v4: calib_a + depthpro_body fallbacks for scale)
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(key_records, n_iters, lr,
                     w_contact, w_chamfer, w_proj, w_depthpro, w_calib_a,
                     w_reg_scale, w_reg_trans,
                     trim_ratio, device, scale_init=8.0, use_contact=True):
    """
    Joint optimization over all keyframes.

    Scale constraints (priority per frame, mutually exclusive):
      1. contact  — foot COLMAP depth vs scale * foot_z_m     [best accuracy]
      2. calib_a  — background DepthPro calib slope vs scale  [when contact biased]
      3. depthpro — body DepthPro depth vs scale * romp_z_m   [last resort]
    t_world:
      - chamfer: SMPL verts ↔ DepthPro human pts
      - proj:    soft silhouette IoU
    use_contact=False when contact estimates are inconsistent with calib_a.
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

        loss_contact_sum  = torch.zeros(1, dtype=torch.float32, device=device)
        loss_calib_a_sum  = torch.zeros(1, dtype=torch.float32, device=device)
        loss_depthpro_sum = torch.zeros(1, dtype=torch.float32, device=device)
        loss_chamfer_sum  = torch.zeros(1, dtype=torch.float32, device=device)
        loss_proj_sum     = torch.zeros(1, dtype=torch.float32, device=device)
        n_contact = 0; n_calib_a = 0; n_depthpro = 0; iou_sum = 0.0; n_kf = 0

        for rec in key_records:
            R = rec['R_t']; t = rec['t_t']

            verts_cam_cu = scale * rec['verts_cam_t']
            verts_world  = (R.T @ (verts_cam_cu - t).T).T
            verts_align  = verts_world + t_world

            # ── 1. Contact loss (foot COLMAP depth) ──────────────────────
            if use_contact and w_contact > 0 and rec.get('foot_z_m') is not None:
                foot_z_pred_cu = scale * rec['foot_z_m_t']
                loss_contact_sum = loss_contact_sum + (
                    foot_z_pred_cu - rec['colmap_z_foot_t']) ** 2
                n_contact += 1

            # ── 2. Calib-a loss: background DepthPro slope ≈ scale ───────
            # Active when foot contact is disabled (biased) or unavailable
            elif w_calib_a > 0 and rec.get('calib_a_t') is not None:
                loss_calib_a_sum = loss_calib_a_sum + (
                    scale - rec['calib_a_t']) ** 2
                n_calib_a += 1

            # ── 3. DepthPro body fallback ─────────────────────────────────
            elif w_depthpro > 0 and rec.get('depthpro_body_z_cu') is not None:
                body_z_pred_cu = scale * rec['romp_trans_z_m_t']
                loss_depthpro_sum = loss_depthpro_sum + (
                    body_z_pred_cu - rec['depthpro_body_z_cu_t']) ** 2
                n_depthpro += 1

            # ── 4. Chamfer: SMPL verts ↔ DepthPro human pts ──────────────
            if w_chamfer > 0 and rec.get('human_pts_t') is not None:
                hp = rec['human_pts_t']
                if hp.shape[0] >= 32:
                    loss_chamfer_sum = loss_chamfer_sum + chamfer_trimmed(
                        verts_align, hp, trim_ratio=trim_ratio)

            # ── 5. Soft silhouette IoU ────────────────────────────────────
            if w_proj > 0:
                verts_cam_proj = (R @ verts_align.T).T + t
                l_proj, iou = soft_silhouette_loss(
                    verts_cam_proj, rec['K'], rec['H'], rec['W'],
                    rec['mask_t'], device=device)
                loss_proj_sum = loss_proj_sum + l_proj
                iou_sum += iou

            n_kf += 1

        if n_kf == 0: continue
        nf  = float(n_kf)
        nc  = float(max(n_contact, 1))
        nca = float(max(n_calib_a, 1))
        nd  = float(max(n_depthpro, 1))

        loss_contact  = loss_contact_sum  / nc
        loss_calib_a  = loss_calib_a_sum  / nca
        loss_depthpro = loss_depthpro_sum / nd
        loss_chamfer  = loss_chamfer_sum  / nf
        loss_proj     = loss_proj_sum     / nf

        reg_scale = (scale - torch.tensor(scale_init, device=device)) ** 2
        reg_trans = (t_world ** 2).sum()

        total = (w_contact  * loss_contact
               + w_calib_a  * loss_calib_a
               + w_depthpro * loss_depthpro
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
                  f'  contact={float(loss_contact.item()):.4f}'
                  f'  calib_a={float(loss_calib_a.item()):.4f}'
                  f'  chamfer={float(loss_chamfer.item()):.4f}'
                  f'  iou={iou_sum/nf:.3f}'
                  f'  scale={sc:.4f}  t={[round(x,3) for x in tw]}')

        log.append({'iter': it, 'loss': cur,
                    'contact': float(loss_contact.item()),
                    'calib_a': float(loss_calib_a.item()),
                    'depthpro': float(loss_depthpro.item()),
                    'chamfer': float(loss_chamfer.item()),
                    'proj_iou': iou_sum / nf,
                    'scale': sc, 't_world': tw})

    return best, log


# ─────────────────────────────────────────────────────────────────────────────
# Visualization (unchanged from v3)
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
    fig, axes = plt.subplots(1, 7, figsize=(28, 4))
    iters = [e['iter'] for e in log]
    for ax, key, title in zip(axes,
        ['loss', 'contact', 'calib_a', 'depthpro', 'chamfer', 'proj_iou', 'scale'],
        ['Total', 'Contact', 'CalibA', 'DepthPro', 'Chamfer', 'ProjIoU', 'Scale']):
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
    p.add_argument('--out_dir', default='output/coarse_align_v4')
    p.add_argument('--n_keyframes', type=int, default=15)
    p.add_argument('--n_iters', type=int, default=500)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--w_contact',   type=float, default=10.0)
    p.add_argument('--w_calib_a',   type=float, default=8.0,
                   help='Calib-a loss weight: background DepthPro slope as scale constraint')
    p.add_argument('--w_depthpro',  type=float, default=5.0,
                   help='DepthPro body-center loss weight (last resort)')
    p.add_argument('--w_chamfer',   type=float, default=1.0)
    p.add_argument('--w_proj',      type=float, default=3.0)
    p.add_argument('--w_reg_scale', type=float, default=0.02)
    p.add_argument('--w_reg_trans', type=float, default=0.001)
    p.add_argument('--trim_ratio',  type=float, default=0.10)
    p.add_argument('--contact_calib_discrepancy_thr', type=float, default=0.20,
                   help='If |contact_median - calib_a_median| / calib_a > thr, '
                        'treat foot contact as biased and switch to calib_a')
    p.add_argument('--contact_radius_px', type=int, default=50)
    p.add_argument('--contact_min_colmap', type=int, default=5)
    p.add_argument('--bottom_frac', type=float, default=0.04)
    p.add_argument('--depthpro_pct_lo', type=float, default=20.0,
                   help='Lower percentile for IQR body depth (remove feet/lower body)')
    p.add_argument('--depthpro_pct_hi', type=float, default=80.0,
                   help='Upper percentile for IQR body depth (remove head)')
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
    calib_a_estimates = []
    depthpro_scale_estimates = []

    for frame_name, smpl_path in keyframes:
        cf    = colmap_frames[frame_name]
        K     = colmap_cams[cf['cam_id']]['K']
        W     = colmap_cams[cf['cam_id']]['W']
        H     = colmap_cams[cf['cam_id']]['H']
        R_w2c = cf['R_w2c']
        t_w2c = cf['t_w2c']

        romp = load_romp(smpl_path, K)
        verts_cam_m = romp['verts'] + romp['trans'][None, :]

        seg = cv2.imread(os.path.join(seg_dir, frame_name), cv2.IMREAD_GRAYSCALE)
        if seg is None: seg = np.zeros((H, W), dtype=np.uint8)
        mask_human = seg < 128
        mask_bg    = seg > 127

        print(f'  {frame_name}:')

        # ── Foot contact scale estimate ────────────────────────────────────
        foot_z_m = None; colmap_z_foot = None; foot_uv_np = None
        foot_info = estimate_scale_from_foot_contact(
            verts_cam_m, scene_pts_np, R_w2c, t_w2c, K, mask_human, W, H,
            radius_px=args.contact_radius_px, verbose=True)
        if foot_info is not None:
            scale_est, foot_uv_np, foot_z_m, colmap_z_foot = foot_info
            # Range 0.5~20: accepts all NeuMan scenes (min GT scale is bike=1.53)
            if 0.5 < scale_est < 20.0:
                contact_scale_estimates.append(scale_est)
            else:
                print(f'    → foot scale_est {scale_est:.2f} out of range, skipped')
                foot_z_m = None

        # ── DepthPro: calibrate + extract human pts ────────────────────────
        stem = os.path.splitext(frame_name)[0]
        dp_path = os.path.join(depth_dir, stem + '.npy')
        human_pts_np = np.zeros((0, 3), dtype=np.float32)
        calib = None
        calib_a_val          = None
        depthpro_body_z_cu   = None
        romp_trans_z_m_val   = None

        if os.path.exists(dp_path):
            depth_m = np.load(dp_path).astype(np.float32)
            calib = calibrate_depthpro(
                depth_m, scene_pts_np, R_w2c, t_w2c, K, W, H, mask_bg)
            if calib is not None:
                human_pts_np = extract_human_pts_depthpro(
                    depth_m, calib, R_w2c, t_w2c, K, mask_human,
                    max_pts=args.max_human_pts, seed=args.seed)

                # ── ★ v4: calib slope a ≈ scale (primary new signal) ──────
                a_val = float(calib[0])
                if 0.5 < a_val < 20.0:
                    calib_a_estimates.append(a_val)
                    calib_a_val = a_val
                    print(f'    depthpro calib a={calib[0]:.3f} b={calib[1]:.3f}  '
                          f'calib_a_scale={a_val:.3f}  human_pts={len(human_pts_np)}')
                else:
                    print(f'    depthpro calib a={calib[0]:.3f} out of range, skip')

                # ── DepthPro body depth (last-resort fallback) ─────────────
                dp_body = estimate_scale_from_depthpro_body(
                    romp['trans'], depth_m, calib, mask_human,
                    pct_lo=args.depthpro_pct_lo, pct_hi=args.depthpro_pct_hi,
                    verbose=True)
                if dp_body is not None:
                    dp_scale, z_body_cu, z_romp_m = dp_body
                    depthpro_scale_estimates.append(dp_scale)
                    depthpro_body_z_cu = z_body_cu
                    romp_trans_z_m_val = z_romp_m

        img = cv2.imread(os.path.join(img_dir, frame_name))

        rec = {
            'frame_name': frame_name,
            'K': K, 'W': W, 'H': H,
            'R_w2c': R_w2c, 't_w2c': t_w2c,
            'R_t': torch.from_numpy(R_w2c).to(device),
            't_t': torch.from_numpy(t_w2c).to(device),
            'verts_cam_t': torch.from_numpy(verts_cam_m).to(device),
            'mask_t': torch.from_numpy(mask_human.astype(np.float32)).to(device
                       ).unsqueeze(0).unsqueeze(0),
            'human_pts_np': human_pts_np,
            'human_pts_t': (torch.from_numpy(human_pts_np).to(device)
                            if len(human_pts_np) >= 32 else None),
            'scene_pts_np': scene_pts_np,
            'img': img,
            'foot_uv': foot_uv_np,
            # foot contact tensors
            'foot_z_m': foot_z_m,
            'foot_z_m_t': (torch.tensor(foot_z_m, dtype=torch.float32, device=device)
                           if foot_z_m is not None else None),
            'colmap_z_foot_t': (torch.tensor(colmap_z_foot, dtype=torch.float32, device=device)
                                if colmap_z_foot is not None else None),
            # calib_a: background DepthPro slope ≈ scale
            'calib_a': calib_a_val,
            'calib_a_t': (torch.tensor(calib_a_val, dtype=torch.float32, device=device)
                          if calib_a_val is not None else None),
            # depthpro body fallback tensors
            'depthpro_body_z_cu': depthpro_body_z_cu,
            'depthpro_body_z_cu_t': (torch.tensor(depthpro_body_z_cu,
                                                    dtype=torch.float32, device=device)
                                      if depthpro_body_z_cu is not None else None),
            'romp_trans_z_m_t': (torch.tensor(romp_trans_z_m_val,
                                               dtype=torch.float32, device=device)
                                  if romp_trans_z_m_val is not None else None),
        }
        key_records.append(rec)

    if not key_records:
        raise RuntimeError('No valid keyframes.')

    # ── 4. Scale init ─────────────────────────────────────────────────────────
    print(f'\n[3/5] Scale estimation:')
    print(f'  Contact estimates  ({len(contact_scale_estimates)}): '
          f'{[f"{s:.3f}" for s in contact_scale_estimates]}')
    print(f'  CalibA  estimates  ({len(calib_a_estimates)}): '
          f'{[f"{s:.3f}" for s in calib_a_estimates]}')
    print(f'  DepthPro estimates ({len(depthpro_scale_estimates)}): '
          f'{[f"{s:.3f}" for s in depthpro_scale_estimates]}')

    # ── Detect foot-contact bias by comparing with calib_a ────────────────────
    # Physical reason: when COLMAP has no ground-level points near foot pixels
    # (e.g. jogging on concrete), foot contact returns depths from walls/objects
    # at closer range → scale underestimated. Calib-a (background DepthPro slope)
    # measures the global COLMAP/metric ratio and is unaffected by this bias.
    use_contact = True
    scale_method = 'unknown'

    if contact_scale_estimates and calib_a_estimates:
        contact_med = float(np.median(contact_scale_estimates))
        calib_a_med = float(np.median(calib_a_estimates))
        discrepancy = abs(contact_med - calib_a_med) / (calib_a_med + 1e-6)
        print(f'  Contact median={contact_med:.4f}  CalibA median={calib_a_med:.4f}  '
              f'discrepancy={discrepancy*100:.1f}%  thr={args.contact_calib_discrepancy_thr*100:.0f}%')

        if discrepancy > args.contact_calib_discrepancy_thr:
            print(f'  ⚠  Foot contact biased (>{args.contact_calib_discrepancy_thr*100:.0f}% from calib_a)'
                  f' → switching to calib_a')
            scale_init  = calib_a_med
            scale_method = 'calib_a'
            use_contact  = False
        else:
            scale_init  = contact_med
            scale_method = 'foot_contact'
            print(f'  → foot-contact consistent with calib_a, using foot_contact')

    elif contact_scale_estimates:
        scale_init   = float(np.median(contact_scale_estimates))
        scale_method = 'foot_contact'
        print(f'  → foot_contact (no calib_a available): scale_init={scale_init:.4f}')

    elif calib_a_estimates:
        scale_init   = float(np.median(calib_a_estimates))
        scale_method = 'calib_a'
        use_contact  = False
        print(f'  → calib_a (no contact): scale_init={scale_init:.4f}')

    elif depthpro_scale_estimates:
        scale_init   = float(np.median(depthpro_scale_estimates))
        scale_method = 'depthpro_body'
        use_contact  = False
        print(f'  → depthpro_body (last resort): scale_init={scale_init:.4f}')

    else:
        scale_init   = 8.0
        scale_method = 'hardcoded_fallback'
        use_contact  = False
        print(f'  → hardcoded fallback: scale_init={scale_init}')

    n_contact_kf  = sum(1 for r in key_records if r['foot_z_m'] is not None)
    n_calib_a_kf  = sum(1 for r in key_records if r['calib_a'] is not None)
    n_depthpro_kf = sum(1 for r in key_records if r['depthpro_body_z_cu'] is not None)
    n_human_kf    = sum(1 for r in key_records if r['human_pts_t'] is not None)
    print(f'\n[4/5] Optimizing: {len(key_records)} KFs  '
          f'({n_contact_kf} contact, {n_calib_a_kf} calib_a, '
          f'{n_depthpro_kf} dp_body, {n_human_kf} human_pts), '
          f'{args.n_iters} iters')
    print(f'  scale_method={scale_method}  use_contact={use_contact}  '
          f'scale_init={scale_init:.4f}')
    print(f'  w_contact={args.w_contact}  w_calib_a={args.w_calib_a}  '
          f'w_depthpro={args.w_depthpro}  w_chamfer={args.w_chamfer}  '
          f'w_proj={args.w_proj}')

    # ── 5. Optimization ───────────────────────────────────────────────────────
    # When using calib_a for scale, DepthPro human pts are also depth-biased
    # (same scene-specific DepthPro error), so Chamfer would pull scale wrong.
    # Keep only IoU for t_world in that case.
    w_chamfer_eff = 0.0 if (scale_method == 'calib_a') else args.w_chamfer

    best, log = run_optimization(
        key_records, args.n_iters, args.lr,
        args.w_contact, w_chamfer_eff, args.w_proj, args.w_depthpro,
        args.w_calib_a,
        args.w_reg_scale, args.w_reg_trans,
        args.trim_ratio, device,
        scale_init=scale_init, use_contact=use_contact)

    global_scale = best['scale']
    global_t     = best['t_world']

    print(f'\n[Result]')
    print(f'  global_scale   = {global_scale:.4f}  (method: {scale_method})')
    print(f'  global_t_world = {global_t.tolist()}')
    print(f'  best_iter={best["iter"]}  best_loss={best["loss"]:.5f}')

    # ── GT comparison ─────────────────────────────────────────────────────────
    gt_path = os.path.join(seq_dir, '4d_humans', 'smpl_optimized_aligned_scale.npz')
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        gt_scale = float(gt['scale'].mean())
        print(f'\n[GT comparison]')
        print(f'  scale: ours={global_scale:.4f}  GT={gt_scale:.4f}  '
              f'ratio={global_scale/gt_scale:.3f}  '
              f'err={abs(global_scale-gt_scale)/gt_scale*100:.1f}%')

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        'seq': args.seq,
        'global_scale': float(global_scale),
        'global_t_world': [float(x) for x in global_t.tolist()],
        'scale_init': float(scale_init),
        'scale_method': scale_method,
        'use_contact': use_contact,
        'n_contact_keyframes': n_contact_kf,
        'n_calib_a_keyframes': n_calib_a_kf,
        'n_depthpro_keyframes': n_depthpro_kf,
        'contact_scale_estimates': contact_scale_estimates,
        'calib_a_estimates': calib_a_estimates,
        'depthpro_scale_estimates': depthpro_scale_estimates,
        'best_iter': int(best['iter']),
        'best_loss': float(best['loss']),
    }
    result_path = os.path.join(out_dir, 'coarse_align_v4.json')
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
