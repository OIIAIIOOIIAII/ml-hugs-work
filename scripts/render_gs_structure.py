#!/usr/bin/env python3
"""Visualize the 2D projected structure of human and scene Gaussians.

For each validation frame, draws the Gaussian ellipses (projected 2D covariance)
on top of the rendered image:
  - Human GS → red ellipses  (or distance-colormap if --dist-cmap is set)
  - Scene GS  → blue ellipses (or distance-colormap if --dist-cmap is set)

Each Gaussian's ellipse size and orientation exactly mirrors its 3D shape as seen
from the camera (via the Jacobian of perspective projection).

When --dist-cmap is given, each Gaussian is colored by its 3D distance to the
nearest human GS center, using the specified matplotlib colormap.
  close → colormap low end (e.g. warm colors for "plasma_r")
  far   → colormap high end

Output files (in <run-dir>/val_gs_struct/):
  struct_{idx:03d}.png   -- ellipses overlaid on the combined RGB render
  struct_gt_{idx:03d}.png -- ellipses overlaid on the GT image

Usage:
  python scripts/render_gs_structure.py --run-dir <training_output_dir>
  python scripts/render_gs_structure.py --run-dir <path> --frames 0 1 2
  python scripts/render_gs_structure.py --run-dir <path> --sigma 2.0 --max-scene 8000
  python scripts/render_gs_structure.py --run-dir <path> --dist-cmap plasma_r --dist-max 0.5
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps as _mpl_cmaps

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Checkpoint helpers (same pattern as other export scripts)
# ---------------------------------------------------------------------------

def natural_ckpt_key(path: str) -> tuple[int, str]:
    name = Path(path).stem
    if name.endswith("final"):
        return (10**12, name)
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else -1, name)


def find_latest_ckpt(run_dir: Path, pattern: str) -> str | None:
    files = glob.glob(str(run_dir / "ckpt" / pattern))
    files += glob.glob(str(run_dir / pattern))
    files = sorted(set(files), key=natural_ckpt_key)
    return files[-1] if files else None


def configure_from_run(args: argparse.Namespace):
    from omegaconf import OmegaConf
    from hugs.cfg.config import cfg as default_cfg

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config_train.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config_path))
    cfg.eval = True
    cfg.train.anim_interval = -1
    cfg.train.save_progress_images = False
    if str(getattr(cfg.human, "name", "")) == "hugs_triplane":
        cfg.human.name = "hugs_trimlp"
    cfg.logdir = str(run_dir)
    cfg.logdir_ckpt = str(run_dir / "ckpt")

    human_ckpt = find_latest_ckpt(run_dir, "human_*.pth")
    scene_ckpt = find_latest_ckpt(run_dir, "scene_*.pth")
    if not human_ckpt:
        raise FileNotFoundError(f"No human checkpoint found under {run_dir / 'ckpt'}")
    if not scene_ckpt:
        raise FileNotFoundError(f"No scene checkpoint found under {run_dir / 'ckpt'}")
    cfg.human.ckpt = human_ckpt
    cfg.scene.ckpt = scene_ckpt

    found_anchor = (
        str(args.anchor_ckpt.resolve()) if args.anchor_ckpt
        else find_latest_ckpt(run_dir, "anchor_attention_*.pth")
    )
    apply_anchor = False if args.no_anchor else bool(found_anchor)

    if apply_anchor:
        if not found_anchor or not Path(found_anchor).is_file():
            raise FileNotFoundError("Anchor checkpoint not found; pass --no-anchor to skip.")
        cfg.anchor_attention.use_anchors = True
        cfg.anchor_attention.use_anchor_token_encoder = True
        cfg.anchor_attention.use_scene_query = True
        cfg.anchor_attention.use_cross_attention = True
        cfg.anchor_attention.ckpt = ""

    print(f"human ckpt : {human_ckpt}")
    print(f"scene ckpt : {scene_ckpt}")
    print(f"anchor     : {'skipped' if not apply_anchor else found_anchor}")
    return cfg, apply_anchor, found_anchor if apply_anchor else None


# ---------------------------------------------------------------------------
# 2D Gaussian projection
# ---------------------------------------------------------------------------

def project_centers(xyz: torch.Tensor, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D Gaussian centers to pixel coordinates via full_proj_transform.

    Returns px, py (pixel coords, float32 numpy) and valid (bool mask).
    """
    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)  # (N, 4)

    clip = xyz_h @ data['full_proj_transform'].to(xyz.device)  # (N, 4)
    w = clip[:, 3]
    valid_t = w > 1e-3

    W = int(data['image_width'])
    H = int(data['image_height'])

    px = torch.zeros(N, device=xyz.device)
    py = torch.zeros(N, device=xyz.device)

    ndc_x = clip[valid_t, 0] / w[valid_t]
    ndc_y = clip[valid_t, 1] / w[valid_t]
    px[valid_t] = (ndc_x + 1.0) * 0.5 * W
    # 3DGS NDC: y=+1 → pixel bottom (matches ndc2Pix in rasterizer)
    py[valid_t] = (ndc_y + 1.0) * 0.5 * H

    in_image = valid_t & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    return px.cpu().numpy(), py.cpu().numpy(), in_image.cpu().numpy()


def compute_2d_cov(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rotq: torch.Tensor,
    data: dict,
) -> np.ndarray:
    """Compute 2D projected covariance Σ2D = J W Σ3D W^T J^T for each Gaussian.

    Reference: 3DGS forward.cu computeCov2D.
    world_view_transform stores W2C.T for right-multiplication with row vectors,
    so the actual camera rotation R_wc = world_view_transform[:3,:3].T.

    Returns (N, 2, 2) numpy array.
    """
    from hugs.utils.rotations import quaternion_to_matrix

    device = xyz.device
    N = xyz.shape[0]
    W_img = int(data['image_width'])
    H_img = int(data['image_height'])
    tanfovx = math.tan(float(data['fovx']) * 0.5)
    tanfovy = math.tan(float(data['fovy']) * 0.5)
    focal_x = W_img / (2.0 * tanfovx)
    focal_y = H_img / (2.0 * tanfovy)

    # Camera-space positions: p_cam = xyz_h @ world_view_transform
    wvt = data['world_view_transform'].to(device)
    ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    pts_cam = (torch.cat([xyz, ones], dim=1) @ wvt)[:, :3]  # (N, 3)

    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2].clamp(min=1e-3)

    # 3D covariance: Σ3D = R diag(s²) R^T
    R = quaternion_to_matrix(rotq)  # (N, 3, 3)
    Sigma3D = R @ torch.diag_embed(scales ** 2) @ R.transpose(-1, -2)  # (N, 3, 3)

    # Camera rotation R_wc = world_view_transform[:3,:3].T
    R_wc = wvt[:3, :3].T  # (3, 3)

    # Perspective projection Jacobian (2×3)
    J = torch.zeros(N, 2, 3, device=device, dtype=xyz.dtype)
    J[:, 0, 0] = focal_x / z
    J[:, 0, 2] = -focal_x * x / (z * z)
    J[:, 1, 1] = focal_y / z
    J[:, 1, 2] = -focal_y * y / (z * z)

    # Σ2D = J R_wc Σ3D R_wc^T J^T
    JW = J @ R_wc.unsqueeze(0)  # (N, 2, 3)
    Sigma2D = JW @ Sigma3D @ JW.transpose(-1, -2)  # (N, 2, 2)
    # Low-pass regularization (same as 3DGS rasterizer)
    Sigma2D[:, 0, 0] += 0.3
    Sigma2D[:, 1, 1] += 0.3

    return Sigma2D.cpu().numpy()


# ---------------------------------------------------------------------------
# Distance-based per-Gaussian coloring
# ---------------------------------------------------------------------------

def compute_dist_colors(
    xyz: torch.Tensor,
    ref_xyz: torch.Tensor,
    cmap_name: str,
    dist_min: float,
    dist_max: float,
    ref_max_sample: int = 8000,
) -> np.ndarray:
    """Return (N, 3) BGR uint8 array: each Gaussian colored by its distance to nearest ref point.

    Uses a subsampled KDTree for efficiency (ref_max_sample controls subsample size).
    """
    from scipy.spatial import KDTree

    M = ref_xyz.shape[0]
    if M > ref_max_sample:
        idx = torch.randperm(M, device="cpu")[:ref_max_sample]
        ref_sub = ref_xyz[idx].cpu().float().numpy()
    else:
        ref_sub = ref_xyz.cpu().float().numpy()

    xyz_np = xyz.cpu().float().numpy()
    tree = KDTree(ref_sub)
    dists, _ = tree.query(xyz_np, k=1, workers=-1)

    span = max(dist_max - dist_min, 1e-6)
    t = np.clip((dists - dist_min) / span, 0.0, 1.0)

    cmap = _mpl_cmaps[cmap_name]
    rgba = cmap(t)             # (N, 4) float [0,1], RGBA
    bgr = (rgba[:, [2, 1, 0]] * 255).astype(np.uint8)   # RGB→BGR
    return bgr


# ---------------------------------------------------------------------------
# Orbital novel-view camera
# ---------------------------------------------------------------------------

def _make_w2c_lookat(cam_pos: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """4×4 W2C in OpenCV convention (y-down, z-forward), for row-vector left-multiplication."""
    fwd = (target - cam_pos).astype(np.float64)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(world_up, fwd)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([1., 0., 0.]), fwd)
    right /= np.linalg.norm(right)
    down = np.cross(right, fwd)          # camera y points DOWN (matches NDC y=+1 → pixel bottom)
    down /= np.linalg.norm(down)
    R = np.stack([right, down, fwd]).astype(np.float32)  # rows = world-to-cam axes
    t = -(R @ cam_pos.astype(np.float32))
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R
    W2C[:3, 3] = t
    return W2C


def make_orbital_camera(
    data: dict,
    human_xyz: torch.Tensor,
    azimuth_deg: float,
    elevation_offset_deg: float = 0.0,
    radius_scale: float = 1.0,
) -> dict:
    """Return a copy of data with camera matrices replaced by an orbital novel view.

    The camera is placed on a sphere of radius = (original distance to human center) * radius_scale,
    rotated by azimuth_deg around world Y (0 = original direction, 180 = opposite side).
    elevation_offset_deg tilts the camera up (+) or down (-) relative to original elevation.
    """
    wvt = data['world_view_transform']   # (4,4) W2C.T, float32 tensor
    fpt = data['full_proj_transform']    # (4,4)

    wvt_np = wvt.cpu().float().numpy()
    fpt_np = fpt.cpu().float().numpy()

    # Recover shared projection matrix: fpt = wvt @ P_right  =>  P_right = wvt^{-1} @ fpt
    P_right = np.linalg.solve(wvt_np, fpt_np).astype(np.float32)

    # Original camera position in world
    W2C = wvt_np.T
    cam_pos = -(W2C[:3, :3].T @ W2C[:3, 3]).astype(np.float64)

    # Human body center
    center = human_xyz.mean(0).cpu().float().numpy().astype(np.float64)

    world_up = np.array([0., 1., 0.])   # orbit axis = world Y

    v = cam_pos - center
    radius = float(np.linalg.norm(v)) * radius_scale

    # Decompose into vertical (along world_up) and horizontal components
    v_vert = float(np.dot(v, world_up)) * world_up
    v_horiz = v - v_vert
    if np.linalg.norm(v_horiz) < 1e-4:
        # Camera almost directly above/below; pick arbitrary horizontal
        v_horiz = np.array([1., 0., 0.]) - float(np.dot([1., 0., 0.], world_up)) * world_up

    h_len = float(np.linalg.norm(v_horiz))
    h_dir = v_horiz / h_len
    h_perp = np.cross(world_up, h_dir)    # 90° CCW from h_dir in horizontal plane
    h_perp /= np.linalg.norm(h_perp)

    az = math.radians(azimuth_deg)
    v_horiz_rot = math.cos(az) * v_horiz + math.sin(az) * h_len * h_perp

    if elevation_offset_deg:
        elev = math.radians(elevation_offset_deg)
        v_vert_new = v_vert + math.tan(elev) * h_len * world_up
    else:
        v_vert_new = v_vert

    v_new = v_horiz_rot + v_vert_new
    new_cam_pos = center + v_new / np.linalg.norm(v_new) * radius

    new_W2C = _make_w2c_lookat(new_cam_pos, center, world_up)
    new_wvt = new_W2C.T                                              # world_view_transform
    new_fpt = (new_wvt @ P_right).astype(np.float32)

    novel = dict(data)
    novel['world_view_transform'] = torch.from_numpy(new_wvt).to(wvt.device)
    novel['full_proj_transform']  = torch.from_numpy(new_fpt).to(wvt.device)
    return novel


# ---------------------------------------------------------------------------
# Ellipse drawing with OpenCV
# ---------------------------------------------------------------------------

def draw_ellipses(
    canvas: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    cov2d: np.ndarray,
    mask: np.ndarray,
    color_bgr: "tuple[int, int, int] | np.ndarray",
    sigma: float,
    alpha: float,
    max_count: int,
    rng: np.random.Generator,
) -> None:
    """Draw Gaussian ellipses onto canvas (H,W,3 uint8) in-place using an overlay.

    color_bgr can be a single (B,G,R) tuple (uniform color) or an (N,3) uint8 numpy
    array for per-Gaussian colors (indexed by the original Gaussian index i).
    """
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return
    if len(indices) > max_count:
        indices = rng.choice(indices, max_count, replace=False)

    per_elem = isinstance(color_bgr, np.ndarray) and color_bgr.ndim == 2
    overlay = canvas.copy()

    for i in indices:
        c = cov2d[i]  # (2, 2)
        eigenvalues, eigenvectors = np.linalg.eigh(c)
        eigenvalues = np.maximum(eigenvalues, 0.01)

        # Semi-axes in pixels at `sigma` standard deviations
        # eigh returns values in ascending order; eigenvectors[:,k] = k-th eigenvector
        axis_major = sigma * math.sqrt(eigenvalues[1])
        axis_minor = sigma * math.sqrt(eigenvalues[0])

        # Angle of major axis (eigenvector of largest eigenvalue) w.r.t. x-axis
        ex, ey = eigenvectors[:, 1]
        angle_deg = math.degrees(math.atan2(ey, ex))

        cx = int(round(float(px[i])))
        cy = int(round(float(py[i])))

        color = (int(color_bgr[i, 0]), int(color_bgr[i, 1]), int(color_bgr[i, 2])) if per_elem else color_bgr

        cv2.ellipse(
            overlay,
            center=(cx, cy),
            axes=(max(1, int(round(axis_major))), max(1, int(round(axis_minor)))),
            angle=angle_deg,
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_frame(
    trainer,
    data: dict,
    eval_iter: int,
    cfg,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (struct_on_render, struct_on_gt) as H×W×3 uint8 numpy arrays."""
    from hugs.renderer.gs_renderer import render_human_scene

    bg_black = torch.zeros(3, dtype=torch.float32, device="cuda")

    human_gs_out = trainer.human_gs.forward(
        global_orient=data["global_orient"],
        body_pose=data["body_pose"],
        betas=data["betas"],
        transl=data["transl"],
        smpl_scale=data["smpl_scale"][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )
    scene_gs_out = trainer.scene_gs.forward()

    human_gs_out, _ = trainer.maybe_apply_anchor_attention(
        human_gs_out, scene_gs_out, cfg.mode, eval_iter,
    )

    # Combined render as background
    render_pkg = render_human_scene(
        data=data,
        human_gs_out=human_gs_out,
        scene_gs_out=scene_gs_out,
        bg_color=bg_black,
        render_mode=cfg.mode,
    )
    render_np = (render_pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

    gt_np = (data["rgb"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)

    canvas_render = render_bgr.copy()
    canvas_gt = gt_bgr.copy()

    # --- Human GS ellipses ---
    h_xyz = human_gs_out["xyz"]
    h_scales = human_gs_out["scales"]
    h_rotq = human_gs_out["rotq"]
    h_opacity = human_gs_out["opacity"].squeeze(-1).sigmoid()

    h_px, h_py, h_in_image = project_centers(h_xyz, data)
    h_cov2d = compute_2d_cov(h_xyz, h_scales, h_rotq, data)

    h_mask = h_in_image & (h_opacity.cpu().numpy() > args.opacity_thresh)

    h_colors = (
        int(args.human_color[2] * 255),
        int(args.human_color[1] * 255),
        int(args.human_color[0] * 255),
    )

    draw_ellipses(canvas_render, h_px, h_py, h_cov2d, h_mask,
                  h_colors, args.sigma, args.alpha, args.max_human, rng)
    draw_ellipses(canvas_gt, h_px, h_py, h_cov2d, h_mask,
                  h_colors, args.sigma, args.alpha, args.max_human, rng)

    # --- Scene GS ellipses ---
    s_xyz = scene_gs_out["xyz"]
    s_scales = scene_gs_out["scales"]
    s_rotq = scene_gs_out["rotq"]
    s_opacity = scene_gs_out["opacity"].squeeze(-1).sigmoid()

    s_px, s_py, s_in_image = project_centers(s_xyz, data)
    s_cov2d = compute_2d_cov(s_xyz, s_scales, s_rotq, data)

    s_mask = s_in_image & (s_opacity.cpu().numpy() > args.opacity_thresh)
    n_scene_visible_total = int(s_mask.sum())

    if args.dist_cmap:
        s_colors = compute_dist_colors(
            s_xyz, h_xyz, args.dist_cmap, args.dist_min, args.dist_max,
        )
    else:
        s_colors = (
            int(args.scene_color[2] * 255),
            int(args.scene_color[1] * 255),
            int(args.scene_color[0] * 255),
        )

    W_img = int(data['image_width'])
    H_img = int(data['image_height'])

    if args.scene_occlude_only:
        # Keep only scene GS that (a) project into the human mask and
        # (b) are closer to the camera than the human front surface.
        gt_mask_2d = (data['mask'] > 0.5).cpu().numpy()  # (H, W) bool
        s_px_int = np.clip(s_px.astype(int), 0, W_img - 1)
        s_py_int = np.clip(s_py.astype(int), 0, H_img - 1)
        in_human_mask_2d = gt_mask_2d[s_py_int, s_px_int] & s_in_image

        # Camera-space z for scene and human GS
        wvt = data['world_view_transform'].to(s_xyz.device)
        ones_s = torch.ones(s_xyz.shape[0], 1, device=s_xyz.device, dtype=s_xyz.dtype)
        s_z_cam = (torch.cat([s_xyz, ones_s], dim=1) @ wvt)[:, 2].cpu().numpy()

        ones_h = torch.ones(h_xyz.shape[0], 1, device=h_xyz.device, dtype=h_xyz.dtype)
        h_z_cam = (torch.cat([h_xyz, ones_h], dim=1) @ wvt)[:, 2].cpu().numpy()
        h_front_z = np.percentile(h_z_cam, args.human_depth_percentile)

        s_mask = in_human_mask_2d & (s_z_cam < h_front_z) & (s_opacity.cpu().numpy() > args.opacity_thresh)
        max_scene_use = args.max_scene_near

    elif args.scene_near_extend >= 0:
        # Keep only scene GS inside the human GS bounding box (+ extension)
        h_min = h_xyz.min(dim=0).values - args.scene_near_extend
        h_max = h_xyz.max(dim=0).values + args.scene_near_extend
        in_human_bbox = (
            (s_xyz >= h_min).all(dim=1) & (s_xyz <= h_max).all(dim=1)
        ).cpu().numpy()
        s_mask = s_mask & in_human_bbox
        max_scene_use = args.max_scene_near

    else:
        max_scene_use = args.max_scene

    draw_ellipses(canvas_render, s_px, s_py, s_cov2d, s_mask,
                  s_colors, args.sigma, args.alpha, max_scene_use, rng)
    draw_ellipses(canvas_gt, s_px, s_py, s_cov2d, s_mask,
                  s_colors, args.sigma, args.alpha, max_scene_use, rng)

    # Print stats
    n_human_drawn = int(h_mask.sum())
    n_scene_near = int(s_mask.sum())
    n_scene_capped = min(n_scene_near, max_scene_use)
    print(f"    human GS drawn: {n_human_drawn} / {len(h_mask)}")
    if args.scene_occlude_only:
        print(f"    scene GS occluding human: {n_scene_near} / {n_scene_visible_total} visible, drawn: {n_scene_capped}")
    elif args.scene_near_extend >= 0:
        print(f"    scene GS near human: {n_scene_near} / {n_scene_visible_total} visible "
              f"(extend={args.scene_near_extend:.2f}), drawn: {n_scene_capped}")
    else:
        print(f"    scene GS drawn: {n_scene_capped} / {n_scene_visible_total} visible (total {len(s_mask)})")

    return canvas_render, canvas_gt


# ---------------------------------------------------------------------------
# Novel-view render loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_novel_views(
    trainer,
    data: dict,
    eval_iter: int,
    cfg,
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_dir: Path,
    frame_idx: int,
) -> None:
    """For each azimuth in args.azimuths, render an orbital novel view with GS overlay."""
    from hugs.renderer.gs_renderer import render_human_scene

    bg_black = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Forward pass once; GS positions are fixed in world space for all azimuths
    human_gs_out = trainer.human_gs.forward(
        global_orient=data["global_orient"],
        body_pose=data["body_pose"],
        betas=data["betas"],
        transl=data["transl"],
        smpl_scale=data["smpl_scale"][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )
    scene_gs_out = trainer.scene_gs.forward()
    human_gs_out, _ = trainer.maybe_apply_anchor_attention(
        human_gs_out, scene_gs_out, cfg.mode, eval_iter,
    )

    h_xyz    = human_gs_out["xyz"]
    h_scales = human_gs_out["scales"]
    h_rotq   = human_gs_out["rotq"]
    h_opacity = human_gs_out["opacity"].squeeze(-1).sigmoid()

    s_xyz    = scene_gs_out["xyz"]
    s_scales = scene_gs_out["scales"]
    s_rotq   = scene_gs_out["rotq"]
    s_opacity = scene_gs_out["opacity"].squeeze(-1).sigmoid()

    # Pre-compute per-GS colors (same for all azimuths)
    if args.dist_cmap:
        h_colors = compute_dist_colors(h_xyz, h_xyz, args.dist_cmap, args.dist_min, args.dist_max)
        s_colors = compute_dist_colors(s_xyz, h_xyz, args.dist_cmap, args.dist_min, args.dist_max)
    else:
        def _bgr(rgb):
            return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        h_colors = _bgr(args.human_color)
        s_colors = _bgr(args.scene_color)

    for az in args.azimuths:
        print(f"  [{frame_idx:03d}] azimuth={az:+.1f}°")
        nd = make_orbital_camera(data, h_xyz, az, args.elevation_offset, args.radius_scale)

        # Render combined scene from novel viewpoint (used as background)
        render_pkg = render_human_scene(
            data=nd, human_gs_out=human_gs_out, scene_gs_out=scene_gs_out,
            bg_color=bg_black, render_mode=cfg.mode,
        )
        render_np = (render_pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        canvas = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

        # For --scene-occlude-only: derive proxy human mask from human-only render
        if args.scene_occlude_only:
            h_pkg = render_human_scene(
                data=nd, human_gs_out=human_gs_out, scene_gs_out=scene_gs_out,
                bg_color=bg_black, render_mode="human",
            )
            proxy_mask = (h_pkg["render"].max(0).values > 0.05).float()
            nd = {**nd, "mask": proxy_mask}

        # Human GS ellipses
        h_px, h_py, h_in_image = project_centers(h_xyz, nd)
        h_cov2d = compute_2d_cov(h_xyz, h_scales, h_rotq, nd)
        h_mask = h_in_image & (h_opacity.cpu().numpy() > args.opacity_thresh)
        draw_ellipses(canvas, h_px, h_py, h_cov2d, h_mask, h_colors,
                      args.sigma, args.alpha, args.max_human, rng)

        # Scene GS ellipses
        s_px, s_py, s_in_image = project_centers(s_xyz, nd)
        s_cov2d = compute_2d_cov(s_xyz, s_scales, s_rotq, nd)
        s_mask = s_in_image & (s_opacity.cpu().numpy() > args.opacity_thresh)
        n_visible = int(s_mask.sum())

        W_img = int(nd["image_width"])
        H_img = int(nd["image_height"])

        if args.scene_occlude_only:
            gt_mask_2d = (nd["mask"] > 0.5).cpu().numpy()
            sx_int = np.clip(s_px.astype(int), 0, W_img - 1)
            sy_int = np.clip(s_py.astype(int), 0, H_img - 1)
            in_hmask = gt_mask_2d[sy_int, sx_int] & s_in_image
            wvt_dev = nd["world_view_transform"].to(s_xyz.device)
            ones_s = torch.ones(s_xyz.shape[0], 1, device=s_xyz.device, dtype=s_xyz.dtype)
            ones_h = torch.ones(h_xyz.shape[0], 1, device=h_xyz.device, dtype=h_xyz.dtype)
            s_zcam = (torch.cat([s_xyz, ones_s], 1) @ wvt_dev)[:, 2].cpu().numpy()
            h_zcam = (torch.cat([h_xyz, ones_h], 1) @ wvt_dev)[:, 2].cpu().numpy()
            h_front_z = np.percentile(h_zcam, args.human_depth_percentile)
            s_mask = in_hmask & (s_zcam < h_front_z) & (s_opacity.cpu().numpy() > args.opacity_thresh)
            max_scene_use = args.max_scene_near
        elif args.scene_near_extend >= 0:
            h_min = h_xyz.min(0).values - args.scene_near_extend
            h_max = h_xyz.max(0).values + args.scene_near_extend
            in_bbox = ((s_xyz >= h_min).all(1) & (s_xyz <= h_max).all(1)).cpu().numpy()
            s_mask = s_mask & in_bbox
            max_scene_use = args.max_scene_near
        else:
            max_scene_use = args.max_scene

        draw_ellipses(canvas, s_px, s_py, s_cov2d, s_mask, s_colors,
                      args.sigma, args.alpha, max_scene_use, rng)

        print(f"          scene GS: {int(s_mask.sum())}/{n_visible} visible drawn")

        # Output: novel_{idx:03d}_azP030.png  (P=+, N=- to avoid filesystem issues)
        az_str = f"{az:+.0f}".replace("+", "P").replace("-", "N")
        cv2.imwrite(str(out_dir / f"novel_{frame_idx:03d}_az{az_str}.png"), canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    cfg, apply_anchor, anchor_ckpt_path = configure_from_run(args)

    from hugs.trainer import GaussianTrainer

    trainer = GaussianTrainer(cfg)
    trainer.human_gs.eval()

    if apply_anchor and anchor_ckpt_path:
        state = torch.load(anchor_ckpt_path)
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing:
            print(f"anchor ckpt missing keys: {len(missing)}")
        if unexpected:
            print(f"anchor ckpt unexpected keys: {len(unexpected)}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else args.run_dir.resolve() / "val_gs_struct"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"sigma={args.sigma}, opacity_thresh={args.opacity_thresh}, "
          f"max_human={args.max_human}, alpha={args.alpha}")
    if args.dist_cmap:
        print(f"distance colormap: {args.dist_cmap}, dist=[{args.dist_min}, {args.dist_max}]m")
    if args.scene_occlude_only:
        print(f"scene filter: occlude-only (human_depth_percentile={args.human_depth_percentile})")
    elif args.scene_near_extend >= 0:
        print(f"scene filter: near-human bbox (extend={args.scene_near_extend}, max_scene_near={args.max_scene_near})")
    else:
        print(f"scene filter: none (max_scene={args.max_scene})")

    val_dataset = trainer.val_dataset
    frame_indices = args.frames if args.frames else list(range(len(val_dataset)))
    eval_iter = cfg.train.num_steps
    rng = np.random.default_rng(42)

    if args.novel_view:
        print(f"mode: novel-view  azimuths={args.azimuths}  "
              f"elevation_offset={args.elevation_offset}°  radius_scale={args.radius_scale}")

    for idx in frame_indices:
        if idx >= len(val_dataset):
            print(f"Frame {idx} out of range ({len(val_dataset)} frames), skipping.")
            continue

        data = val_dataset[idx]

        if args.novel_view:
            render_novel_views(trainer, data, eval_iter, cfg, args, rng, out_dir, idx)
        else:
            print(f"  [{idx:03d}]")
            canvas_render, canvas_gt = render_frame(trainer, data, eval_iter, cfg, args, rng)
            cv2.imwrite(str(out_dir / f"struct_{idx:03d}.png"), canvas_render)
            cv2.imwrite(str(out_dir / f"struct_gt_{idx:03d}.png"), canvas_gt)

    print(f"\nDone. {len(frame_indices)} frames written to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="HUGS training output directory (contains config_train.yaml and ckpt/).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory. Defaults to <run-dir>/val_gs_struct/.")
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="Frame indices to process (default: all val frames).")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Draw ellipses at this many standard deviations (default: 1.0).")
    p.add_argument("--opacity-thresh", type=float, default=0.1,
                   help="Min opacity (after sigmoid) to include a Gaussian (default: 0.1).")
    p.add_argument("--max-human", type=int, default=50000,
                   help="Max human Gaussians to draw per frame (default: 50000 = all).")
    p.add_argument("--max-scene", type=int, default=5000,
                   help="Max scene Gaussians to draw per frame when not using near-human filter (default: 5000).")
    p.add_argument("--scene-occlude-only", action="store_true",
                   help="Only draw scene GS whose 2D projection falls inside the human mask AND "
                        "whose camera-space depth is less than the human front surface depth. "
                        "Takes priority over --scene-near-extend.")
    p.add_argument("--human-depth-percentile", type=float, default=30.0,
                   help="Percentile of human GS camera-z used as front-surface depth threshold "
                        "for --scene-occlude-only (default: 30).")
    p.add_argument("--scene-near-extend", type=float, default=0.5,
                   help="Extend the human-GS bounding box by this amount (3D units) and only draw "
                        "scene GS inside that region. Set to -1 to disable and show all scene GS (default: 0.5).")
    p.add_argument("--max-scene-near", type=int, default=100000,
                   help="Max scene Gaussians to draw when near-human filter is active (default: 100000, i.e. dense).")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Ellipse opacity in the overlay (default: 0.6).")
    p.add_argument("--human-color", type=float, nargs=3, default=[1.0, 0.2, 0.1],
                   metavar=("R", "G", "B"), help="Human GS ellipse color (default: red).")
    p.add_argument("--scene-color", type=float, nargs=3, default=[0.1, 0.5, 1.0],
                   metavar=("R", "G", "B"), help="Scene GS ellipse color (default: blue).")
    p.add_argument("--anchor-ckpt", type=Path, default=None,
                   help="Override anchor-attention checkpoint path.")
    p.add_argument("--no-anchor", action="store_true",
                   help="Skip anchor-attention even if a checkpoint exists.")

    n = p.add_argument_group("orbital novel views (--novel-view)")
    n.add_argument("--novel-view", action="store_true",
                   help="Render orbital novel views around the human body center "
                        "instead of the default val-camera views.")
    n.add_argument("--azimuths", type=float, nargs="+", default=[30., 60., 90., 120., 150., 180.],
                   metavar="DEG",
                   help="Azimuth offsets in degrees (default: 30 60 90 120 150 180). "
                        "0 = original camera direction, 180 = opposite side.")
    n.add_argument("--elevation-offset", type=float, default=0.,
                   help="Tilt the orbital camera up (+) or down (-) by this many degrees (default: 0).")
    n.add_argument("--radius-scale", type=float, default=1.,
                   help="Scale camera distance to human center (default: 1.0 = same distance).")

    g = p.add_argument_group("distance colormap")
    g.add_argument("--dist-cmap", type=str, default=None, metavar="CMAP",
                   help="matplotlib colormap name for distance-based coloring "
                        "(e.g. plasma_r, RdYlGn_r, coolwarm). "
                        "If set, --human-color / --scene-color are ignored; "
                        "each Gaussian is colored by its 3D distance to the nearest human GS center.")
    g.add_argument("--dist-min", type=float, default=0.0,
                   help="Distance (m) mapped to colormap low end (default: 0.0).")
    g.add_argument("--dist-max", type=float, default=0.5,
                   help="Distance (m) mapped to colormap high end (default: 0.5). "
                        "Gaussians beyond this distance are clamped to the high end.")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
