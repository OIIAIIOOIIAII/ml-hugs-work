#!/usr/bin/env python3
"""Evaluate human Gaussian ellipsoid penetration through the fitted ground plane.

Unlike mesh-vertex metrics, each Gaussian is treated as a 3-D ellipsoid with volume.
For a Gaussian with world-space center mu, rotation R (from quaternion), and
scales s, the signed distance from its closest surface point to the plane (n, d) is:

    s_surface = (n · mu + d) - ||s × (R^T @ n)||

where:
  - (n · mu + d) > 0  →  center is ABOVE the ground  (positive = good)
  - ||s × (R^T @ n)||  =  sqrt(n^T Σ n)  =  ellipsoid "radius" toward ground
  - s_surface < 0  →  ellipsoid penetrates the ground

Coordinate convention: NeuMan uses Y-down (OpenCV).  The ground plane normal
points upward, i.e. its Y component is negative.

Usage
-----
    python scripts/eval_gs_ground_contact.py \\
        --run-dir output/human_scene/neuman/lab/hugs_trimlp/<exp>/<ts> \\
        --seq lab \\
        [--plane eval_contact/lab/ground_plane.npz]  # auto-derived from --seq \\
        [--foot-band 1.0]    # keep Gaussians whose center is within this distance of the floor \\
        [--min-opacity 0.05] # ignore nearly-transparent Gaussians \\
        [--scale-factor 1.0] # ellipsoid scale multiplier (1 = 1-sigma ellipsoid) \\
        [--apply-anchor-attention auto|yes|no] \\
        [--save-colored-ply]  # save PLY with penetrating Gaussians coloured red (frame 0)
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Quaternion utils  (3DGS convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [w, x, y, z] → rotation matrix.

    Args:
        q: (N, 4) float tensor

    Returns:
        R: (N, 3, 3) float tensor  (R @ v rotates vector v)
    """
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R   # (N, 3, 3)


# ---------------------------------------------------------------------------
# Core ellipsoid-plane contact metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def ellipsoid_plane_contact(
    mu: torch.Tensor,       # (N, 3)  world-space centers
    scales: torch.Tensor,   # (N, 3)  actual scales (NOT log-scale)
    rotq: torch.Tensor,     # (N, 4)  quaternions [w,x,y,z]
    opacity: torch.Tensor,  # (N,)    sigmoid-ed opacity
    normal: np.ndarray,     # (3,)    unit plane normal (points upward)
    plane_d: float,         # scalar  plane offset  (n·x + d = 0)
    foot_band: float,
    min_opacity: float,
    scale_factor: float,
) -> dict:
    """Compute contact statistics between human Gaussians and the ground plane.

    Returns a dict of scalar stats and tensors for this frame.
    """
    device = mu.device
    n = torch.tensor(normal, dtype=torch.float32, device=device)  # (3,)

    # ---- valid mask: opacity threshold ------------------------------------
    op = opacity.squeeze(-1) if opacity.dim() > 1 else opacity   # (N,)
    valid = op >= min_opacity                                      # (N,)

    # ---- center-to-plane signed distance ----------------------------------
    s_center = mu @ n + plane_d                                    # (N,)

    # ---- foot-region mask: center within foot_band ABOVE the plane ------
    # Use s_center < foot_band (not abs) so deeply penetrating Gaussians
    # (s_center << 0) are still captured in the foot region.
    foot_mask = valid & (s_center < foot_band)                    # (N,)

    # ---- ellipsoid radius in plane-normal direction -----------------------
    R = quat_to_rotmat(rotq)                                       # (N,3,3)
    # n in local Gaussian frame: R^T @ n
    n_local = torch.einsum('nji,j->ni', R, n)                     # (N,3)
    # radius = ||s * (R^T @ n)||  (scales already in world units after LBS)
    r = (scales * scale_factor * n_local).norm(dim=-1)             # (N,)

    # ---- surface signed distance & penetration ----------------------------
    s_surface = s_center - r                                       # (N,)
    penet = torch.clamp(-s_surface, min=0.0)                      # (N,)  ≥0

    def _stats(mask: torch.Tensor) -> dict:
        if not mask.any():
            return dict(n=0, n_pen=0, max_pen=0.0, mean_pen=0.0,
                        pen_ratio=0.0, opacity_weighted_pen=0.0)
        p  = penet[mask]
        op_m = op[mask]
        n_gs = int(mask.sum())
        n_pen = int((p > 0).sum())
        return dict(
            n=n_gs,
            n_pen=n_pen,
            max_pen=float(p.max()),
            mean_pen=float(p[p > 0].mean()) if n_pen > 0 else 0.0,
            pen_ratio=n_pen / n_gs,
            opacity_weighted_pen=float((op_m * p).sum() / (op_m.sum() + 1e-8)),
        )

    return dict(
        all=_stats(valid),
        foot=_stats(foot_mask),
        # keep tensors for optional PLY export
        _s_center=s_center,
        _s_surface=s_surface,
        _penet=penet,
        _foot_mask=foot_mask,
        _valid=valid,
    )


# ---------------------------------------------------------------------------
# Checkpoint / trainer helpers  (adapted from export_human_scene_ply.py)
# ---------------------------------------------------------------------------

def _natural_ckpt_key(path: str):
    name = Path(path).stem
    if name.endswith("final"):
        return (10**12, name)
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else -1, name)


def _find_latest_ckpt(run_dir: Path, pattern: str) -> str | None:
    files = glob.glob(str(run_dir / "ckpt" / pattern))
    files += glob.glob(str(run_dir / pattern))
    files = sorted(set(files), key=_natural_ckpt_key)
    return files[-1] if files else None


def load_trainer(run_dir: Path, apply_anchor_attention: str, anchor_ckpt_override: Path | None):
    from hugs.cfg.config import cfg as default_cfg
    from hugs.trainer import GaussianTrainer

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

    human_ckpt = _find_latest_ckpt(run_dir, "human_*.pth")
    scene_ckpt = _find_latest_ckpt(run_dir, "scene_*.pth")
    if not human_ckpt:
        raise FileNotFoundError(f"No human checkpoint under {run_dir}")
    cfg.human.ckpt = human_ckpt
    cfg.scene.ckpt = scene_ckpt or ""

    anchor_ckpt_path = (
        str(anchor_ckpt_override.resolve())
        if anchor_ckpt_override
        else _find_latest_ckpt(run_dir, "anchor_attention_*.pth")
    )
    use_anchor = (
        True  if apply_anchor_attention == "yes"
        else False if apply_anchor_attention == "no"
        else bool(anchor_ckpt_path)
    )
    if use_anchor:
        if not anchor_ckpt_path or not Path(anchor_ckpt_path).is_file():
            raise FileNotFoundError("Anchor attention requested but no checkpoint found.")
        cfg.anchor_attention.use_anchors = True
        cfg.anchor_attention.use_anchor_token_encoder = True
        cfg.anchor_attention.use_scene_query = True
        cfg.anchor_attention.use_cross_attention = True
        cfg.anchor_attention.ckpt = ""

    print(f"  human ckpt : {human_ckpt}")
    print(f"  scene ckpt : {scene_ckpt}")
    print(f"  anchor attn: {'enabled → ' + str(anchor_ckpt_path) if use_anchor else 'disabled'}")

    trainer = GaussianTrainer(cfg)
    if use_anchor and anchor_ckpt_path:
        state = torch.load(anchor_ckpt_path, map_location="cpu")
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing:
            print(f"  [anchor] missing keys: {len(missing)}")
        if unexpected:
            print(f"  [anchor] unexpected keys: {len(unexpected)}")

    return trainer, cfg, use_anchor


# ---------------------------------------------------------------------------
# Optional: save coloured PLY for one frame
# ---------------------------------------------------------------------------

def save_colored_ply(
    run_dir: Path,
    trainer,
    frame_idx: int,
    normal: np.ndarray,
    plane_d: float,
    foot_band: float,
    min_opacity: float,
    scale_factor: float,
    use_anchor: bool,
    device: str,
):
    """Save a PLY where penetrating foot Gaussians are coloured red."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("  [warn] plyfile not installed — skipping coloured PLY export.")
        return

    dataset = trainer.all_dataset
    data = dataset[frame_idx]

    human_out = trainer.human_gs.forward(
        global_orient=data["global_orient"],
        body_pose=data["body_pose"],
        betas=data["betas"],
        transl=data["transl"],
        smpl_scale=data["smpl_scale"][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )
    if use_anchor:
        human_out, _ = trainer.maybe_apply_anchor_attention(
            human_out, trainer.scene_gs.forward(), trainer.cfg.mode,
            trainer.cfg.train.num_steps,
        )

    mu     = human_out["xyz"].to(device)
    scales = human_out["scales"].to(device)
    rotq   = human_out["rotq"].to(device)
    opacity = human_out["opacity"].to(device)

    stats = ellipsoid_plane_contact(
        mu, scales, rotq, opacity, normal, plane_d,
        foot_band, min_opacity, scale_factor,
    )

    # Build PLY with red-coloured penetrating foot Gaussians
    xyz_np = mu.cpu().numpy()
    n_gs   = xyz_np.shape[0]
    colors = np.ones((n_gs, 3), dtype=np.uint8) * 128  # grey default

    foot_mask  = stats["_foot_mask"].cpu().numpy()
    penet_mask = (stats["_s_surface"] < 0).cpu().numpy()
    colors[foot_mask & ~penet_mask] = [0, 255, 0]   # foot, no penetration → green
    colors[foot_mask &  penet_mask] = [255, 0, 0]   # foot, penetrating    → red
    colors[~foot_mask]              = [80, 80, 80]   # non-foot             → dark grey

    dtype = [("x","f4"),("y","f4"),("z","f4"),
             ("red","u1"),("green","u1"),("blue","u1")]
    arr = np.empty(n_gs, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz_np[:,0], xyz_np[:,1], xyz_np[:,2]
    arr["red"]   = colors[:,0]
    arr["green"] = colors[:,1]
    arr["blue"]  = colors[:,2]

    out = run_dir / "eval_gs_contact" / f"frame{frame_idx:04d}_penetration.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(out)
    n_pen = int((foot_mask & penet_mask).sum())
    n_foot = int(foot_mask.sum())
    print(f"  saved coloured PLY → {out}  ({n_pen}/{n_foot} foot Gaussians penetrating)")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(args):
    run_dir = Path(args.run_dir).resolve()
    device  = args.device

    # ---- ground plane ----------------------------------------------------
    plane_path = Path(args.plane) if args.plane else Path("eval_contact") / args.seq / "ground_plane.npz"
    if not plane_path.is_file():
        raise FileNotFoundError(
            f"Ground plane not found: {plane_path}\n"
            f"Run eval_ground_contact.py --seq {args.seq} first."
        )
    plane = np.load(plane_path)
    normal  = plane["normal"].astype(np.float64)   # (3,) unit normal, points upward
    plane_d = float(plane["d"])
    print(f"[plane] normal=[{normal[0]:.4f},{normal[1]:.4f},{normal[2]:.4f}]  d={plane_d:.4f}")
    print(f"        inlier_ratio={float(plane['inlier_ratio']):.3f}")

    # ---- trainer ---------------------------------------------------------
    print("[trainer] loading model ...")
    trainer, cfg, use_anchor = load_trainer(
        run_dir,
        args.apply_anchor_attention,
        Path(args.anchor_ckpt) if args.anchor_ckpt else None,
    )
    dataset = trainer.all_dataset
    n_frames = len(dataset)
    print(f"[dataset] {n_frames} frames")

    # ---- per-frame loop --------------------------------------------------
    rows = []
    for fi in range(n_frames):
        data = dataset[fi]
        human_out = trainer.human_gs.forward(
            global_orient=data["global_orient"],
            body_pose=data["body_pose"],
            betas=data["betas"],
            transl=data["transl"],
            smpl_scale=data["smpl_scale"][None],
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
        )
        if use_anchor:
            scene_out = trainer.scene_gs.forward()
            human_out, _ = trainer.maybe_apply_anchor_attention(
                human_out, scene_out, cfg.mode, cfg.train.num_steps,
            )

        mu     = human_out["xyz"].to(device)
        scales = human_out["scales"].to(device)
        rotq   = human_out["rotq"].to(device)
        opacity = human_out["opacity"].to(device)

        st = ellipsoid_plane_contact(
            mu, scales, rotq, opacity,
            normal, plane_d,
            args.foot_band, args.min_opacity, args.scale_factor,
        )
        a = st["all"]
        f = st["foot"]
        rows.append({
            "frame":                fi,
            # all valid Gaussians
            "all_n":                a["n"],
            "all_n_pen":            a["n_pen"],
            "all_pen_ratio":        a["pen_ratio"],
            "all_max_pen":          a["max_pen"],
            "all_mean_pen":         a["mean_pen"],
            "all_opw_pen":          a["opacity_weighted_pen"],
            # foot region
            "foot_n":               f["n"],
            "foot_n_pen":           f["n_pen"],
            "foot_pen_ratio":       f["pen_ratio"],
            "foot_max_pen":         f["max_pen"],
            "foot_mean_pen":        f["mean_pen"],
            "foot_opw_pen":         f["opacity_weighted_pen"],
        })
        if (fi + 1) % 20 == 0 or fi == n_frames - 1:
            print(f"  frame {fi+1:3d}/{n_frames}  foot: {f['n_pen']}/{f['n']} pen, "
                  f"max={f['max_pen']:.4f}, opw={f['opacity_weighted_pen']:.4f}")

    # ---- output dir ------------------------------------------------------
    out_dir = run_dir / "eval_gs_contact"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV -------------------------------------------------------------
    import csv
    csv_path = out_dir / "gs_ground_contact.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[out] CSV → {csv_path}")

    # ---- summary statistics ----------------------------------------------
    foot_max_pen  = np.array([r["foot_max_pen"]  for r in rows])
    foot_mean_pen = np.array([r["foot_mean_pen"] for r in rows])
    foot_pen_rat  = np.array([r["foot_pen_ratio"]  for r in rows])
    foot_opw_pen  = np.array([r["foot_opw_pen"]  for r in rows])
    all_max_pen   = np.array([r["all_max_pen"]   for r in rows])

    print("\n========== Summary (foot region, foot-band={:.2f}) ==========".format(args.foot_band))
    print(f"  foot Gaussians per frame   : {np.mean([r['foot_n'] for r in rows]):.0f} avg")
    print(f"  frames with any penetration: {(foot_max_pen > 0).sum()} / {n_frames}")
    print(f"  penetrating GS ratio       : {foot_pen_rat.mean():.3f} ± {foot_pen_rat.std():.3f}")
    print(f"  max penetration (world)    : {foot_max_pen.max():.4f}  avg={foot_max_pen.mean():.4f}")
    print(f"  mean penetration (world)   : {foot_mean_pen.mean():.4f}")
    print(f"  opacity-weighted penetration: {foot_opw_pen.mean():.4f}")
    print(f"  [scale_factor={args.scale_factor}, min_opacity={args.min_opacity}]")

    # ---- plot ------------------------------------------------------------
    frames = np.arange(n_frames)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(frames, foot_pen_rat,  color="tab:red",   linewidth=1.5, label="foot penetrating ratio")
    ax.set_ylabel("Penetrating GS fraction")
    ax.set_title(f"Human GS ground penetration  —  {run_dir.name}\n"
                 f"foot-band={args.foot_band}m  scale-factor={args.scale_factor}  min-opacity={args.min_opacity}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(frames, foot_max_pen,  color="crimson",   linewidth=1.5, label="foot max penetration depth")
    ax.plot(frames, foot_mean_pen, color="salmon",    linewidth=1.0, linestyle="--", label="foot mean penetration (over pen.)")
    ax.set_ylabel("Penetration depth (world units)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(frames, foot_opw_pen, color="darkorange", linewidth=1.5, label="opacity-weighted penetration")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Opac.-weighted penetration")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "gs_ground_contact.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[out] plot → {plot_path}")

    # ---- optional coloured PLY -------------------------------------------
    if args.save_colored_ply:
        ply_frame = args.colored_ply_frame
        print(f"[ply] saving coloured PLY for frame {ply_frame} ...")
        save_colored_ply(
            run_dir, trainer, ply_frame,
            normal, plane_d,
            args.foot_band, args.min_opacity, args.scale_factor,
            use_anchor, device,
        )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate human Gaussian ellipsoid penetration through the ground plane."
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="HUGS training output directory (contains config_train.yaml).")
    p.add_argument("--seq", default="lab",
                   help="NeuMan sequence name, used to locate the ground plane file.")
    p.add_argument("--plane", default=None,
                   help="Path to ground_plane.npz (default: eval_contact/{seq}/ground_plane.npz).")
    p.add_argument("--foot-band", type=float, default=1.0,
                   help="Consider Gaussians whose center is within this signed distance of the "
                        "ground plane (world units, default 1.0).  Increasing this captures "
                        "more of the leg; decreasing focuses on the sole.")
    p.add_argument("--min-opacity", type=float, default=0.05,
                   help="Skip Gaussians with opacity below this threshold (default 0.05).")
    p.add_argument("--scale-factor", type=float, default=1.0,
                   help="Multiply all scales by this factor before computing the ellipsoid "
                        "radius (default 1.0 = 1-sigma ellipsoid).  Use 2 or 3 for a larger "
                        "bounding ellipsoid.")
    p.add_argument("--apply-anchor-attention", choices=("auto", "yes", "no"), default="auto",
                   help="Apply anchor-attention correction (default: auto).")
    p.add_argument("--anchor-ckpt", default=None,
                   help="Override path to anchor_attention checkpoint.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-colored-ply", action="store_true",
                   help="Save a PLY with penetrating foot Gaussians coloured red.")
    p.add_argument("--colored-ply-frame", type=int, default=0,
                   help="Frame index to use for the coloured PLY (default 0).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
