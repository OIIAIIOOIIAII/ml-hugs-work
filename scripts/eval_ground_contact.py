#!/usr/bin/env python3
"""Evaluate foot-ground contact distance using a COLMAP-fitted ground plane.

Coordinate convention
---------------------
NeuMan uses the OpenCV / COLMAP convention: Y points DOWNWARD.
The ground (floor) therefore has the MAXIMUM Y value among scene points.
The bottom of the foot also has the MAXIMUM Y value among foot vertices.
Signed distance is defined as:
    dist = (foot_Y - ground_Y) / |normal|
where dist ≈ 0 means contact, dist > 0 means foot is below the ground (penetration),
and dist < 0 means foot is above the ground.

Pipeline
--------
1. Parse COLMAP sparse/points3D.txt → 3-D world-space point cloud.
2. Run SMPL forward for all frames → foot-bottom vertex positions (world coords).
3. Estimate ground plane height from foot-bottom statistics (prior).
4. Filter COLMAP points near the estimated floor height.
5. RANSAC plane fitting on the filtered points → ground plane (n, d).
6. Compute signed distance from foot-bottom to plane per frame.
7. Write per-frame CSV + matplotlib plot.

Usage
-----
    python scripts/eval_ground_contact.py --seq lab
    python scripts/eval_ground_contact.py --seq lab --contact-thresh 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# COLMAP parser
# ---------------------------------------------------------------------------

def load_colmap_points(path: str) -> np.ndarray:
    """Return (N, 3) float32 array of COLMAP world-space points."""
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# RANSAC plane fitting (arbitrary orientation, no sklearn dependency)
# ---------------------------------------------------------------------------

def ransac_plane(
    points: np.ndarray,
    n_iter: int = 3000,
    threshold: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane via RANSAC, then refine with SVD on inliers.

    Returns (unit_normal, d, inlier_mask) where  normal·x + d = 0.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    best_normal, best_d, best_count = None, 0.0, 0

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p = points[idx]
        v1, v2 = p[1] - p[0], p[2] - p[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -float(np.dot(normal, p[0]))
        count = int((np.abs(points @ normal + d) < threshold).sum())
        if count > best_count:
            best_count, best_normal, best_d = count, normal.copy(), d

    if best_normal is None:
        raise RuntimeError("RANSAC failed — no valid plane found.")

    # Refine with SVD on all inliers
    mask = np.abs(points @ best_normal + best_d) < threshold
    if mask.sum() >= 3:
        centroid = points[mask].mean(0)
        _, _, Vt = np.linalg.svd(points[mask] - centroid)
        best_normal = Vt[-1]
        best_d = -float(np.dot(best_normal, centroid))
        mask = np.abs(points @ best_normal + best_d) < threshold

    return best_normal, best_d, mask


# ---------------------------------------------------------------------------
# SMPL forward
# ---------------------------------------------------------------------------

def build_smpl(smpl_dir: str = "data/smpl", device: str = "cpu"):
    from hugs.models.modules.smpl_layer import SMPL
    return SMPL(smpl_dir).to(device)


@torch.no_grad()
def compute_foot_bottom_world(
    smpl,
    smpl_params: dict,
    device: str = "cpu",
    n_bottom_verts: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame foot-bottom positions in world coordinates.

    Y is DOWN (OpenCV convention), so the foot bottom = maximum-Y vertices.

    Returns
    -------
    foot_bottom : (N_frames, 3) mean position of the n_bottom_verts lowest
                  (max-Y) foot vertices per frame.
    all_verts_world : (N_frames, V, 3) full vertex array in world coords
                      (may be used for diagnostics).
    """
    n_frames = smpl_params["global_orient"].shape[0]
    foot_bottoms = []

    for i in range(n_frames):
        go    = torch.tensor(smpl_params["global_orient"][i:i+1], dtype=torch.float32, device=device)
        bp    = torch.tensor(smpl_params["body_pose"][i:i+1],     dtype=torch.float32, device=device)
        betas = torch.tensor(smpl_params["betas"][i:i+1],         dtype=torch.float32, device=device)
        transl = smpl_params["transl"][i]   # (3,) numpy
        scale  = float(smpl_params["scale"][i])

        out   = smpl(global_orient=go, body_pose=bp, betas=betas)
        verts = out.vertices[0].detach().cpu().numpy()  # (V, 3)

        # World transform: world = canonical * scale + transl
        wv = verts * scale + transl  # (V, 3)

        # Y is DOWN → foot bottom = maximum Y vertices
        top_y_idx = np.argsort(wv[:, 1])[-n_bottom_verts:]
        foot_bottoms.append(wv[top_y_idx].mean(0))  # (3,)

    return np.stack(foot_bottoms, axis=0)  # (N, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="lab")
    p.add_argument("--data-root", default="data/neuman/dataset")
    p.add_argument("--smpl-dir",  default="data/smpl")
    p.add_argument("--ransac-thresh", type=float, default=0.3,
                   help="RANSAC inlier threshold in world units (default 0.3)")
    p.add_argument("--ransac-iter",   type=int,   default=3000)
    p.add_argument("--floor-band",    type=float, default=2.0,
                   help="Half-width of the Y band around the floor prior used to "
                        "filter COLMAP points before RANSAC (default 2.0)")
    p.add_argument("--n-bottom-verts", type=int, default=20,
                   help="Number of lowest foot vertices averaged for the contact point")
    p.add_argument("--contact-thresh", type=float, default=None,
                   help="If given, frames with dist < this are labelled as 'in contact'")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    seq_dir = Path(args.data_root) / args.seq
    out_dir = Path(args.out_dir) if args.out_dir else Path("eval_contact") / args.seq
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. COLMAP point cloud
    # ------------------------------------------------------------------
    pts_path = seq_dir / "sparse" / "points3D.txt"
    print(f"[1] Loading COLMAP points: {pts_path}")
    pts = load_colmap_points(str(pts_path))
    print(f"    {len(pts)} points | Y ∈ [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")

    # ------------------------------------------------------------------
    # 2. SMPL → foot-bottom world positions (used as floor prior)
    # ------------------------------------------------------------------
    smpl_npz = seq_dir / "4d_humans" / "smpl_optimized_aligned_scale.npz"
    print(f"[2] Loading SMPL params: {smpl_npz}")
    smpl_params = dict(np.load(smpl_npz))
    n_frames = smpl_params["global_orient"].shape[0]
    print(f"    {n_frames} frames | scale ∈ [{smpl_params['scale'].min():.2f}, {smpl_params['scale'].max():.2f}]")

    print(f"    Running SMPL forward ({n_frames} frames) ...")
    smpl = build_smpl(args.smpl_dir, args.device)
    foot_bottom = compute_foot_bottom_world(
        smpl, smpl_params, args.device, args.n_bottom_verts
    )  # (N, 3)

    # Y is DOWN: foot bottom = max Y among foot vertices
    floor_prior_Y = np.median(foot_bottom[:, 1])
    print(f"    Foot-bottom world Y: min={foot_bottom[:,1].min():.3f}  "
          f"max={foot_bottom[:,1].max():.3f}  median={floor_prior_Y:.3f}")

    # ------------------------------------------------------------------
    # 3. Filter COLMAP points near floor height → RANSAC
    # ------------------------------------------------------------------
    band = args.floor_band
    floor_mask = np.abs(pts[:, 1] - floor_prior_Y) < band
    pts_floor = pts[floor_mask]
    print(f"[3] COLMAP pts in Y ∈ [{floor_prior_Y-band:.2f}, {floor_prior_Y+band:.2f}]: "
          f"{floor_mask.sum()} / {len(pts)}")

    if len(pts_floor) < 10:
        print("    WARNING: too few points in floor band — using ALL COLMAP points.")
        pts_floor = pts

    print(f"    RANSAC (iter={args.ransac_iter}, thresh={args.ransac_thresh}) ...")
    normal, d, inlier_mask = ransac_plane(
        pts_floor, n_iter=args.ransac_iter, threshold=args.ransac_thresh
    )
    inlier_ratio = inlier_mask.sum() / len(pts_floor)
    print(f"    Normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]  d={d:.4f}")
    print(f"    Inliers: {inlier_mask.sum()} / {len(pts_floor)} ({100*inlier_ratio:.1f}%)")

    # Orient normal so it points UPWARD (negative Y in Y-down convention)
    if normal[1] > 0:
        normal = -normal
        d = -d
    print(f"    Normal (up-oriented): [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]  d={d:.4f}")

    # Plane equation: normal·x + d = 0
    # For a point ON the floor: normal·x_floor + d = 0
    # Signed distance: dist = normal·x + d
    #   dist > 0 → above the floor (normal points up = negative Y direction)
    #   dist < 0 → below the floor (penetration)

    # ------------------------------------------------------------------
    # 4. Signed distance per frame
    # ------------------------------------------------------------------
    dists = foot_bottom @ normal + d   # (N,)

    # ------------------------------------------------------------------
    # 5. CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / "foot_ground_dist.csv"
    with open(csv_path, "w") as f:
        f.write("frame,foot_bottom_Y_world,signed_dist\n")
        for i in range(n_frames):
            f.write(f"{i},{foot_bottom[i,1]:.4f},{dists[i]:.4f}\n")
    print(f"[5] Saved CSV → {csv_path}")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    frames = np.arange(n_frames)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.plot(frames, foot_bottom[:, 1], color="steelblue", linewidth=1.5, label="foot-bottom world Y")
    ax.axhline(floor_prior_Y, color="orange", linewidth=1.0, linestyle="--",
               label=f"floor prior Y={floor_prior_Y:.2f}")
    # Ground plane Y at x=foot_x, z=foot_z (using the plane equation)
    ground_y_at_foot = -(normal[0]*foot_bottom[:,0] + normal[2]*foot_bottom[:,2] + d) / (normal[1] + 1e-12)
    ax.plot(frames, ground_y_at_foot, color="red", linewidth=1.0, linestyle=":",
            label="fitted ground Y (at foot XZ)")
    ax.set_ylabel("World Y (down-positive)")
    ax.set_title(f"{args.seq} — Foot-bottom vs ground plane  (Y-down convention)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()   # visually flip so "above ground" = higher on screen

    ax2 = axes[1]
    ax2.plot(frames, dists, color="tab:purple", linewidth=1.5, label="signed dist (+ = above ground)")
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--", label="ground plane")
    if args.contact_thresh is not None:
        contact = dists < args.contact_thresh
        ax2.fill_between(frames, 0, dists, where=contact,
                         alpha=0.2, color="red", label=f"contact (dist<{args.contact_thresh})")
        ax2.axhline(args.contact_thresh, color="r", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Signed distance")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "foot_ground_dist.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[6] Saved plot → {plot_path}")

    # ------------------------------------------------------------------
    # 7. Save plane
    # ------------------------------------------------------------------
    plane_path = out_dir / "ground_plane.npz"
    np.savez(plane_path, normal=normal, d=np.array(d),
             floor_prior_Y=np.array(floor_prior_Y),
             inlier_ratio=np.array(inlier_ratio),
             ransac_thresh=np.array(args.ransac_thresh))
    print(f"[7] Saved plane → {plane_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"Ground plane : normal=[{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]  d={d:.4f}")
    print(f"Inlier ratio : {100*inlier_ratio:.1f}%")
    print(f"Signed dist  : min={dists.min():.3f}  max={dists.max():.3f}  mean={dists.mean():.3f}")
    if args.contact_thresh is not None:
        n_c = (dists < args.contact_thresh).sum()
        print(f"Contact frames: {n_c} / {n_frames}  (thresh={args.contact_thresh})")


if __name__ == "__main__":
    main()
