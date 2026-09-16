#!/usr/bin/env python3
"""Visualize the fitted ground plane as a PLY file.

Exports:
  - COLMAP point cloud: inliers (green), outliers (grey)
  - Ground plane grid: flat mesh over the XZ extent of the point cloud
  - SMPL foot-bottom positions per frame: red dots

Open the output PLY in MeshLab, CloudCompare, or any PLY viewer.

Usage
-----
    python scripts/vis_ground_plane.py --seq lab
    python scripts/vis_ground_plane.py --seq lab --grid-res 80 --grid-alpha 180
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# COLMAP loader (same as eval_ground_contact.py)
# ---------------------------------------------------------------------------

def load_colmap_points(path: str) -> np.ndarray:
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# SMPL foot positions (reuse logic from eval_ground_contact.py)
# ---------------------------------------------------------------------------

def load_foot_positions(seq_dir: Path, smpl_dir: str, device: str = "cpu",
                        n_bottom: int = 20) -> np.ndarray:
    import torch
    from hugs.models.modules.smpl_layer import SMPL

    smpl_npz = seq_dir / "4d_humans" / "smpl_optimized_aligned_scale.npz"
    params = dict(np.load(smpl_npz))
    smpl = SMPL(smpl_dir).to(device)
    n_frames = params["global_orient"].shape[0]
    foot_pts = []

    with torch.no_grad():
        for i in range(n_frames):
            go = torch.tensor(params["global_orient"][i:i+1], dtype=torch.float32, device=device)
            bp = torch.tensor(params["body_pose"][i:i+1],     dtype=torch.float32, device=device)
            be = torch.tensor(params["betas"][i:i+1],         dtype=torch.float32, device=device)
            tr = params["transl"][i]
            sc = float(params["scale"][i])
            verts = smpl(global_orient=go, body_pose=bp, betas=be).vertices[0].cpu().numpy()
            wv = verts * sc + tr
            idx = np.argsort(wv[:, 1])[-n_bottom:]   # Y-down: max Y = foot bottom
            foot_pts.append(wv[idx].mean(0))

    return np.stack(foot_pts, axis=0)   # (N, 3)


# ---------------------------------------------------------------------------
# PLY writer helpers
# ---------------------------------------------------------------------------

def make_vertex_array(xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """xyz (N,3) float32, rgb (N,3) uint8 → structured array."""
    dtype = [("x","f4"),("y","f4"),("z","f4"),
             ("red","u1"),("green","u1"),("blue","u1")]
    arr = np.empty(len(xyz), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    arr["red"]   = rgb[:,0]
    arr["green"] = rgb[:,1]
    arr["blue"]  = rgb[:,2]
    return arr


def make_face_array(faces: np.ndarray) -> np.ndarray:
    """faces (F,3) int → structured array with list field."""
    dtype = [("vertex_indices", "O")]
    arr = np.empty(len(faces), dtype=dtype)
    for i, f in enumerate(faces):
        arr[i] = (f.astype(np.int32),)
    return arr


def write_ply_with_mesh(out_path: Path,
                        vertices: np.ndarray,   # (N,3) float32
                        colors:   np.ndarray,   # (N,3) uint8
                        faces:    np.ndarray | None = None):  # (F,3) int
    """Write a PLY that includes vertex colours and optional triangular faces."""
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if faces is not None and len(faces):
        header_lines += [
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    vdata = np.zeros(len(vertices),
                     dtype=[("x","f4"),("y","f4"),("z","f4"),
                            ("r","u1"),("g","u1"),("b","u1")])
    vdata["x"] = vertices[:,0]
    vdata["y"] = vertices[:,1]
    vdata["z"] = vertices[:,2]
    vdata["r"] = colors[:,0]
    vdata["g"] = colors[:,1]
    vdata["b"] = colors[:,2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vdata.tobytes())
        if faces is not None and len(faces):
            for face in faces:
                f.write(np.array([3], dtype=np.uint8).tobytes())
                f.write(face.astype(np.int32).tobytes())

    print(f"  saved → {out_path}  ({len(vertices)} vertices"
          + (f", {len(faces)} triangles" if faces is not None else "") + ")")


# ---------------------------------------------------------------------------
# Build ground plane grid mesh
# ---------------------------------------------------------------------------

def build_plane_grid(normal: np.ndarray, d: float,
                     pts: np.ndarray,
                     res: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of a flat grid lying on the plane.

    The grid covers the XZ (horizontal) extent of *pts*, projected onto
    the plane.
    """
    # Two orthogonal basis vectors in the plane
    # n is the plane normal; choose u, v perpendicular to n.
    n = normal / np.linalg.norm(normal)
    # pick a vector not parallel to n
    ref = np.array([0., 1., 0.]) if abs(n[1]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u);   v /= np.linalg.norm(v)

    # Project all COLMAP points onto the plane, get their (u,v) coordinates
    # project onto plane: p_plane = p - (n·p + d)*n
    plane_pts = pts - (pts @ n + d)[:, None] * n
    u_coords = plane_pts @ u
    v_coords = plane_pts @ v

    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    # add 10% margin
    um, vm = (u_max - u_min) * 0.1, (v_max - v_min) * 0.1
    u_min -= um; u_max += um
    v_min -= vm; v_max += vm

    # Build grid
    us = np.linspace(u_min, u_max, res)
    vs = np.linspace(v_min, v_max, res)
    uu, vv = np.meshgrid(us, vs)  # (res, res)
    # Grid origin: nearest point on plane to world origin
    origin = -d * n

    grid_pts = (origin[None, None, :]
                + uu[:, :, None] * u
                + vv[:, :, None] * v)  # (res, res, 3)
    vertices = grid_pts.reshape(-1, 3).astype(np.float32)  # (res*res, 3)

    # Build faces (two triangles per quad)
    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            tl = i * res + j
            tr = tl + 1
            bl = tl + res
            br = bl + 1
            faces.append([tl, tr, bl])
            faces.append([tr, br, bl])
    faces = np.array(faces, dtype=np.int32)

    return vertices, faces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="lab")
    p.add_argument("--data-root", default="data/neuman/dataset")
    p.add_argument("--smpl-dir", default="data/smpl")
    p.add_argument("--plane-dir", default=None,
                   help="Directory containing ground_plane.npz (default: eval_contact/{seq})")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: eval_contact/{seq})")
    p.add_argument("--grid-res", type=int, default=60,
                   help="Ground plane grid resolution (default 60)")
    p.add_argument("--grid-color", nargs=3, type=int, default=[255, 200, 0],
                   metavar=("R","G","B"),
                   help="Plane grid vertex colour (default: yellow 255 200 0)")
    p.add_argument("--ransac-thresh", type=float, default=0.3,
                   help="Inlier threshold for colouring COLMAP points (default 0.3)")
    p.add_argument("--no-smpl", action="store_true",
                   help="Skip SMPL foot point export (faster, no model loading)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    seq_dir   = Path(args.data_root) / args.seq
    plane_dir = Path(args.plane_dir) if args.plane_dir else Path("eval_contact") / args.seq
    out_dir   = Path(args.out_dir) if args.out_dir else plane_dir

    # 1. Load ground plane
    plane = np.load(plane_dir / "ground_plane.npz")
    normal  = plane["normal"].astype(np.float64)
    plane_d = float(plane["d"])
    thresh  = args.ransac_thresh
    print(f"[plane] n=[{normal[0]:.4f},{normal[1]:.4f},{normal[2]:.4f}]  d={plane_d:.4f}")

    # 2. Load COLMAP points
    pts_path = seq_dir / "sparse" / "points3D.txt"
    print(f"[colmap] loading {pts_path} ...")
    pts = load_colmap_points(str(pts_path))
    dists = pts @ normal + plane_d       # signed dist to plane
    inlier = np.abs(dists) < thresh     # (N,)
    print(f"  {inlier.sum()} inliers / {len(pts)} total  (thresh={thresh})")

    # Colour COLMAP points
    colmap_colors = np.zeros((len(pts), 3), dtype=np.uint8)
    colmap_colors[inlier]  = [80, 220, 80]   # green  = inlier
    colmap_colors[~inlier] = [120, 120, 120]  # grey   = outlier

    # 3. Build ground plane grid
    print(f"[plane-grid] building {args.grid_res}×{args.grid_res} grid ...")
    grid_verts, grid_faces = build_plane_grid(normal, plane_d, pts, res=args.grid_res)
    grid_colors = np.array([args.grid_color] * len(grid_verts), dtype=np.uint8)

    # 4. SMPL foot positions
    foot_verts  = np.empty((0, 3), dtype=np.float32)
    foot_colors = np.empty((0, 3), dtype=np.uint8)
    if not args.no_smpl:
        print("[smpl] computing foot-bottom world positions ...")
        try:
            foot_pts = load_foot_positions(seq_dir, args.smpl_dir, args.device)
            foot_verts  = foot_pts.astype(np.float32)
            foot_colors = np.full((len(foot_verts), 3), [220, 60, 60], dtype=np.uint8)
            print(f"  {len(foot_verts)} foot positions")
        except Exception as e:
            print(f"  [warn] SMPL failed ({e}), skipping foot points.")

    # 5. Assemble and write PLY  ----------------------------------------
    # A: point cloud (COLMAP + foot) with colours, no faces
    all_pts    = np.concatenate([pts, foot_verts], axis=0)
    all_colors = np.concatenate([colmap_colors, foot_colors], axis=0)

    cloud_path = out_dir / "ground_plane_cloud.ply"
    write_ply_with_mesh(cloud_path, all_pts, all_colors)

    # B: plane grid mesh (separate file, easier to toggle in viewer)
    plane_path = out_dir / "ground_plane_mesh.ply"
    write_ply_with_mesh(plane_path, grid_verts, grid_colors, grid_faces)

    # C: combined (cloud + plane mesh) in one file
    combined_v = np.concatenate([all_pts, grid_verts], axis=0)
    combined_c = np.concatenate([all_colors, grid_colors], axis=0)
    # shift face indices by len(all_pts)
    shifted_faces = grid_faces + len(all_pts)
    combined_path = out_dir / "ground_plane_combined.ply"
    write_ply_with_mesh(combined_path, combined_v, combined_c, shifted_faces)

    print("\nDone. Files saved:")
    print(f"  point cloud  : {cloud_path}")
    print(f"  plane mesh   : {plane_path}")
    print(f"  combined     : {combined_path}")
    print("\nLegend:")
    print("  Green  = COLMAP inlier points (used for plane fitting)")
    print("  Grey   = COLMAP outlier points")
    print("  Yellow = fitted ground plane grid")
    print("  Red    = SMPL foot-bottom positions per frame")


if __name__ == "__main__":
    main()
