import argparse
import importlib
import json
import os
import os.path as osp
import sys
from glob import glob

import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None

from demo_align_smplx_pointcloud import (
    apply_coord_transform,
    apply_mask_if_possible,
    backproject_depth_to_points,
    build_binary_human_mask,
    build_smplx_mesh,
    chamfer_trimmed,
    draw_mesh_overlay,
    draw_projected_points,
    ensure_dir,
    filter_points_by_mask_projection,
    load_json,
    load_mask,
    load_pointcloud,
    maybe_downsample,
    points_to_nx3,
    projection_consistency_loss,
    project_points,
    remove_far_outliers_camera_depth,
    render_mesh_projection_mask,
    resolve_default_paths,
    resolve_optional_json_path,
    resolve_smplx_root_from_param,
    resolve_subject_id,
    sanitize_points,
    save_overlay_png,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global keyframe SMPL-X alignment: optimize one global scale + translation for multiple keyframes."
    )
    parser.add_argument("--subject_id", type=str, default="", help="Subject id, e.g. yogaball.")
    parser.add_argument("--seq", type=str, default="", help="Alias of subject_id.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for global alignment results.")
    parser.add_argument("--pointcloud_dir", type=str, default="", help="Optional manual point cloud directory.")
    parser.add_argument("--mask_dir", type=str, default="", help="Optional manual mask directory.")
    parser.add_argument(
        "--scene_pointcloud_path",
        type=str,
        default="",
        help="Optional scene/background point cloud file for visualization (.txt/.npy/.npz/.ply).",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--smplx_backend",
        type=str,
        default="avatar",
        choices=["avatar", "fitting"],
        help="SMPL-X backend for alignment mesh generation. Use avatar to match train geometry.",
    )
    parser.add_argument(
        "--smplx_gender",
        type=str,
        default="male",
        choices=["male", "female", "neutral"],
        help="SMPL-X gender used when backend supports gender-specific layers.",
    )

    parser.add_argument("--frame_ids", type=str, default="", help="Optional comma-separated frame ids to limit candidates.")
    parser.add_argument("--keyframe_count", type=int, default=6)
    parser.add_argument("--vis_count", type=int, default=4, help="How many keyframes to export before/after visualizations.")
    parser.add_argument(
        "--scene_vis_max_points",
        type=int,
        default=30000,
        help="Max scene points used in scene-enhanced visualization per frame.",
    )
    parser.add_argument(
        "--scene_vis_radius_scale",
        type=float,
        default=4.0,
        help="Keep scene points within radius_scale * mesh_radius around mesh center for visualization.",
    )
    parser.add_argument(
        "--scene_vis_view",
        type=str,
        default="free",
        choices=["free", "gt_camera"],
        help="free: 3D orbit view; gt_camera: render from GT camera projection view.",
    )
    parser.add_argument(
        "--export_scene_human_mesh_ply",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: export before/after PLY per vis frame, containing scene points + human points + human mesh.",
    )
    parser.add_argument(
        "--scene_vis_elev",
        type=float,
        default=15.0,
        help="Elevation used by free-view scene visualization.",
    )
    parser.add_argument(
        "--scene_vis_azims",
        type=str,
        default="-65,20,105",
        help="Comma-separated azimuth list for multi-view scene visualization, e.g. '-65,20,105'.",
    )

    parser.add_argument("--w_chamfer", type=float, default=1.0)
    parser.add_argument("--lambda_scale", type=float, default=1e-2)
    parser.add_argument("--lambda_trans", type=float, default=1e-3)
    parser.add_argument("--lambda_proj", type=float, default=5.0)
    parser.add_argument("--proj_silh_h", type=int, default=192)
    parser.add_argument("--proj_silh_w", type=int, default=256)
    parser.add_argument("--proj_silh_sigma", type=float, default=1.8)
    parser.add_argument("--proj_silh_chunk", type=int, default=512)
    parser.add_argument("--trim_ratio", type=float, default=0.10)

    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--mask_white_is_human", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--remove_far_outliers", type=int, default=1, choices=[0, 1])
    parser.add_argument("--far_outlier_percentile", type=float, default=98.5)
    parser.add_argument("--far_outlier_mad_k", type=float, default=3.5)
    parser.add_argument("--far_outlier_min_keep", type=int, default=256)

    parser.add_argument(
        "--coord_transform",
        type=str,
        default="none",
        choices=["none", "swap_yz", "neg_x", "neg_y", "neg_z"],
    )

    parser.add_argument(
        "--export_per_frame_meta",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: also export per-frame align_result.json under per_frame_meta/<frame>/.",
    )
    return parser.parse_args()


def frame_stem_candidates(frame_id_str):
    cands = [frame_id_str]
    try:
        cands.append(str(int(frame_id_str)))
    except Exception:
        pass
    out = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def parse_scene_vis_azims(azims_text):
    values = []
    for token in str(azims_text).split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            values.append(float(token))
        except Exception:
            continue
    if len(values) == 0:
        values = [-65.0, 20.0, 105.0]
    return values


def pick_frame_file_from_dir(frame_id, folder, exts):
    stems = frame_stem_candidates(frame_id)
    for stem in stems:
        for ext in exts:
            p = osp.join(folder, f"{stem}.{ext}")
            if osp.isfile(p):
                return p
    files = []
    for ext in exts:
        files.extend(glob(osp.join(folder, f"*.{ext}")))
    for stem in stems:
        for p in files:
            base = osp.splitext(osp.basename(p))[0]
            if base == stem or base.startswith(stem + "_") or base.endswith("_" + stem):
                return p
    return ""


def pick_first_frame_file_from_dirs(frame_id, dirs, exts):
    for d in dirs:
        if not osp.isdir(d):
            continue
        p = pick_frame_file_from_dir(frame_id, d, exts)
        if p:
            return p
    return ""


def resolve_pointcloud_for_frame(frame_id, paths):
    p = pick_first_frame_file_from_dirs(frame_id, paths["pointcloud_dirs"], ["npy", "npz", "ply"])
    if p:
        return {"mode": "file", "path": p}

    depth = pick_first_frame_file_from_dirs(frame_id, paths["depth_dirs"], ["npy", "npz"])
    cam = pick_first_frame_file_from_dirs(frame_id, paths["cam_dirs"], ["json"])
    if depth and cam:
        return {"mode": "depth_backproject", "path": depth, "cam_path": cam}

    return {"mode": "none"}


def collect_frame_infos(args, paths):
    smplx_jsons = []
    for root in paths["smplx_roots"]:
        smplx_jsons.extend(glob(osp.join(root, "smplx_params_smoothed", "*.json")))
        smplx_jsons.extend(glob(osp.join(root, "smplx_params", "*.json")))

    by_frame = {}
    for p in sorted(smplx_jsons):
        stem = osp.splitext(osp.basename(p))[0]
        try:
            fid = str(int(stem))
        except Exception:
            fid = stem
        if fid not in by_frame:
            by_frame[fid] = p

    if args.frame_ids.strip() != "":
        requested = [x.strip() for x in args.frame_ids.split(",") if x.strip() != ""]
        requested_norm = set()
        for r in requested:
            requested_norm.add(r)
            try:
                requested_norm.add(str(int(r)))
            except Exception:
                pass
        frame_ids = [f for f in sorted(by_frame.keys(), key=lambda x: int(x) if x.isdigit() else x) if f in requested_norm]
    else:
        frame_ids = sorted(by_frame.keys(), key=lambda x: int(x) if x.isdigit() else x)

    out = []
    for fid in frame_ids:
        smplx_param_path = by_frame[fid]
        cam_path = pick_first_frame_file_from_dirs(fid, paths["cam_dirs"], ["json"])
        mask_path = pick_first_frame_file_from_dirs(fid, paths["mask_dirs"], ["png", "jpg", "jpeg", "bmp"])
        pc_info = resolve_pointcloud_for_frame(fid, paths)
        if (cam_path == "") or (mask_path == "") or (pc_info["mode"] == "none"):
            continue
        out.append(
            {
                "frame_id": fid,
                "smplx_param_path": smplx_param_path,
                "cam_path": cam_path,
                "mask_path": mask_path,
                "pc_info": pc_info,
            }
        )
    return out


def select_keyframes(frame_infos, keyframe_count):
    n = len(frame_infos)
    if n == 0:
        return []
    if keyframe_count >= n:
        return frame_infos
    idx = np.linspace(0, n - 1, keyframe_count)
    idx = np.round(idx).astype(np.int64)
    uniq = []
    for i in idx.tolist():
        if i not in uniq:
            uniq.append(i)
    return [frame_infos[i] for i in uniq]


def resolve_scene_pointcloud_for_vis(args, paths):
    manual = args.scene_pointcloud_path.strip()
    if manual != "":
        p = osp.abspath(osp.expanduser(manual))
        if not osp.isfile(p):
            p2 = osp.join(paths["subject_root"], manual)
            p2 = osp.abspath(osp.expanduser(p2))
            if osp.isfile(p2):
                p = p2
        if not osp.isfile(p):
            raise FileNotFoundError(f"--scene_pointcloud_path not found: {manual}")
        return {"path": p, "format": "auto", "source": "manual"}

    colmap_p = osp.join(paths["subject_root"], "sparse", "points3D.txt")
    if osp.isfile(colmap_p):
        return {"path": colmap_p, "format": "colmap_points3D", "source": "auto_colmap"}

    bkg_p = osp.join(paths["subject_root"], "bkg_point_cloud.txt")
    if osp.isfile(bkg_p):
        return {"path": bkg_p, "format": "xyzrgb_txt", "source": "auto_bkg_point_cloud"}

    return {"path": "", "format": "none", "source": "none"}


def reservoir_sample_txt_xyz(path, xyz_cols, max_points, seed):
    rng = np.random.default_rng(seed)
    if max_points <= 0:
        pts = []
    else:
        pts = np.zeros((max_points, 3), dtype=np.float32)
    seen = 0

    with open(path, "r") as f:
        for line in f:
            if line == "":
                continue
            if line.startswith("#"):
                continue
            sp = line.strip().split()
            if len(sp) <= max(xyz_cols):
                continue
            try:
                p = np.array(
                    [
                        float(sp[xyz_cols[0]]),
                        float(sp[xyz_cols[1]]),
                        float(sp[xyz_cols[2]]),
                    ],
                    dtype=np.float32,
                )
            except Exception:
                continue
            if not np.isfinite(p).all():
                continue

            if max_points <= 0:
                pts.append(p)
                seen += 1
                continue

            if seen < max_points:
                pts[seen] = p
            else:
                j = int(rng.integers(0, seen + 1))
                if j < max_points:
                    pts[j] = p
            seen += 1

    if max_points <= 0:
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32), 0
        return np.asarray(pts, dtype=np.float32), int(len(pts))

    keep = min(seen, max_points)
    if keep <= 0:
        return np.zeros((0, 3), dtype=np.float32), 0
    return pts[:keep].copy(), int(seen)


def load_scene_pointcloud_for_vis(args, paths):
    info = resolve_scene_pointcloud_for_vis(args, paths)
    if info["format"] == "none":
        return np.zeros((0, 3), dtype=np.float32), {"source": "none", "raw_count": 0, "path": ""}

    path = info["path"]
    ext = osp.splitext(path)[1].lower()

    if ext in [".npy", ".npz", ".ply"]:
        raw = load_pointcloud(path)
        pts, _ = points_to_nx3(raw)
        pts = sanitize_points(pts)
        hard_cap = max(int(args.scene_vis_max_points) * 4, int(args.scene_vis_max_points), 1)
        pts, _ = maybe_downsample(pts, hard_cap, seed=args.seed)
        return pts, {"source": info["source"], "raw_count": int(pts.shape[0]), "path": path}

    if ext == ".txt":
        if info["format"] == "colmap_points3D" or osp.basename(path) == "points3D.txt":
            xyz_cols = (1, 2, 3)
        else:
            xyz_cols = (0, 1, 2)
        hard_cap = max(int(args.scene_vis_max_points) * 4, int(args.scene_vis_max_points), 1)
        pts, seen = reservoir_sample_txt_xyz(path, xyz_cols, hard_cap, seed=args.seed)
        pts = sanitize_points(pts)
        return pts, {"source": info["source"], "raw_count": int(seen), "path": path}

    raise ValueError(f"Unsupported scene point cloud extension for visualization: {path}")


def select_local_scene_points(scene_points, mesh_verts, max_points, radius_scale, seed):
    pts = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts

    mv = np.asarray(mesh_verts, dtype=np.float32).reshape(-1, 3)
    center = np.mean(mv, axis=0)
    mesh_radius = float(np.max(np.linalg.norm(mv - center[None, :], axis=1)))
    if (not np.isfinite(mesh_radius)) or (mesh_radius < 1e-4):
        mesh_radius = 1.0

    radius = max(mesh_radius * max(radius_scale, 1.0), 1e-3)
    dist = np.linalg.norm(pts - center[None, :], axis=1)
    keep = dist <= radius
    local = pts[keep] if np.any(keep) else pts

    if max_points > 0:
        local, _ = maybe_downsample(local, int(max_points), seed=seed)
    return local


def set_axes_equal(ax, xyz):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if xyz.shape[0] == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        return
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_before_after_with_scene_png(
    path,
    scene_points,
    human_points,
    verts_before,
    verts_after,
    faces,
    elev=15,
    azim=-65,
    vis_view="free",
    cam_param=None,
):
    try:
        scene_points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
        human_points = np.asarray(human_points, dtype=np.float32).reshape(-1, 3)
        verts_before = np.asarray(verts_before, dtype=np.float32).reshape(-1, 3)
        verts_after = np.asarray(verts_after, dtype=np.float32).reshape(-1, 3)

        keep_scene = np.isfinite(scene_points).all(axis=1) if scene_points.shape[0] > 0 else np.zeros((0,), dtype=bool)
        keep_human = np.isfinite(human_points).all(axis=1) if human_points.shape[0] > 0 else np.zeros((0,), dtype=bool)
        keep_before = np.isfinite(verts_before).all(axis=1)
        keep_after = np.isfinite(verts_after).all(axis=1)

        scene_points = scene_points[keep_scene] if scene_points.shape[0] > 0 else scene_points
        human_points = human_points[keep_human] if human_points.shape[0] > 0 else human_points
        verts_before = verts_before[keep_before]
        verts_after = verts_after[keep_after]

        if verts_before.shape[0] == 0 or verts_after.shape[0] == 0:
            return False, "invalid verts: non-finite or empty"

        if vis_view == "gt_camera":
            if (cv2 is None) or (cam_param is None):
                return False, "gt_camera view needs cv2 and cam_param"

            h = int(round(float(cam_param["princpt"][1] * 2)))
            w = int(round(float(cam_param["princpt"][0] * 2)))
            h = max(h, 512)
            w = max(w, 512)
            bg = np.full((h, w, 3), 18, dtype=np.uint8)

            pts_all = human_points
            if scene_points.shape[0] > 0:
                pts_all = np.concatenate([scene_points, human_points], axis=0)

            uv, _, valid = project_points(pts_all, cam_param)
            n_scene = int(scene_points.shape[0])
            valid_scene = valid[:n_scene] if n_scene > 0 else valid[:0]
            valid_human = valid[n_scene:]

            base = bg.copy()
            if n_scene > 0:
                base = draw_projected_points(base, uv[:n_scene], valid_scene, color=(240, 166, 17), radius=1, max_draw=30000)
            if human_points.shape[0] > 0:
                base = draw_projected_points(base, uv[n_scene:], valid_human, color=(70, 180, 255), radius=1, max_draw=14000)

            before = draw_mesh_overlay(base, verts_before, faces, cam_param, color=(45, 45, 220), alpha=0.40)
            after = draw_mesh_overlay(base, verts_after, faces, cam_param, color=(40, 210, 40), alpha=0.40)

            cv2.putText(before, "Before + Scene (GT camera)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(after, "After + Scene (GT camera)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (240, 240, 240), 2, cv2.LINE_AA)
            cmp_img = np.concatenate([before, after], axis=1)
            ok = bool(cv2.imwrite(path, cmp_img))
            return ok, ""

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        tri = np.asarray(faces, dtype=np.int64)
        if tri.shape[0] > 50000:
            step = max(1, int(np.ceil(float(tri.shape[0]) / 50000.0)))
            tri = tri[::step]

        def _panel_xyz(verts):
            chunks = []
            if scene_points.shape[0] > 0:
                chunks.append(scene_points)
            if human_points.shape[0] > 0:
                chunks.append(human_points)
            chunks.append(verts)
            return np.concatenate(chunks, axis=0)

        def _draw_wireframe(ax, verts, tri, color):
            n_faces = tri.shape[0]
            if n_faces <= 0:
                return
            keep_faces = min(12000, n_faces)
            step = max(1, int(np.ceil(float(n_faces) / float(keep_faces))))
            tri_show = tri[::step]
            for f in tri_show:
                i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
                p0 = verts[i0]
                p1 = verts[i1]
                p2 = verts[i2]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=0.2, alpha=0.55)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=0.2, alpha=0.55)
                ax.plot([p2[0], p0[0]], [p2[1], p0[1]], [p2[2], p0[2]], color=color, linewidth=0.2, alpha=0.55)

        def _draw(ax, verts, title, color):
            if scene_points.shape[0] > 0:
                ax.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2], s=0.30, c="#f59e0b", alpha=0.38)
            if human_points.shape[0] > 0:
                ax.scatter(human_points[:, 0], human_points[:, 1], human_points[:, 2], s=0.55, c="#38bdf8", alpha=0.72)

            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=tri,
                color=color,
                linewidth=0.0,
                alpha=0.50,
                antialiased=False,
                shade=True,
            )

            _draw_wireframe(ax, verts, tri, color="#1f2937")
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.12, c=color, alpha=0.85)

            set_axes_equal(ax, _panel_xyz(verts))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title(title)

        _draw(ax1, verts_before, "Before + Scene", "red")
        _draw(ax2, verts_after, "After + Scene", "green")

        plt.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True, ""
    except Exception as e:
        return False, str(e)


def save_single_with_scene_3d_png(
    path,
    scene_points,
    human_points,
    verts,
    faces,
    title,
    mesh_color,
    elev,
    azim,
):
    try:
        scene_points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
        human_points = np.asarray(human_points, dtype=np.float32).reshape(-1, 3)
        verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)

        keep_scene = np.isfinite(scene_points).all(axis=1) if scene_points.shape[0] > 0 else np.zeros((0,), dtype=bool)
        keep_human = np.isfinite(human_points).all(axis=1) if human_points.shape[0] > 0 else np.zeros((0,), dtype=bool)
        keep_verts = np.isfinite(verts).all(axis=1)

        scene_points = scene_points[keep_scene] if scene_points.shape[0] > 0 else scene_points
        human_points = human_points[keep_human] if human_points.shape[0] > 0 else human_points
        verts = verts[keep_verts]
        if verts.shape[0] == 0:
            return False, "invalid verts: non-finite or empty"

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tri = np.asarray(faces, dtype=np.int64)
        if tri.shape[0] > 50000:
            step = max(1, int(np.ceil(float(tri.shape[0]) / 50000.0)))
            tri = tri[::step]

        fig = plt.figure(figsize=(7.2, 7.2))
        ax = fig.add_subplot(111, projection="3d")

        if scene_points.shape[0] > 0:
            ax.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2], s=0.30, c="#f59e0b", alpha=0.40)
        if human_points.shape[0] > 0:
            ax.scatter(human_points[:, 0], human_points[:, 1], human_points[:, 2], s=0.55, c="#38bdf8", alpha=0.74)

        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=tri,
            color=mesh_color,
            linewidth=0.0,
            alpha=0.52,
            antialiased=False,
            shade=True,
        )
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.12, c=mesh_color, alpha=0.86)

        chunks = []
        if scene_points.shape[0] > 0:
            chunks.append(scene_points)
        if human_points.shape[0] > 0:
            chunks.append(human_points)
        chunks.append(verts)
        set_axes_equal(ax, np.concatenate(chunks, axis=0))

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True, ""
    except Exception as e:
        return False, str(e)


def save_camera_before_after(path_before, path_after, points, verts_before, verts_after, faces, cam_param):
    if cv2 is None:
        return False
    h = int(round(float(cam_param["princpt"][1] * 2)))
    w = int(round(float(cam_param["princpt"][0] * 2)))
    h = max(h, 512)
    w = max(w, 512)
    bg = np.full((h, w, 3), 18, dtype=np.uint8)

    uv, _, valid = project_points(points, cam_param)
    base = draw_projected_points(bg, uv, valid, color=(145, 145, 145), radius=1, max_draw=9000)
    before = draw_mesh_overlay(base, verts_before, faces, cam_param, color=(40, 40, 220), alpha=0.35)
    after = draw_mesh_overlay(base, verts_after, faces, cam_param, color=(40, 200, 40), alpha=0.35)

    cv2.putText(before, "Before (cam view)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(after, "After (cam view)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    ok1 = bool(cv2.imwrite(path_before, before))
    ok2 = bool(cv2.imwrite(path_after, after))
    return ok1 and ok2


def write_ascii_ply_vertex(path, vertices_xyz, vertices_rgb):
    vertices_xyz = np.asarray(vertices_xyz, dtype=np.float32).reshape(-1, 3)
    vertices_rgb = np.asarray(vertices_rgb, dtype=np.uint8).reshape(-1, 3)

    if vertices_xyz.shape[0] != vertices_rgb.shape[0]:
        raise ValueError("vertices_xyz and vertices_rgb size mismatch")

    ensure_dir(osp.dirname(path))
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {int(vertices_xyz.shape[0])}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(vertices_xyz.shape[0]):
            x, y, z = vertices_xyz[i]
            r, g, b = vertices_rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def save_scene_human_mesh_ply(
    path,
    scene_points,
    human_points,
    mesh_verts,
    mesh_faces,
    mesh_rgb=(210, 80, 220),
    scene_rgb=(245, 158, 11),
    human_rgb=(56, 189, 248),
):
    try:
        scene_points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
        human_points = np.asarray(human_points, dtype=np.float32).reshape(-1, 3)
        mesh_verts = np.asarray(mesh_verts, dtype=np.float32).reshape(-1, 3)
        _ = np.asarray(mesh_faces, dtype=np.int64).reshape(-1, 3)

        if scene_points.shape[0] > 0:
            scene_points = scene_points[np.isfinite(scene_points).all(axis=1)]
        if human_points.shape[0] > 0:
            human_points = human_points[np.isfinite(human_points).all(axis=1)]
        if mesh_verts.shape[0] == 0:
            return False, "empty mesh verts"
        if not np.isfinite(mesh_verts).all():
            return False, "mesh verts contain non-finite values"

        n_scene = int(scene_points.shape[0])
        n_human = int(human_points.shape[0])
        n_mesh = int(mesh_verts.shape[0])

        vertices_xyz = np.concatenate([scene_points, human_points, mesh_verts], axis=0)
        scene_color = np.tile(np.asarray(scene_rgb, dtype=np.uint8).reshape(1, 3), (n_scene, 1))
        human_color = np.tile(np.asarray(human_rgb, dtype=np.uint8).reshape(1, 3), (n_human, 1))
        mesh_color = np.tile(np.asarray(mesh_rgb, dtype=np.uint8).reshape(1, 3), (n_mesh, 1))
        vertices_rgb = np.concatenate([scene_color, human_color, mesh_color], axis=0)

        write_ascii_ply_vertex(path, vertices_xyz, vertices_rgb)
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scene_vis_azims = parse_scene_vis_azims(args.scene_vis_azims)

    subject_id = resolve_subject_id(args)
    paths = resolve_default_paths(args, subject_id)

    out_root = osp.join(args.out_dir, f"global_align_{subject_id}")
    ensure_dir(out_root)

    frame_infos = collect_frame_infos(args, paths)
    if len(frame_infos) == 0:
        raise RuntimeError("No valid frames found with smplx+cam+mask+pointcloud/depth.")

    key_infos = select_keyframes(frame_infos, args.keyframe_count)
    if len(key_infos) == 0:
        raise RuntimeError("No keyframes selected.")

    print("[INFO] total valid frames:", len(frame_infos))
    print("[INFO] selected keyframes:", [x["frame_id"] for x in key_infos])

    # Setup SMPL-X backend.
    repo_root = paths["repo_root"]
    if args.smplx_backend == "avatar":
        sys.path.insert(0, osp.join(repo_root, "avatar", "main"))
        sys.path.insert(0, osp.join(repo_root, "avatar", "common"))
        cfg_module = importlib.import_module("config")
        cfg = cfg_module.cfg
        cfg.smplx_gender = args.smplx_gender
        cfg.human_model_path = osp.join(repo_root, "avatar", "common", "human_model_files")
        smpl_module = importlib.import_module("utils.smpl_x")
        smpl_x = smpl_module.smpl_x
    else:
        sys.path.insert(0, osp.join(repo_root, "fitting", "main"))
        sys.path.insert(0, osp.join(repo_root, "fitting", "common"))
        cfg_module = importlib.import_module("config")
        cfg = cfg_module.cfg
        cfg.human_model_path = osp.join(repo_root, "fitting", "common", "utils", "human_model_files")
        smpl_module = importlib.import_module("utils.smpl_x")
        smpl_x = smpl_module.smpl_x
    setattr(smpl_x, "align_gender", args.smplx_gender)
    print(f"[INFO] SMPL-X backend: {args.smplx_backend}, gender: {args.smplx_gender}")

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] CUDA unavailable, fallback to CPU.")
        device = "cpu"
    device = torch.device(device)
    if isinstance(smpl_x.layer, dict):
        for k in list(smpl_x.layer.keys()):
            smpl_x.layer[k] = smpl_x.layer[k].to(device)
    else:
        smpl_x.layer = smpl_x.layer.to(device)

    # ID-specific files from first keyframe smplx root.
    smplx_root = resolve_smplx_root_from_param(key_infos[0]["smplx_param_path"])
    shape_param_path = resolve_optional_json_path("", osp.join(smplx_root, "shape_param.json"))
    joint_offset_path = resolve_optional_json_path("", osp.join(smplx_root, "joint_offset.json"))
    face_offset_path = resolve_optional_json_path("", osp.join(smplx_root, "face_offset.json"))
    locator_offset_path = resolve_optional_json_path("", osp.join(smplx_root, "locator_offset.json"))

    shape_param = load_json(shape_param_path) if shape_param_path else [0.0] * smpl_x.shape_param_dim
    joint_offset = load_json(joint_offset_path) if joint_offset_path else None
    face_offset = load_json(face_offset_path) if face_offset_path else None
    locator_offset = load_json(locator_offset_path) if locator_offset_path else None

    key_records = []
    faces_np = None

    for k in key_infos:
        frame_id = k["frame_id"]
        cam_param = load_json(k["cam_path"])
        mask = load_mask(k["mask_path"])
        human_mask_bin = build_binary_human_mask(mask, args.mask_threshold, bool(args.mask_white_is_human))

        if k["pc_info"]["mode"] == "file":
            points_raw = load_pointcloud(k["pc_info"]["path"])
        else:
            depth = load_pointcloud(k["pc_info"]["path"])
            if isinstance(depth, np.ndarray) and depth.ndim == 3:
                depth = depth[..., 0]
            points_raw = backproject_depth_to_points(depth, cam_param)

        points_nx3, hw_shape = points_to_nx3(points_raw)
        human_points, mask_applied = apply_mask_if_possible(
            points_raw,
            points_nx3,
            hw_shape,
            mask,
            args.mask_threshold,
            args.mask_white_is_human,
        )
        if (not mask_applied):
            human_points = filter_points_by_mask_projection(
                human_points,
                cam_param,
                human_mask_bin,
                min_keep_points=32,
                fallback_to_input=True,
            )

        human_points = sanitize_points(human_points)
        if args.remove_far_outliers == 1:
            human_points, _, _ = remove_far_outliers_camera_depth(
                human_points,
                cam_param,
                percentile=args.far_outlier_percentile,
                mad_k=args.far_outlier_mad_k,
                min_keep=args.far_outlier_min_keep,
            )
        human_points, _ = maybe_downsample(human_points, args.max_points, seed=args.seed)
        human_points, _ = apply_coord_transform(human_points, args.coord_transform)
        if human_points.shape[0] < 32:
            print(f"[WARN] skip keyframe {frame_id}: too few points ({human_points.shape[0]})")
            continue

        smplx_param = load_json(k["smplx_param_path"])
        verts_init, faces_t, _ = build_smplx_mesh(
            smpl_x,
            smplx_param,
            shape_param,
            joint_offset,
            face_offset,
            locator_offset,
            device,
        )
        verts_init_np = verts_init.detach().cpu().numpy()
        if faces_np is None:
            faces_np = faces_t.detach().cpu().numpy()

        # Keep optimization in world coordinates, projection computed via camera extrinsic.
        R_t = torch.tensor(cam_param["R"], dtype=torch.float32, device=device)
        t_t = torch.tensor(cam_param["t"], dtype=torch.float32, device=device)
        focal_t = torch.tensor(cam_param["focal"], dtype=torch.float32, device=device)
        princpt_t = torch.tensor(cam_param["princpt"], dtype=torch.float32, device=device)

        target_mask = render_mesh_projection_mask(
            verts_init_np,
            faces_np,
            cam_param,
            image_shape=(int(human_mask_bin.shape[0]), int(human_mask_bin.shape[1])),
        ).astype(np.float32)
        target_mask_t = torch.from_numpy(target_mask).to(device=device, dtype=torch.float32)[None, None, :, :]

        key_records.append(
            {
                "frame_id": frame_id,
                "cam_param": cam_param,
                "points_np": human_points,
                "points_t": torch.from_numpy(human_points).to(device),
                "verts_init": verts_init,
                "verts_init_np": verts_init_np,
                "R_t": R_t,
                "t_t": t_t,
                "focal_t": focal_t,
                "princpt_t": princpt_t,
                "target_mask_t": target_mask_t,
                "image_shape": (int(human_mask_bin.shape[0]), int(human_mask_bin.shape[1])),
            }
        )

    if len(key_records) == 0:
        raise RuntimeError("No keyframes survived preprocessing.")

    print("[INFO] usable keyframes:", [x["frame_id"] for x in key_records])

    scene_points_vis = np.zeros((0, 3), dtype=np.float32)
    scene_meta = {"source": "none", "raw_count": 0, "path": ""}
    if args.scene_pointcloud_path.strip() != "":
        scene_points_vis, scene_meta = load_scene_pointcloud_for_vis(args, paths)
    else:
        try:
            scene_points_vis, scene_meta = load_scene_pointcloud_for_vis(args, paths)
        except Exception as e:
            print(f"[WARN] failed to auto-load scene point cloud for visualization: {e}")
            scene_points_vis = np.zeros((0, 3), dtype=np.float32)
            scene_meta = {"source": "none", "raw_count": 0, "path": ""}

    if scene_points_vis.shape[0] > 0:
        print(
            f"[INFO] scene point cloud for vis loaded: {scene_points_vis.shape[0]} sampled points "
            f"(source={scene_meta['source']}, raw={scene_meta['raw_count']}, path={scene_meta['path']})"
        )
    else:
        print("[WARN] scene point cloud for vis unavailable; scene-enhanced images will be skipped.")

    log_scale = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))
    t_world = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_scale, t_world], lr=args.lr)

    best = {
        "loss": float("inf"),
        "iter": -1,
        "scale": 1.0,
        "t_world": np.zeros(3, dtype=np.float32),
    }

    for it in range(args.iters):
        optimizer.zero_grad()
        scale = torch.exp(log_scale)

        loss_chamfer_sum = torch.zeros(1, dtype=torch.float32, device=device)
        loss_proj_sum = torch.zeros(1, dtype=torch.float32, device=device)
        proj_iou_sum = torch.zeros(1, dtype=torch.float32, device=device)

        for rec in key_records:
            verts = scale * rec["verts_init"] + t_world[None, :]
            d_vm, d_pv = chamfer_trimmed(verts, rec["points_t"], trim_ratio=args.trim_ratio, chunk_size=2048)
            loss_chamfer_sum = loss_chamfer_sum + 0.5 * (d_vm + d_pv)

            if args.lambda_proj > 0:
                verts_cam = torch.matmul(verts, rec["R_t"].t()) + rec["t_t"].view(1, 3)
                loss_proj, _, _, proj_iou = projection_consistency_loss(
                    verts_cam,
                    {"focal": rec["focal_t"], "princpt": rec["princpt_t"]},
                    rec["target_mask_t"],
                    image_shape=rec["image_shape"],
                    silh_shape=(args.proj_silh_h, args.proj_silh_w),
                    sigma=args.proj_silh_sigma,
                    chunk_size=args.proj_silh_chunk,
                )
                loss_proj_sum = loss_proj_sum + loss_proj
                proj_iou_sum = proj_iou_sum + proj_iou

        kf_num = float(len(key_records))
        loss_chamfer = loss_chamfer_sum / kf_num
        loss_proj = loss_proj_sum / kf_num
        proj_iou_avg = proj_iou_sum / kf_num

        reg_scale = (scale - 1.0) ** 2
        reg_trans = torch.sum(t_world ** 2)

        total_loss = (
            args.w_chamfer * loss_chamfer
            + args.lambda_proj * loss_proj
            + args.lambda_scale * reg_scale
            + args.lambda_trans * reg_trans
        )

        cur = float(total_loss.detach().cpu().item())
        if cur < best["loss"]:
            best["loss"] = cur
            best["iter"] = it
            best["scale"] = float(scale.detach().cpu().item())
            best["t_world"] = t_world.detach().cpu().numpy().astype(np.float32)

        total_loss.backward()
        optimizer.step()

        if (it + 1) % 20 == 0 or it == 0 or (it + 1) == args.iters:
            print(
                f"[OPT] iter {it+1:04d}/{args.iters} "
                f"loss={cur:.6f} "
                f"chamfer={float(loss_chamfer.detach().cpu().item()):.6f} "
                f"proj={float(loss_proj.detach().cpu().item()):.6f} "
                f"proj_iou={float(proj_iou_avg.detach().cpu().item()):.4f} "
                f"scale={float(scale.detach().cpu().item()):.6f} "
                f"t_world={t_world.detach().cpu().numpy().tolist()}"
            )

    global_scale = float(best["scale"])
    global_t_world = best["t_world"].reshape(3)

    # Build train/export-compatible per-frame records.
    frame_records = {}
    for fi in frame_infos:
        frame_id = str(int(fi["frame_id"]))
        cam_param = load_json(fi["cam_path"])
        R = np.asarray(cam_param["R"], dtype=np.float32)
        t = np.asarray(cam_param["t"], dtype=np.float32).reshape(3)

        t_align_cam = (R @ global_t_world.reshape(3, 1)).reshape(3) + (1.0 - global_scale) * t
        rec = {
            "frame_id": int(frame_id),
            "subject_id": subject_id,
            "optimize_in_cam": True,
            "scale": float(global_scale),
            "alpha": 0.0,
            "view_dir_cam": [0.0, 0.0, 1.0],
            "t_align_cam": [float(x) for x in t_align_cam.tolist()],
            "cam_R": R.tolist(),
            "cam_t": t.tolist(),
            "focal": np.asarray(cam_param["focal"], dtype=np.float32).tolist(),
            "princpt": np.asarray(cam_param["princpt"], dtype=np.float32).tolist(),
        }
        frame_records[frame_id] = rec

    summary = {
        "subject_id": subject_id,
        "global": {
            "smplx_backend": args.smplx_backend,
            "smplx_gender": args.smplx_gender,
            "best_iter": int(best["iter"]),
            "best_loss": float(best["loss"]),
            "scale": float(global_scale),
            "t_world": [float(x) for x in global_t_world.tolist()],
            "num_valid_frames": int(len(frame_infos)),
            "num_keyframes": int(len(key_records)),
            "keyframes": [int(x["frame_id"]) for x in key_records],
        },
        "frames": frame_records,
    }

    summary_path = osp.join(out_root, "global_align_for_train.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] saved global alignment metadata:", summary_path)

    if args.export_per_frame_meta == 1:
        for frame_id, rec in frame_records.items():
            fp = osp.join(out_root, "per_frame_meta", frame_id)
            ensure_dir(fp)
            with open(osp.join(fp, "align_result.json"), "w") as f:
                json.dump(rec, f, indent=2)
        print("[INFO] saved per-frame metadata under:", osp.join(out_root, "per_frame_meta"))

    # Visualization for a subset of keyframes.
    vis_num = max(1, min(args.vis_count, len(key_records)))
    vis_records = select_keyframes(key_records, vis_num)
    vis_report = []

    for rec in vis_records:
        frame_id = rec["frame_id"]
        vis_dir = osp.join(out_root, "vis", str(frame_id).zfill(6))
        ensure_dir(vis_dir)

        verts_before = rec["verts_init_np"]
        verts_after = (global_scale * rec["verts_init"] + torch.from_numpy(global_t_world).to(device)[None, :]).detach().cpu().numpy()
        points_vis = rec["points_np"]

        save_overlay_png(osp.join(vis_dir, "before_3d.png"), points_vis, verts_before, faces_np, mesh_color="red")
        save_overlay_png(osp.join(vis_dir, "after_3d.png"), points_vis, verts_after, faces_np, mesh_color="green")

        ok_scene_3d = False
        ok_scene_multiview = False
        ok_before_scene_human_mesh_ply = False
        ok_after_scene_human_mesh_ply = False
        scene_multiview_count = 0
        scene_local_count = 0
        scene_local = np.zeros((0, 3), dtype=np.float32)
        if scene_points_vis.shape[0] > 0:
            seed_i = int(args.seed)
            try:
                seed_i += int(frame_id)
            except Exception:
                pass
            scene_local = select_local_scene_points(
                scene_points_vis,
                verts_before,
                max_points=args.scene_vis_max_points,
                radius_scale=args.scene_vis_radius_scale,
                seed=seed_i,
            )
            scene_local_count = int(scene_local.shape[0])
            ok_scene_3d, _ = save_before_after_with_scene_png(
                osp.join(
                    vis_dir,
                    "before_after_cam_with_scene.png" if args.scene_vis_view == "gt_camera" else "before_after_3d_with_scene.png",
                ),
                scene_local,
                points_vis,
                verts_before,
                verts_after,
                faces_np,
                elev=float(args.scene_vis_elev),
                vis_view=args.scene_vis_view,
                cam_param=rec["cam_param"],
            )

            if args.scene_vis_view == "free":
                scene_mv_dir = osp.join(vis_dir, "scene_multiview")
                ensure_dir(scene_mv_dir)
                ok_flags = []
                for vidx, azim in enumerate(scene_vis_azims):
                    before_name = f"before_view_{vidx:02d}_e{int(round(float(args.scene_vis_elev)))}_a{int(round(float(azim)))}.png"
                    after_name = f"after_view_{vidx:02d}_e{int(round(float(args.scene_vis_elev)))}_a{int(round(float(azim)))}.png"

                    ok_b, _ = save_single_with_scene_3d_png(
                        osp.join(scene_mv_dir, before_name),
                        scene_local,
                        points_vis,
                        verts_before,
                        faces_np,
                        title=f"Before + Scene (view {vidx+1})",
                        mesh_color="red",
                        elev=float(args.scene_vis_elev),
                        azim=float(azim),
                    )
                    ok_a, _ = save_single_with_scene_3d_png(
                        osp.join(scene_mv_dir, after_name),
                        scene_local,
                        points_vis,
                        verts_after,
                        faces_np,
                        title=f"After + Scene (view {vidx+1})",
                        mesh_color="green",
                        elev=float(args.scene_vis_elev),
                        azim=float(azim),
                    )
                    ok_flags.append(bool(ok_b and ok_a))

                scene_multiview_count = int(len(scene_vis_azims))
                ok_scene_multiview = bool(len(ok_flags) > 0 and all(ok_flags))

        if args.export_scene_human_mesh_ply == 1:
            ok_before_scene_human_mesh_ply, _ = save_scene_human_mesh_ply(
                osp.join(vis_dir, "before_scene_human_mesh.ply"),
                scene_local,
                points_vis,
                verts_before,
                faces_np,
                mesh_rgb=(210, 80, 220),
            )
            ok_after_scene_human_mesh_ply, _ = save_scene_human_mesh_ply(
                osp.join(vis_dir, "after_scene_human_mesh.ply"),
                scene_local,
                points_vis,
                verts_after,
                faces_np,
                mesh_rgb=(210, 80, 220),
            )

        ok_cam = save_camera_before_after(
            osp.join(vis_dir, "before_cam.png"),
            osp.join(vis_dir, "after_cam.png"),
            points_vis,
            verts_before,
            verts_after,
            faces_np,
            rec["cam_param"],
        )

        mask_before = render_mesh_projection_mask(
            verts_before,
            faces_np,
            rec["cam_param"],
            image_shape=rec["image_shape"],
        ).astype(np.float32)
        mask_after = render_mesh_projection_mask(
            verts_after,
            faces_np,
            rec["cam_param"],
            image_shape=rec["image_shape"],
        ).astype(np.float32)
        inter = float(np.logical_and(mask_before > 0.5, mask_after > 0.5).sum())
        union = float(np.logical_or(mask_before > 0.5, mask_after > 0.5).sum())
        iou = inter / max(union, 1.0)

        vis_report.append(
            {
                "frame_id": int(frame_id),
                "projection_iou_before_after": float(iou),
                "camera_vis_saved": bool(ok_cam),
                "scene_enhanced_vis_saved": bool(ok_scene_3d),
                "scene_multiview_saved": bool(ok_scene_multiview),
                "scene_multiview_count": int(scene_multiview_count),
                "before_scene_human_mesh_ply_saved": bool(ok_before_scene_human_mesh_ply),
                "after_scene_human_mesh_ply_saved": bool(ok_after_scene_human_mesh_ply),
                "scene_points_used": int(scene_local_count),
            }
        )

    vis_report_path = osp.join(out_root, "vis", "vis_report.json")
    ensure_dir(osp.dirname(vis_report_path))
    with open(vis_report_path, "w") as f:
        json.dump(vis_report, f, indent=2)

    print("[INFO] saved visualization report:", vis_report_path)
    print("[DONE] outputs in:", out_root)


if __name__ == "__main__":
    main()
