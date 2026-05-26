#!/usr/bin/env python
"""Align a converted Human3R scene point cloud into a NeuMan scene frame."""

import argparse
import json
from pathlib import Path

import numpy as np

from hugs.datasets.neuman_utils import neuman_helper


def umeyama(src, dst):
    """Return scale, rotation, translation mapping src -> dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, s, vh = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vh
    var = (src_c ** 2).sum() / src.shape[0]
    scale = float((s * d).sum() / max(var, 1e-12))
    t = dst_mean - scale * (r @ src_mean)
    return scale, r.astype(np.float64), t.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human3r-hugs-dir", required=True)
    parser.add_argument("--neuman-seq-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-points", type=int, default=400000)
    parser.add_argument("--step-percentile", type=float, default=90.0)
    args = parser.parse_args()

    h3r_dir = Path(args.human3r_hugs_dir)
    neuman_dir = Path(args.neuman_seq_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    h3r_cams = np.load(h3r_dir / "cameras.npz")
    h3r_centers = h3r_cams["c2w"][:, :3, 3].astype(np.float64)

    scene = neuman_helper.NeuManReader.read_scene(
        str(neuman_dir),
        tgt_size=None,
        normalize=False,
        smpl_type="optimized",
    )
    neu_centers = np.stack(
        [cap.cam_pose.camera_center_in_world for cap in scene.captures],
        axis=0,
    ).astype(np.float64)

    n = min(len(h3r_centers), len(neu_centers))
    h3r_centers = h3r_centers[:n]
    neu_centers = neu_centers[:n]

    keep = np.ones(n, dtype=bool)
    if n > 2:
        h3r_steps = np.linalg.norm(np.diff(h3r_centers, axis=0), axis=1)
        thr = np.percentile(h3r_steps, args.step_percentile)
        bad_edge = h3r_steps > thr
        keep[:-1] &= ~bad_edge
        keep[1:] &= ~bad_edge
    if keep.sum() < 6:
        keep[:] = True

    scale, rot, trans = umeyama(h3r_centers[keep], neu_centers[keep])

    pts_npz = np.load(h3r_dir / "scene" / "points.npz")
    points = pts_npz["points"].astype(np.float64)
    colors = pts_npz["colors"].astype(np.float32)
    if args.max_points > 0 and points.shape[0] > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=args.max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    aligned = scale * (points @ rot.T) + trans
    np.savez_compressed(out, points=aligned.astype(np.float32), colors=colors.astype(np.float32))

    pred = scale * (h3r_centers @ rot.T) + trans
    err = np.linalg.norm(pred - neu_centers, axis=1)
    report = {
        "human3r_hugs_dir": str(h3r_dir),
        "neuman_seq_dir": str(neuman_dir),
        "out": str(out),
        "num_camera_pairs": int(n),
        "num_used_pairs": int(keep.sum()),
        "step_percentile": args.step_percentile,
        "sim3_scale": scale,
        "sim3_rotation": rot.tolist(),
        "sim3_translation": trans.tolist(),
        "camera_fit_error_mean": float(err.mean()),
        "camera_fit_error_median": float(np.median(err)),
        "camera_fit_error_max": float(err.max()),
        "num_points": int(aligned.shape[0]),
        "point_min": aligned.min(axis=0).tolist(),
        "point_max": aligned.max(axis=0).tolist(),
        "point_std": aligned.std(axis=0).tolist(),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
