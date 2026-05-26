#!/usr/bin/env python3
"""Evaluate scene Gaussian center alignment to a depth-prior surface.

This script is intentionally geometry-first: it ignores rendered RGB metrics and
asks whether scene Gaussian centers are supported by a scaled depth map when
projected into the training cameras.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from hugs.datasets.neuman_utils import neuman_helper


def parse_frames(text, nframes):
    if text == "all":
        return list(range(nframes))
    frames = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            vals = [int(v) for v in item.split(":")]
            if len(vals) == 2:
                start, stop = vals
                step = 1
            elif len(vals) == 3:
                start, stop, step = vals
            else:
                raise ValueError(f"Bad frame range: {item}")
            frames.extend(range(start, stop, step))
        else:
            frames.append(int(item))
    return [f for f in frames if 0 <= f < nframes]


def load_scene_points(ply_path, max_points=0, seed=0):
    ply = PlyData.read(str(ply_path), mmap=True)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    if "part" in v.dtype.names:
        xyz = xyz[v["part"] == 0]
    total = len(xyz)
    if max_points and total > max_points:
        rng = np.random.default_rng(seed)
        xyz = xyz[rng.choice(total, max_points, replace=False)]
    return xyz, total


def evaluate_one(ply_path, scene, seq_dir, depth_dir, frames, max_points, mask_human, seed):
    points, total_points = load_scene_points(ply_path, max_points=max_points, seed=seed)
    n = len(points)
    best_abs = np.full(n, np.inf, dtype=np.float64)
    best_signed = np.full(n, np.nan, dtype=np.float64)
    hit_count = np.zeros(n, dtype=np.int32)

    depth_dir = Path(depth_dir)
    mask_dir = Path(seq_dir) / "4d_humans" / "sam_segmentations"
    masks = sorted(mask_dir.glob("*.png"))

    homog = np.concatenate([points, np.ones((n, 1), dtype=np.float64)], axis=1)
    frame_summaries = []

    for fid in frames:
        depth_path = depth_dir / f"{fid:05d}_depth.npy"
        if not depth_path.exists():
            depth_path = depth_dir / f"{fid:06d}.npy"
        if not depth_path.exists():
            continue
        depth = np.load(depth_path).astype(np.float64)

        cap = scene.captures[fid]
        k = cap.intrinsic_matrix.astype(np.float64)
        w2c = cap.cam_pose.world_to_camera.astype(np.float64)
        height, width = int(cap.size[0]), int(cap.size[1])
        if depth.shape != (height, width):
            depth = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float64)

        valid_depth = np.isfinite(depth) & (depth > 0)
        if mask_human and fid < len(masks):
            mask = cv2.imread(str(masks[fid]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                valid_depth &= mask < 127

        cam = (w2c @ homog.T).T[:, :3]
        z = cam[:, 2]
        positive = z > 1e-4
        u = np.rint(k[0, 0] * (cam[:, 0] / np.maximum(z, 1e-8)) + k[0, 2]).astype(np.int64)
        v = np.rint(k[1, 1] * (cam[:, 1] / np.maximum(z, 1e-8)) + k[1, 2]).astype(np.int64)
        inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not inside.any():
            frame_summaries.append({"frame": fid, "supported_points": 0})
            continue

        idx = np.where(inside)[0]
        uu = u[idx]
        vv = v[idx]
        valid = valid_depth[vv, uu]
        if not valid.any():
            frame_summaries.append({"frame": fid, "supported_points": 0})
            continue

        idx = idx[valid]
        uu = u[idx]
        vv = v[idx]
        residual = z[idx] - depth[vv, uu]
        abs_residual = np.abs(residual)
        hit_count[idx] += 1

        improve = abs_residual < best_abs[idx]
        update_idx = idx[improve]
        best_abs[update_idx] = abs_residual[improve]
        best_signed[update_idx] = residual[improve]

        frame_summaries.append({
            "frame": fid,
            "supported_points": int(len(idx)),
            "residual_abs_p50": float(np.percentile(abs_residual, 50)),
            "residual_abs_p90": float(np.percentile(abs_residual, 90)),
        })

    supported = np.isfinite(best_abs)
    supported_abs = best_abs[supported]
    supported_signed = best_signed[supported]
    out = {
        "ply": str(ply_path),
        "scene_points_total_in_ply": int(total_points),
        "scene_points_sampled": int(n),
        "frames": frames,
        "supported_frac": float(supported.mean()) if n else 0.0,
        "unsupported_frac": float((~supported).mean()) if n else 0.0,
        "frame_summaries": frame_summaries,
    }
    if supported_abs.size:
        out.update({
            "best_abs_residual_p50": float(np.percentile(supported_abs, 50)),
            "best_abs_residual_p75": float(np.percentile(supported_abs, 75)),
            "best_abs_residual_p90": float(np.percentile(supported_abs, 90)),
            "best_abs_residual_p95": float(np.percentile(supported_abs, 95)),
            "best_abs_residual_mean": float(supported_abs.mean()),
            "best_signed_residual_p10": float(np.percentile(supported_signed, 10)),
            "best_signed_residual_p50": float(np.percentile(supported_signed, 50)),
            "best_signed_residual_p90": float(np.percentile(supported_signed, 90)),
            "behind_depth_frac": float((supported_signed > 0).mean()),
            "in_front_of_depth_frac": float((supported_signed < 0).mean()),
            "mean_valid_frame_hits": float(hit_count[supported].mean()),
        })
        for thr in [0.25, 0.5, 1.0, 2.0, 5.0]:
            out[f"supported_frac_abs_lt_{thr}"] = float((supported_abs < thr).mean())
            out[f"all_points_frac_abs_lt_{thr}"] = float(((best_abs < thr) & supported).mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", default="data/neuman/dataset/lab")
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--frames", default="0:103:3")
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--mask-human", action="store_true")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("plys", nargs="+")
    args = parser.parse_args()

    scene = neuman_helper.NeuManReader.read_scene(
        args.seq_dir,
        tgt_size=None,
        normalize=False,
        smpl_type="optimized",
    )
    frames = parse_frames(args.frames, len(scene.captures))
    rows = [
        evaluate_one(p, scene, args.seq_dir, args.depth_dir, frames, args.max_points, args.mask_human, seed=i)
        for i, p in enumerate(args.plys)
    ]
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    for row in rows:
        keep = {k: v for k, v in row.items() if k != "frame_summaries"}
        print(json.dumps(keep, indent=2))


if __name__ == "__main__":
    main()
