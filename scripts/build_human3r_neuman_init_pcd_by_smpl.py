#!/usr/bin/env python
"""Align a Human3R scene point cloud to NeuMan/HUGS coordinates using SMPL bodies."""

import argparse
import json
from pathlib import Path

import numpy as np

from hugs.datasets.neuman_utils.smpl import SMPL


def umeyama(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, s, vh = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        sign[-1] = -1
    rot = u @ np.diag(sign) @ vh
    var = (src_c ** 2).sum() / src.shape[0]
    scale = float((s * sign).sum() / max(var, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def load_params(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def smpl_world_vertices(model, params, frame_ids, vertex_ids):
    verts = []
    for frame_id in frame_ids:
        full_pose = np.concatenate(
            [params["global_orient"][frame_id], params["body_pose"][frame_id]],
            axis=0,
        )[None].astype(np.float32)
        betas = params["betas"][frame_id][None].astype(np.float32)
        if betas.shape[1] < model.shapedirs.shape[-1]:
            betas = np.pad(betas, ((0, 0), (0, model.shapedirs.shape[-1] - betas.shape[1])))
        v = model(
            poses=full_pose,
            betas=betas,
            return_tensor=False,
        )
        v = v[vertex_ids]
        scale = float(params["scale"][frame_id])
        transl = params["transl"][frame_id].astype(np.float64)
        verts.append(v.astype(np.float64) * scale + transl)
    return np.concatenate(verts, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human3r-hugs-dir", required=True)
    parser.add_argument("--neuman-smpl", default="data/neuman/dataset/lab/4d_humans/smpl_optimized_aligned_scale.npz")
    parser.add_argument("--smpl-model-dir", default="data/smpl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-points", type=int, default=400000)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--vertex-sample", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    h3r_dir = Path(args.human3r_hugs_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    h3r_params = load_params(h3r_dir / "4d_humans" / "smpl_optimized_aligned_scale.npz")
    neu_params = load_params(Path(args.neuman_smpl))

    n = min(len(h3r_params["global_orient"]), len(neu_params["global_orient"]))
    frame_ids = np.arange(0, n, max(args.frame_stride, 1), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    all_vertex_ids = np.arange(6890)
    if args.vertex_sample > 0 and args.vertex_sample < len(all_vertex_ids):
        vertex_ids = np.sort(rng.choice(all_vertex_ids, args.vertex_sample, replace=False))
    else:
        vertex_ids = all_vertex_ids

    model = SMPL(args.smpl_model_dir, gender="neutral", device="cpu")
    src = smpl_world_vertices(model, h3r_params, frame_ids, vertex_ids)
    dst = smpl_world_vertices(model, neu_params, frame_ids, vertex_ids)
    scale, rot, trans = umeyama(src, dst)

    pred = scale * (src @ rot.T) + trans
    err = np.linalg.norm(pred - dst, axis=1)

    pts_npz = np.load(h3r_dir / "scene" / "points.npz")
    points = pts_npz["points"].astype(np.float64)
    colors = pts_npz["colors"].astype(np.float32)
    if args.max_points > 0 and len(points) > args.max_points:
        idx = rng.choice(len(points), args.max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    aligned = scale * (points @ rot.T) + trans
    np.savez_compressed(out, points=aligned.astype(np.float32), colors=colors.astype(np.float32))

    report = {
        "human3r_hugs_dir": str(h3r_dir),
        "neuman_smpl": str(args.neuman_smpl),
        "out": str(out),
        "num_frames_total": int(n),
        "num_frames_used": int(len(frame_ids)),
        "frame_stride": int(args.frame_stride),
        "num_vertices_used_per_frame": int(len(vertex_ids)),
        "sim3_scale": scale,
        "sim3_rotation": rot.tolist(),
        "sim3_translation": trans.tolist(),
        "smpl_fit_error_mean": float(err.mean()),
        "smpl_fit_error_median": float(np.median(err)),
        "smpl_fit_error_p95": float(np.percentile(err, 95)),
        "smpl_fit_error_max": float(err.max()),
        "num_points": int(len(aligned)),
        "point_min": aligned.min(axis=0).tolist(),
        "point_max": aligned.max(axis=0).tolist(),
        "point_std": aligned.std(axis=0).tolist(),
        "point_mean": aligned.mean(axis=0).tolist(),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
