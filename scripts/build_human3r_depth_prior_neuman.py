#!/usr/bin/env python3
"""Build NeuMan-view pseudo depth maps from Human3R depth outputs.

This uses Human3R depth/intrinsics/c2w to backproject scene pixels, applies a saved
Human3R->NeuMan Sim3 alignment, then z-buffers the points into NeuMan cameras.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from hugs.datasets.neuman_utils import neuman_helper


def resize_crop_like_human3r(img, target_height, target_width, interpolation):
    in_height, in_width = img.shape[:2]
    long_edge = max(target_height, target_width)
    scale = long_edge / max(in_height, in_width)
    resized_width = int(round(in_width * scale))
    resized_height = int(round(in_height * scale))
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=interpolation)
    y0 = max((resized_height - target_height) // 2, 0)
    x0 = max((resized_width - target_width) // 2, 0)
    cropped = resized[y0:y0 + target_height, x0:x0 + target_width]
    if cropped.shape[:2] != (target_height, target_width):
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)
    return cropped


def backproject_h3r_depth(result_dir, sim3, conf_threshold, stride, mask_dir=None, dilate_mask=15, max_points=0, seed=0):
    result_dir = Path(result_dir)
    stems = [p.stem for p in sorted((result_dir / 'depth').glob('*.npy'))]
    all_pts = []
    rng = np.random.default_rng(seed)
    scale = float(sim3['sim3_scale'])
    rot = np.asarray(sim3['sim3_rotation'], dtype=np.float64)
    trans = np.asarray(sim3['sim3_translation'], dtype=np.float64)
    for frame_i, stem in enumerate(stems):
        depth = np.load(result_dir / 'depth' / f'{stem}.npy').astype(np.float32)
        conf = np.load(result_dir / 'conf' / f'{stem}.npy').astype(np.float32)
        cam = np.load(result_dir / 'camera' / f'{stem}.npz')
        k = cam['intrinsics'].astype(np.float32)
        c2w = cam['pose'].astype(np.float32)
        h, w = depth.shape
        valid = np.isfinite(depth) & np.isfinite(conf) & (depth > 0) & (conf >= conf_threshold)
        if mask_dir:
            mpath = Path(mask_dir) / f'{int(stem):06d}.png'
            if not mpath.exists():
                mpath = Path(mask_dir) / f'image_{int(stem):04d}.png'
            if mpath.exists():
                mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                mask = resize_crop_like_human3r(mask, h, w, cv2.INTER_NEAREST) > 127
                if dilate_mask > 0:
                    kernel = np.ones((dilate_mask, dilate_mask), np.uint8)
                    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
                valid &= ~mask
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[yy, xx]
        v = valid[yy, xx]
        if not np.any(v):
            continue
        x = xx[v].astype(np.float32)
        y = yy[v].astype(np.float32)
        z = z[v].astype(np.float32)
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]
        x_cam = (x - cx) * z / fx
        y_cam = (y - cy) * z / fy
        p_cam = np.stack([x_cam, y_cam, z], axis=-1).astype(np.float64)
        p_h3r = p_cam @ c2w[:3, :3].T + c2w[:3, 3]
        p_neu = scale * (p_h3r @ rot.T) + trans
        all_pts.append(p_neu.astype(np.float32))
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.concatenate(all_pts, axis=0)
    if max_points > 0 and len(pts) > max_points:
        keep = rng.choice(len(pts), max_points, replace=False)
        pts = pts[keep]
    return pts


def zbuffer_depth(points, cap):
    k = cap.intrinsic_matrix.astype(np.float64)
    w2c = cap.cam_pose.world_to_camera.astype(np.float64)
    height, width = cap.size[0], cap.size[1]
    homog = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    cam = (w2c @ homog.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-4
    cam = cam[valid]
    z = z[valid]
    uv = (k @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if len(z):
        np.minimum.at(depth, (v, u), z.astype(np.float32))
    depth[~np.isfinite(depth)] = 0.0
    return depth


def depth_to_vis(depth, pmin=None, pmax=None):
    valid = depth > 0
    if valid.any():
        if pmin is None:
            pmin = np.percentile(depth[valid], 2)
        if pmax is None:
            pmax = np.percentile(depth[valid], 98)
        x = (depth - pmin) / max(pmax - pmin, 1e-6)
        x = np.clip(x, 0, 1)
        vis = (255 * (1 - x)).astype(np.uint8)
        vis[~valid] = 0
    else:
        vis = np.zeros_like(depth, dtype=np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human3r-result-dir', default='data/neuman/dataset/lab/human3r')
    ap.add_argument('--neuman-seq-dir', default='data/neuman/dataset/lab')
    ap.add_argument('--sim3-json', default='data/human3r_hugs/hybrid_init/lab_h3r1024_scene_to_neuman_by_smpl_pcd.json')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--mask-dir', default='data/neuman/dataset/lab/4d_humans/sam_segmentations')
    ap.add_argument('--conf-threshold', type=float, default=1.5)
    ap.add_argument('--source-stride', type=int, default=2)
    ap.add_argument('--max-source-points', type=int, default=800000)
    ap.add_argument('--dilate-mask', type=int, default=15)
    ap.add_argument('--vis-frames', default='0,20,40,60,80')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    depth_dir = out_dir / 'depth'
    vis_dir = out_dir / 'vis'
    depth_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    sim3 = json.loads(Path(args.sim3_json).read_text())
    points = backproject_h3r_depth(
        args.human3r_result_dir, sim3, args.conf_threshold, args.source_stride,
        mask_dir=args.mask_dir, dilate_mask=args.dilate_mask, max_points=args.max_source_points,
    )
    np.savez_compressed(out_dir / 'source_points_neuman.npz', points=points)
    scene = neuman_helper.NeuManReader.read_scene(args.neuman_seq_dir, tgt_size=None, normalize=False, smpl_type='optimized')
    coverage = []
    for i, cap in enumerate(scene.captures):
        depth = zbuffer_depth(points, cap)
        np.save(depth_dir / f'{i:06d}.npy', depth)
        valid = depth > 0
        coverage.append(float(valid.mean()))
    vis_ids = [int(x) for x in args.vis_frames.split(',') if x.strip()]
    for i in vis_ids:
        depth = np.load(depth_dir / f'{i:06d}.npy')
        vis = depth_to_vis(depth)
        cv2.imwrite(str(vis_dir / f'{i:06d}_depth_vis.png'), vis)
    meta = {
        'human3r_result_dir': args.human3r_result_dir,
        'neuman_seq_dir': args.neuman_seq_dir,
        'sim3_json': args.sim3_json,
        'num_source_points': int(len(points)),
        'conf_threshold': args.conf_threshold,
        'source_stride': args.source_stride,
        'max_source_points': args.max_source_points,
        'coverage_mean': float(np.mean(coverage)),
        'coverage_min': float(np.min(coverage)),
        'coverage_max': float(np.max(coverage)),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
