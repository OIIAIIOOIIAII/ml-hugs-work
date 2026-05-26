# Build an evaluation-only scene proxy for contact metrics.

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from contact_eval_lib import (
    dump_json,
    get_rgb_mask_bbox,
    make_trainer,
    read_colmap_points3d,
    write_ply,
)


def sample_depth_proxy(trainer, depth_dir, max_frames, frame_stride, pixel_stride, max_points):
    depth_dir = Path(depth_dir)
    points = []
    colors = []
    frame_ids = range(0, len(trainer.all_dataset), max(1, frame_stride))
    if max_frames > 0:
        frame_ids = list(frame_ids)[:max_frames]
    for idx in tqdm(frame_ids, desc='Depth proxy'):
        data = trainer.all_dataset[idx]
        frame_idx = int(data['frame_idx'].detach().cpu())
        candidates = [
            depth_dir / f'{frame_idx:05d}_depth.npy',
            depth_dir / f'{frame_idx:06d}.npy',
            depth_dir / f'{frame_idx:05d}.npy',
        ]
        depth_path = next((p for p in candidates if p.exists()), None)
        if depth_path is None:
            continue
        depth = np.load(depth_path).astype(np.float32)
        rgb, mask, _ = get_rgb_mask_bbox(trainer.all_dataset, data)
        rgb_np = rgb.detach().cpu().permute(1, 2, 0).numpy()
        if depth.shape != rgb_np.shape[:2]:
            depth = cv2.resize(depth, (rgb_np.shape[1], rgb_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            bg = mask.detach().cpu().numpy() < 0.5
        else:
            bg = np.ones(depth.shape, dtype=bool)
        valid = np.isfinite(depth) & (depth > 0) & bg
        ys, xs = np.where(valid)
        if ys.size == 0:
            continue
        take = np.arange(0, ys.size, max(1, pixel_stride))
        ys, xs = ys[take], xs[take]
        z = depth[ys, xs]
        K = data['cam_intrinsics'].detach().cpu().numpy()
        x_cam = (xs.astype(np.float32) - K[0, 2]) / K[0, 0] * z
        y_cam = (ys.astype(np.float32) - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1).astype(np.float32)
        c2w = data['c2w'].detach().cpu().numpy().astype(np.float32)
        world = cam @ c2w.T
        points.append(world[:, :3])
        colors.append((rgb_np[ys, xs] * 255.0).clip(0, 255).astype(np.uint8))
    if not points:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    pts = np.concatenate(points, axis=0)
    cols = np.concatenate(colors, axis=0)
    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts.astype(np.float32), cols.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Build eval-only scene proxy for contact metrics.')
    parser.add_argument('-o', '--output-dir', help='HUGS output directory; required for depth proxy and metadata.')
    parser.add_argument('--dataset-path', default='data/neuman/dataset/lab')
    parser.add_argument('--colmap-points', default='')
    parser.add_argument('--depth-dir', default='')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-depth-frames', type=int, default=24)
    parser.add_argument('--frame-stride', type=int, default=4)
    parser.add_argument('--pixel-stride', type=int, default=24)
    parser.add_argument('--max-points', type=int, default=200000)
    parser.add_argument('--source', choices=['colmap', 'depth', 'combined'], default='combined')
    args, extras = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_points = []
    all_colors = []
    sources = []

    if args.source in ['colmap', 'combined']:
        colmap_path = Path(args.colmap_points) if args.colmap_points else Path(args.dataset_path) / 'sparse' / 'points3D.txt'
        pts, cols = read_colmap_points3d(colmap_path)
        if pts.shape[0] > 0:
            all_points.append(pts)
            all_colors.append(cols)
            sources.append({'source': 'colmap', 'path': str(colmap_path), 'num_points': int(pts.shape[0])})

    if args.source in ['depth', 'combined'] and args.depth_dir:
        if not args.output_dir:
            raise ValueError('--output-dir is required when building depth proxy')
        trainer, ckpts = make_trainer(args.output_dir, extras=extras, enable_anchors=False, enable_attention=False)
        pts, cols = sample_depth_proxy(
            trainer,
            args.depth_dir,
            args.max_depth_frames,
            args.frame_stride,
            args.pixel_stride,
            args.max_points,
        )
        if pts.shape[0] > 0:
            all_points.append(pts)
            all_colors.append(cols)
            sources.append({'source': 'depth', 'path': str(args.depth_dir), 'num_points': int(pts.shape[0])})

    if not all_points:
        raise RuntimeError('No proxy points were built. Check COLMAP/depth paths.')
    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
    if args.max_points > 0 and points.shape[0] > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=args.max_points, replace=False)
        points, colors = points[idx], colors[idx]

    np.savez_compressed(out_dir / 'scene_proxy_points.npz', points=points, colors=colors)
    write_ply(out_dir / 'scene_proxy_points.ply', points, colors)
    summary = {
        'num_points': int(points.shape[0]),
        'bounds_min': points.min(axis=0).astype(float).tolist(),
        'bounds_max': points.max(axis=0).astype(float).tolist(),
        'sources': sources,
        'note': 'Eval-only proxy; do not treat as ground-truth contact surface without confidence checks.',
    }
    dump_json(out_dir / 'scene_proxy_summary.json', summary)
    print(f'PROXY_NPZ={out_dir / "scene_proxy_points.npz"}')
    print(f'PROXY_PLY={out_dir / "scene_proxy_points.ply"}')


if __name__ == '__main__':
    main()
