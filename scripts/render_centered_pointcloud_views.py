#!/usr/bin/env python3
"""Project exported colored point-cloud PLY from GT camera views and crop around human bbox."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from hugs.datasets.neuman_utils import neuman_helper


def load_points(ply_path, max_points=0, seed=0):
    ply = PlyData.read(str(ply_path))
    v = ply['vertex'].data
    pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    colors = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
    part = v['part'].astype(np.uint8) if 'part' in v.dtype.names else np.zeros(len(pts), dtype=np.uint8)
    if max_points and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        # Keep all/most human points, subsample scene first. This preserves the human-centered view.
        human_idx = np.where(part == 1)[0]
        scene_idx = np.where(part != 1)[0]
        n_human = min(len(human_idx), max_points // 2)
        n_scene = max_points - n_human
        keep_h = human_idx if len(human_idx) <= n_human else rng.choice(human_idx, n_human, replace=False)
        keep_s = scene_idx if len(scene_idx) <= n_scene else rng.choice(scene_idx, n_scene, replace=False)
        keep = np.concatenate([keep_s, keep_h])
        pts, colors, part = pts[keep], colors[keep], part[keep]
    return pts, colors, part


def project_points(points, colors, K, w2c, width, height, point_size=2):
    homog = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    cam = (w2c @ homog.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-4
    cam = cam[valid]
    cols = colors[valid]
    z = z[valid]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u = np.rint(uv[:, 0]).astype(np.int32)
    vv = np.rint(uv[:, 1]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (vv >= 0) & (vv < height)
    u, vv, z, cols = u[inside], vv[inside], z[inside], cols[inside]

    order = np.argsort(z)[::-1]  # far to near, near overwrites
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    r = max(0, point_size // 2)
    for idx in order:
        x, y = u[idx], vv[idx]
        if r <= 0:
            img[y, x] = cols[idx]
        else:
            img[max(0, y-r):min(height, y+r+1), max(0, x-r):min(width, x+r+1)] = cols[idx]
    return img


def make_crop(img, bbox, pad=1.8, out_size=900):
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox.astype(float)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    bw, bh = max(1, xmax - xmin), max(1, ymax - ymin)
    side = max(bw, bh) * pad
    x0 = int(max(0, round(cx - side / 2)))
    x1 = int(min(w, round(cx + side / 2)))
    y0 = int(max(0, round(cy - side / 2)))
    y1 = int(min(h, round(cy + side / 2)))
    crop = img[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = img.copy()
        x0 = y0 = 0
    pil = Image.fromarray(crop)
    pil.thumbnail((out_size, out_size), Image.Resampling.LANCZOS)
    return np.asarray(pil), (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--seq-dir', default='data/neuman/dataset/lab')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--frames', default='0,20,40,60,80')
    ap.add_argument('--max-points', type=int, default=1200000)
    ap.add_argument('--point-size', type=int, default=2)
    ap.add_argument('--crop-pad', type=float, default=1.8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points, colors, part = load_points(Path(args.ply), args.max_points)
    print(f'loaded_points={len(points)} human={int((part==1).sum())} scene={int((part!=1).sum())}')

    scene = neuman_helper.NeuManReader.read_scene(
        args.seq_dir,
        tgt_size=None,
        normalize=False,
        smpl_type='optimized',
    )
    mask_dir = Path(args.seq_dir) / '4d_humans' / 'sam_segmentations'
    masks = sorted(mask_dir.glob('*.png'))
    frames = [int(x) for x in args.frames.split(',') if x.strip()]
    for fid in frames:
        cap = scene.captures[fid]
        K = cap.intrinsic_matrix.astype(np.float64)
        w2c = cap.cam_pose.world_to_camera.astype(np.float64)
        height, width = cap.size[0], cap.size[1]
        img = project_points(points, colors, K, w2c, width, height, point_size=args.point_size)
        m = cv2.imread(str(masks[fid]), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(m > 127)
        if len(xs) > 0:
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)
        else:
            bbox = np.array([0, 0, width - 1, height - 1], dtype=float)
        crop, crop_box = make_crop(img, bbox, pad=args.crop_pad)
        Image.fromarray(img).save(out_dir / f'frame_{fid:03d}_full.png')
        Image.fromarray(crop).save(out_dir / f'frame_{fid:03d}_human_centered.png')

        # overlay bbox on full image for debugging
        overlay = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(tuple(bbox.tolist()), outline=(0, 0, 0), width=4)
        draw.rectangle(crop_box, outline=(255, 0, 0), width=4)
        overlay.save(out_dir / f'frame_{fid:03d}_full_bbox.png')
        print(f'wrote frame={fid} bbox={bbox.astype(int).tolist()} crop={list(crop_box)}')


if __name__ == '__main__':
    main()
