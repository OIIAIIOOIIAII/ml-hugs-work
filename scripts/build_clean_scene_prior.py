#!/usr/bin/env python3
"""Build a cleaner scene prior from an aligned scene point cloud.

The script removes points close to a posed human point cloud, optionally trims
extreme outliers by percentile, voxel-downsamples, and writes an npz suitable for
NeumanDataset scene.init_pcd_path plus a colored PLY for inspection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def read_xyz_part_ply(path):
    ply = PlyData.read(str(path), mmap=True)
    v = ply['vertex'].data
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    part = v['part'].astype(np.uint8) if 'part' in v.dtype.names else np.zeros(len(xyz), dtype=np.uint8)
    return xyz, part


def voxel_downsample(points, colors, voxel_size, max_points, seed):
    if voxel_size > 0:
        keys = np.floor(points / voxel_size).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        keep = np.sort(keep)
        points, colors = points[keep], colors[keep]
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors


def write_xyzrgb_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(colors * 255.0 if colors.max() <= 1.0 else colors, 0, 255).astype(np.uint8)
    verts = np.empty(len(points), dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    verts['x'], verts['y'], verts['z'] = points[:,0], points[:,1], points[:,2]
    verts['red'], verts['green'], verts['blue'] = colors[:,0], colors[:,1], colors[:,2]
    PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-npz', required=True)
    ap.add_argument('--human-ply', required=True, help='colored pointcloud PLY with part=1 for posed human')
    ap.add_argument('--out-npz', required=True)
    ap.add_argument('--out-ply', required=True)
    ap.add_argument('--human-radius', type=float, default=0.18)
    ap.add_argument('--trim-percentile', type=float, default=99.5)
    ap.add_argument('--voxel-size', type=float, default=0.03)
    ap.add_argument('--max-points', type=int, default=120000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    src = np.load(args.input_npz)
    points = src['points'].astype(np.float32)
    colors = src['colors'].astype(np.float32)
    report = {'input_points': int(len(points))}

    if args.trim_percentile and args.trim_percentile < 100:
        med = np.median(points, axis=0)
        dist = np.linalg.norm(points - med[None], axis=1)
        thr = np.percentile(dist, args.trim_percentile)
        keep = dist <= thr
        points, colors = points[keep], colors[keep]
        report['after_percentile_trim'] = int(len(points))
        report['percentile_trim_threshold'] = float(thr)

    human_xyz, part = read_xyz_part_ply(args.human_ply)
    if part.max() > 0:
        human_xyz = human_xyz[part == 1]
    report['human_reference_points'] = int(len(human_xyz))
    if len(human_xyz) > 0 and args.human_radius > 0:
        tree = cKDTree(human_xyz)
        d, _ = tree.query(points, k=1, workers=-1)
        keep = d > args.human_radius
        report['removed_near_human'] = int((~keep).sum())
        points, colors = points[keep], colors[keep]
        report['after_human_exclusion'] = int(len(points))
        report['human_radius'] = float(args.human_radius)

    before_voxel = len(points)
    points, colors = voxel_downsample(points, colors, args.voxel_size, args.max_points, args.seed)
    report['before_voxel_or_max'] = int(before_voxel)
    report['final_points'] = int(len(points))
    report['voxel_size'] = float(args.voxel_size)
    report['max_points'] = int(args.max_points)
    report['point_min'] = points.min(axis=0).tolist()
    report['point_max'] = points.max(axis=0).tolist()
    report['point_mean'] = points.mean(axis=0).tolist()

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, points=points.astype(np.float32), colors=colors.astype(np.float32))
    write_xyzrgb_ply(args.out_ply, points, colors)
    Path(args.out_npz).with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
