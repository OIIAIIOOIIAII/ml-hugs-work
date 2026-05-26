#!/usr/bin/env python3
"""Export white scene geometry and simple plane-fit diagnostics for HUGS PLYs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_gaussian_ply(path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    out = {"points": points}
    if "opacity" in names:
        out["opacity_logit"] = np.asarray(vertex["opacity"], dtype=np.float64)
        out["opacity"] = sigmoid(out["opacity_logit"])
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(name in names for name in scale_names):
        scales_log = np.stack([vertex[name] for name in scale_names], axis=1).astype(np.float64)
        out["scales_log"] = scales_log
        out["scales"] = np.exp(scales_log)
    normal_names = ["nx", "ny", "nz"]
    if all(name in names for name in normal_names):
        normals = np.stack([vertex[name] for name in normal_names], axis=1).astype(np.float64)
        out["normals"] = normals
    return out


def subsample_indices(n_items: int, max_items: int, seed: int) -> np.ndarray:
    if n_items <= max_items:
        return np.arange(n_items)
    rng = np.random.default_rng(seed)
    return rng.choice(n_items, size=max_items, replace=False)


def subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    return points[subsample_indices(len(points), max_points, seed)]


def make_point_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(points), 1)))
    return pcd


def fit_plane_ransac(points: np.ndarray, threshold: float, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    pcd = make_point_cloud(points, (1.0, 1.0, 1.0))
    plane, inliers = pcd.segment_plane(
        distance_threshold=threshold,
        ransac_n=3,
        num_iterations=iterations,
    )
    return np.asarray(plane, dtype=np.float64), np.asarray(inliers, dtype=np.int64)


def plane_distances(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    normal = plane[:3]
    denom = np.linalg.norm(normal)
    if denom == 0:
        raise ValueError("Invalid plane with zero normal.")
    return np.abs(points @ normal + plane[3]) / denom


def write_white_point_cloud(path: Path, points: np.ndarray) -> None:
    pcd = make_point_cloud(points, (0.86, 0.86, 0.86))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def write_plane_inliers(path: Path, points: np.ndarray, inlier_mask: np.ndarray) -> None:
    colors = np.zeros((len(points), 3), dtype=np.float64)
    colors[:] = (0.35, 0.35, 0.35)
    colors[inlier_mask] = (0.92, 0.92, 0.92)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def write_poisson_mesh(
    path: Path,
    points: np.ndarray,
    poisson_depth: int,
    density_quantile: float,
    target_triangles: int,
) -> dict[str, int | float]:
    pcd = make_point_cloud(points, (0.86, 0.86, 0.86))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.35, max_nn=40))
    pcd.orient_normals_consistent_tangent_plane(50)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    densities = np.asarray(densities)
    if density_quantile > 0:
        remove = densities < np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(remove)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    if target_triangles > 0 and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
    mesh.paint_uniform_color((0.86, 0.86, 0.86))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False, write_vertex_normals=True)
    return {
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
    }


def summarize(name: str, ply_path: Path, args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    data = read_gaussian_ply(ply_path)
    points = data["points"]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    metric_idx = subsample_indices(len(points), args.max_metric_points, args.seed)
    metric_points = points[metric_idx]
    plane, inliers = fit_plane_ransac(metric_points, args.plane_threshold, args.ransac_iters)
    distances = plane_distances(metric_points, plane)
    inlier_mask = distances <= args.plane_threshold

    white_points = subsample(points, args.max_export_points, args.seed + 1)
    write_white_point_cloud(out_dir / f"{name}_white_points.ply", white_points)
    write_plane_inliers(out_dir / f"{name}_plane_inliers_white.ply", metric_points, inlier_mask)

    mesh_stats = {}
    if args.export_poisson:
        mesh_points = metric_points[inlier_mask]
        if len(mesh_points) < args.min_mesh_points:
            mesh_points = metric_points
        mesh_points = subsample(mesh_points, args.max_mesh_points, args.seed + 2)
        mesh_stats = write_poisson_mesh(
            out_dir / f"{name}_white_poisson_mesh.ply",
            mesh_points,
            args.poisson_depth,
            args.density_quantile,
            args.target_triangles,
        )

    opacity_stats = {}
    if "opacity" in data:
        opacity = data["opacity"][finite]
        metric_opacity = opacity[metric_idx]
        opacity_stats = {
            "opacity_mean": float(np.mean(opacity)),
            "opacity_median": float(np.median(opacity)),
            "opacity_p10": float(np.quantile(opacity, 0.10)),
            "opacity_p90": float(np.quantile(opacity, 0.90)),
        }
        if len(metric_opacity) == len(distances):
            opacity_stats["opacity_weighted_mean_plane_distance"] = float(
                np.sum(distances * metric_opacity) / max(np.sum(metric_opacity), 1e-12)
            )

    scale_stats = {}
    if "scales" in data:
        scales = data["scales"][finite]
        thickness = np.min(scales, axis=1)
        footprint = np.max(scales, axis=1)
        scale_stats = {
            "scale_min_median": float(np.median(thickness)),
            "scale_max_median": float(np.median(footprint)),
            "scale_thickness_ratio_median": float(np.median(thickness / np.maximum(footprint, 1e-12))),
        }

    return {
        "name": name,
        "source_ply": str(ply_path),
        "num_points": int(len(points)),
        "metric_points": int(len(metric_points)),
        "plane_abcd": [float(x) for x in plane],
        "plane_threshold": float(args.plane_threshold),
        "plane_inlier_ratio": float(np.mean(inlier_mask)),
        "plane_distance_mean": float(np.mean(distances)),
        "plane_distance_median": float(np.median(distances)),
        "plane_distance_p90": float(np.quantile(distances, 0.90)),
        "plane_distance_p95": float(np.quantile(distances, 0.95)),
        "plane_distance_max": float(np.max(distances)),
        "white_points_ply": str(out_dir / f"{name}_white_points.ply"),
        "plane_inliers_ply": str(out_dir / f"{name}_plane_inliers_white.ply"),
        "poisson_mesh_ply": str(out_dir / f"{name}_white_poisson_mesh.ply") if args.export_poisson else None,
        **opacity_stats,
        **scale_stats,
        **mesh_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hugs-ply", type=Path, required=True)
    parser.add_argument("--sugar-ply", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-metric-points", type=int, default=250_000)
    parser.add_argument("--max-export-points", type=int, default=600_000)
    parser.add_argument("--max-mesh-points", type=int, default=180_000)
    parser.add_argument("--min-mesh-points", type=int, default=10_000)
    parser.add_argument("--plane-threshold", type=float, default=0.05)
    parser.add_argument("--ransac-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--export-poisson", action="store_true")
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--density-quantile", type=float, default=0.05)
    parser.add_argument("--target-triangles", type=int, default=250_000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        summarize("hugs_original_15000", args.hugs_ply, args, args.out_dir),
        summarize("sugar_hugs_15000", args.sugar_ply, args, args.out_dir),
    ]

    json_path = args.out_dir / "geometry_metrics.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.out_dir / "geometry_metrics.csv"
    keys = sorted({key for row in results for key in row.keys() if not isinstance(row.get(key), list)})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in keys})

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
