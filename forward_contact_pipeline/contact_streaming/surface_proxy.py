from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceEstimate:
    point: np.ndarray
    normal: np.ndarray
    signed_distance: float
    confidence: float
    num_points: int


def scene_points_from_frame(
    point_map: np.ndarray,
    confidence: np.ndarray,
    human_mask: np.ndarray | None = None,
    min_confidence: float = 0.0,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(point_map)[::stride, ::stride].reshape(-1, 3)
    conf = np.asarray(confidence)[::stride, ::stride].reshape(-1)
    valid = np.isfinite(points).all(1) & np.isfinite(conf) & (conf > min_confidence)
    if human_mask is not None:
        mask = np.asarray(human_mask).squeeze()[::stride, ::stride].reshape(-1) > 0.5
        valid &= ~mask
    return points[valid].astype(np.float32), conf[valid].astype(np.float32)


def fit_local_plane(
    anchor: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray | None = None,
    k: int = 256,
    radius: float = 0.35,
    up_axis: int = 1,
) -> SurfaceEstimate:
    """Fit a weighted local PCA plane around an anchor."""
    anchor = np.asarray(anchor, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return SurfaceEstimate(anchor.copy(), np.eye(3, dtype=np.float32)[up_axis], np.nan, 0.0, len(points))
    distances = np.linalg.norm(points - anchor, axis=1)
    indices = np.argsort(distances)[: min(k, len(points))]
    indices = indices[distances[indices] <= radius]
    if len(indices) < 8:
        indices = np.argsort(distances)[: min(max(8, k // 4), len(points))]
    local = points[indices]
    if weights is None:
        local_weights = np.ones(len(local), dtype=np.float64)
    else:
        local_weights = np.maximum(np.asarray(weights)[indices].astype(np.float64), 1e-6)
    spatial = np.exp(-np.square(np.linalg.norm(local - anchor, axis=1) / max(radius, 1e-4)))
    local_weights *= spatial
    local_weights /= max(local_weights.sum(), 1e-12)
    center = (local * local_weights[:, None]).sum(0)
    centered = local - center
    covariance = (centered * local_weights[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if normal[up_axis] < 0:
        normal = -normal
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    signed_distance = float(np.dot(anchor - center, normal))
    planarity = 1.0 - float(eigenvalues[0] / max(eigenvalues.sum(), 1e-10))
    support = min(1.0, len(local) / 64.0)
    proximity = float(np.exp(-max(0.0, distances[indices].mean() - radius) / max(radius, 1e-4)))
    confidence = float(np.clip(planarity * support * proximity, 0.0, 1.0))
    projection = anchor - signed_distance * normal
    return SurfaceEstimate(projection.astype(np.float32), normal.astype(np.float32), signed_distance, confidence, len(local))


def estimate_feet_surfaces(
    anchors: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    estimates = [fit_local_plane(anchor, points, weights, **kwargs) for anchor in anchors]
    return {
        "surface_point": np.stack([item.point for item in estimates]),
        "surface_normal": np.stack([item.normal for item in estimates]),
        "surface_distance": np.asarray([item.signed_distance for item in estimates], dtype=np.float32),
        "surface_confidence": np.asarray([item.confidence for item in estimates], dtype=np.float32),
        "surface_support": np.asarray([item.num_points for item in estimates], dtype=np.int32),
    }
