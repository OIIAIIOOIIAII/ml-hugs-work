from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    residual: float
    confidence: float
    method: str

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        return self.scale * (points @ self.rotation.T) + self.translation

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both have shape [N,3]")
    if len(src) < 3:
        raise ValueError("At least three correspondences are required for Sim(3)")
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.square(src_centered).sum() / len(src)
    scale = float((singular * sign).sum() / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def robust_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    max_iterations: int = 8,
    trim_fraction: float = 0.8,
) -> Sim3:
    """Trimmed Umeyama alignment for known 3D correspondences."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    valid = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[valid], dst[valid]
    if len(src) < 3:
        raise ValueError("Fewer than three finite correspondences")
    keep = np.arange(len(src))
    for _ in range(max_iterations):
        scale, rotation, translation = _umeyama(src[keep], dst[keep])
        prediction = scale * (src @ rotation.T) + translation
        errors = np.linalg.norm(prediction - dst, axis=1)
        count = max(3, int(round(len(src) * trim_fraction)))
        next_keep = np.argsort(errors)[:count]
        if np.array_equal(np.sort(keep), np.sort(next_keep)):
            break
        keep = next_keep
    scale, rotation, translation = _umeyama(src[keep], dst[keep])
    prediction = scale * (src @ rotation.T) + translation
    residual = float(np.median(np.linalg.norm(prediction - dst, axis=1)))
    scene_extent = float(np.median(np.linalg.norm(dst - np.median(dst, axis=0), axis=1)))
    relative = residual / max(scene_extent, 1e-4)
    confidence = float(np.exp(-4.0 * relative) * min(1.0, len(keep) / 8.0))
    return Sim3(scale, rotation.astype(np.float32), translation.astype(np.float32), residual, confidence, "robust_umeyama")


def translation_alignment(src_root: np.ndarray, target_root: np.ndarray, scale: float = 1.0) -> Sim3:
    src_root = np.asarray(src_root, dtype=np.float32)
    target_root = np.asarray(target_root, dtype=np.float32)
    translation = target_root - scale * src_root
    return Sim3(float(scale), np.eye(3, dtype=np.float32), translation, float("nan"), 0.25, "translation_fallback")


def human_cloud_centroid(point_map: np.ndarray, human_mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    points = np.asarray(point_map).reshape(-1, 3)
    mask = np.asarray(human_mask).squeeze().reshape(-1) > 0.5
    conf = np.asarray(confidence).reshape(-1)
    valid = mask & (conf > 0) & np.isfinite(points).all(1)
    if valid.sum() < 20:
        raise ValueError("Not enough valid human point-map samples for fallback alignment")
    return np.median(points[valid], axis=0).astype(np.float32)
