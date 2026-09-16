"""Sensor-calibrated translation hypothesis from frozen model reprojections."""
import numpy as np


def reproject_translation(vertices, source_k, target_k):
    """Keep pose/shape fixed, solve translation using predicted pixels, never GT.

    This fits the model's own 2D predictions. It cannot recover incorrect pose
    or supply actual observed image keypoints that the frontend did not find.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > .01)
    v = vertices[valid]
    if len(v) < 4:
        raise ValueError('Insufficient positive-depth predicted vertices')
    pixels = v @ np.asarray(source_k).T
    pixels = pixels / pixels[:, 2:3]
    rays = pixels @ np.linalg.inv(target_k).T
    rays = rays / rays[:, 2:3]
    a = np.zeros((len(v), 2, 3))
    a[:, 0, 0] = 1; a[:, 1, 1] = 1; a[:, :, 2] = -rays[:, :2]
    b = rays[:, :2] * v[:, 2:3] - v[:, :2]
    flat = a.reshape(-1, 3)
    if np.linalg.cond(flat) > 1e8:
        raise ValueError('Degenerate translation system')
    translation = np.linalg.lstsq(flat, b.reshape(-1), rcond=None)[0]
    if (v[:, 2] + translation[2] <= .01).any():
        raise ValueError('Reprojection gives nonpositive depth')
    return translation
