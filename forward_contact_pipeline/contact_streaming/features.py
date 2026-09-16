from __future__ import annotations

import numpy as np


def finite_difference(values: np.ndarray, timestamps: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.zeros_like(values)
    if len(values) < 2:
        return output
    if timestamps is None:
        dt = np.ones(len(values) - 1, dtype=np.float32)
    else:
        dt = np.maximum(np.diff(np.asarray(timestamps, dtype=np.float32)), 1e-4)
    output[1:] = np.diff(values, axis=0) / dt.reshape((-1,) + (1,) * (values.ndim - 1))
    output[0] = output[1]
    return output


def build_features(
    anchor_position: np.ndarray,
    anchor_orientation: np.ndarray,
    surface_point: np.ndarray,
    surface_normal: np.ndarray,
    surface_distance: np.ndarray,
    surface_confidence: np.ndarray,
    root: np.ndarray,
    alignment_residual: np.ndarray,
    alignment_confidence: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build fixed 58D per-frame features used by GRU/TCN."""
    anchor_velocity = finite_difference(anchor_position, timestamps)
    root_velocity = finite_difference(root, timestamps)
    previous_contact = np.zeros_like(surface_distance, dtype=np.float32)
    previous_residual = np.zeros_like(anchor_position, dtype=np.float32)
    feet = []
    for foot in range(2):
        foot_feature = np.concatenate(
            [
                anchor_position[:, foot], anchor_velocity[:, foot], anchor_orientation[:, foot],
                surface_point[:, foot], surface_normal[:, foot], surface_distance[:, foot, None],
                surface_confidence[:, foot, None], alignment_confidence[:, None],
                previous_contact[:, foot, None], previous_residual[:, foot],
            ], axis=1,
        )
        feet.append(foot_feature)
    global_features = np.concatenate(
        [root, root_velocity, alignment_residual[:, None], alignment_confidence[:, None]], axis=1
    )
    features = np.concatenate(feet + [global_features], axis=1).astype(np.float32)
    aux = {"anchor_velocity": anchor_velocity, "root_velocity": root_velocity}
    return features, aux
