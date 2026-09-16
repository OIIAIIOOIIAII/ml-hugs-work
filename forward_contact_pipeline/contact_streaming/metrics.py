from __future__ import annotations

from typing import Dict

import numpy as np


def binary_contact_metrics(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    prediction = np.asarray(probability) >= threshold
    target = np.asarray(target) >= 0.5
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"contact_precision": precision, "contact_recall": recall, "contact_f1": f1}


def transition_f1(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = np.asarray(probability) >= threshold
    gt = np.asarray(target) >= 0.5
    if len(pred) < 2:
        return 0.0
    return binary_contact_metrics(np.abs(np.diff(pred.astype(np.int8), axis=0)), np.abs(np.diff(gt.astype(np.int8), axis=0)))["contact_f1"]


def geometric_metrics(
    probability: np.ndarray,
    target_contact: np.ndarray,
    predicted_distance: np.ndarray,
    target_distance: np.ndarray,
    predicted_root: np.ndarray,
    target_root: np.ndarray,
) -> Dict[str, float]:
    metrics = binary_contact_metrics(probability, target_contact)
    metrics.update(
        {
            "contact_transition_f1": transition_f1(probability, target_contact),
            "surface_distance_mae": float(np.mean(np.abs(predicted_distance - target_distance))),
            "root_residual_rmse": float(np.sqrt(np.mean(np.square(predicted_root - target_root)))),
            "penetration_ratio_pred": float((predicted_distance < 0).mean()),
            "penetration_depth_pred": float(np.maximum(-predicted_distance, 0).mean()),
        }
    )
    return metrics


def foot_sliding(anchor_position: np.ndarray, contact: np.ndarray, timestamps: np.ndarray | None = None) -> float:
    """Mean contact-foot speed in m/s (not unnormalised displacement/frame)."""
    if len(anchor_position) < 2:
        return 0.0
    displacement = np.linalg.norm(np.diff(anchor_position, axis=0), axis=-1)
    if timestamps is None:
        # Callers without timing metadata receive a per-frame proxy; current
        # contact experiments pass PROX timestamps and report true m/s.
        velocity = displacement
    else:
        dt = np.maximum(np.diff(np.asarray(timestamps, dtype=np.float32)), 1e-4)
        velocity = displacement / dt[:, None]
    active = np.asarray(contact)[1:] >= 0.5
    return float(velocity[active].mean()) if active.any() else 0.0
