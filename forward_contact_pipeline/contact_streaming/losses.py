from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .schema import ContactOutput


DEFAULT_WEIGHTS = {
    "state": 1.0,
    "surface": 1.0,
    "velocity": 0.5,
    "penetration": 0.5,
    "pose": 0.2,
    "temporal": 0.2,
    "uncertainty": 0.1,
}


def contact_loss(output: ContactOutput, batch: Dict[str, torch.Tensor], weights=None) -> Dict[str, torch.Tensor]:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    contact = batch["contact"]
    state = F.binary_cross_entropy_with_logits(output.contact_logits, contact)
    point = F.smooth_l1_loss(output.contact_point, batch["contact_point"])
    distance = F.smooth_l1_loss(output.surface_distance, batch["surface_distance"])
    normal = 1.0 - F.cosine_similarity(output.surface_normal, batch["surface_normal"], dim=-1).mean()
    surface = point + distance + normal
    penetration = F.binary_cross_entropy_with_logits(output.penetration_risk, batch["penetration_risk"])
    pose = F.smooth_l1_loss(output.pose_residual, batch["pose_residual"])
    root = F.smooth_l1_loss(output.root_residual, batch["root_residual"])
    uncertainty = F.binary_cross_entropy_with_logits(output.uncertainty_logits, batch["uncertainty"])
    # Velocity/temporal supervision is represented by low residual magnitude in v1.
    velocity = torch.mean(torch.abs(output.pose_residual[..., :6]))
    temporal = torch.mean(torch.abs(output.root_residual))
    total = (
        weights["state"] * state + weights["surface"] * surface
        + weights["velocity"] * velocity + weights["penetration"] * penetration
        + weights["pose"] * (pose + root) + weights["temporal"] * temporal
        + weights["uncertainty"] * uncertainty
    )
    return {
        "total": total, "state": state, "surface": surface, "penetration": penetration,
        "pose": pose, "root": root, "velocity": velocity, "temporal": temporal,
        "uncertainty": uncertainty,
    }
