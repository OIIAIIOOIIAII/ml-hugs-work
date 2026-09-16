from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


OUTPUT_DIMS: Dict[str, int] = {
    "contact_logits": 2,
    "contact_point": 6,
    "surface_distance": 2,
    "surface_normal": 6,
    "pose_residual": 12,
    "root_residual": 6,
    "penetration_risk": 2,
    "uncertainty_logits": 2,
}
OUTPUT_DIM = sum(OUTPUT_DIMS.values())


@dataclass
class ContactOutput:
    contact_logits: torch.Tensor
    contact_point: torch.Tensor
    surface_distance: torch.Tensor
    surface_normal: torch.Tensor
    pose_residual: torch.Tensor
    root_residual: torch.Tensor
    penetration_risk: torch.Tensor
    uncertainty_logits: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__.copy()


def split_model_output(flat: torch.Tensor) -> ContactOutput:
    """Split ``[..., 38]`` model output into typed tensors."""
    if flat.shape[-1] != OUTPUT_DIM:
        raise ValueError(f"Expected output dim {OUTPUT_DIM}, got {flat.shape[-1]}")
    chunks = torch.split(flat, list(OUTPUT_DIMS.values()), dim=-1)
    values = dict(zip(OUTPUT_DIMS, chunks))
    shape = flat.shape[:-1]
    values["contact_point"] = values["contact_point"].reshape(*shape, 2, 3)
    values["surface_normal"] = values["surface_normal"].reshape(*shape, 2, 3)
    return ContactOutput(**values)
