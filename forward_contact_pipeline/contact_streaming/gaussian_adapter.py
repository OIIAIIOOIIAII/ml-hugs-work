from __future__ import annotations

import numpy as np
import torch


def blend_lbs_transforms(weights: torch.Tensor, joint_transforms: torch.Tensor) -> torch.Tensor:
    """Blend joint transforms. weights=[N,J], transforms=[B,J,4,4]."""
    if weights.ndim != 2 or joint_transforms.ndim != 4:
        raise ValueError("Expected weights [N,J] and transforms [B,J,4,4]")
    return torch.einsum("nj,bjxy->bnxy", weights, joint_transforms)


def apply_lbs_to_gaussians(
    xyz_zero: torch.Tensor,
    lbs_weights: torch.Tensor,
    joint_transforms: torch.Tensor,
    translation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SMPL-X joint transforms to canonical Gaussian centers."""
    if xyz_zero.ndim == 2:
        xyz_zero = xyz_zero.unsqueeze(0).expand(joint_transforms.shape[0], -1, -1)
    transforms = blend_lbs_transforms(lbs_weights, joint_transforms)
    homogeneous = torch.cat([xyz_zero, torch.ones_like(xyz_zero[..., :1])], dim=-1)
    posed = torch.einsum("bnxy,bny->bnx", transforms, homogeneous)[..., :3]
    if translation is not None:
        posed = posed + translation[:, None]
    return posed


def apply_root_residual(xyz: np.ndarray, root_residual: np.ndarray) -> np.ndarray:
    """Apply translation and Rodrigues rotation around the point-cloud center."""
    from scipy.spatial.transform import Rotation

    xyz = np.asarray(xyz, dtype=np.float32)
    residual = np.asarray(root_residual, dtype=np.float32)
    rotation = Rotation.from_rotvec(residual[3:6]).as_matrix().astype(np.float32)
    center = xyz.mean(0, keepdims=True)
    return (xyz - center) @ rotation.T + center + residual[:3]
