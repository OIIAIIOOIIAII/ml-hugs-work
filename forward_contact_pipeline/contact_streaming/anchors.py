from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


SMPLX_JOINTS = {"pelvis": 0, "left_ankle": 7, "right_ankle": 8, "left_foot": 10, "right_foot": 11}


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm > 1e-8 else fallback).astype(np.float32)


def anchors_from_joints(joints: np.ndarray, up_axis: int = 1) -> Dict[str, np.ndarray]:
    joints = np.asarray(joints, dtype=np.float32)
    left_ankle, right_ankle = joints[7], joints[8]
    left_foot, right_foot = joints[10], joints[11]
    up = np.zeros(3, dtype=np.float32)
    up[up_axis] = 1.0
    left_forward = _normalize(left_foot - left_ankle, np.array([0, 0, 1], np.float32))
    right_forward = _normalize(right_foot - right_ankle, np.array([0, 0, 1], np.float32))
    orientation = np.stack([np.concatenate([left_forward, up]), np.concatenate([right_forward, up])])
    return {
        "root": joints[0],
        "anchor_position": np.stack([left_foot, right_foot]),
        "ankle_position": np.stack([left_ankle, right_ankle]),
        "anchor_orientation": orientation.astype(np.float32),
    }


class SMPLXAnchorExtractor:
    """Small SMPL-X wrapper matching Human3R's 53-axis-angle output."""

    def __init__(self, model_dir: str | Path, device: str = "cpu", gender: str = "neutral"):
        import smplx

        self.device = torch.device(device)
        self.model = smplx.create(
            str(model_dir), model_type="smplx", gender=gender, use_pca=False,
            flat_hand_mean=True, num_betas=10, num_expression_coeffs=10,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        shape: np.ndarray,
        rotvec: np.ndarray,
        transl: np.ndarray,
        expression: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        pose = torch.as_tensor(rotvec, dtype=torch.float32, device=self.device).reshape(1, -1, 3)
        if pose.shape[1] != 53:
            raise ValueError(f"Expected Human3R rotvec [53,3], got {tuple(pose.shape[1:])}")
        kwargs = {
            "betas": torch.as_tensor(shape, dtype=torch.float32, device=self.device).reshape(1, -1)[:, :10],
            "global_orient": pose[:, 0],
            "body_pose": pose[:, 1:22].reshape(1, -1),
            "jaw_pose": pose[:, 22],
            "left_hand_pose": pose[:, 23:38].reshape(1, -1),
            "right_hand_pose": pose[:, 38:53].reshape(1, -1),
            "transl": torch.as_tensor(transl, dtype=torch.float32, device=self.device).reshape(1, 3),
            "return_verts": True,
        }
        if expression is not None:
            kwargs["expression"] = torch.as_tensor(expression, dtype=torch.float32, device=self.device).reshape(1, -1)[:, :10]
        output = self.model(**kwargs)
        joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
        vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
        result = anchors_from_joints(joints)
        result.update({"joints": joints, "vertices": vertices})
        return result
