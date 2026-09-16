"""Trainable geometry initialization using frozen predictions, with no GT inputs."""
from __future__ import annotations

import torch
from torch import nn
import roma


def cap_vectors(value, limit):
    scale = (limit / value.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1)
    return value * scale


def apply_residual(inputs, residual):
    """Return valid SMPL-X parameters; hands/expression stay from the frontend."""
    delta = cap_vectors(residual[:, :66].reshape(-1, 22, 3), 1.2)
    rotation = torch.cat((roma.rotvec_to_rotmat(delta) @ inputs['rotation'][:, :22],
                          inputs['rotation'][:, 22:]), dim=1)
    beta = inputs['shape'] + residual[:, 66:76].clamp(-3, 3)
    head = inputs['head'] + cap_vectors(residual[:, 76:79], .25) * inputs['head'][:, 2:3].clamp_min(.5)
    return rotation, beta, head


class ResidualMLP(nn.Module):
    def __init__(self, dimension, mean, std, hidden=128, dropout=.1):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).float())
        self.register_buffer('std', torch.as_tensor(std).float().clamp_min(.01))
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, 79))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('output_scale', torch.tensor([.2] * 66 + [1.] * 10 + [.05] * 3))

    def forward(self, value):
        return self.network(((value - self.mean) / self.std).clamp(-10, 10)) * self.output_scale


class DifferentiableHumanGS(nn.Module):
    """Batched HumanGS FK with its inference-only no_grad decorator bypassed.

    Evaluation still uses the released per-frame export and checks exact zero
    correction. Batched matrix kernels can have different TF32 rounding.
    """
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.layer = mesh.layer['neutral']
        self.layer.requires_grad_(False)

    def forward(self, inputs, residual):
        rot, beta, head_camera = apply_residual(inputs, residual)
        rv = roma.rotmat_to_rotvec(rot)
        c2w = inputs['c2w']
        camera_rot = c2w[:, :3, :3]
        p = dict(betas=beta, expr=inputs['expression'],
                 root_pose=roma.rotmat_to_rotvec(camera_rot @ rot[:, 0]),
                 body_pose=rv[:, 1:22], lhand_pose=rv[:, 22:37], rhand_pose=rv[:, 37:52],
                 jaw_pose=rv[:, 52], leye_pose=torch.zeros_like(rv[:, 52]),
                 reye_pose=torch.zeros_like(rv[:, 52]))
        # Public inference method is decorated with no_grad. Removing that wrapper
        # here is essential: head FK must differentiate with pose and shape too.
        _, head_fk, _ = type(self.mesh).get_target_transform.__wrapped__(self.mesh, p)
        body = self.layer(betas=beta, expression=p['expr'], global_orient=p['root_pose'],
                          body_pose=p['body_pose'], left_hand_pose=p['lhand_pose'],
                          right_hand_pose=p['rhand_pose'], jaw_pose=p['jaw_pose'],
                          leye_pose=p['leye_pose'], reye_pose=p['reye_pose'])
        world_head = (camera_rot @ head_camera[..., None]).squeeze(-1) + c2w[:, :3, 3]
        world = body.vertices + (world_head - head_fk)[:, None]
        camera = (world - c2w[:, None, :3, 3]) @ camera_rot
        return camera, head_camera
