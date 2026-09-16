"""Differentiable causal rollout utilities for the E1b foot-lock experiment."""

from __future__ import annotations

import torch


# 58D feature layout; each foot has previous-contact plus previous-residual.
PREVIOUS = ((21, 22), (46, 47))


def rollout(
    model,
    features: torch.Tensor,
    normals: torch.Tensor,
    history: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_action_m: float,
    hard_gate: float | None = None,
    external_gate: torch.Tensor | None = None,
):
    """Roll out one sequence without teacher previous-contact/residual state.

    `features` is [T,58].  At each frame we overwrite the recurrent-state
    fields with our own previous contact/action.  Actions are projected into
    the local scene tangent plane and bounded before being fed back.
    """
    dynamic = []
    probabilities, actions = [], []
    previous_probability = torch.zeros(2, dtype=features.dtype, device=features.device)
    previous_action = torch.zeros(2, 3, dtype=features.dtype, device=features.device)
    for frame in range(features.shape[0]):
        value = features[frame].clone()
        for foot, (contact_idx, residual_idx) in enumerate(PREVIOUS):
            value[contact_idx] = previous_probability[foot]
            value[residual_idx : residual_idx + 3] = previous_action[foot]
        dynamic.append(value)
        window = dynamic[max(0, frame - history + 1) : frame + 1]
        if len(window) < history:
            window = [window[0]] * (history - len(window)) + window
        x = torch.stack(window).unsqueeze(0)
        output = model((x - mean) / std)
        probability = torch.sigmoid(output.contact_logits[0])
        raw = output.pose_residual[0, :6].reshape(2, 3)
        normal = normals[frame]
        tangent = raw - (raw * normal).sum(dim=-1, keepdim=True) * normal
        action = float(max_action_m) * torch.tanh(tangent / float(max_action_m))
        # Keep the gate explicitly [foot, 1].  A bare [foot] tensor would
        # broadcast against the xyz dimension and fail when hard gating is
        # enabled during evaluation.
        gate = (probability >= hard_gate).to(action.dtype) if hard_gate is not None else probability
        if external_gate is not None:
            gate = gate * external_gate[frame].to(action.dtype)
        applied = action * gate[:, None]
        probabilities.append(probability)
        actions.append(applied)
        previous_probability = probability
        previous_action = applied
    return torch.stack(probabilities), torch.stack(actions)
