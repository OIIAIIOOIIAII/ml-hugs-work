from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from .packet import packet_from_prediction


@dataclass
class RuntimeResult:
    contact_probability: np.ndarray
    root_residual: np.ndarray
    pose_residual: np.ndarray
    uncertainty: np.ndarray
    packet: bytes


class ContactRuntime:
    def __init__(self, model, history: int, feature_mean: np.ndarray, feature_std: np.ndarray, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.history = int(history)
        self.device = torch.device(device)
        self.mean = np.asarray(feature_mean, dtype=np.float32)
        self.std = np.asarray(feature_std, dtype=np.float32)
        self.memory = deque(maxlen=self.history)

    def reset(self) -> None:
        self.memory.clear()

    @torch.no_grad()
    def step(self, feature: np.ndarray, frame_id: int, timestamp_sec: float, mode: str = "Tracking") -> RuntimeResult:
        normalized = (np.asarray(feature, np.float32) - self.mean) / self.std
        self.memory.append(normalized)
        while len(self.memory) < self.history:
            self.memory.appendleft(normalized.copy())
        tensor = torch.from_numpy(np.stack(self.memory)[None]).to(self.device)
        output = self.model(tensor)
        probability = torch.sigmoid(output.contact_logits[0]).cpu().numpy()
        uncertainty = torch.sigmoid(output.uncertainty_logits[0]).cpu().numpy()
        root = output.root_residual[0].cpu().numpy()
        pose = output.pose_residual[0].cpu().numpy()
        packet = packet_from_prediction(frame_id, timestamp_sec, probability, root, pose, uncertainty, mode).encode()
        return RuntimeResult(probability, root, pose, uncertainty, packet)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[torch.nn.Module, Dict]:
    from .models import build_model

    # Checkpoints are produced locally by train_contact.py and contain NumPy
    # feature statistics in addition to tensor weights.  PyTorch >=2.6 defaults
    # to weights_only=True, which rejects that trusted local metadata.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = build_model(
        config["model"], config["input_dim"], config["hidden_dim"], config["layers"], config["dropout"]
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
