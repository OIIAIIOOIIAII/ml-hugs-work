from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_KEYS = [
    "contact", "contact_point", "surface_distance", "surface_normal",
    "pose_residual", "root_residual", "penetration_risk", "uncertainty",
]


def discover_sequences(root: str | Path, split: str) -> List[Path]:
    root = Path(root)
    split_file = root / "splits.json"
    if split_file.exists():
        split_map = json.loads(split_file.read_text(encoding="utf-8"))
        return [root / item for item in split_map[split]]
    split_dir = root / split
    if split_dir.exists():
        return sorted(split_dir.glob("*.npz"))
    return sorted(root.glob("*.npz"))


class ContactWindowDataset(Dataset):
    def __init__(self, root: str | Path, split: str, history: int = 8):
        self.history = int(history)
        self.sequences = []
        self.index = []
        for path in discover_sequences(root, split):
            with np.load(path, allow_pickle=False) as archive:
                sequence = {key: archive[key] for key in archive.files}
            if "features" not in sequence:
                raise KeyError(f"Missing features in {path}")
            seq_idx = len(self.sequences)
            self.sequences.append((path, sequence))
            for end in range(self.history - 1, len(sequence["features"])):
                self.index.append((seq_idx, end))
        if not self.index:
            raise RuntimeError(f"No windows found under {root} for split={split}")

    @property
    def input_dim(self) -> int:
        return int(self.sequences[0][1]["features"].shape[-1])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_idx, end = self.index[index]
        _, sequence = self.sequences[seq_idx]
        start = end - self.history + 1
        item = {"features": torch.from_numpy(sequence["features"][start : end + 1]).float()}
        for key in TARGET_KEYS:
            if key in sequence:
                item[key] = torch.from_numpy(np.asarray(sequence[key][end])).float()
        item["frame_index"] = torch.tensor(end, dtype=torch.long)
        item["sequence_index"] = torch.tensor(seq_idx, dtype=torch.long)
        return item


def feature_stats(dataset: ContactWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    all_features = np.concatenate([sequence["features"] for _, sequence in dataset.sequences], axis=0)
    mean = all_features.mean(0).astype(np.float32)
    std = all_features.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std
