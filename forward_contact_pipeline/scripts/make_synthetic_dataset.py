#!/usr/bin/env python3
"""Generate deterministic synthetic contact sequences for end-to-end tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.features import build_features
from contact_streaming.io import save_npz, write_json
from contact_streaming.rules import geometric_correction


def generate(seed: int, frames: int):
    rng = np.random.default_rng(seed)
    time = np.arange(frames, dtype=np.float32) / 30.0
    phase = 2 * np.pi * time / 1.2
    contact = np.stack([(np.sin(phase) < 0), (np.sin(phase + np.pi) < 0)], axis=1).astype(np.float32)
    root = np.stack([0.25 * time, 1.0 + 0.01 * np.sin(phase), np.zeros_like(time)], axis=1)
    anchors = np.zeros((frames, 2, 3), np.float32)
    anchors[:, :, 0] = root[:, None, 0] + np.array([-0.10, 0.10])[None]
    anchors[:, :, 1] = 0.02 + (1 - contact) * (0.10 + 0.12 * np.maximum(np.sin(phase[:, None] + np.array([0, np.pi])), 0))
    anchors[:, :, 2] = np.array([-0.04, 0.04])[None]
    anchors += rng.normal(0, 0.003, anchors.shape)
    surface_point = anchors.copy(); surface_point[:, :, 1] = 0
    surface_normal = np.zeros_like(anchors); surface_normal[:, :, 1] = 1
    distance = anchors[:, :, 1] + rng.normal(0, 0.002, (frames, 2))
    confidence = np.full((frames, 2), 0.95, np.float32)
    orientation = np.zeros((frames, 2, 6), np.float32)
    orientation[:, :, 2] = 1; orientation[:, :, 4] = 1
    alignment_residual = np.full(frames, 0.01, np.float32)
    alignment_confidence = np.full(frames, 0.95, np.float32)
    features, aux = build_features(
        anchors, orientation, surface_point, surface_normal, distance, confidence,
        root, alignment_residual, alignment_confidence, time,
    )
    root_residual, pose_residual = [], []
    for index in range(frames):
        root_delta, pose_delta, _ = geometric_correction(contact[index] > 0, distance[index], surface_normal[index], confidence[index])
        root_residual.append(root_delta); pose_residual.append(pose_delta)
    return {
        "features": features,
        "timestamps": time,
        "anchor_position": anchors,
        "anchor_velocity": aux["anchor_velocity"],
        "contact": contact,
        "contact_point": surface_point,
        "surface_distance": distance.astype(np.float32),
        "surface_normal": surface_normal,
        "pose_residual": np.stack(pose_residual),
        "root_residual": np.stack(root_residual),
        "penetration_risk": (distance < 0).astype(np.float32),
        "uncertainty": np.full((frames, 2), 0.05, np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(args.sequences):
        name = f"synthetic_{index:03d}.npz"
        save_npz(args.output / name, generate(index, args.frames))
        names.append(name)
    n_train = max(1, int(0.7 * len(names)))
    n_val = max(n_train + 1, int(0.85 * len(names))) if len(names) > 2 else n_train
    splits = {"train": names[:n_train], "val": names[n_train:n_val] or names[-1:], "test": names[n_val:] or names[-1:]}
    write_json(args.output / "splits.json", splits)
    print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()
