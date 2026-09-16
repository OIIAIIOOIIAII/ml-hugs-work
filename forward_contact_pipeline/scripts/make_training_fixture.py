#!/usr/bin/env python3
"""Small numeric fixtures for software checks, never a real experiment dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_streaming.training.data import digest


def create_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17)
    records = []
    for i, split in enumerate(("train", "train", "val", "test")):
        shape = (4, 2, 8)
        distance = rng.uniform(-.05, .05, shape).astype(np.float32)
        vertices = rng.normal(0, .1, (*shape, 3)).astype(np.float32)
        vertices[..., 2] = distance + rng.normal(0, .005, shape)
        normals = np.zeros_like(vertices); normals[..., 2] = 1
        points = np.repeat(vertices[..., None, :], 4, axis=-2)
        points[..., 2] = 0
        point_normals = np.repeat(normals[..., None, :], 4, axis=-2)
        inputs = {"vertex_local": vertices, "vertex_normal": normals, "point_local": points,
                  "point_normal": point_normals, "point_valid": np.ones((*shape, 4), np.float32),
                  "rgb_tokens": rng.normal(size=(4, 2, 3, 6)).astype(np.float32),
                  "gaussian_features": rng.normal(size=(*shape, 4, 3)).astype(np.float32)}
        targets = {"contact": (np.abs(distance) < .02).astype(np.float32), "contact_valid": np.ones(shape, np.float32),
                   "proximity": distance, "proximity_valid": np.ones(shape, np.float32)}
        ip, tp = root / f"{i}.inputs.npz", root / f"{i}.targets.npz"
        np.savez(ip, **inputs); np.savez(tp, **targets)
        records.append({"id": f"fixture-{i}", "split": split, "scene_id": f"scene-{i}",
                        "sequence_id": f"sequence-{i}", "subject_id": f"subject-{i}",
                        "input": ip.name, "target": tp.name, "input_sha256": digest(ip), "target_sha256": digest(tp)})
    index = {"schema_version": 1, "dataset": "synthetic_software_fixture", "provenance": {"kind": "synthetic"}, "records": records}
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(create_fixture(parser.parse_args().output))
