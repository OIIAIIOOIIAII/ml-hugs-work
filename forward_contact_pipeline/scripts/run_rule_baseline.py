#!/usr/bin/env python3
"""Evaluate geometry-rule contact labels and low-dimensional corrections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.metrics import binary_contact_metrics, foot_sliding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.sequence, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    probability = data.get("contact_probability_rule", data["contact"])
    metrics = binary_contact_metrics(probability, data["contact"])
    metrics.update(
        {
            "frames": int(len(data["contact"])),
            "foot_sliding": foot_sliding(data["anchor_position"], data["contact"], data.get("timestamps")) if "anchor_position" in data else None,
            "penetration_ratio": float((data["surface_distance"] < 0).mean()),
            "penetration_depth": float(np.maximum(-data["surface_distance"], 0).mean()),
            "mean_abs_root_correction": float(np.linalg.norm(data["root_residual"][:, :3], axis=-1).mean()),
        }
    )
    text = json.dumps(metrics, indent=2); print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
