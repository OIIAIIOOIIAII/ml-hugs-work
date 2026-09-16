#!/usr/bin/env python3
"""Audit contact-label prevalence and geometry-rule separability on synthetic drift.

Thresholds are selected only on the train scenes, then reported unchanged on
validation and held-out test scenes.  This is a diagnostic baseline, not a
front-end/GUSH3R result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.metrics import binary_contact_metrics


def arrays(root: Path, split: str) -> dict[str, np.ndarray]:
    files = sorted((root / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {root / split}")
    collected: dict[str, list[np.ndarray]] = {"contact": [], "distance": [], "speed": [], "confidence": []}
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            collected["contact"].append(data["contact"].astype(np.float32).reshape(-1))
            # The observed features are intentionally used here: noisy anchor
            # velocity and noisy signed distance, not clean teacher targets.
            collected["distance"].append(data["features"][:, [18, 43]].astype(np.float32).reshape(-1))
            velocity = data["anchor_velocity"].astype(np.float32)
            collected["speed"].append(np.linalg.norm(velocity, axis=-1).reshape(-1))
            collected["confidence"].append((data["surface_confidence"] * data["alignment_confidence"][:, None]).reshape(-1))
    return {key: np.concatenate(value) for key, value in collected.items()}


def evaluate(values: dict[str, np.ndarray], distance: float, speed: float, confidence: float) -> dict[str, float]:
    probability = ((np.abs(values["distance"]) <= distance) & (values["speed"] <= speed) & (values["confidence"] >= confidence)).astype(np.float32)
    result = binary_contact_metrics(probability, values["contact"])
    result["predicted_positive_rate"] = float(probability.mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-grid", type=float, nargs="+", default=[0.01, 0.02, 0.035, 0.05, 0.07, 0.10, 0.15])
    parser.add_argument("--speed-grid", type=float, nargs="+", default=[0.10, 0.16, 0.25, 0.40, 0.60, 1.0, 2.0])
    parser.add_argument("--confidence-grid", type=float, nargs="+", default=[0.0, 0.10, 0.25, 0.50])
    args = parser.parse_args()
    by_split = {split: arrays(args.data, split) for split in ("train", "val", "test")}
    candidates = []
    for distance in args.distance_grid:
        for speed in args.speed_grid:
            for confidence in args.confidence_grid:
                score = evaluate(by_split["train"], distance, speed, confidence)
                candidates.append((score["contact_f1"], distance, speed, confidence, score))
    _, distance, speed, confidence, train_score = max(candidates, key=lambda item: (item[0], item[1], item[2], -item[3]))
    report = {
        "scope": "PROX synthetic-drift B-layer diagnostic only; no Human3R/GUSH3R claim",
        "selection": "distance/speed/confidence grid chosen by train contact F1 only; val/test untouched",
        "selected_thresholds": {"abs_distance_m": distance, "speed_mps": speed, "confidence": confidence},
        "splits": {
            split: {
                "samples_per_foot": int(values["contact"].size),
                "positive_rate": float(values["contact"].mean()),
                "distance_abs_quantiles_m": [float(x) for x in np.quantile(np.abs(values["distance"]), [0.1, 0.5, 0.9])],
                "speed_quantiles_mps": [float(x) for x in np.quantile(values["speed"], [0.1, 0.5, 0.9])],
                "fixed_threshold_metrics": evaluate(values, distance, speed, confidence),
            }
            for split, values in by_split.items()
        },
        "train_selection_metrics": train_score,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
