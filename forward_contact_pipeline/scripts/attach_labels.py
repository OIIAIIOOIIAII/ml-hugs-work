#!/usr/bin/env python3
"""Attach BEDLAM/PROX GT labels to a prepared sequence.

The label NPZ may contain any subset of the standard target keys. Frame counts
must match; unknown keys are rejected to prevent silent supervision mistakes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.data import TARGET_KEYS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-source", required=True, choices=["BEDLAM_GT", "PROX_GT", "NeuMan_pseudo", "manual"])
    parser.add_argument("--label-confidence", type=float, default=1.0)
    args = parser.parse_args()
    with np.load(args.sequence, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.labels, allow_pickle=False) as archive:
        labels = {key: archive[key] for key in archive.files}
    unknown = set(labels) - set(TARGET_KEYS)
    if unknown:
        raise KeyError(f"Unknown label keys: {sorted(unknown)}")
    frames = len(data["features"])
    for key, value in labels.items():
        if len(value) != frames:
            raise ValueError(f"{key}: {len(value)} labels for {frames} frames")
        data[key] = value
    data["label_confidence"] = np.full((frames, 2), args.label_confidence, np.float32)
    data["label_source_id"] = np.full(frames, {"BEDLAM_GT": 1, "PROX_GT": 2, "NeuMan_pseudo": 3, "manual": 4}[args.label_source], np.int16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Attached {sorted(labels)} from {args.label_source}: {args.output}")


if __name__ == "__main__":
    main()
