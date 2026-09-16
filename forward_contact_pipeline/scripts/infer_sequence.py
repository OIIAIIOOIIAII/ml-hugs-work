#!/usr/bin/env python3
"""Run causal contact inference and write predictions plus binary packets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.io import save_npz, write_json
from contact_streaming.runtime import ContactRuntime, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with np.load(args.sequence, allow_pickle=False) as archive:
        sequence = {key: archive[key] for key in archive.files}
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    runtime = ContactRuntime(
        model, checkpoint["config"]["history"], checkpoint["feature_mean"], checkpoint["feature_std"], args.device
    )
    predictions = {key: [] for key in ["contact_probability", "root_residual", "pose_residual", "uncertainty"]}
    packet_dir = args.output / "packets"; packet_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); packet_bytes = 0
    timestamps = sequence.get("timestamps", np.arange(len(sequence["features"])) / 30.0)
    for frame_id, (timestamp, feature) in enumerate(zip(timestamps, sequence["features"])):
        result = runtime.step(feature, frame_id, float(timestamp))
        for key in predictions:
            predictions[key].append(getattr(result, key))
        payload = result.packet; packet_bytes += len(payload)
        (packet_dir / f"{frame_id:06d}.hct").write_bytes(payload)
    elapsed = time.perf_counter() - start
    save_npz(args.output / "predictions.npz", {key: np.stack(value) for key, value in predictions.items()})
    report = {
        "frames": len(sequence["features"]), "seconds": elapsed, "fps": len(sequence["features"]) / max(elapsed, 1e-9),
        "packet_bytes_total": packet_bytes, "packet_bytes_per_frame": packet_bytes / len(sequence["features"]),
    }
    write_json(args.output / "runtime.json", report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
