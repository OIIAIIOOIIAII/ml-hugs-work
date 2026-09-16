#!/usr/bin/env python3
"""Build controlled point-map-like local-geometry degradation for E2.

The clean contact target and anchor drift stay fixed.  Only the scene proxy
observed by the controller is corrupted, allowing safety/abstention tests
without claiming a real Splat-SAP result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.features import build_features


PRESETS = {
    "mild": {"normal_deg": 8.0, "distance_noise_m": 0.008, "distance_bias_m": 0.004, "dropout": 0.08, "delay": 1},
    "medium": {"normal_deg": 18.0, "distance_noise_m": 0.020, "distance_bias_m": 0.010, "dropout": 0.20, "delay": 2},
    "severe": {"normal_deg": 35.0, "distance_noise_m": 0.040, "distance_bias_m": 0.020, "dropout": 0.38, "delay": 4},
}


def rotate(normals: np.ndarray, rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=normals.shape).astype(np.float32)
    axis -= (axis * normals).sum(-1, keepdims=True) * normals
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-6)
    theta = rng.uniform(-max_deg, max_deg, size=normals.shape[:-1])[..., None].astype(np.float32) * np.pi / 180.0
    return (normals * np.cos(theta) + np.cross(axis, normals) * np.sin(theta)).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="E1b dataset root with train/val/test NPZ files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preset", choices=tuple(PRESETS), required=True)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--normal-deg", type=float)
    p.add_argument("--distance-noise-m", type=float)
    p.add_argument("--distance-bias-m", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument("--delay", type=int)
    p.add_argument("--isolate", action="store_true", help="Set all non-overridden corruption axes to zero for a one-factor audit.")
    args = p.parse_args(); cfg = PRESETS[args.preset].copy()
    if args.isolate:
        cfg = {"normal_deg": 0.0, "distance_noise_m": 0.0, "distance_bias_m": 0.0, "dropout": 0.0, "delay": 0}
    for name in ("normal_deg", "distance_noise_m", "distance_bias_m", "dropout", "delay"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    for split in ("train", "val", "test"):
        for source in sorted((args.input / split).glob("*.npz")):
            rng = np.random.default_rng(args.seed + sum(source.name.encode()))
            with np.load(source, allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
            t = len(data["timestamps"]); delay = cfg["delay"]
            index = np.maximum(np.arange(t) - delay, 0)
            clean_normal = data["surface_normal"].copy(); clean_point = data["surface_point"].copy(); clean_distance = data["surface_distance"].copy()
            normal = rotate(clean_normal[index], rng, cfg["normal_deg"])
            distance = clean_distance[index] + rng.normal(0.0, cfg["distance_noise_m"], clean_distance.shape).astype(np.float32) + cfg["distance_bias_m"]
            point = clean_point[index] + normal * distance[..., None]
            dropout = rng.random(clean_distance.shape) < cfg["dropout"]
            confidence = np.clip(data["surface_confidence"][index] * (1.0 - cfg["normal_deg"] / 60.0) * (1.0 - dropout.astype(np.float32)), 0.0, 1.0).astype(np.float32)
            point[dropout] = 0.0; normal[dropout] = 0.0; distance[dropout] = 0.0
            features, aux = build_features(data["anchor_position"], data["anchor_orientation"], point, normal, distance, confidence,
                                           data["root"], data["alignment_residual"], data["alignment_confidence"], data["timestamps"])
            data.update({"surface_point": point.astype(np.float32), "surface_normal": normal.astype(np.float32), "surface_distance": distance.astype(np.float32),
                         "surface_confidence": confidence, "features": features, "anchor_velocity": aux["anchor_velocity"],
                         "proxy_dropout": dropout.astype(np.uint8), "proxy_delay_frames": np.full((t, 2), delay, np.int16),
                         "proxy_normal_error_deg": np.full((t, 2), cfg["normal_deg"], np.float32), "proxy_preset": np.full(t, ("mild", "medium", "severe").index(args.preset), np.int16)})
            target = args.output / split / source.name; target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **data)
    manifest = {"stage": "E2", "scope": "controlled point-map proxy degradation; not Splat-SAP output", "input": str(args.input), "preset": args.preset, "seed": args.seed, **cfg}
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
