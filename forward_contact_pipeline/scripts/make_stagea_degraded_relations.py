#!/usr/bin/env python3
"""Create controlled local-patch degradations for Stage-A robustness splits.

The labels remain the clean PROX SDF labels.  Only the scene evidence supplied
to Stage A is corrupted.  This is deliberately distinct from a real frozen
front-end error distribution and is recorded as a `proxy_*` error family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)


def rodrigues(values: np.ndarray, angle_deg: float, rng: np.random.Generator) -> np.ndarray:
    axes = normalize(rng.normal(size=values.shape[:-2] + (3,)).astype(np.float32))[..., None, :]
    angle = np.float32(np.deg2rad(angle_deg))
    return (
        values * np.cos(angle)
        + np.cross(axes, values) * np.sin(angle)
        + axes * np.sum(axes * values, axis=-1, keepdims=True) * (1.0 - np.cos(angle))
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("normal18", "distance", "dropout20"), required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    manifests = sorted(args.input_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(args.input_root)
    args.output_root.mkdir(parents=True)
    for index, manifest_path in enumerate(manifests):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(manifest_path.parent / manifest["payload"], allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        points = arrays["scene_point_local"]
        normals = arrays["scene_normal_local"]
        rng = np.random.default_rng(args.seed + index * 1009)
        valid = np.ones(points.shape[:-1], dtype=np.float32)
        if args.kind == "normal18":
            points = rodrigues(points, 18.0, rng)
            normals = normalize(rodrigues(normals, 18.0, rng))
        elif args.kind == "distance":
            # One bias per foot plus per-point depth noise, all in the local
            # normal axis; source labels remain untouched.
            bias = rng.normal(0.01, 0.002, size=points.shape[:2] + (1, 1, 1)).astype(np.float32)
            noise = rng.normal(0.0, 0.02, size=points.shape[:-1] + (1,)).astype(np.float32)
            points = points.copy()
            points[..., 2:3] += bias + noise
        else:
            valid = (rng.random(size=points.shape[:-1]) >= 0.20).astype(np.float32)
            points = points * valid[..., None]
            normals = normals * valid[..., None]
        arrays["scene_point_local"] = points.astype(np.float32)
        arrays["scene_normal_local"] = normals.astype(np.float32)
        arrays["scene_point_valid"] = valid
        arrays["scene_point_distance"] = np.linalg.norm(points, axis=-1).astype(np.float32)
        destination = args.output_root / manifest["sequence"]
        destination.mkdir()
        payload_name = manifest["payload"]
        np.savez_compressed(destination / payload_name, **arrays)
        manifest.update({
            "payload": payload_name,
            "parent_clean_manifest": str(manifest_path),
            "error_family": f"proxy_{args.kind}",
            "degradation": {"kind": args.kind, "seed": args.seed + index * 1009},
            "note": "Controlled local-scene evidence degradation; labels remain clean PROX oracle labels. Not a real-front-end result.",
        })
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"kind": args.kind, "assets": len(manifests), "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
