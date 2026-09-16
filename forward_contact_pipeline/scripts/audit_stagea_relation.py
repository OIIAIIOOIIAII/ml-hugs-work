#!/usr/bin/env python3
"""Numerically audit Stage-A relation assets; never reads RGB images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for manifest_path in sorted(args.root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(manifest_path.parent / manifest["payload"], allow_pickle=False) as payload:
            required = {
                "roi_vertex_local", "roi_vertex_normal_local", "vertex_contact", "vertex_proximity",
                "scene_point_local", "scene_normal_local", "scene_point_distance", "teacher_contact", "timestamps",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"{manifest_path}: missing {sorted(missing)}")
            arrays = {key: payload[key] for key in required}
        frames, feet, roi_vertices, xyz = arrays["roi_vertex_local"].shape
        patch_shape = arrays["scene_point_local"].shape
        expected_patch = (frames, feet, roi_vertices, manifest["scene_points_per_vertex"], 3)
        if (feet, xyz) != (2, 3) or patch_shape != expected_patch:
            raise ValueError(f"{manifest_path}: incompatible shapes roi={arrays['roi_vertex_local'].shape} patch={patch_shape}")
        finite = all(np.isfinite(value).all() for value in arrays.values())
        normal_length = np.linalg.norm(arrays["scene_normal_local"], axis=-1)
        records.append({
            "sequence": manifest["sequence"],
            "scene": manifest["scene"],
            "error_family": manifest["error_family"],
            "frames": frames,
            "vertex_contact_rate": float(arrays["vertex_contact"].mean()),
            "roi_contact_rate": float(arrays["teacher_contact"].mean()),
            "mean_patch_distance_m": float(arrays["scene_point_distance"].mean()),
            "p95_patch_distance_m": float(np.quantile(arrays["scene_point_distance"], 0.95)),
            "mean_scene_normal_length": float(normal_length.mean()),
            "finite": bool(finite),
            "rgb_references_present": int(sum(bool(path) for path in manifest["rgb_paths"])),
        })
    if not records:
        raise FileNotFoundError(f"No manifests under {args.root}")
    scenes = [record["scene"] for record in records]
    report = {
        "schema": "hugs.forward_contact.stage_a_relation.audit.v1",
        "assets": len(records),
        "frames": int(sum(record["frames"] for record in records)),
        "unique_scenes": len(set(scenes)),
        "scene_duplicates": sorted({scene for scene in scenes if scenes.count(scene) > 1}),
        "error_families": sorted({record["error_family"] for record in records}),
        "all_finite": bool(all(record["finite"] for record in records)),
        "records": records,
        "verdict": (
            "clean-oracle relation contract valid; insufficient for error-family-separated robustness training until degraded relation assets are added"
            if len({record["error_family"] for record in records}) == 1
            else "relation assets include multiple error families; construct and freeze split before training"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
