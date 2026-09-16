#!/usr/bin/env python3
"""Audit E0 PROX SDF teacher labels without decoding RGB/depth images.

The audit validates the serialized coordinate/label contract.  It deliberately
does not claim that a separate learned front end is aligned to PROX world.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_LABELS = {
    "contact", "contact_point", "surface_distance", "surface_normal",
    "penetration_risk", "uncertainty",
}
REQUIRED_TEACHER = {
    "timestamps", "root", "anchor_position", "anchor_orientation",
    "surface_point", "surface_normal", "surface_distance", "contact",
    "penetration_risk", "uncertainty",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    failures = []
    for path in sorted(args.labels.glob("*.npz")):
        if path.name.endswith(".teacher.npz"):
            continue
        meta_path = path.with_suffix(".json")
        teacher_path = path.with_suffix(".teacher.npz")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(path, allow_pickle=False) as labels:
                missing = REQUIRED_LABELS - set(labels.files)
                if missing:
                    raise ValueError(f"missing labels={sorted(missing)}")
                frames = int(labels["contact"].shape[0])
                if frames <= 1 or labels["contact"].shape != (frames, 2):
                    raise ValueError("invalid foot label shape")
                if not all(np.isfinite(labels[key]).all() for key in ("surface_distance", "surface_normal", "contact_point")):
                    raise ValueError("non-finite label geometry")
                has_teacher = teacher_path.exists()
                teacher = None
                if has_teacher:
                    with np.load(teacher_path, allow_pickle=False) as archive:
                        missing_teacher = REQUIRED_TEACHER - set(archive.files)
                        if missing_teacher:
                            raise ValueError(f"missing teacher={sorted(missing_teacher)}")
                        teacher = {key: archive[key] for key in archive.files}
                    if teacher["timestamps"].shape != (frames,) or teacher["anchor_position"].shape != (frames, 2, 3):
                        raise ValueError("invalid teacher frame shape")
                    if not all(np.isfinite(teacher[key]).all() for key in ("surface_distance", "surface_normal", "surface_point", "anchor_position", "root")):
                        raise ValueError("non-finite teacher geometry")
                    if not np.all(np.diff(teacher["timestamps"]) >= 0):
                        raise ValueError("non-monotonic timestamps")
                normal_norm = np.linalg.norm(labels["surface_normal"], axis=-1)
                rows.append({
                    "sequence": meta.get("sequence", path.stem),
                    "scene": meta.get("scene"),
                    "frames": frames,
                    "coordinate_frame": meta.get("coordinate_frame"),
                    "label_source": meta.get("label_source"),
                    "teacher_npz_present": has_teacher,
                    "valid_foot_fraction": float(meta.get("valid_foot_fraction", np.nan)),
                    "contact_rate": np.asarray(labels["contact"], dtype=np.float32).mean(axis=0).tolist(),
                    "penetration_rate": np.asarray(labels["penetration_risk"], dtype=np.float32).mean(axis=0).tolist(),
                    "median_abs_sdf_m": np.median(np.abs(labels["surface_distance"]), axis=0).astype(float).tolist(),
                    "normal_unit_error_mean": float(np.mean(np.abs(normal_norm - 1.0))),
                    "timestamp_min_step_s": float(np.min(np.diff(teacher["timestamps"]))) if teacher is not None else None,
                })
        except Exception as exc:  # audit must report all broken sequences
            failures.append({"path": str(path), "error": str(exc)})

    report = {
        "stage": "E0",
        "scope": "PROX PROXD-to-scene-SDF teacher contract; no Human3R/GUSH3R alignment claim",
        "coordinate_contract": "PROX_scene_world",
        "num_sequences": len(rows),
        "num_failures": len(failures),
        "sequences": rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
