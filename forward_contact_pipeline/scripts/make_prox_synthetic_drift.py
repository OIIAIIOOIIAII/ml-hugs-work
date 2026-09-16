#!/usr/bin/env python3
"""Create a controlled PROX contact-correction sequence from SDF teacher geometry.

This B-layer dataset simulates a frozen front-end's low-frequency root drift
and foot-local error.  Its correction targets are known by construction, so it
is appropriate for testing GRU/TCN mechanics before Human3R-on-PROX is ready.
It must never be reported as a real front-end or GUSH3R result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.features import build_features


def ar_noise(rng: np.random.Generator, frames: int, shape: tuple[int, ...], sigma: float, rho: float) -> np.ndarray:
    values = np.zeros((frames,) + shape, dtype=np.float32)
    innovation = sigma * np.sqrt(max(1.0 - rho * rho, 1e-6))
    values[0] = rng.normal(0.0, sigma, size=shape)
    for frame in range(1, frames):
        values[frame] = rho * values[frame - 1] + rng.normal(0.0, innovation, size=shape)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True, help="*.teacher.npz from build_prox_geometry_labels.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-sigma-m", type=float, default=0.035)
    parser.add_argument("--foot-sigma-m", type=float, default=0.020)
    parser.add_argument("--temporal-rho", type=float, default=0.94)
    parser.add_argument("--penetration-depth-m", type=float, default=0.005)
    parser.add_argument(
        "--correction-target-mode", choices=("injected_components", "contact_normal_projection", "contact_anchor_lock"),
        default="injected_components",
        help="The legacy injected-component target is not generally identifiable. "
             "contact_normal_projection defines an observable SDF-normal correction for E1.",
    )
    args = parser.parse_args()
    with np.load(args.teacher, allow_pickle=False) as data:
        teacher = {key: data[key].copy() for key in data.files}
    required = {"timestamps", "root", "anchor_position", "anchor_orientation", "surface_point", "surface_normal", "surface_distance", "contact", "uncertainty"}
    missing = required - set(teacher)
    if missing:
        raise KeyError(f"Teacher data missing {sorted(missing)}")
    frames = len(teacher["timestamps"])
    rng = np.random.default_rng(args.seed)
    root_error = ar_noise(rng, frames, (3,), args.root_sigma_m, args.temporal_rho)
    foot_error = ar_noise(rng, frames, (2, 3), args.foot_sigma_m, args.temporal_rho)
    if args.correction_target_mode == "contact_anchor_lock":
        # E1b: tangential drift exists only while the clean teacher says the
        # foot is in contact.  Normal clearance remains nearly unchanged, so
        # a normal-only corrector cannot solve this task.
        foot_error = foot_error - np.sum(foot_error * teacher["surface_normal"], axis=-1, keepdims=True) * teacher["surface_normal"]
        foot_error *= (teacher["contact"] > 0.5)[..., None]
    total_error = root_error[:, None, :] + foot_error
    anchor_position = teacher["anchor_position"] + total_error
    root = teacher["root"] + root_error
    observed_distance = teacher["surface_distance"] + np.sum(total_error * teacher["surface_normal"], axis=-1)
    alignment_residual = np.linalg.norm(root_error, axis=-1).astype(np.float32)
    alignment_confidence = np.exp(-alignment_residual / max(args.root_sigma_m * 3.0, 1e-6)).astype(np.float32)
    surface_confidence = np.clip(1.0 - teacher["uncertainty"], 0.0, 1.0).astype(np.float32)
    features, aux = build_features(
        anchor_position, teacher["anchor_orientation"], teacher["surface_point"], teacher["surface_normal"],
        observed_distance, surface_confidence, root, alignment_residual, alignment_confidence, teacher["timestamps"],
    )
    root_residual = np.zeros((frames, 6), dtype=np.float32)
    pose_residual = np.zeros((frames, 12), dtype=np.float32)
    contact_anchor_world = np.full_like(anchor_position, np.nan)
    if args.correction_target_mode == "injected_components":
        root_residual[:, :3] = -root_error
        pose_residual[:, :6] = -foot_error.reshape(frames, 6)
    elif args.correction_target_mode == "contact_normal_projection":
        # E1 observable mechanism target: only correct a true contact foot
        # along the known local SDF normal.  The common root correction is the
        # mean active-foot displacement and the per-foot residual is its
        # deterministic remainder, removing the root-vs-foot gauge ambiguity
        # present in the legacy random-component target.
        active = (teacher["contact"] > 0.5) & (surface_confidence > 0.5)
        total = -observed_distance[..., None] * teacher["surface_normal"] * active[..., None]
        active_count = np.maximum(active.sum(axis=1, keepdims=True), 1)
        root_xyz = total.sum(axis=1) / active_count
        root_residual[:, :3] = root_xyz.astype(np.float32)
        pose_residual[:, :6] = (total - root_xyz[:, None, :]).reshape(frames, 6).astype(np.float32)
    else:
        # Each contact episode receives a fixed clean world anchor at entry.
        # We deliberately put the complete correction in the per-foot head and
        # keep root residual zero, avoiding a root/foot gauge ambiguity while
        # testing causal foot-lock execution.
        lock_delta = np.zeros_like(anchor_position)
        for foot in range(2):
            active_anchor = None
            for frame in range(frames):
                active = bool(teacher["contact"][frame, foot] > 0.5 and surface_confidence[frame, foot] > 0.5)
                if active and active_anchor is None:
                    active_anchor = teacher["anchor_position"][frame, foot].copy()
                if active:
                    contact_anchor_world[frame, foot] = active_anchor
                    lock_delta[frame, foot] = active_anchor - anchor_position[frame, foot]
                else:
                    active_anchor = None
        pose_residual[:, :6] = lock_delta.reshape(frames, 6).astype(np.float32)
    arrays = {
        "features": features,
        "timestamps": teacher["timestamps"].astype(np.float32),
        "anchor_position": anchor_position.astype(np.float32),
        "anchor_velocity": aux["anchor_velocity"].astype(np.float32),
        "anchor_orientation": teacher["anchor_orientation"].astype(np.float32),
        "root": root.astype(np.float32),
        "surface_point": teacher["surface_point"].astype(np.float32),
        "surface_normal": teacher["surface_normal"].astype(np.float32),
        "surface_distance": teacher["surface_distance"].astype(np.float32),
        "surface_confidence": surface_confidence,
        "alignment_residual": alignment_residual,
        "alignment_confidence": alignment_confidence,
        "contact": teacher["contact"].astype(np.float32),
        "contact_point": teacher["surface_point"].astype(np.float32),
        "pose_residual": pose_residual,
        "root_residual": root_residual,
        "penetration_risk": (observed_distance < -args.penetration_depth_m).astype(np.float32),
        "uncertainty": teacher["uncertainty"].astype(np.float32),
        "label_source_id": np.full(frames, 5, np.int16),  # PROX synthetic drift; not an attach_labels source.
    }
    if args.correction_target_mode == "contact_anchor_lock":
        # Teacher-forced recurrent state for E1b training.  The later rollout
        # experiment must replace these fields with previous predictions.
        previous_contact = np.zeros((frames, 2), np.float32)
        previous_foot_residual = np.zeros((frames, 2, 3), np.float32)
        previous_contact[1:] = teacher["contact"][:-1]
        previous_foot_residual[1:] = pose_residual[:-1, :6].reshape(frames - 1, 2, 3)
        for foot, offset in enumerate((0, 25)):
            arrays["features"][:, offset + 21] = previous_contact[:, foot]
            arrays["features"][:, offset + 22:offset + 25] = previous_foot_residual[:, foot]
        arrays["contact_anchor_world"] = contact_anchor_world.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "dataset": "PROX_SYNTHETIC_DRIFT",
        "teacher": str(args.teacher),
        "frames": frames,
        "seed": args.seed,
        "root_sigma_m": args.root_sigma_m,
        "foot_sigma_m": args.foot_sigma_m,
        "temporal_rho": args.temporal_rho,
        "input_is_frontend_prediction": False,
        "target_residual_definition": (
            "negative of injected root/foot error"
            if args.correction_target_mode == "injected_components"
            else ("contact-gated projection of observed signed distance along the teacher SDF normal"
                  if args.correction_target_mode == "contact_normal_projection"
                  else "teacher-contact episode anchor minus tangentially drifted foot anchor; root residual fixed to zero")
        ),
        "correction_target_mode": args.correction_target_mode,
        "note": "Mechanism-only B-layer data; do not report as Human3R/GUSH3R performance.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
