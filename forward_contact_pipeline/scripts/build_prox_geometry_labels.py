#!/usr/bin/env python3
"""Build auditable PROX foot-contact geometry labels from PROXD and scene SDF.

This is a *teacher-label* builder, not a Human3R/GUSH3R evaluation script.
PROXD provides fitted SMPL-X parameters and PROX provides an official scene
SDF.  We transform fitted body vertices from the recording camera to the
official scene world, query the SDF at left/right leg vertices, and export
only labels that have genuine PROX geometric supervision.

The output intentionally does not fabricate pose/root-residual targets:
those require a corrupted/predicted front-end input and are created by a
separate perturbation or front-end-alignment experiment.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox-root", type=Path, default=ROOT.parent / "PROX")
    parser.add_argument("--sequence", required=True, help="PROX sequence, e.g. BasementSittingBooth_00142_01")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz containing only valid GT target fields")
    parser.add_argument("--smplx-model-dir", type=Path, default=ROOT.parent / "Human3R" / "src" / "models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all matched frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--contact-distance-m", type=float, default=0.02)
    parser.add_argument("--foot-velocity-mps", type=float, default=0.15)
    parser.add_argument("--penetration-depth-m", type=float, default=0.005)
    return parser.parse_args()


def scene_name(sequence: str) -> str:
    parts = sequence.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Expected '<scene>_<subject>_<take>', got {sequence!r}")
    return parts[0]


def frame_seconds(frame_name: str, fallback: int) -> float:
    """Parse PROX names like s001_frame_00001__00.00.00.029."""
    match = re.search(r"__(\d+)\.(\d+)\.(\d+)\.(\d+)$", frame_name)
    if not match:
        return float(fallback) / 30.0
    hour, minute, second, millisecond = (int(value) for value in match.groups())
    return hour * 3600.0 + minute * 60.0 + second + millisecond / 1000.0


def load_segment_indices(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = np.asarray(payload["verts_ind"], dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError(f"Invalid verts_ind in {path}")
    return indices


class SDFGrid:
    """Trilinear sampler for PROX's flattened 256^3 signed-distance grid."""

    def __init__(self, path: Path, metadata_path: Path):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.minimum = np.asarray(metadata["min"], dtype=np.float32)
        self.maximum = np.asarray(metadata["max"], dtype=np.float32)
        self.dim = int(metadata["dim"])
        expected = self.dim**3
        values = np.load(path, mmap_mode="r")
        if values.size != expected:
            raise ValueError(f"{path}: expected {expected} values, got {values.size}")
        self.grid = values.reshape((self.dim, self.dim, self.dim))
        self.spacing = (self.maximum - self.minimum) / float(self.dim - 1)

    def sample(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float32)
        grid = (points - self.minimum) / (self.maximum - self.minimum) * (self.dim - 1)
        valid = np.all((grid >= 0.0) & (grid <= self.dim - 1.0), axis=-1)
        values = np.full(len(points), np.nan, np.float32)
        if not valid.any():
            return values, valid
        coordinate = grid[valid]
        lo = np.floor(coordinate).astype(np.int64)
        hi = np.minimum(lo + 1, self.dim - 1)
        frac = coordinate - lo
        c000 = self.grid[lo[:, 0], lo[:, 1], lo[:, 2]]
        c001 = self.grid[lo[:, 0], lo[:, 1], hi[:, 2]]
        c010 = self.grid[lo[:, 0], hi[:, 1], lo[:, 2]]
        c011 = self.grid[lo[:, 0], hi[:, 1], hi[:, 2]]
        c100 = self.grid[hi[:, 0], lo[:, 1], lo[:, 2]]
        c101 = self.grid[hi[:, 0], lo[:, 1], hi[:, 2]]
        c110 = self.grid[hi[:, 0], hi[:, 1], lo[:, 2]]
        c111 = self.grid[hi[:, 0], hi[:, 1], hi[:, 2]]
        wx, wy, wz = frac[:, 0], frac[:, 1], frac[:, 2]
        c00 = c000 * (1 - wz) + c001 * wz
        c01 = c010 * (1 - wz) + c011 * wz
        c10 = c100 * (1 - wz) + c101 * wz
        c11 = c110 * (1 - wz) + c111 * wz
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy
        values[valid] = c0 * (1 - wx) + c1 * wx
        return values.astype(np.float32), valid

    def normal(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return outward (increasing-SDF) normals with central differences."""
        points = np.asarray(points, dtype=np.float32)
        gradient = np.zeros_like(points, dtype=np.float32)
        valid = np.ones(len(points), dtype=bool)
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float32)
            offset[axis] = self.spacing[axis]
            plus, valid_plus = self.sample(points + offset)
            minus, valid_minus = self.sample(points - offset)
            valid &= valid_plus & valid_minus
            gradient[:, axis] = (plus - minus) / (2.0 * self.spacing[axis])
        length = np.linalg.norm(gradient, axis=-1, keepdims=True)
        valid &= length[:, 0] > 1e-6
        normals = np.zeros_like(gradient, dtype=np.float32)
        normals[valid] = gradient[valid] / length[valid]
        return normals, valid


def make_smplx(model_dir: Path, device: str):
    import smplx

    return smplx.create(
        str(model_dir), model_type="smplx", gender="neutral", ext="npz",
        use_pca=True, num_pca_comps=12, batch_size=1,
    ).to(torch.device(device)).eval()


MODEL_KEYS = {
    "betas", "body_pose", "global_orient", "transl", "left_hand_pose",
    "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose", "expression",
}


def fitted_body(model, parameters: Dict[str, np.ndarray], camera_to_world: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    inputs = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in parameters.items() if key in MODEL_KEYS
    }
    with torch.no_grad():
        output = model(return_verts=True, **inputs)
    vertices = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0].detach().cpu().numpy()
    def transform(points: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate([points, np.ones((len(points), 1), np.float32)], axis=1)
        return (homogeneous @ camera_to_world.T)[:, :3].astype(np.float32)
    return transform(vertices), transform(joints)


def sequence_pickles(result_root: Path, stride: int, max_frames: int) -> Iterable[Path]:
    paths = sorted(result_root.glob("*/000.pkl"))[::stride]
    return paths[:max_frames] if max_frames > 0 else paths


def main() -> None:
    args = parse_args()
    prox_root = args.prox_root.resolve()
    scene = scene_name(args.sequence)
    result_root = prox_root / "PROXD" / args.sequence / "results"
    if not result_root.is_dir():
        raise FileNotFoundError(result_root)
    camera_path = prox_root / "cam2world" / f"{scene}.json"
    camera_to_world = np.asarray(json.loads(camera_path.read_text(encoding="utf-8")), dtype=np.float32)
    if camera_to_world.shape != (4, 4):
        raise ValueError(f"Expected [4,4] cam2world in {camera_path}, got {camera_to_world.shape}")
    sdf = SDFGrid(prox_root / "sdf" / f"{scene}_sdf.npy", prox_root / "sdf" / f"{scene}.json")
    left_indices = load_segment_indices(prox_root / "body_segments" / "L_Leg.json")
    right_indices = load_segment_indices(prox_root / "body_segments" / "R_Leg.json")
    model = make_smplx(args.smplx_model_dir, args.device)

    records = []
    for index, path in enumerate(sequence_pickles(result_root, args.stride, args.max_frames)):
        with path.open("rb") as handle:
            parameters = pickle.load(handle, encoding="latin1")
        vertices, joints = fitted_body(model, parameters, camera_to_world, args.device)
        per_foot = []
        for indices in (left_indices, right_indices):
            candidates = vertices[indices]
            values, valid = sdf.sample(candidates)
            if not valid.any():
                per_foot.append((np.full(3, np.nan, np.float32), np.full(3, np.nan, np.float32), np.nan, False))
                continue
            # The contact surface is the foot vertex closest to the scene boundary,
            # regardless of whether it is a small positive gap or a penetration.
            local = np.nanargmin(np.abs(values))
            foot_point = candidates[local]
            distance = float(values[local])
            normal, normal_valid = sdf.normal(foot_point[None])
            usable = bool(normal_valid[0])
            surface_point = foot_point - distance * normal[0] if usable else np.full(3, np.nan, np.float32)
            per_foot.append((surface_point.astype(np.float32), normal[0].astype(np.float32), distance, usable))
        frame_name = path.parent.name
        # SMPL-X joints 10/11 are left/right foot.  Use these fixed anatomical
        # anchors for temporal velocity; the closest SDF vertex may change
        # identity between frames and is therefore unsuitable for motion labels.
        records.append((frame_name, frame_seconds(frame_name, index), per_foot, joints[[0, 7, 8, 10, 11]]))

    if not records:
        raise RuntimeError(f"No PROXD frames found under {result_root}")
    timestamps = np.asarray([record[1] for record in records], dtype=np.float32)
    points = np.asarray([[foot[0] for foot in record[2]] for record in records], dtype=np.float32)
    normals = np.asarray([[foot[1] for foot in record[2]] for record in records], dtype=np.float32)
    distances = np.asarray([[foot[2] for foot in record[2]] for record in records], dtype=np.float32)
    usable = np.asarray([[foot[3] for foot in record[2]] for record in records], dtype=bool)
    selected_joints = np.asarray([record[3] for record in records], dtype=np.float32)
    roots = selected_joints[:, 0]
    ankles = selected_joints[:, 1:3]
    foot_anchors = selected_joints[:, 3:5]
    forward = foot_anchors - ankles
    forward_norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    forward = np.divide(forward, forward_norm, out=np.zeros_like(forward), where=forward_norm > 1e-6)
    forward[forward_norm[..., 0] <= 1e-6] = np.array([0.0, 0.0, 1.0], np.float32)
    anchor_orientation = np.concatenate([forward, normals], axis=-1).astype(np.float32)
    velocity = np.zeros((len(records), 2), dtype=np.float32)
    if len(records) > 1:
        delta_time = np.maximum(np.diff(timestamps), 1e-4)
        displacement = np.linalg.norm(np.diff(foot_anchors, axis=0), axis=-1)
        velocity[1:] = displacement / delta_time[:, None]
        velocity[0] = velocity[1]
    near_surface = np.abs(distances) <= args.contact_distance_m
    contact = (near_surface & (velocity <= args.foot_velocity_mps) & usable).astype(np.float32)
    penetration = ((distances < -args.penetration_depth_m) & usable).astype(np.float32)
    uncertainty = (~usable).astype(np.float32)
    labels = {
        "contact": contact,
        "contact_point": points,
        "surface_distance": distances,
        "surface_normal": normals,
        "penetration_risk": penetration,
        "uncertainty": uncertainty,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **labels)
    teacher_path = args.output.with_suffix(".teacher.npz")
    np.savez_compressed(
        teacher_path,
        timestamps=timestamps,
        root=roots,
        anchor_position=foot_anchors,
        anchor_orientation=anchor_orientation,
        surface_point=points,
        surface_normal=normals,
        surface_distance=distances,
        contact=contact,
        penetration_risk=penetration,
        uncertainty=uncertainty,
    )
    report = {
        "dataset": "PROX",
        "label_source": "PROX_GT_SDF_from_PROXD",
        "sequence": args.sequence,
        "scene": scene,
        "frames": len(records),
        "stride": args.stride,
        "camera_transform": str(camera_path),
        "sdf": str((prox_root / "sdf" / f"{scene}_sdf.npy")),
        "coordinate_frame": "PROX_scene_world",
        "foot_regions": {"left": "L_Leg.json", "right": "R_Leg.json"},
        "thresholds_m": {
            "contact_distance": args.contact_distance_m,
            "foot_velocity": args.foot_velocity_mps,
            "penetration_depth": args.penetration_depth_m,
        },
        "valid_foot_fraction": float(usable.mean()),
        "contact_rate": [float(contact[:, foot].mean()) for foot in range(2)],
        "penetration_rate": [float(penetration[:, foot].mean()) for foot in range(2)],
        "distance_abs_median_m": [float(np.nanmedian(np.abs(distances[:, foot]))) for foot in range(2)],
        "distance_signed_median_m": [float(np.nanmedian(distances[:, foot])) for foot in range(2)],
        "teacher_geometry": str(teacher_path),
        "note": "Only geometry-supervised targets are exported. Pose/root residual targets require a separate corrupted or front-end prediction experiment.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
