#!/usr/bin/env python3
"""Convert Human3R FrameInput into aligned geometry and learning tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contact_streaming.alignment import human_cloud_centroid, robust_sim3, translation_alignment
from contact_streaming.anchors import SMPLXAnchorExtractor
from contact_streaming.features import build_features, finite_difference
from contact_streaming.io import frame_paths, load_frame, read_json, save_npz, write_json
from contact_streaming.rules import ContactStateMachine, RuleConfig, geometric_correction
from contact_streaming.surface_proxy import estimate_feet_surfaces, scene_points_from_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Human3R FrameInput directory")
    parser.add_argument("--output", type=Path, required=True, help="Output sequence .npz")
    parser.add_argument("--smplx-model-dir", type=Path, default=ROOT.parent / "Human3R" / "src" / "models")
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--correspondences", type=Path, help="Optional npz containing src/dst [N,3]")
    parser.add_argument("--up-axis", type=int, default=1)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--surface-radius", type=float, default=0.35)
    parser.add_argument("--surface-k", type=int, default=256)
    parser.add_argument("--min-point-confidence", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = frame_paths(args.input)
    if not paths:
        raise RuntimeError(f"No FrameInput frames under {args.input}")
    manifest = read_json(args.input / "manifest.json")
    timestamps = np.asarray(
        [record.get("timestamp_sec") if record.get("timestamp_sec") is not None else index for index, record in enumerate(manifest["frames"])],
        dtype=np.float32,
    )[: len(paths)]
    extractor = SMPLXAnchorExtractor(args.smplx_model_dir, args.device)
    raw_frames, body = [], []
    for path in paths:
        frame = load_frame(path)
        person = args.person_index
        if person >= len(frame["smpl_shape"]):
            raise IndexError(f"person-index {person} unavailable in {path}")
        body_frame = extractor(
            frame["smpl_shape"][person], frame["smpl_rotvec"][person], frame["smpl_transl"][person],
            frame.get("smpl_expression", None)[person] if "smpl_expression" in frame else None,
        )
        raw_frames.append(frame)
        body.append(body_frame)

    if args.correspondences:
        with np.load(args.correspondences, allow_pickle=False) as corr:
            alignment = robust_sim3(corr["src"], corr["dst"])
    else:
        first = raw_frames[0]
        target = human_cloud_centroid(first["point_map"], first["smpl_mask"], first["confidence"])
        alignment = translation_alignment(body[0]["root"], target)

    anchors, orientations, roots = [], [], []
    for item in body:
        anchors.append(alignment.transform(item["anchor_position"]))
        roots.append(alignment.transform(item["root"]))
        orientation = item["anchor_orientation"].reshape(2, 2, 3) @ alignment.rotation.T
        orientations.append(orientation.reshape(2, 6))
    anchors = np.stack(anchors).astype(np.float32)
    orientations = np.stack(orientations).astype(np.float32)
    roots = np.stack(roots).astype(np.float32)
    # surface_radius is specified in the native Human3R metric frame.  When a
    # Sim(3) maps it into a reference scene (NeuMan/COLMAP can use a different
    # world scale), its radius must be scaled with the same factor.  Otherwise
    # anchors and points share a frame but the local-neighbourhood query does
    # not share units, yielding near-zero proxy confidence.
    reference_surface_radius = float(args.surface_radius * alignment.scale)

    surface_records = []
    for frame, frame_anchors in zip(raw_frames, anchors):
        points, weights = scene_points_from_frame(
            frame["point_map"], frame["confidence"], frame.get("smpl_mask"),
            args.min_point_confidence, args.point_stride,
        )
        # FrameInput point maps are in the native Human3R frame.  Anchors above
        # have already been mapped into the reference scene frame, therefore the
        # local proxy must be mapped by the identical Sim(3) before querying it.
        # Mixing the two frames silently produces plausible-looking but invalid
        # foot-to-surface distances.
        points = alignment.transform(points)
        surface_records.append(
            estimate_feet_surfaces(
                frame_anchors, points, weights, k=args.surface_k, radius=reference_surface_radius, up_axis=args.up_axis
            )
        )
    surface = {key: np.stack([record[key] for record in surface_records]) for key in surface_records[0]}
    alignment_residual = np.full(len(paths), alignment.residual if np.isfinite(alignment.residual) else 0.0, np.float32)
    alignment_confidence = np.full(len(paths), alignment.confidence, np.float32)
    features, aux = build_features(
        anchors, orientations, surface["surface_point"], surface["surface_normal"], surface["surface_distance"],
        surface["surface_confidence"], roots, alignment_residual, alignment_confidence, timestamps,
    )

    machine = ContactStateMachine(RuleConfig())
    speed = np.linalg.norm(aux["anchor_velocity"], axis=-1)
    contact, contact_probability, root_residual, pose_residual, modes = [], [], [], [], []
    for index in range(len(paths)):
        state, probability, mode = machine.update(
            surface["surface_distance"][index], speed[index], surface["surface_confidence"][index], alignment_confidence[index]
        )
        root_delta, pose_delta, _ = geometric_correction(
            state, surface["surface_distance"][index], surface["surface_normal"][index],
            surface["surface_confidence"][index] * alignment_confidence[index],
        )
        contact.append(state.astype(np.float32)); contact_probability.append(probability)
        root_residual.append(root_delta); pose_residual.append(pose_delta); modes.append(mode)

    arrays = {
        "features": features,
        "timestamps": timestamps,
        "anchor_position": anchors,
        "anchor_velocity": aux["anchor_velocity"],
        "anchor_orientation": orientations,
        "root": roots,
        "surface_point": surface["surface_point"].astype(np.float32),
        "surface_normal": surface["surface_normal"].astype(np.float32),
        "surface_distance": surface["surface_distance"].astype(np.float32),
        "surface_confidence": surface["surface_confidence"].astype(np.float32),
        "surface_support": surface["surface_support"].astype(np.int32),
        "alignment_residual": alignment_residual,
        "alignment_confidence": alignment_confidence,
        "contact": np.stack(contact),
        "contact_probability_rule": np.stack(contact_probability),
        "contact_point": surface["surface_point"].astype(np.float32),
        "pose_residual": np.stack(pose_residual),
        "root_residual": np.stack(root_residual),
        "penetration_risk": (surface["surface_distance"] < 0).astype(np.float32),
        "uncertainty": 1.0 - np.clip(surface["surface_confidence"] * alignment_confidence[:, None], 0, 1),
    }
    save_npz(args.output, arrays)
    metadata = {
        "source": str(args.input.resolve()), "num_frames": len(paths), "person_index": args.person_index,
        "feature_dim": int(features.shape[1]), "alignment": {
            "method": alignment.method, "scale": alignment.scale, "rotation": alignment.rotation.tolist(),
            "translation": alignment.translation.tolist(), "residual": alignment.residual if np.isfinite(alignment.residual) else None,
            "confidence": alignment.confidence,
        },
        "geometry_frame": "reference_scene" if args.correspondences else "human3r_native_translation_fallback",
        "proxy_points_transformed_with_alignment": True,
        "surface_radius_native": args.surface_radius,
        "surface_radius_reference": reference_surface_radius,
        "state_modes": {mode: modes.count(mode) for mode in sorted(set(modes))},
        "label_source": "geometry_rule", "label_confidence": "surface_confidence * alignment_confidence",
    }
    write_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
