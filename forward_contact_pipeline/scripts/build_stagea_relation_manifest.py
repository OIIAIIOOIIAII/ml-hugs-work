#!/usr/bin/env python3
"""Build a contact-local Stage-A relation dataset from PROX geometry teachers.

This builder is deliberately a *data-contract* step, not a claim that the
current Human3R/GUSH3R front end is aligned well enough for training.  It
creates, for every PROXD frame and foot, (a) SMPL-X foot ROI vertices, (b)
local scene-mesh patches, and (c) vertex/ROI contact and proximity labels in a
normal/tangent contact-local frame.  RGB paths are retained as references for
the later frozen-backbone token extraction stage; this script never reads RGB
pixels and therefore cannot silently train a new image encoder.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prox_geometry_labels import SDFGrid, fitted_body, load_segment_indices, make_smplx, scene_name, sequence_pickles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox-root", type=Path, default=ROOT.parent / "PROX")
    parser.add_argument("--teacher", type=Path, required=True, help="PROX *.teacher.npz")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, default=ROOT.parent / "Human3R" / "src" / "models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--roi-vertices", type=int, default=64)
    parser.add_argument("--scene-points", type=int, default=128)
    parser.add_argument("--patch-radius-m", type=float, default=0.35)
    parser.add_argument("--contact-distance-m", type=float, default=0.02)
    parser.add_argument(
        "--roi-selection",
        choices=("near_anchor", "full_leg"),
        default="near_anchor",
        help="Fixed SMPL-X identities selected near the contact anchor (training) or across full leg (representation audit).",
    )
    return parser.parse_args()


def make_frame(normal: np.ndarray) -> np.ndarray:
    """Columns are a deterministic tangent-1, tangent-2, outward normal."""
    normal = np.asarray(normal, dtype=np.float32)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Invalid surface normal")
    normal = normal / norm
    axis = np.array([0.0, 0.0, 1.0], np.float32)
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], np.float32)
    tangent1 = np.cross(axis, normal)
    tangent1 /= max(float(np.linalg.norm(tangent1)), 1e-6)
    tangent2 = np.cross(normal, tangent1)
    return np.stack([tangent1, tangent2, normal], axis=1).astype(np.float32)


def exact_count(indices: np.ndarray, count: int) -> np.ndarray:
    """Deterministic repeat-pad used only when a local patch is sparse."""
    if len(indices) == 0:
        raise ValueError("No candidate points")
    if len(indices) >= count:
        return indices[:count]
    return np.resize(indices, count)


def farthest_subset(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic spatial coverage indices, computed once in frame zero."""
    chosen = [0]
    nearest = np.sum((points - points[0]) ** 2, axis=1)
    for _ in range(1, min(count, len(points))):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((points - points[index]) ** 2, axis=1))
    return exact_count(np.asarray(chosen, dtype=np.int64), count)


def main() -> None:
    args = parse_args()
    if args.roi_vertices <= 0 or args.scene_points <= 0:
        raise ValueError("--roi-vertices and --scene-points must be positive")
    prox_root = args.prox_root.resolve()
    scene = scene_name(args.sequence)
    with np.load(args.teacher, allow_pickle=False) as loaded:
        teacher = {key: loaded[key].copy() for key in loaded.files}
    required = {"timestamps", "surface_point", "surface_normal", "contact", "uncertainty"}
    missing = required - set(teacher)
    if missing:
        raise KeyError(f"Teacher missing {sorted(missing)}")
    frames = len(teacher["timestamps"])
    if teacher["surface_point"].shape != (frames, 2, 3):
        raise ValueError("Teacher surface point shape is incompatible")

    try:
        import trimesh
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("Requires trimesh and scipy in the selected environment") from exc
    mesh = trimesh.load(prox_root / "scenes" / f"{scene}.ply", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh for {scene}")
    scene_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    scene_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if len(scene_vertices) == 0 or len(scene_normals) != len(scene_vertices):
        raise ValueError("Scene mesh has invalid vertices/normals")
    # Some PROX PLY files expose area-weighted normals (length ~= 2*pi) rather
    # than unit normals.  Stage A consumes directional evidence, so normalize
    # explicitly and reject zero/invalid entries instead of leaking mesh scale
    # into its reliability head.
    scene_normal_length = np.linalg.norm(scene_normals, axis=1)
    # A few scanned meshes contain isolated degenerate vertices.  They carry
    # no directional evidence, so remove them from the *scene-patch candidate*
    # set instead of inventing a normal or failing an otherwise valid scene.
    valid_scene_normal = np.isfinite(scene_normal_length) & (scene_normal_length > 1e-6)
    discarded_scene_vertices = int((~valid_scene_normal).sum())
    scene_vertices = scene_vertices[valid_scene_normal]
    scene_normals = scene_normals[valid_scene_normal] / scene_normal_length[valid_scene_normal, None]
    if len(scene_vertices) == 0:
        raise ValueError("Scene mesh has no valid directional vertices")
    scene_tree = cKDTree(scene_vertices)

    camera_to_world = np.asarray(json.loads((prox_root / "cam2world" / f"{scene}.json").read_text(encoding="utf-8")), dtype=np.float32)
    model = make_smplx(args.smplx_model_dir, args.device)
    segment_indices = [
        load_segment_indices(prox_root / "body_segments" / "L_Leg.json"),
        load_segment_indices(prox_root / "body_segments" / "R_Leg.json"),
    ]
    pickles = list(sequence_pickles(prox_root / "PROXD" / args.sequence / "results", stride=1, max_frames=frames))
    if len(pickles) != frames:
        raise RuntimeError(f"Teacher has {frames} frames but found {len(pickles)} PROXD frames")

    roi_local = np.zeros((frames, 2, args.roi_vertices, 3), np.float32)
    roi_normal_local = np.zeros_like(roi_local)
    roi_contact = np.zeros((frames, 2, args.roi_vertices), np.float32)
    roi_proximity = np.zeros((frames, 2, args.roi_vertices), np.float32)
    scene_local = np.zeros((frames, 2, args.roi_vertices, args.scene_points, 3), np.float32)
    scene_normal_local = np.zeros_like(scene_local)
    scene_distance = np.zeros((frames, 2, args.roi_vertices, args.scene_points), np.float32)
    rgb_paths: list[str] = []
    sparse_patch_count = 0
    roi_selection: list[np.ndarray | None] = [None, None]

    sdf = SDFGrid(prox_root / "sdf" / f"{scene}_sdf.npy", prox_root / "sdf" / f"{scene}.json")
    for frame, path in enumerate(pickles):
        with path.open("rb") as handle:
            parameters = pickle.load(handle, encoding="latin1")
        vertices, _ = fitted_body(model, parameters, camera_to_world, args.device)
        frame_name = path.parent.name
        rgb = prox_root / "recordings" / args.sequence / "Color" / f"{frame_name}.jpg"
        rgb_paths.append(str(rgb.relative_to(prox_root)) if rgb.is_file() else "")
        for foot, indices in enumerate(segment_indices):
            surface = teacher["surface_point"][frame, foot]
            normal = teacher["surface_normal"][frame, foot]
            if teacher["uncertainty"][frame, foot] > 0.5 or not np.isfinite(surface).all() or not np.isfinite(normal).all():
                raise RuntimeError(f"{args.sequence} frame {frame} foot {foot}: invalid teacher geometry")
            local_frame = make_frame(normal)
            candidates = vertices[indices]
            # Do not choose the currently closest SDF vertices: that made all
            # per-vertex labels equal to the ROI label.  Select a fixed,
            # spatially spread foot/ankle subset in frame zero, then retain
            # those SMPL-X vertex identities for all frames.
            if roi_selection[foot] is None:
                if args.roi_selection == "near_anchor":
                    anchor = teacher["surface_point"][frame, foot]
                    near = np.argsort(np.sum((candidates - anchor) ** 2, axis=1), kind="stable")[: min(512, len(candidates))]
                    roi_selection[foot] = near[farthest_subset(candidates[near], args.roi_vertices)]
                else:
                    roi_selection[foot] = farthest_subset(candidates, args.roi_vertices)
            chosen = candidates[roi_selection[foot]]
            vertex_sdf, vertex_usable = sdf.sample(chosen)
            if not vertex_usable.all():
                raise RuntimeError("Chosen ROI vertex lies outside the scene SDF")
            vertex_normals, normal_ok = sdf.normal(chosen)
            if not normal_ok.all():
                raise RuntimeError("ROI vertex normal query failed")
            roi_local[frame, foot] = (chosen - surface) @ local_frame
            roi_normal_local[frame, foot] = vertex_normals @ local_frame
            roi_proximity[frame, foot] = vertex_sdf
            roi_contact[frame, foot] = ((np.abs(vertex_sdf) <= args.contact_distance_m) & (teacher["contact"][frame, foot] > 0.5)).astype(np.float32)
            for vertex_id, vertex in enumerate(chosen):
                nearby = scene_tree.query_ball_point(vertex, r=args.patch_radius_m)
                if not nearby:
                    _, nearest = scene_tree.query(vertex, k=1)
                    nearby = [int(nearest)]
                nearby = np.asarray(nearby, dtype=np.int64)
                order = np.argsort(np.sum((scene_vertices[nearby] - vertex) ** 2, axis=1), kind="stable")
                selected = nearby[exact_count(order, args.scene_points)]
                if len(nearby) < args.scene_points:
                    sparse_patch_count += 1
                points = scene_vertices[selected]
                scene_local[frame, foot, vertex_id] = (points - vertex) @ local_frame
                scene_normal_local[frame, foot, vertex_id] = scene_normals[selected] @ local_frame
                scene_distance[frame, foot, vertex_id] = np.linalg.norm(points - vertex, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload_path = args.output_dir / f"{args.sequence}.npz"
    np.savez_compressed(
        payload_path,
        roi_vertex_local=roi_local,
        roi_vertex_normal_local=roi_normal_local,
        vertex_contact=roi_contact,
        vertex_proximity=roi_proximity,
        scene_point_local=scene_local,
        scene_normal_local=scene_normal_local,
        scene_point_distance=scene_distance,
        teacher_contact=teacher["contact"].astype(np.float32),
        timestamps=teacher["timestamps"].astype(np.float32),
    )
    manifest = {
        "schema": "hugs.forward_contact.stage_a_relation.v1",
        "dataset": "PROX",
        "sequence": args.sequence,
        "scene": scene,
        "frames": frames,
        "payload": payload_path.name,
        "coordinate_frame": "PROX_scene_world -> per-foot contact-local tangent-normal frame",
        "rgb_reference_root": str(prox_root),
        "rgb_paths": rgb_paths,
        "rgb_tokens": "not_extracted: frozen backbone token extraction is a separate, auditable stage",
        "label_source": "PROXD SMPL-X + PROX official SDF/scene mesh",
        "input_is_real_frontend": False,
        "error_family": "clean_oracle_geometry",
        "roi_vertices": args.roi_vertices,
        "scene_points_per_vertex": args.scene_points,
        "patch_radius_m": args.patch_radius_m,
        "contact_distance_m": args.contact_distance_m,
        "sparse_patch_repeat_pad_count": sparse_patch_count,
        "roi_selection": (
            "fixed frame-zero full-leg spatial-FPS SMPL-X vertex identities"
            if args.roi_selection == "full_leg"
            else "fixed frame-zero foot/ankle spatial-FPS SMPL-X vertex identities; never SDF-reranked per frame"
        ),
        "discarded_invalid_scene_normal_vertices": discarded_scene_vertices,
        "note": "Data-contract/oracle-label asset only; not a Human3R/GUSH3R result or training claim.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("sequence", "scene", "frames", "roi_vertices", "scene_points_per_vertex", "sparse_patch_repeat_pad_count", "payload")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
