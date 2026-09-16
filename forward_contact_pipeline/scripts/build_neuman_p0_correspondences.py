#!/usr/bin/env python3
"""Build and independently validate Human3R -> NeuMan Sim(3) correspondences.

This is a P0 calibration utility, not a contact-label generator.  It uses the
same-frame, same-topology SMPL body vertices from a Human3R-converted sequence
and NeuMan's scene-aligned 4D-Humans result.  Frames are split before fitting:
the output NPZ contains only training correspondences for ``prepare_sequence``;
the JSON report evaluates the fitted transform on held-out frames.

Both input files must describe the identical ordered NeuMan frames and use the
HUGS SMPL parameter convention (global_orient, body_pose, betas, scale, transl).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contact_streaming.alignment import robust_sim3


def load_smpl_class():
    """Load NeuMan's standalone SMPL implementation without importing hugs.datasets.

    Importing the package executes unrelated dataset registrations (and their
    optional logging dependencies), which makes this calibration-only tool fail
    in otherwise valid lightweight environments.
    """
    module_path = PROJECT_ROOT / "hugs" / "datasets" / "neuman_utils" / "smpl.py"
    spec = importlib.util.spec_from_file_location("neuman_p0_smpl", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SMPL module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SMPL


REQUIRED_KEYS = ("global_orient", "body_pose", "betas", "scale", "transl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-smpl", type=Path, required=True,
                        help="Human3R-converted HUGS SMPL parameters.")
    parser.add_argument("--target-smpl", type=Path, required=True,
                        help="NeuMan scene-aligned SMPL parameters.")
    parser.add_argument("--smpl-model-dir", type=Path, default=PROJECT_ROOT / "data" / "smpl")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output correspondence NPZ; JSON report is written beside it.")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--vertex-sample", type=int, default=512)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_params(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} misses required keys: {missing}")
        return {key: data[key] for key in REQUIRED_KEYS}


def world_vertices(model, params: dict[str, np.ndarray], frame_ids: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for frame_id in frame_ids:
        pose = np.concatenate([params["global_orient"][frame_id], params["body_pose"][frame_id]], axis=0)[None].astype(np.float32)
        betas = params["betas"][frame_id][None].astype(np.float32)
        if betas.shape[1] < model.shapedirs.shape[-1]:
            betas = np.pad(betas, ((0, 0), (0, model.shapedirs.shape[-1] - betas.shape[1])))
        vertices = model(poses=pose, betas=betas, return_tensor=False)[vertex_ids].astype(np.float64)
        chunks.append(vertices * float(params["scale"][frame_id]) + params["transl"][frame_id].astype(np.float64))
    return np.concatenate(chunks, axis=0)


def error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(errors.max()),
    }


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1 or args.vertex_sample < 3:
        raise ValueError("frame-stride must be >= 1 and vertex-sample must be >= 3")
    if not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("holdout-fraction must be in (0, 0.5)")

    source, target = load_params(args.source_smpl), load_params(args.target_smpl)
    frames = min(len(source["global_orient"]), len(target["global_orient"]))
    if args.max_frames > 0:
        frames = min(frames, args.max_frames)
    frame_ids = np.arange(0, frames, args.frame_stride, dtype=np.int64)
    if len(frame_ids) < 4:
        raise ValueError("Need at least four sampled matching frames for fit/hold-out validation")

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(frame_ids)
    holdout_count = max(1, int(round(len(frame_ids) * args.holdout_fraction)))
    holdout_frames = np.sort(shuffled[:holdout_count])
    train_frames = np.sort(shuffled[holdout_count:])
    if len(train_frames) < 3:
        raise ValueError("Too few training frames after hold-out split")
    vertex_ids = np.sort(rng.choice(6890, size=min(args.vertex_sample, 6890), replace=False)).astype(np.int64)

    model = load_smpl_class()(args.smpl_model_dir, gender="neutral", device="cpu")
    train_src = world_vertices(model, source, train_frames, vertex_ids)
    train_dst = world_vertices(model, target, train_frames, vertex_ids)
    holdout_src = world_vertices(model, source, holdout_frames, vertex_ids)
    holdout_dst = world_vertices(model, target, holdout_frames, vertex_ids)

    alignment = robust_sim3(train_src, train_dst)
    train_error = np.linalg.norm(alignment.transform(train_src) - train_dst, axis=1)
    holdout_error = np.linalg.norm(alignment.transform(holdout_src) - holdout_dst, axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        src=train_src.astype(np.float32), dst=train_dst.astype(np.float32),
        train_frame_ids=train_frames, holdout_frame_ids=holdout_frames,
        vertex_ids=vertex_ids,
    )
    report = {
        "purpose": "P0 coordinate calibration only; this file does not provide contact ground truth.",
        "source_smpl": str(args.source_smpl.resolve()),
        "target_smpl": str(args.target_smpl.resolve()),
        "frames_available": int(frames),
        "train_frame_ids": train_frames.tolist(),
        "holdout_frame_ids": holdout_frames.tolist(),
        "vertex_sample": int(len(vertex_ids)),
        "training_correspondences": int(len(train_src)),
        "sim3": {
            "scale": alignment.scale,
            "rotation": alignment.rotation.tolist(),
            "translation": alignment.translation.tolist(),
            "training_residual_median": alignment.residual,
            "training_confidence": alignment.confidence,
        },
        "training_vertex_error": error_summary(train_error),
        "holdout_vertex_error": error_summary(holdout_error),
        "validation": {
            "independent": True,
            "method": "held-out frames; same SMPL topology and frame ordering",
            "next_checks": ["SMPL re-projection against NeuMan images/masks", "aligned Human3R point-map versus NeuMan scene diagnostics"],
        },
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"CORRESPONDENCES={args.out}")
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
