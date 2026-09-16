#!/usr/bin/env python3
"""Run Human3R and export a versioned per-frame FrameInput dataset.

This deliberately reuses Human3R's demo preparation/post-processing code, but
does not start the interactive viewer.  The exported ``frames/*.npz`` files
contain model outputs needed by the contact pipeline; ``manifest.json`` keeps
paths, timing, source/checkpoint metadata and coordinate conventions together.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human3r-root", type=Path, default=Path("../Human3R"))
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--seq-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--reset-interval", type=int, default=10_000_000)
    p.add_argument("--use-ttt3r", action="store_true")
    return p.parse_args()


def _as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _video_metadata(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": None, "frame_count": None, "source_frame_indices": None}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count or None,
        "source_frame_indices": "video_frame_index = output_index * subsample",
    }


def _make_pointcloud_video(outdir: Path, pts3ds_other, fps: float | None):
    """Save an image-space point-map/depth visualization video."""
    raw_dir = outdir / "human3r_raw"
    video_path = outdir / "pointcloud_depth.mp4"
    first = cv2.imread(str(raw_dir / "color" / "000000.png"))
    if first is None:
        raise RuntimeError("Cannot create point-cloud video: first color frame missing")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps or 24.0), (w * 2, h)
    )
    for i, point_map in enumerate(pts3ds_other):
        name = f"{i:06d}"
        rgb = cv2.imread(str(raw_dir / "color" / f"{name}.png"))
        conf = np.load(raw_dir / "conf" / f"{name}.npy")
        pts = _as_numpy(point_map)
        if pts.ndim == 4 and pts.shape[0] == 1:
            pts = pts[0]
        depth = pts[..., 2].astype(np.float32)
        valid = np.isfinite(depth) & np.isfinite(pts).all(axis=-1) & (conf > 0)
        if valid.any():
            lo, hi = np.percentile(depth[valid], [2, 98])
            norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            depth_u8 = ((1.0 - norm) * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            depth_vis[~valid] = 0
        else:
            depth_vis = np.zeros_like(rgb)
        cv2.putText(rgb, "RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(depth_vis, "Human3R point-map depth", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        writer.write(np.concatenate([rgb, depth_vis], axis=1))
    writer.release()
    return video_path


def main():
    args = parse_args()
    repo = args.human3r_root.resolve()
    if args.model_path is None:
        model_path = repo / "src" / "human3r_896L.pth"
    else:
        model_path = args.model_path.resolve()
    seq_path = args.seq_path.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)

    # Import after adding Human3R to sys.path, matching its demo.py layout.
    import demo  # type: ignore
    from add_ckpt_path import add_path_to_dust3r

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", flush=True)
        device = "cpu"

    add_path_to_dust3r(str(model_path))
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    img_paths, tmpdir = demo.parse_seq_path(str(seq_path))
    if args.max_frames is not None:
        img_paths = img_paths[: args.max_frames]
    img_paths = img_paths[:: args.subsample]
    if not img_paths:
        raise RuntimeError(f"No input frames found: {seq_path}")

    print(f"Loading Human3R from {model_path}", flush=True)
    model = ARCroco3DStereo.from_pretrained(str(model_path)).to(device)
    model.eval()
    img_res = getattr(model, "mhmr_img_res", None)
    views = demo.prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    if tmpdir is not None:
        shutil.rmtree(tmpdir)

    print(f"Running Human3R on {len(img_paths)} frames", flush=True)
    start = time.time()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            views, model, device, use_ttt3r=args.use_ttt3r
        )
    inference_seconds = time.time() - start

    # Human3R's post-processing writes the established depth/conf/camera/smpl
    # files and returns world-transformed point maps for each output frame.
    raw_dir = outdir / "human3r_raw"
    (raw_dir).mkdir(parents=True, exist_ok=True)
    pts3d_other, _, _, _, _, _, _, _ = demo.prepare_output(
        outputs, str(raw_dir), 1, True, True, True, True, img_res, args.subsample
    )

    frame_dir = outdir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    fps = None
    source_meta = _video_metadata(seq_path) if seq_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else {}
    fps = source_meta.get("fps")
    frame_records = []

    for output_idx, point_map in enumerate(pts3d_other):
        frame_name = f"{output_idx:06d}"
        depth = np.load(raw_dir / "depth" / f"{frame_name}.npy")
        confidence = np.load(raw_dir / "conf" / f"{frame_name}.npy")
        camera = np.load(raw_dir / "camera" / f"{frame_name}.npz")
        smpl = np.load(raw_dir / "smpl" / f"{frame_name}.npz", allow_pickle=True)
        color = cv2.cvtColor(
            cv2.imread(str(raw_dir / "color" / f"{frame_name}.png")), cv2.COLOR_BGR2RGB
        )
        point_map_np = _as_numpy(point_map)
        if point_map_np.ndim == 4 and point_map_np.shape[0] == 1:
            point_map_np = point_map_np[0]

        frame_path = frame_dir / f"{frame_name}.npz"
        arrays = {
            "color_rgb": color,
            "depth": depth,
            "confidence": confidence,
            "point_map": point_map_np,
            "camera_pose_c2w": camera["pose"],
            "intrinsics": camera["intrinsics"],
            "smpl_scores": smpl["scores"],
            "smpl_mask": smpl["msk"],
            "smpl_shape": smpl["shape"],
            "smpl_rotvec": smpl["rotvec"],
            "smpl_transl": smpl["transl"],
            "smpl_expression": smpl["expression"],
        }
        np.savez_compressed(frame_path, **arrays)

        source_frame_index = output_idx * args.subsample
        frame_records.append(
            {
                "frame_id": output_idx,
                "source_frame_index": source_frame_index,
                "timestamp_sec": (source_frame_index / fps) if fps else None,
                "file": str(frame_path.relative_to(outdir)),
                "image_file": str((raw_dir / "color" / f"{frame_name}.png").relative_to(outdir)),
                "num_humans": int(arrays["smpl_shape"].shape[0]),
                "image_size_hw": [int(color.shape[0]), int(color.shape[1])],
            }
        )

    manifest = {
        "schema": "hugs.forward_contact.FrameInput",
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "seq_path": str(seq_path),
            "source_type": "video" if seq_path.is_file() else "image_directory",
            **source_meta,
        },
        "model": {
            "name": "Human3R",
            "checkpoint": str(model_path),
            "checkpoint_size_bytes": model_path.stat().st_size,
            "input_size": args.size,
            "device": device,
            "use_ttt3r": args.use_ttt3r,
        },
        "processing": {
            "num_frames": len(frame_records),
            "subsample": args.subsample,
            "reset_interval": args.reset_interval,
            "inference_seconds": inference_seconds,
            "average_seconds_per_frame": inference_seconds / len(frame_records),
        },
        "coordinate_convention": {
            "camera_pose": "Human3R demo post-processing c2w pose",
            "point_map": "Human3R world-transformed point map",
            "smpl_translation": "Human3R output convention; Sim(3) alignment not applied",
            "alignment_status": "raw_unaligned",
        },
        "fields_per_frame": {
            "color_rgb": "H,W,3 uint8",
            "depth": "H,W float",
            "confidence": "H,W float",
            "point_map": "H,W,3 float",
            "camera_pose_c2w": "4,4 float",
            "intrinsics": "3,3 float",
            "smpl_shape": "N,betas float",
            "smpl_rotvec": "N,53,3 float",
            "smpl_transl": "N,3 float",
            "smpl_expression": "N,E float or object/None",
        },
        "frames": frame_records,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pointcloud_video = _make_pointcloud_video(outdir, pts3d_other, fps)
    print(f"Saved point-map visualization: {pointcloud_video}", flush=True)
    mesh_video = raw_dir / "output_video.mp4"
    if mesh_video.exists():
        shutil.copy2(mesh_video, outdir / "mesh_overlay.mp4")
        print(f"Saved mesh visualization: {outdir / 'mesh_overlay.mp4'}", flush=True)
    print(f"Saved FrameInput dataset: {outdir}", flush=True)
    print(f"Frames: {len(frame_records)}, inference: {inference_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
