"""Minimal human-first progressive scene streaming experiment.

The receiver first displays the human representation, then adds scene GS
chunks ordered by distance to the median SMPL translation.  This is an
engineering baseline, intentionally without contact-aware scheduling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from codec import decode_frame, encode_frame  # noqa: E402
from baseline_benchmark import transfer_seconds  # noqa: E402
from eval_method5_psnr import load_models  # noqa: E402


CKPT = ROOT / "output/human_scene/neuman/bike_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_12-52-08/ckpt"
DELTA = ROOT / "output/delta_mu_analysis/bike/delta_mu_f16.npy"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bandwidth-mbps", type=float, default=1000.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--scene-chunks", type=int, default=10)
    p.add_argument("--scene-fraction", type=float, default=0.2,
                   help="fraction of post-init link reserved for scene chunks")
    p.add_argument("--human-init-bytes", type=int, default=16_060_000,
                   help="H-canonical + H-lbs + H-smpl + first correction")
    p.add_argument("--scene-bytes", type=int, default=486_000_000)
    p.add_argument("--rtt-ms", type=float, default=20.0)
    p.add_argument("--out-dir", type=Path, default=Path("transmission_system/results/progressive_center"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def image_from_render(render: torch.Tensor) -> np.ndarray:
    image = render.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (image[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8)


def subset_scene(scene: dict, indices: np.ndarray, n_points: int) -> dict:
    out = {}
    index = torch.from_numpy(indices).long()
    for key, value in scene.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n_points:
            out[key] = value[index.to(value.device)]
        else:
            out[key] = value
    return out


@torch.no_grad()
def main() -> None:
    a = args()
    device = torch.device(a.device)
    out_dir = a.out_dir / f"bike_{int(a.bandwidth_mbps)}mbps_{a.scene_chunks}chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = np.load(DELTA).astype(np.float32)
    dm = savgol_filter(dm, 11, 2, axis=0).astype(np.float32)
    payloads, headers = [], []
    previous = None
    for frame in dm:
        header, payload, previous = encode_frame(frame, previous)
        payloads.append(payload)
        headers.append(header)

    # Separate logical streams: correction gets (1-fraction), scene gets
    # fraction. This keeps the baseline simple and reproducible.
    init_time = transfer_seconds(a.human_init_bytes, a.bandwidth_mbps, a.rtt_ms)
    corr_bw = a.bandwidth_mbps * (1.0 - a.scene_fraction)
    scene_bw = a.bandwidth_mbps * a.scene_fraction
    corr_arrivals = []
    t = init_time
    for payload in payloads:
        t += transfer_seconds(len(payload), corr_bw, a.rtt_ms)
        corr_arrivals.append(t)
    chunk_bytes = a.scene_bytes / a.scene_chunks
    scene_arrivals = []
    t = init_time
    for _ in range(a.scene_chunks):
        t += transfer_seconds(int(chunk_bytes), scene_bw, a.rtt_ms)
        scene_arrivals.append(t)

    print(f"human first-frame network time: {init_time:.3f}s", flush=True)
    print(f"scene chunk arrivals: {[round(x, 3) for x in scene_arrivals]}", flush=True)

    from hugs.datasets.neuman import NeumanDataset
    from hugs.renderer.gs_renderer import render_human_scene

    load_start = time.perf_counter()
    human_gs, scene_out, anchor_attn, cfg = load_models(CKPT, str(device))
    human_gs.eval()
    dataset = NeumanDataset(seq=cfg.dataset.seq, split="all")
    load_seconds = time.perf_counter() - load_start

    n_scene = int(scene_out["xyz"].shape[0])
    # Center loading baseline: order scene points by distance to the median
    # human translation, rather than the arbitrary scene bounding-box center.
    trajectory = dataset.smpl_params["transl"].numpy()
    center = np.median(trajectory, axis=0)
    xyz = scene_out["xyz"].detach().cpu().numpy()
    order = np.argsort(np.sum((xyz - center[None]) ** 2, axis=1))
    chunk_indices = [order[: int(np.ceil(n_scene * k / a.scene_chunks))] for k in range(1, a.scene_chunks + 1)]

    anchor_ids = human_gs.anchor_ids.to(device).long()
    anchor_weights = human_gs.anchor_weights.to(device)
    iteration = int(cfg.train.num_steps)
    bg_color = torch.zeros(3, dtype=torch.float32, device=device)
    first_arrival = init_time
    writer = None
    previous_decoded = None
    decoded_index = -1
    render_times = []
    chunk_counts = []
    frames_late = 0
    max_frames = len(dataset)
    content_duration = max(max_frames / a.fps, scene_arrivals[-1] - first_arrival)
    total_slots = int(np.ceil(content_duration * a.fps))
    tail_slots = max(0, total_slots - max_frames)
    cached_key = None
    cached_frame = None

    for output_index in range(total_slots):
        deadline = first_arrival + output_index / a.fps
        available_corr = [i for i, t in enumerate(corr_arrivals) if t <= deadline]
        latest_corr = max(available_corr) if available_corr else -1
        available_scene = sum(t <= deadline for t in scene_arrivals)
        current_frame = min(output_index, max_frames - 1)
        corr_index = min(latest_corr, current_frame) if latest_corr >= 0 else -1
        if output_index < max_frames and corr_index < current_frame:
            frames_late += 1
        if corr_index < 0:
            corr_index = 0
        if corr_index > decoded_index:
            for packet_index in range(decoded_index + 1, corr_index + 1):
                previous_decoded = decode_frame(headers[packet_index], payloads[packet_index], previous_decoded)
            decoded_index = corr_index

        state_key = (current_frame, available_scene, corr_index)
        if state_key == cached_key:
            writer.write(cached_frame)
            chunk_counts.append(available_scene)
            continue

        data = dataset[current_frame]
        data_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        human_out = human_gs.forward(
            global_orient=data_gpu["global_orient"], body_pose=data_gpu["body_pose"],
            betas=data_gpu["betas"], transl=data_gpu["transl"],
            smpl_scale=data_gpu["smpl_scale"][None], dataset_idx=-1,
            is_train=False, ext_tfs=None,
        )
        corrected, stats = anchor_attn(
            human_out, scene_out, anchor_ids, anchor_weights,
            iteration=iteration, frame_idx=output_index,
        )
        original = stats["delta_mu"]
        corrected["xyz"] = corrected["xyz"] + float(stats["gamma_mu_eff"]) * (
            torch.from_numpy(previous_decoded).to(device) - original
        )

        if available_scene:
            scene_current = subset_scene(scene_out, chunk_indices[available_scene - 1], n_scene)
            mode = "human_scene"
        else:
            scene_current = scene_out
            mode = "human"
        start = time.perf_counter()
        pkg = render_human_scene(
            data=data_gpu, human_gs_out=corrected, scene_gs_out=scene_current,
            bg_color=bg_color, render_mode=mode,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        render_times.append(time.perf_counter() - start)
        frame = image_from_render(pkg["render"])
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / "progressive_center.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
            blank = np.zeros_like(frame)
            for _ in range(int(round(first_arrival * a.fps))):
                writer.write(blank)
        writer.write(frame)
        cached_key = state_key
        cached_frame = frame
        chunk_counts.append(available_scene)
        if output_index % 10 == 0 or output_index >= max_frames:
            print(f"rendered {output_index + 1}/{total_slots}: scene_chunks={available_scene}, corr={corr_index}", flush=True)

    if writer:
        writer.release()
    result = {
        "bandwidth_mbps": a.bandwidth_mbps,
        "fps": a.fps,
        "human_init_bytes": a.human_init_bytes,
        "human_first_frame_seconds": init_time,
        "receiver_model_load_seconds": load_seconds,
        "scene_chunks": a.scene_chunks,
        "scene_fraction": a.scene_fraction,
        "first_scene_chunk_seconds": scene_arrivals[0],
        "full_scene_seconds": scene_arrivals[-1],
        "scene_chunk_arrivals_seconds": scene_arrivals,
        "frames": max_frames,
        "video_slots": total_slots,
        "tail_slots_after_human": tail_slots,
        "video_duration_seconds": first_arrival + content_duration,
        "frames_late": frames_late,
        "avg_render_ms": float(np.mean(render_times) * 1000),
        "render_fps_if_continuous": float(1.0 / np.mean(render_times)),
        "video": str(out_dir / "progressive_center.mp4"),
        "note": "center-distance scene loading baseline; no contact-aware scheduling",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
