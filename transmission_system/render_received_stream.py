"""Render a received correction stream through the real HUGS renderer.

The network is simulated analytically (static assets first, then frame
packets).  The receiver uses the latest packet whose simulated arrival time is
before each presentation deadline; if a packet is late, the previous decoded
correction is reused.  This produces a video with the requested playback FPS
and exposes both network stalls and HUGS GPU rendering cost.
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
from eval_method5_psnr import get_dataset, load_models  # noqa: E402


CKPTS = {
    "bike": ROOT / "output/human_scene/neuman/bike_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_12-52-08/ckpt",
    "jogging": ROOT / "output/human_scene/neuman/jogging_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_21-51-55/ckpt",
    "seattle": ROOT / "output/human_scene/neuman/seattle_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_17-28-48/ckpt",
    "lab": ROOT / "output/human_scene/neuman/lab_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_01-16-10/ckpt",
    "parkinglot": ROOT / "output/human_scene/neuman/parkinglot_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_03-29-29/ckpt",
    "citron": ROOT / "output/human_scene/neuman/citron_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_05-32-10/ckpt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=sorted(CKPTS), default="bike")
    p.add_argument("--delta", type=Path)
    p.add_argument("--ckpt-dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("results/received_render"))
    p.add_argument("--bandwidth-mbps", type=float, default=1000.0)
    p.add_argument("--rtt-ms", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--static-bytes", type=int, default=502_000_000)
    p.add_argument("--savgol-window", type=int, default=11)
    p.add_argument("--savgol-polyorder", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--split", choices=["all", "val"], default="all")
    p.add_argument("--no-startup-delay", dest="include_startup_delay", action="store_false",
                   help="do not prepend the simulated static-asset loading interval to the video")
    p.set_defaults(include_startup_delay=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def image_from_render(render: torch.Tensor) -> np.ndarray:
    image = render.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (image[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8)  # RGB -> BGR


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred.float().clamp(0, 1) - target.float().clamp(0, 1)) ** 2)
    return float(10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12)))


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    delta_path = args.delta or ROOT / "output/delta_mu_analysis" / args.scene / "delta_mu_f16.npy"
    ckpt_dir = args.ckpt_dir or CKPTS[args.scene]
    out_dir = args.out_dir / f"{args.scene}_savgol{args.savgol_window}_{int(args.bandwidth_mbps)}mbps"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = np.load(delta_path).astype(np.float32)
    if args.savgol_window:
        dm = savgol_filter(dm, args.savgol_window, args.savgol_polyorder, axis=0).astype(np.float32)

    # Encode once as the sender would.  Arrival timestamps model a serial link.
    payloads, headers = [], []
    previous = None
    encode_start = time.perf_counter()
    for frame in dm:
        header, payload, previous = encode_frame(frame, previous)
        payloads.append(payload)
        headers.append(header)
    encode_seconds = time.perf_counter() - encode_start
    arrivals = []
    clock = transfer_seconds(args.static_bytes, args.bandwidth_mbps, args.rtt_ms)
    for payload in payloads:
        clock += transfer_seconds(len(payload), args.bandwidth_mbps, args.rtt_ms)
        arrivals.append(clock)

    # Static client initialization is timed separately from network startup.
    load_start = time.perf_counter()
    human_gs, scene_out, anchor_attn, cfg = load_models(ckpt_dir, str(device))
    # HUGS_TRIMLP is a custom container rather than torch.nn.Module; its
    # tensors are placed on CUDA internally during construction.
    human_gs.eval()
    if args.split == "all":
        from hugs.datasets.neuman import NeumanDataset
        dataset = NeumanDataset(
            seq=cfg.dataset.seq,
            split="all",
            mono_depth_dir=getattr(cfg.dataset, "mono_depth_dir", None),
        )
    else:
        dataset = get_dataset(cfg, device)
    static_load_seconds = time.perf_counter() - load_start
    from hugs.renderer.gs_renderer import render_human_scene

    max_frames = min(len(dataset), args.max_frames or len(dataset))
    anchor_ids = human_gs.anchor_ids.to(device).long()
    anchor_weights = human_gs.anchor_weights.to(device)
    iteration = int(cfg.train.num_steps)
    bg_color = torch.zeros(3, dtype=torch.float32, device=device)

    writer = None
    previous_decoded = None
    decoded_index = -1
    render_times = []
    psnrs = []
    packet_indices = []
    frames_late = 0
    frames_reused = 0
    first_arrival = arrivals[0]
    total_start = time.perf_counter()

    for output_index in range(max_frames):
        data = dataset[output_index]
        data_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        global_index = int(data_gpu["frame_idx"].item())
        presentation_deadline = first_arrival + output_index / args.fps
        available = [i for i, t in enumerate(arrivals) if t <= presentation_deadline]
        latest_available = min(max(available), len(payloads) - 1) if available else -1
        # Never apply a future correction to the current display frame.  A
        # receiver may have prefetched future packets, but playback must stay
        # causal with respect to the requested frame index.
        available_index = min(latest_available, global_index) if latest_available >= 0 else -1
        if latest_available < global_index:
            frames_late += 1
        if available_index < 0:
            available_index = 0
        if available_index == decoded_index:
            frames_reused += 1
        else:
            for packet_index in range(decoded_index + 1, available_index + 1):
                previous_decoded = decode_frame(headers[packet_index], payloads[packet_index], previous_decoded)
            decoded_index = available_index
        packet_indices.append(decoded_index)

        human_out = human_gs.forward(
            global_orient=data_gpu["global_orient"],
            body_pose=data_gpu["body_pose"],
            betas=data_gpu["betas"],
            transl=data_gpu["transl"],
            smpl_scale=data_gpu["smpl_scale"][None],
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
        )
        corrected, stats = anchor_attn(
            human_out, scene_out, anchor_ids, anchor_weights,
            iteration=iteration, frame_idx=output_index,
        )
        if stats and "delta_mu" in stats and "gamma_mu_eff" in stats:
            original = stats["delta_mu"]
            corrected["xyz"] = corrected["xyz"] + float(stats["gamma_mu_eff"]) * (
                torch.from_numpy(previous_decoded).to(device) - original
            )

        render_start = time.perf_counter()
        pkg = render_human_scene(
            data=data_gpu,
            human_gs_out=corrected,
            scene_gs_out=scene_out,
            bg_color=bg_color,
            render_mode="human_scene",
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        render_times.append(time.perf_counter() - render_start)
        frame = image_from_render(pkg["render"])
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_dir / "received_stream.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError("cannot open output video writer")
            if args.include_startup_delay:
                # A real client has no decoded frame during static-asset
                # loading.  Keep the time axis honest with a loading/black
                # placeholder instead of silently deleting this interval.
                blank = np.zeros_like(frame)
                for _ in range(int(round(first_arrival * args.fps))):
                    writer.write(blank)
        writer.write(frame)
        if "rgb" in data_gpu:
            psnrs.append(psnr(pkg["render"], data_gpu["rgb"]))
        print(f"rendered {output_index + 1}/{max_frames} packet={decoded_index} late={available_index < global_index}", flush=True)

    if writer is not None:
        writer.release()
    wall_seconds = time.perf_counter() - total_start
    result = {
        "scene": args.scene,
        "delta_path": str(delta_path),
        "video": str(out_dir / "received_stream.mp4"),
        "include_startup_delay": args.include_startup_delay,
        "video_duration_seconds": (max_frames / args.fps + first_arrival
                                    if args.include_startup_delay else max_frames / args.fps),
        "frames": max_frames,
        "target_fps": args.fps,
        "bandwidth_mbps": args.bandwidth_mbps,
        "savgol_window": args.savgol_window,
        "static_network_startup_seconds": first_arrival,
        "receiver_model_load_seconds": static_load_seconds,
        "sender_encode_seconds": encode_seconds,
        "avg_render_ms": float(np.mean(render_times) * 1000),
        "p95_render_ms": float(np.percentile(render_times, 95) * 1000),
        "render_fps_if_continuous": float(1.0 / np.mean(render_times)),
        "wall_clock_render_fps": max_frames / wall_seconds,
        "mean_psnr_db": float(np.mean(psnrs)) if psnrs else None,
        "split": args.split,
        "frames_late": frames_late,
        "frames_reused": frames_reused,
        "mean_packet_index": float(np.mean(packet_indices)),
        "note": "video follows target presentation FPS; late packets reuse the latest decoded correction; no scene chunk streaming yet",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
