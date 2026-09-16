"""Baseline streaming benchmark: static assets once, correction every frame.

This deliberately measures the transport/decode pipeline separately from HUGS
GPU rendering.  The resulting ``decode_fps`` is not advertised as render FPS;
attach a renderer callback later for the true client viewing FPS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from codec import decode_frame, encode_frame


def mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def transfer_seconds(num_bytes: int, bandwidth_mbps: float, rtt_ms: float) -> float:
    return num_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0) + rtt_ms / 1000.0


def make_sequence(path: Path | None, frames: int, points: int) -> np.ndarray:
    if path:
        sequence = np.load(path, allow_pickle=False)
        if sequence.ndim != 3 or sequence.shape[-1] != 3:
            raise ValueError("--delta must have shape [T,N,3]")
        return sequence.astype(np.float32, copy=False)
    # Size-controlled fallback for a first architecture benchmark.
    return np.zeros((frames, points, 3), dtype=np.float32)


def benchmark(args: argparse.Namespace) -> dict:
    sequence = make_sequence(args.delta, args.frames, args.points)
    if args.savgol_window:
        if args.savgol_window >= len(sequence) or args.savgol_window % 2 == 0:
            raise ValueError("--savgol-window must be odd and smaller than the number of frames")
        from scipy.signal import savgol_filter
        sequence = savgol_filter(
            sequence.astype(np.float32),
            window_length=args.savgol_window,
            polyorder=args.savgol_polyorder,
            axis=0,
        ).astype(np.float32)
    frame_count = len(sequence)

    if args.static_bytes is None:
        static_bytes = sum(path.stat().st_size for path in args.static_file)
    else:
        static_bytes = args.static_bytes

    payloads: list[bytes] = []
    headers = []
    raw_dynamic_bytes = int(sequence.nbytes)
    previous = None
    encode_start = time.perf_counter()
    for frame in sequence:
        if args.codec == "raw":
            payloads.append(frame.tobytes(order="C"))
            headers.append(None)
            previous = frame
        else:
            header, payload, previous = encode_frame(frame, previous)
            headers.append(header)
            payloads.append(payload)
    encode_seconds = time.perf_counter() - encode_start

    dynamic_bytes = sum(len(payload) for payload in payloads)
    static_seconds = transfer_seconds(static_bytes, args.bandwidth_mbps, args.rtt_ms)

    # Sequential single-stream baseline: static assets occupy the link first,
    # then frame packets are sent in order.  Playback begins at frame 0 arrival.
    clock = static_seconds
    arrivals = []
    for payload in payloads:
        clock += transfer_seconds(len(payload), args.bandwidth_mbps, args.rtt_ms)
        arrivals.append(clock)
    first_frame_time = arrivals[0]
    playback_deadline = [first_frame_time + i / args.fps for i in range(frame_count)]
    on_time = sum(arrival <= deadline + args.buffer_seconds for arrival, deadline in zip(arrivals, playback_deadline))
    stream_end = max(arrivals[-1] if arrivals else static_seconds, first_frame_time + max(0, frame_count - 1) / args.fps)

    decode_start = time.perf_counter()
    previous = None
    for frame_id, payload in enumerate(payloads):
        if args.codec == "raw":
            _ = np.frombuffer(payload, dtype=np.float32).reshape(sequence[frame_id].shape)
        else:
            previous = decode_frame(headers[frame_id], payload, previous)
    decode_seconds = time.perf_counter() - decode_start

    result = {
        "mode": "static_once_plus_frame_stream",
        "codec": args.codec,
        "savgol_window": args.savgol_window,
        "savgol_polyorder": args.savgol_polyorder if args.savgol_window else None,
        "frames": frame_count,
        "target_fps": args.fps,
        "bandwidth_mbps": args.bandwidth_mbps,
        "rtt_ms": args.rtt_ms,
        "buffer_seconds": args.buffer_seconds,
        "static_bytes": static_bytes,
        "static_mib": mib(static_bytes),
        "raw_dynamic_bytes": raw_dynamic_bytes,
        "raw_dynamic_mib_per_frame": mib(raw_dynamic_bytes / frame_count),
        "dynamic_payload_bytes": dynamic_bytes,
        "dynamic_payload_mib_per_frame": mib(dynamic_bytes / frame_count),
        "total_bytes": static_bytes + dynamic_bytes,
        "total_mib": mib(static_bytes + dynamic_bytes),
        "payload_ratio_vs_raw_dynamic": dynamic_bytes / raw_dynamic_bytes if raw_dynamic_bytes else None,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "decode_fps": frame_count / decode_seconds if decode_seconds else None,
        "startup_to_first_frame_seconds": first_frame_time,
        "static_transfer_seconds": static_seconds,
        "on_time_frames": on_time,
        "on_time_ratio": on_time / frame_count if frame_count else None,
        "delivery_span_seconds": stream_end - first_frame_time if frame_count else 0.0,
        "received_fps_over_delivery_span": ((frame_count - 1) / (arrivals[-1] - first_frame_time)
                                             if frame_count > 1 and arrivals[-1] > first_frame_time else None),
        "note": "decode_fps excludes HUGS GPU rendering; use a renderer hook for true viewing FPS",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, help="real delta_mu .npy with shape [T,N,3]")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--points", type=int, default=500_000)
    parser.add_argument("--static-file", type=Path, action="append", default=[], help="one or more static files")
    parser.add_argument("--static-bytes", type=int, default=502_000_000, help="static size when no files are given")
    parser.add_argument("--bandwidth-mbps", type=float, default=100.0)
    parser.add_argument("--rtt-ms", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--buffer-seconds", type=float, default=0.0)
    parser.add_argument("--codec", choices=["raw", "causal_int8_zlib"], default="raw")
    parser.add_argument("--savgol-window", type=int, help="offline Savitzky-Golay preprocessing window")
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.static_file and args.static_bytes == 502_000_000:
        args.static_bytes = None
    result = benchmark(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
