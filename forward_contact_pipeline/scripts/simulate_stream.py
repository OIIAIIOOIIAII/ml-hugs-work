#!/usr/bin/env python3
"""Simulate packet loss/latency and report contact-stream recovery metrics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.packet import ContactPacket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loss-rate", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=20.0)
    parser.add_argument("--jitter-ms", type=float, default=5.0)
    parser.add_argument("--late-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); rng = random.Random(args.seed)
    files = sorted(args.packets.glob("*.hct")); arrivals = []; dropped = 0; late = 0
    last_delivered = None; recovery_frames = []
    for path in files:
        packet = ContactPacket.decode(path.read_bytes())
        if rng.random() < args.loss_rate:
            dropped += 1; continue
        network_ms = max(0.0, rng.gauss(args.latency_ms, args.jitter_ms))
        if network_ms > args.late_ms:
            late += 1
        if last_delivered is not None and packet.frame_id > last_delivered + 1:
            recovery_frames.append(packet.frame_id - last_delivered - 1)
        last_delivered = packet.frame_id
        arrivals.append(network_ms)
    total_bytes = sum(path.stat().st_size for path in files)
    duration = len(files) / args.fps if files else 0
    report = {
        "packets": len(files), "delivered": len(arrivals), "dropped": dropped,
        "loss_rate_actual": dropped / max(len(files), 1), "late_ratio": late / max(len(files), 1),
        "latency_ms_mean": float(np.mean(arrivals)) if arrivals else None,
        "latency_ms_p95": float(np.percentile(arrivals, 95)) if arrivals else None,
        "recovery_frames_mean": float(np.mean(recovery_frames)) if recovery_frames else 0.0,
        "bitrate_kbps": total_bytes * 8 / max(duration, 1e-9) / 1000,
    }
    text = json.dumps(report, indent=2); print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
