"""localhost TCP sender for delta_mu sequences."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np

from codec import encode_frame
from protocol import send_message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    sequence = np.load(args.input, allow_pickle=False)
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError("input must have shape [T, N, 3]")
    with socket.create_connection((args.host, args.port)) as sock:
        send_message(sock, {"kind": "session_start", "frames": len(sequence), "shape": list(sequence.shape[1:]), "fps": args.fps})
        previous = None
        for frame_id, delta in enumerate(sequence):
            header, payload, previous = encode_frame(delta, previous)
            header["frame_id"] = frame_id
            send_message(sock, header, payload)
            print(f"sent frame {frame_id + 1}/{len(sequence)} payload={len(payload)}", flush=True)
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
        send_message(sock, {"kind": "session_end"})


if __name__ == "__main__":
    main()

