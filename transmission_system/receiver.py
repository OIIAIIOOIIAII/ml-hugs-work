"""localhost TCP receiver; renderer integration attaches after decode_frame."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import numpy as np

from codec import decode_frame
from protocol import recv_message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    with socket.create_server((args.host, args.port)) as server:
        print(f"listening on {args.host}:{args.port}", flush=True)
        conn, address = server.accept()
        with conn, conn.makefile("rb") as stream:
            previous = None
            while True:
                header, payload = recv_message(stream)
                if header["kind"] == "session_start":
                    print(f"session: {header['frames']} frames, shape={header['shape']}", flush=True)
                elif header["kind"] == "frame":
                    previous = decode_frame(header, payload, previous)
                    frames.append(previous.copy())
                    # TODO: call HUGS renderer here after static assets are loaded.
                    print(f"received frame {header['frame_id'] + 1}", flush=True)
                elif header["kind"] == "session_end":
                    break
                else:
                    raise ValueError(f"unknown message kind: {header['kind']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, np.stack(frames), allow_pickle=False)
    print(f"saved {len(frames)} reconstructed frames -> {args.out}")


if __name__ == "__main__":
    main()

