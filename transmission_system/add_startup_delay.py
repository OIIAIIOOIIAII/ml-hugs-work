"""Prepend a simulated static-asset loading interval to a rendered video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--startup-seconds", type=float, required=True)
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"cannot read video metadata: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output video: {args.output}")
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(int(round(args.startup_seconds * fps))):
        writer.write(blank)
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
    cap.release()
    writer.release()
    print(f"wrote {frames} content frames + {args.startup_seconds:.3f}s startup -> {args.output}")


if __name__ == "__main__":
    main()

