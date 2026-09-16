#!/usr/bin/env python3
"""Export RGB images from FrameInput NPZ files without using a system temp directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="FrameInput directory containing frames/*.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    paths = sorted((args.input / "frames").glob("*.npz"))
    paths = paths[args.start_frame :]
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    if not paths:
        raise RuntimeError(f"No frames under {args.input / 'frames'}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as frame:
            if "color_rgb" not in frame:
                raise KeyError(f"{path} has no color_rgb")
            rgb = frame["color_rgb"]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0.0, 1.0)
            rgb = (rgb * 255.0).astype(np.uint8)
        cv2.imwrite(str(args.output / f"frame_{index:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Exported {len(paths)} RGB frames to {args.output}")


if __name__ == "__main__":
    main()
