"""Create a small deterministic delta_mu sequence for smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--points", type=int, default=64)
    args = parser.parse_args()
    rng = np.random.default_rng(1234)
    base = rng.normal(0.0, 0.02, size=(args.points, 3)).astype(np.float32)
    motion = np.linspace(0.0, 1.0, args.frames, dtype=np.float32)[:, None, None]
    sequence = base[None] + motion * np.array([0.03, -0.01, 0.02], dtype=np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "delta_mu.npy", sequence, allow_pickle=False)
    print(f"created {args.out / 'delta_mu.npy'} shape={sequence.shape}")


if __name__ == "__main__":
    main()

