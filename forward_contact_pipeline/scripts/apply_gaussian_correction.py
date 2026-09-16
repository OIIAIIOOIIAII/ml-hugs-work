#!/usr/bin/env python3
"""Apply predicted root corrections to generic Gaussian-center NPZ files.

For full pose/LBS deformation use GUSH3R's SMPLX_Mesh.get_target_transform and
pose_points_from_zero. This utility is an interchange smoke-test for xyz arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.gaussian_adapter import apply_root_residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussians", type=Path, required=True, help="NPZ with xyz [T,N,3] or [N,3]")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.gaussians, allow_pickle=False) as archive:
        gaussian_data = {key: archive[key] for key in archive.files}
    with np.load(args.predictions, allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    xyz = gaussian_data["xyz"]
    if xyz.ndim == 2:
        xyz = np.repeat(xyz[None], len(predictions["root_residual"]), axis=0)
    if len(xyz) != len(predictions["root_residual"]):
        raise ValueError("Gaussian frame count and prediction frame count differ")
    corrected = np.stack([apply_root_residual(points, residual) for points, residual in zip(xyz, predictions["root_residual"])])
    gaussian_data["xyz_corrected"] = corrected.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **gaussian_data)
    print(f"Saved {len(corrected)} corrected Gaussian frames to {args.output}")


if __name__ == "__main__":
    main()
