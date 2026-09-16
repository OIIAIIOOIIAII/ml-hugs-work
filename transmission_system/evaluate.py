"""Numerical evaluation of reconstructed delta_mu and payload size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    reference = np.load(args.reference, allow_pickle=False).astype(np.float32)
    reconstructed = np.load(args.reconstructed, allow_pickle=False).astype(np.float32)
    error = reconstructed - reference
    result = {
        "shape": list(reference.shape),
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "reference_bytes": int(reference.nbytes),
        "reconstructed_bytes": int(reconstructed.nbytes),
    }
    if args.package:
        manifest = json.loads((args.package / "manifest.json").read_text())
        result["compressed_payload_bytes"] = int(manifest["payload_bytes"])
        result["payload_ratio"] = manifest["payload_bytes"] / reference.nbytes
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

