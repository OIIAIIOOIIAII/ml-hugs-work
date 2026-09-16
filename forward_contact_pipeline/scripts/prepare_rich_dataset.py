#!/usr/bin/env python3
"""Prepare versioned RICH supervision manifests; never starts training/inference."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_streaming.rich_dataset import prepare_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets/RICH")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/data/rich_dataset_v1.json")
    parser.add_argument("--output", type=Path, help="Default: ROOT/processed/dataset_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve(); output = args.output or root / "processed/dataset_v1"
    (root / "processed").mkdir(parents=True, exist_ok=True)
    with (root / "processed/dataset_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(prepare_dataset(root, args.config, output, args.workers), indent=2))
