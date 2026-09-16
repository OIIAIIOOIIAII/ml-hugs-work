#!/usr/bin/env python3
"""Create sequence-safe train/val/test splits without adjacent-frame leakage."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-prefix", action="store_true", help="Keep names before '__' in the same split")
    args = parser.parse_args()
    files = sorted(path.name for path in args.data.glob("*.npz"))
    if len(files) < 3:
        raise RuntimeError("At least three prepared sequence files are required")
    groups = {}
    for name in files:
        key = name.split("__", 1)[0] if args.group_prefix else name
        groups.setdefault(key, []).append(name)
    keys = sorted(groups); random.Random(args.seed).shuffle(keys)
    train_end = max(1, round(len(keys) * args.train_ratio))
    val_end = max(train_end + 1, round(len(keys) * (args.train_ratio + args.val_ratio)))
    partitions = {"train": keys[:train_end], "val": keys[train_end:val_end], "test": keys[val_end:]}
    if not partitions["test"]:
        partitions["test"] = partitions["val"][-1:]
    splits = {split: [name for key in selected for name in groups[key]] for split, selected in partitions.items()}
    write_json(args.data / "splits.json", splits)
    print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()
