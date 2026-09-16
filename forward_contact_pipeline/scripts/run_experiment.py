#!/usr/bin/env python3
"""Versioned Stage-A preflight, training and explicit evaluation entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_streaming.training.config import load_config
from contact_streaming.training.engine import atomic_json, evaluate_checkpoint, prepare, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True, help="Local machine paths YAML (not committed)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, help="Optional preflight/evaluation JSON; never overwrites an existing file")
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths, args.set)
    if args.output and args.output.exists():
        parser.error("--output exists; choose a new result filename")
    if args.command != "train" and args.resume:
        parser.error("--resume is only for train")
    if args.command == "train":
        if args.checkpoint or args.output:
            parser.error("Train writes to configured run_root/name; use --resume for continuation")
        result = train(cfg, args.resume)
    elif args.command == "evaluate":
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        result = evaluate_checkpoint(cfg, args.checkpoint, args.split)
    else:
        _, _, result = prepare(cfg, ("train", "val"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
