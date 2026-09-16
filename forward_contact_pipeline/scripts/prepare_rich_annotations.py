#!/usr/bin/env python3
"""Prepare numeric RICH GT annotations from complete ZIPs, independently of JPGs."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_streaming.rich_annotations import prepare_annotations
from contact_streaming.rich_assets import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets/RICH")
    parser.add_argument("--metadata", type=Path, default=Path(__file__).resolve().parents[1] / "configs/data/rich_metadata")
    parser.add_argument("--output", type=Path, help="Default: ROOT/processed/annotations_v1")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("sample", "start", "run", "status"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(); output = (args.output or root / "processed/annotations_v1").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.status:
        path = output / "state.json"
        if path.exists():
            state = json.loads(path.read_text()); shards = state.pop("shards", []); state.pop("tracks", None)
            state["completed_shards"] = len(shards); print(json.dumps(state, indent=2))
        else:
            print("Full annotation packing has not started")
        return
    with (output / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.start:
            fcntl.flock(lock, fcntl.LOCK_UN)
            with (output / "controller.log").open("ab") as log:
                child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run",
                    "--root", str(root), "--output", str(output), "--metadata", str(args.metadata.resolve())],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save_json(output / "launch.json", {"pid": child.pid, "started": time.time()})
            print(f"RICH annotation preparation started: PID {child.pid}")
        else:
            prepare_annotations(root, args.metadata.resolve(), output, sample=args.sample)


if __name__ == "__main__":
    main()
