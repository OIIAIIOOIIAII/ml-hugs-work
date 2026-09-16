#!/usr/bin/env python3
"""Process final RICH archives now and optionally wait for pending downloads."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_streaming.rich_assets import extract_archive, save_json

ASSETS = [
    ("multicam2world", "common/multicam2world.zip", 4304),
    ("scan_calibration", "common/scan_calibration.zip", 904175585),
    ("train_body", "train/train_body.zip", 9147649220),
    ("train_hsc", "train/train_hsc.zip", 28826902288),
    ("train_jpg", "train/train.tar.gz", 559418993166),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets/RICH")
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("start", "run", "status"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--wait-hours", type=float, default=12)
    parser.add_argument("--stream-jpg", action="store_true", help="Extract complete prefix pieces provisionally while aria2 downloads; publish only after full gzip verification")
    args = parser.parse_args()
    root = args.root.resolve(); logs = root / "processing_logs"; logs.mkdir(parents=True, exist_ok=True)
    if args.status:
        for name, relative, _ in ASSETS:
            state = logs / (name + ".json")
            print(name, state.read_text().strip() if state.exists() else json.dumps({"status": "pending", "archive_exists": (root / "raw" / relative).exists()}))
        return
    with (logs / "processing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.start:
            fcntl.flock(lock, fcntl.LOCK_UN)
            with (logs / "controller.log").open("ab") as stream:
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--root", str(root), "--wait-hours", str(args.wait_hours)]
                if args.stream_jpg:
                    command.append("--stream-jpg")
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save_json(logs / "launch.json", {"pid": child.pid, "started": time.time(), "root": str(root),
                                             "stream_jpg": args.stream_jpg, "wait_hours": args.wait_hours})
            print(f"RICH processing started in background: PID {child.pid}")
            return
        pending = list(ASSETS); deadline = time.monotonic() + args.wait_hours * 3600
        while pending:
            for item in pending[:]:
                name, relative, expected = item; source = root / "raw" / relative
                if name == "train_jpg" and args.stream_jpg and (source.exists() or source.with_name(source.name + ".part").exists()):
                    from contact_streaming.rich_streaming import stream_jpg
                    print("Streaming complete JPG prefix; output remains provisional until full archive CRC verification", flush=True)
                    result = stream_jpg(root, expected_bytes=expected, wait_hours=args.wait_hours)
                    print(json.dumps(result), flush=True)
                    pending.remove(item)
                    continue
                if not source.exists():
                    continue
                if source.stat().st_size != expected:
                    raise ValueError(f"{name}: archive size differs from verified source size")
                print(f"Processing {name}", flush=True)
                result = extract_archive(source, root / "extracted", logs / (name + ".json"), name)
                print(json.dumps(result), flush=True)
                pending.remove(item)
            if pending:
                save_json(logs / "pending.json", {"assets": [x[0] for x in pending], "updated": time.time()})
                if time.monotonic() >= deadline:
                    raise TimeoutError("Waiting deadline reached; rerun to continue existing completed assets")
                time.sleep(30)
        save_json(logs / "pending.json", {"assets": [], "updated": time.time()})
        print("All requested archives extracted and verified", flush=True)


if __name__ == "__main__":
    main()
