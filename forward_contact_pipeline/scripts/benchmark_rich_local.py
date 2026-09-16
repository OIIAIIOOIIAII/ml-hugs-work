#!/usr/bin/env python3
"""Compare sustained JPG-only transfers, then restore the previous download queue.

All samples contribute to the existing archive and preserve its aria2 control file.
The first minute of each sample is excluded from the steady-rate estimate.
"""
import argparse
import json
import re
import signal
import statistics
import subprocess
import sys
import time

from benchmark_rich_connections import rich, SOURCE, stop_owned_controller


def summaries(path):
    content = path.read_text(errors="replace") if path.exists() else ""
    result = []
    for line in re.findall(r"\[#[^\]\r\n]+\]", content):
        match = re.search(r"DL:([\d.]+)(GiB|MiB|KiB|B)", line)
        if match:
            result.append({"line": line, "Bps": float(match[1]) *
                           {"GiB": 2**30, "MiB": 2**20, "KiB": 1024, "B": 1}[match[2]]})
    return result


def sample(item, cookie, connections, seconds):
    partial = rich.ROOT / "raw" / (item["relative_path"] + ".part")
    log = rich.LOGS / f"train_jpg.local_{connections}_{int(time.time())}.log"
    command = [rich.aria2_binary(), "--continue=true", "--auto-file-renaming=false",
               "--file-allocation=none", "--load-cookies=" + str(cookie),
               f"--max-connection-per-server={connections}", f"--split={connections}",
               "--min-split-size=16M", "--max-tries=2", "--retry-wait=5",
               "--timeout=30", "--connect-timeout=20", "--summary-interval=10",
               f"--stop={seconds}", "--show-console-readout=false",
               "--enable-color=false", "--human-readable=false",
               "--dir=" + str(partial.parent), "--out=" + partial.name, item["url"]]
    rich.save_json(rich.LOGS / "train_jpg.json", {
        "name": "train_jpg", "status": "benchmarking", "engine": "aria2",
        "connections": connections, "path": str(partial), "time": rich.now(),
        "benchmark_log": str(log),
    })
    started = time.monotonic()
    process = None
    with log.open("wb") as stream:
        try:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            result = process.wait(timeout=seconds + 45)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                # aria2 must finish saving its control file before the next writer.
                process.wait(timeout=30)
    rows = summaries(log)
    rates = [row["Bps"] for row in rows[6:]]
    errors = re.findall(r"errorCode=\d+", log.read_text(errors="replace"))
    return {"connections": connections, "elapsed_seconds": time.monotonic() - started,
            "return_code": result, "log": str(log), "summaries": rows,
            "return_code_note": "7 is expected for an unfinished timed transfer; inspect error_codes",
            "excluded_startup_seconds": 60, "steady_samples": len(rates),
            "steady_mean_Bps": statistics.mean(rates) if rates else None,
            "steady_median_Bps": statistics.median(rates) if rates else None,
            "steady_min_Bps": min(rates) if rates else None,
            "steady_max_Bps": max(rates) if rates else None, "error_codes": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    if args.seconds < 120 or args.seconds > 600:
        parser.error("--seconds must be between 120 and 600")
    launch = json.loads((rich.LOGS / "launch.json").read_text())
    data = rich.credentials()
    item = next(item for item in rich.assets() if item["name"] == "train_jpg")
    cookie = rich.PRIVATE / "train_jpg.cookies.txt"
    info = rich.probe(item, data, cookie)
    if not info["range_supported"] or not rich.aria2_binary():
        raise RuntimeError("Range and aria2 are required")
    report_path = rich.LOGS / "jpg_local_benchmark.json"
    report = {"started": rich.now(), "expected_bytes": info["expected_bytes"],
              "original_launch": launch, "samples": [],
              "baseline_last_18": summaries(rich.LOGS / "train_jpg.aria2.log")[-18:]}
    rich.save_json(report_path, report)
    guard = None
    stop_owned_controller()
    try:
        guard = rich.lock()
        for count in (16, 8):
            print(f"{rich.now()} JPG-only: {count} connections for {args.seconds}s", flush=True)
            result = sample(item, cookie, count, args.seconds)
            report["samples"].append(result)
            rich.save_json(report_path, report)
            print(json.dumps({k: v for k, v in result.items() if k != "summaries"}), flush=True)
    finally:
        if guard is not None:
            guard.close()
        report["ended"] = rich.now()
        rich.save_json(report_path, report)
        # Keep the prior connection count, priority, and rate cap intact.
        result = subprocess.run([sys.executable, str(SOURCE), "--start", "--engine", "aria2",
                                 "--connections", str(launch.get("connections", 16)),
                                 "--other-limit", launch.get("other_limit", "500K")], check=False)
        report["resume_exit_code"] = result.returncode
        rich.save_json(report_path, report)
        if result.returncode:
            raise RuntimeError("Download controller could not resume")


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt("Interrupted; preserving partials and restoring queue")
    signal.signal(signal.SIGTERM, terminate)
    main()
