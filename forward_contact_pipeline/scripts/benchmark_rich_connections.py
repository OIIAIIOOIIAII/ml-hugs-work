#!/usr/bin/env python3
"""Briefly benchmark the real JPG archive, preserving bytes and resuming the queue."""
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).with_name("acquire_rich.py").resolve()
spec = importlib.util.spec_from_file_location("rich_download", SOURCE)
rich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rich)


def stop_owned_controller():
    launch = json.loads((rich.LOGS / "launch.json").read_text())
    pid = launch["pid"]
    proc = Path("/proc", str(pid))
    if not proc.exists():
        return
    command = (proc / "cmdline").read_bytes().split(b"\0")
    if str(SOURCE).encode() not in command or b"--run" not in command or os.getpgid(pid) != pid:
        raise RuntimeError("Existing controller identity differs; not stopping it")
    os.killpg(pid, signal.SIGTERM)
    for _ in range(100):
        living = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                if os.getpgid(int(path.name)) == pid:
                    status = (path / "stat").read_text().rsplit(")", 1)[1].split()[0]
                    if status != "Z":
                        living.append(int(path.name))
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                pass
        if not living:
            return
        time.sleep(.2)
    raise RuntimeError("Previous transfer processes have not stopped")


def run_sample(item, connections, cookie_path):
    partial = rich.ROOT / "raw" / (item["relative_path"] + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    path = rich.LOGS / f"train_jpg.benchmark{connections}.log"
    command = [rich.aria2_binary(), "--continue=true", "--auto-file-renaming=false",
               "--file-allocation=none", "--load-cookies=" + str(cookie_path),
               "--max-connection-per-server=" + str(connections), "--split=" + str(connections),
               "--min-split-size=16M", "--max-tries=2", "--retry-wait=5", "--timeout=30",
               "--connect-timeout=20", "--summary-interval=10", "--stop=60",
               "--show-console-readout=false", "--enable-color=false",
               "--dir=" + str(partial.parent), "--out=" + partial.name, item["url"]]
    rich.save_json(rich.LOGS / "train_jpg.json", {
        "name": "train_jpg", "status": "benchmarking", "engine": "aria2",
        "connections": connections, "path": str(partial), "time": rich.now(),
    })
    started = time.time()
    with path.open("wb") as logfile:
        result = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT)
    text = path.read_text(errors="replace")
    summaries = [line for line in re.findall(r"\[#[^\]\r\n]+\]", text)
                 if f"CN:{connections} " in line]
    rates = []
    for line in summaries[-4:]:
        match = re.search(r"DL:([\d.]+)(GiB|MiB|KiB|B)", line)
        if match:
            rates.append(float(match[1]) * {"GiB": 2**30, "MiB": 2**20, "KiB": 1024, "B": 1}[match[2]])
    return {"connections": connections, "elapsed_seconds": time.time() - started,
            "return_code": result.returncode, "summaries": summaries,
            "steady_mean_Bps": sum(rates) / len(rates) if rates else None,
            "steady_samples": len(rates), "error_codes": re.findall(r"errorCode=\d+", text)}


def main():
    os.umask(0o077)
    data = rich.credentials()
    items = rich.assets()
    item = next(item for item in items if item["name"] == "train_jpg")
    cookie_path = rich.PRIVATE / "train_jpg.cookies.txt"
    info = rich.probe(item, data, cookie_path)
    if not info["range_supported"] or not rich.aria2_binary():
        raise RuntimeError("Range and aria2 are required for this benchmark")
    selected = 8
    guard = None
    report = {"started": rich.now(), "expected_bytes": info["expected_bytes"], "samples": []}
    stop_owned_controller()
    try:
        guard = rich.lock()
        for connections in (8, 16):
            print(f"{rich.now()} Starting {connections}-connection JPG sample", flush=True)
            sample = run_sample(item, connections, cookie_path)
            report["samples"].append(sample)
            rich.save_json(rich.LOGS / "jpg_connection_benchmark.json", report)
            print(json.dumps(sample), flush=True)
        first, second = report["samples"]
        if (first["steady_samples"] >= 3 and second["steady_samples"] >= 3
                and not second["error_codes"]
                and second["steady_mean_Bps"] > 1.2 * first["steady_mean_Bps"]):
            selected = 16
        report.update(selected_connections=selected, ended=rich.now())
        # Start the large image archive alongside numerical assets after this
        # benchmark. Its downloaded portions remain useful to the real transfer.
        order = {"scan_calibration": 0, "train_jpg": 1, "train_body": 2, "train_hsc": 3}
        backup = rich.PRIVATE / ("protected_assets.before_jpg_priority." + str(int(time.time())) + ".json")
        backup.write_text(rich.MANIFEST.read_text())
        rich.save_json(rich.MANIFEST, sorted(items, key=lambda item: order[item["name"]]))
        report["queue_order"] = [item["name"] for item in sorted(items, key=lambda item: order[item["name"]])]
    finally:
        if guard is not None:
            guard.close()
        report["resume_connections"] = selected
        rich.save_json(rich.LOGS / "jpg_connection_benchmark.json", report)
        result = subprocess.run([sys.executable, str(SOURCE), "--start", "--engine", "aria2",
                                 "--connections", str(selected)], check=False)
        report["resume_exit_code"] = result.returncode
        rich.save_json(rich.LOGS / "jpg_connection_benchmark.json", report)
        if result.returncode:
            raise RuntimeError("Benchmark ended but controller did not resume; inspect authentication/network")


if __name__ == "__main__":
    main()
