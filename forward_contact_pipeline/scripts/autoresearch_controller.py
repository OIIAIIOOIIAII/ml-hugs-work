#!/usr/bin/env python3
"""Restartable, bounded local controller for sequential autoresearch runs.

It deliberately executes a pre-registered JSON plan rather than inventing
unbounded shell commands.  Each completed process writes a durable event and
the controller selects the next declared candidate from metric gates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def satisfied(metrics: dict, gate: dict) -> bool:
    """Metric gate: compare post/pre ratio and minimum contact F1."""
    if metrics.get("contact_f1", 0.0) < gate.get("min_contact_f1", 0.0): return False
    if metrics.get("post_foot_sliding_mps", float("inf")) > metrics.get("pre_foot_sliding_mps", 0.0) * gate.get("max_sliding_ratio", 1.0): return False
    return True


def gpu_is_idle() -> bool:
    """Return true only when NVIDIA reports no compute processes.

    This makes waiting for a shared GPU an explicit controller state rather
    than an implicit race with another user's job.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not result.stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--state", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=6.0); p.add_argument("--poll-seconds", type=float, default=20.0)
    p.add_argument("--wait-for-idle-gpu", action="store_true", help="Do not start a declared CUDA job while another compute process exists.")
    args = p.parse_args(); plan = read(args.plan, {}); state = read(args.state, {"completed": [], "events": []})
    deadline = time.time() + args.max_hours * 3600.0
    while time.time() < deadline:
        pending = [x for x in plan["experiments"] if x["id"] not in state["completed"]]
        if not pending:
            state["status"] = "all_declared_experiments_finished"; state["updated_at"] = now(); write(args.state, state); return
        item = pending[0]; workdir = Path(plan.get("workdir", ".")); log = Path(item["log"]); log.parent.mkdir(parents=True, exist_ok=True)
        if args.wait_for_idle_gpu and not gpu_is_idle():
            state["status"] = "waiting_for_idle_gpu"; state["current"] = item["id"]; state["updated_at"] = now(); write(args.state, state)
            time.sleep(args.poll_seconds)
            continue
        event = {"id": item["id"], "started_at": now(), "command": item["command"]}; state["events"].append(event); write(args.state, state)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{now()}] START {item['id']}\n")
            process = subprocess.Popen(item["command"], cwd=workdir, stdout=handle, stderr=subprocess.STDOUT, text=True)
            while process.poll() is None:
                if time.time() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    event.update({"finished_at":now(),"status":"terminated_at_budget"}); state["status"] = "time_budget_exhausted"; write(args.state,state); return
                time.sleep(args.poll_seconds)
            event.update({"finished_at": now(), "returncode": process.returncode})
        if process.returncode != 0:
            event["status"] = "failed_process"; state["completed"].append(item["id"]); write(args.state,state); continue
        report = Path(item["report"])
        if not report.exists():
            event["status"] = "missing_report"; state["completed"].append(item["id"]); write(args.state,state); continue
        metrics = read(report, {}).get("macro_mean", {})
        event["metrics"] = metrics; event["status"] = "passed" if satisfied(metrics, item.get("gate", {})) else "failed_gate"
        state["completed"].append(item["id"]); state["updated_at"] = now(); write(args.state,state)
    state["status"] = "time_budget_exhausted"; state["updated_at"] = now(); write(args.state,state)


if __name__ == "__main__":
    main()
