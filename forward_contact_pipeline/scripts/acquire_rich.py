#!/usr/bin/env python3
"""Acquire the user's four authorized RICH archives using local POST credentials.

Run --configure in a server terminal once; --start resumes with saved credentials.
The detached --run process uses two jobs, preferring aria2, and preserves partials.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import getpass
import http.cookiejar
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "datasets/RICH"
PRIVATE = ROOT / "download_lists"
LOGS = ROOT / "download_logs"
AUTH = PRIVATE / "credentials.post"
MANIFEST = PRIVATE / "protected_assets.json"
DEFAULT_MANIFEST = PROJECT / "forward_contact_pipeline/configs/data/rich_download_assets.json"
LOCAL_ARIA2 = PROJECT / ".tools/rich-download/bin/aria2c"


def aria2_binary():
    return str(LOCAL_ARIA2) if LOCAL_ARIA2.is_file() else shutil.which("aria2c")


def save_cookies(jar, path):
    """Write Netscape cookies, representing session expiry as 0 for aria2."""
    lines = ["# Netscape HTTP Cookie File"]
    for cookie in jar:
        lines.append("\t".join([
            cookie.domain, "TRUE" if cookie.domain_initial_dot else "FALSE",
            cookie.path, "TRUE" if cookie.secure else "FALSE",
            str(cookie.expires or 0), cookie.name, cookie.value,
        ]))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    path.chmod(0o600)


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def assets():
    result = json.loads((MANIFEST if MANIFEST.exists() else DEFAULT_MANIFEST).read_text())
    for item in result:
        url = urllib.parse.urlsplit(item["url"])
        relative = Path(item["relative_path"])
        if url.scheme != "https" or url.hostname != "download.is.tue.mpg.de":
            raise ValueError("Unexpected resource host in manifest")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid output path in manifest")
    return result


def credentials():
    if AUTH.stat().st_mode & 0o077:
        raise ValueError("Credential file must have permissions 600")
    data = AUTH.read_bytes()
    fields = urllib.parse.parse_qs(data.decode())
    if not fields.get("username") or not fields.get("password"):
        raise ValueError("Run --configure to provide RICH credentials")
    return data


def valid_magic(blob, relative_path):
    if relative_path.endswith(".zip"):
        return blob.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))
    return blob.startswith(b"\x1f\x8b")


def probe(item, data, cookie_path=None):
    # RICH accepts the POST, sets PHPSESSID, then redirects to a GET download.
    # The default urlopen handler follows redirects but does not retain cookies.
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    request = urllib.request.Request(
        item["url"], data=data, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Range": "bytes=0-1023", "User-Agent": "RICH-research-download/1.0"},
    )
    try:
        response = opener.open(request, timeout=45)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{item['name']}: HTTP {error.code}") from None
    with response:
        head = response.read(1024)
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" in content_type or not valid_magic(head, item["relative_path"]):
            raise RuntimeError(f"{item['name']}: login/error page or unexpected archive; check credentials")
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", content_range)
        ranged = response.status == 206 and match is not None
        if ranged and int(match[1]) + 1 != len(head):
            raise RuntimeError("Unexpected Range response length")
        length = response.headers.get("Content-Length")
        total = int(match[2]) if ranged else (int(length) if length else None)
        if response.status not in (200, 206):
            raise RuntimeError(f"Unexpected HTTP status {response.status}")
        if response.status == 206 and not ranged:
            raise RuntimeError("Malformed Content-Range; refusing unsafe resume")
        if cookie_path is not None:
            save_cookies(jar, cookie_path)
        return {"time": now(), "status": response.status, "range_supported": ranged,
                "expected_bytes": total, "content_type": content_type}


def inspect_archive(path, item, total):
    if total is not None and path.stat().st_size != total:
        raise RuntimeError(f"{item['name']}: size mismatch; partial file preserved")
    with path.open("rb") as stream:
        if not valid_magic(stream.read(8), item["relative_path"]):
            raise RuntimeError(f"{item['name']}: invalid archive signature")
    if item["relative_path"].endswith(".zip") and not zipfile.is_zipfile(path):
        raise RuntimeError(f"{item['name']}: ZIP directory is not readable")


def download(item, data, engine="auto", connections=4, other_limit="0"):
    state_path = LOGS / (item["name"] + ".json")
    state = {"name": item["name"], "status": "probing", "time": now()}
    save_json(state_path, state)
    try:
        selected = ("aria2" if aria2_binary() else "wget") if engine == "auto" else engine
        if selected == "aria2" and not aria2_binary():
            raise RuntimeError("aria2 was requested but is not installed")
        cookie_path = PRIVATE / (item["name"] + ".cookies.txt")
        info = probe(item, data, cookie_path if selected == "aria2" else None)
        state.update(info)
        state.update(engine=selected, connections=connections if selected == "aria2" else 1)
        state["rate_limit"] = other_limit if item["name"] != "train_jpg" else "0"
        target = ROOT / "raw" / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        total = info["expected_bytes"]
        if target.exists():
            inspect_archive(target, item, total)
            state["status"] = "existing_archive_checked"
        else:
            if selected == "wget" and partial.with_name(partial.name + ".aria2").exists():
                raise RuntimeError("aria2 partial data requires aria2 control file; cannot resume with wget")
            if partial.exists() and partial.stat().st_size and not info["range_supported"]:
                raise RuntimeError("Server did not support Range; existing partial file preserved")
            if partial.exists() and total is not None and partial.stat().st_size > total:
                raise RuntimeError("Partial file exceeds server size; preserved for inspection")
            state.update(status="downloading", path=str(partial))
            save_json(state_path, state)
            print(f"{now()} {item['name']}: downloading; Range={info['range_supported']}", flush=True)
            log_path = LOGS / (item["name"] + "." + selected + ".log")
            with log_path.open("ab") as logfile:
                if selected == "aria2":
                    command = [aria2_binary(), "--continue=true", "--auto-file-renaming=false",
                               "--file-allocation=none", "--load-cookies=" + str(cookie_path),
                               "--max-connection-per-server=" + str(connections),
                               "--split=" + str(connections), "--min-split-size=16M",
                               "--max-tries=10", "--retry-wait=15", "--timeout=60",
                               "--connect-timeout=30", "--summary-interval=10",
                               "--show-console-readout=false", "--enable-color=false",
                               "--max-download-limit=" + state["rate_limit"],
                               "--dir=" + str(partial.parent), "--out=" + partial.name, item["url"]]
                else:
                    command = ["wget", "--continue", "--post-file=" + str(AUTH),
                               "--timeout=60", "--tries=10", "--waitretry=15",
                               "--retry-connrefused", "--progress=dot:giga",
                               "--output-document=" + str(partial), item["url"]]
                completed = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"{selected} exited {completed.returncode}; see {log_path.name}")
            inspect_archive(partial, item, total)
            partial.replace(target)
            state["status"] = "downloaded_size_and_format_checked"
        state.update(path=str(target), bytes=target.stat().st_size,
                     full_archive_crc="not_yet_checked", time=now())
        print(f"{now()} {item['name']}: {state['status']}", flush=True)
        save_json(state_path, state)
        return True
    except Exception as error:
        # Never echo server bodies or credential values to console or state files.
        message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        state.update(status="failed", error=message, time=now())
        save_json(state_path, state)
        print(f"{now()} {item['name']}: {message}", flush=True)
        return False


def lock():
    stream = (PRIVATE / "download.lock").open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError("RICH downloader is already running") from None
    return stream


def download_with_retries(item, data, engine="auto", connections=4, other_limit="0", rounds=3, delay=60):
    """Refresh the official session only after a transfer process fails.

    Protocol/credential/archive validation failures are not retried blindly.
    aria2 control files remain authoritative; every retry uses the same partial.
    """
    for attempt in range(1, rounds + 1):
        if download(item, data, engine, connections, other_limit):
            return True
        path = LOGS / (item["name"] + ".json")
        state = json.loads(path.read_text())
        error = state.get("error", "")
        if attempt == rounds or not re.match(r"(?:aria2|wget) exited \d+;", error):
            return False
        state.update(status="waiting_retry", next_attempt=attempt + 1, max_attempts=rounds, retry_delay_seconds=delay, time=now())
        save_json(path, state)
        print(f"{now()} {item['name']}: retry {attempt + 1}/{rounds} after {delay}s with a fresh official session", flush=True)
        time.sleep(delay)
    return False


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("configure", "start", "run", "status"):
        group.add_argument("--" + flag, action="store_true")
    parser.add_argument("--engine", choices=("auto", "wget", "aria2"), default="auto")
    parser.add_argument("--connections", type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument("--other-limit", default="0", help="aria2 rate limit for non-JPG assets, e.g. 500K")
    parser.add_argument("--retry-rounds", type=int, default=3, help="Bounded transfer attempts, refreshing login between failures")
    parser.add_argument("--retry-delay", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.retry_rounds <= 10 or args.retry_delay < 0:
        parser.error("retry-rounds must be 1..10 and retry-delay nonnegative")
    if not re.fullmatch(r"\d+[KMG]?", args.other_limit):
        parser.error("--other-limit must be a nonnegative integer optionally followed by K, M or G")
    for path in (PRIVATE, LOGS):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.status:
        launch_path = LOGS / "launch.json"
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
        for item in assets():
            path = LOGS / (item["name"] + ".json")
            state = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
            if (state.get("status") in ("downloading", "probing")
                    and state.get("time", "") < launch.get("time", "")):
                state["status"] = "queued_with_saved_partial"
            partial = ROOT / "raw" / (item["relative_path"] + ".part")
            if partial.exists():
                if state.get("engine") == "aria2" or partial.with_name(partial.name + ".aria2").exists():
                    state["partial_logical_size_not_progress"] = partial.stat().st_size
                    log_path = LOGS / (item["name"] + ".aria2.log")
                    if log_path.exists():
                        with log_path.open("rb") as stream:
                            stream.seek(max(0, log_path.stat().st_size - 65536))
                            tail = stream.read().decode(errors="replace")
                        lines = re.findall(r"\[#[^\]\r\n]+\]", tail)
                        if lines:
                            state["latest_progress"] = lines[-1]
                else:
                    state["partial_bytes_now"] = partial.stat().st_size
            print(item["name"], json.dumps(state))
        return 0
    if not shutil.which("wget"):
        raise RuntimeError("wget is required")
    guard = lock()
    try:
        if args.configure:
            if not sys.stdin.isatty():
                raise RuntimeError("Run --configure in an interactive server terminal")
            username = input("RICH username: ").strip()
            password = getpass.getpass("RICH password (hidden): ")
            if not username or not password:
                raise RuntimeError("Username and password must not be empty")
            data = urllib.parse.urlencode({"username": username, "password": password,
                                          "commit": "Log in"}).encode()
            descriptor = os.open(AUTH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
            AUTH.chmod(0o600)
            del username, password, data
        data = credentials()
        items = assets()
        probe(items[0], data)  # Do not queue GB-scale transfers on a login page.
        if args.run:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda item: download_with_retries(item, data, args.engine, args.connections, args.other_limit, args.retry_rounds, args.retry_delay), items))
            return 0 if all(results) else 1
        guard.close()
        with (LOGS / "controller.log").open("ab") as logfile:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run",
                                      "--engine", args.engine, "--connections", str(args.connections),
                                      "--other-limit", args.other_limit,
                                      "--retry-rounds", str(args.retry_rounds), "--retry-delay", str(args.retry_delay)],
                                     # Credentials remain in local files, never process arguments.
                                     stdin=subprocess.DEVNULL, stdout=logfile,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        save_json(LOGS / "launch.json", {"pid": child.pid, "time": now(), "jobs": 2,
                                       "engine": args.engine, "connections": args.connections,
                                       "other_limit": args.other_limit, "retry_rounds": args.retry_rounds,
                                       "retry_delay": args.retry_delay})
        print(f"RICH downloads started in background (PID {child.pid}); terminal can be closed.")
        print(f"Progress: python {Path(__file__).resolve()} --status")
        return 0
    finally:
        guard.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Cannot start RICH download: {error}", file=sys.stderr)
        raise SystemExit(1)
