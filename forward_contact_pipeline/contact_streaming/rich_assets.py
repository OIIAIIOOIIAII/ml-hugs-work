"""Restartable extraction of complete RICH archives, without decoding images."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import time
import zipfile
import zlib


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def member_path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in path.parts[0]:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def crc32(path):
    value = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xffffffff


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        return value


def extract_archive(source, extracted_root, state_path, name):
    """Read every member (CRC checked), then atomically publish one archive root.

    ZIP restarts validate existing staged files by CRC. TAR restarts re-stream
    the archive, rewriting staged files; gzip CRC is checked by draining to EOF.
    No raw archives are removed and no image pixels are interpreted.
    """
    source, root, state_path = Path(source), Path(extracted_root), Path(state_path)
    if source.name.endswith(".part") or not source.is_file():
        raise ValueError("Only final, complete archive filenames may be processed")
    info = source.stat()
    fingerprint = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("source_fingerprint") != fingerprint:
            raise ValueError("Source changed; use a new extraction/version directory")
        if previous.get("status") == "complete":
            if not Path(previous["output"]).exists():
                raise ValueError("Completed extraction output is missing")
            return previous
    staging = root / ".staging" / name
    marker = staging / ".source.json"
    if staging.exists():
        if not marker.exists() or json.loads(marker.read_text()) != fingerprint:
            raise ValueError("Unowned or mismatched staging directory")
    staging.mkdir(parents=True, exist_ok=True)
    save_json(marker, fingerprint)
    state = {"asset": name, "source": str(source.resolve()), "source_fingerprint": fingerprint,
             "status": "extracting", "files": 0, "uncompressed_bytes": 0, "started": time.time()}
    tops, seen = set(), set()
    last_save = 0.0

    def progress(force=False):
        nonlocal last_save
        if force or time.monotonic() - last_save > 10:
            state["updated"] = time.time(); save_json(state_path, state); last_save = time.monotonic()

    def destination(filename, is_dir):
        relative = member_path(filename)
        tops.add(relative.parts[0])
        if len(tops) > 1 or relative.parts[0].startswith("."):
            raise ValueError("Expected one non-hidden archive root")
        path = staging.joinpath(*relative.parts)
        for parent in (path, *path.parents):
            if parent == staging.parent:
                break
            if parent.is_symlink():
                raise ValueError("Symlink in extraction destination")
        if not is_dir:
            if relative in seen:
                raise ValueError("Duplicate file in archive")
            seen.add(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def copy_member(stream, path, size):
        temporary = path.with_name(path.name + ".extracting")
        with temporary.open("wb") as output:
            shutil.copyfileobj(stream, output, 1024 * 1024)
        if temporary.stat().st_size != size:
            raise ValueError("Truncated extracted member")
        temporary.replace(path)

    progress(True)
    try:
        if source.suffix == ".zip":
            with zipfile.ZipFile(source) as archive:
                required = sum(member.file_size for member in archive.infolist())
                if required > shutil.disk_usage(root).free:
                    raise RuntimeError("Insufficient free space for ZIP extraction")
                for member in archive.infolist():
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError("Archive contains a link or special file")
                    path = destination(member.filename, member.is_dir())
                    if member.is_dir():
                        path.mkdir(exist_ok=True)
                        continue
                    if not (path.is_file() and path.stat().st_size == member.file_size and crc32(path) == member.CRC):
                        with archive.open(member) as stream:
                            copy_member(stream, path, member.file_size)
                    state["files"] += 1; state["uncompressed_bytes"] += member.file_size
                    progress()
            state["archive_sha256"] = sha256(source)
            state["integrity"] = "all ZIP member CRCs verified"
        elif source.name.endswith(".tar.gz"):
            with source.open("rb") as raw:
                reader = HashReader(raw)
                with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
                    free_budget = 0
                    with tarfile.open(fileobj=stream, mode="r|") as archive:
                        for member in archive:
                            if not (member.isfile() or member.isdir()):
                                raise ValueError("TAR contains a link or special file")
                            path = destination(member.name, member.isdir())
                            if member.isdir():
                                path.mkdir(exist_ok=True)
                                continue
                            if state["files"] % 1024 == 0:
                                free_budget = shutil.disk_usage(root).free
                            if member.size > free_budget:
                                raise RuntimeError("Insufficient space for TAR member")
                            free_budget -= member.size
                            with archive.extractfile(member) as member_stream:
                                copy_member(member_stream, path, member.size)
                            state["files"] += 1; state["uncompressed_bytes"] += member.size
                            progress()
                    # tar EOF alone does not guarantee the gzip trailer was read.
                    while stream.read(4 * 1024 * 1024):
                        pass
                state["archive_sha256"] = reader.digest.hexdigest()
            state["integrity"] = "TAR member sizes and complete gzip CRC/trailer verified"
        else:
            raise ValueError("Unsupported archive format")
        if source.stat().st_size != fingerprint["bytes"] or source.stat().st_mtime_ns != fingerprint["mtime_ns"]:
            raise ValueError("Source changed during extraction")
        if len(tops) != 1:
            raise ValueError("Empty archive")
        top = next(iter(tops)); output = root / top
        if output.exists():
            raise ValueError("Refusing to overwrite an existing extracted dataset")
        (staging / top).replace(output)
        marker.unlink(); staging.rmdir()
        state.update(status="complete", output=str(output.resolve()), completed=time.time())
        progress(True)
        return state
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        progress(True)
        raise
