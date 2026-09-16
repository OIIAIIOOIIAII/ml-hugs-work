"""Extract provisional RICH images through only aria2-complete prefix pieces.

The decompressor waits at the first hole. Publication requires the final archive
filename, complete gzip CRC/trailer verification and a compressed-file SHA256.
JPEG reads inspect only marker headers, never decode or display image pixels.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tarfile
import time

from .rich_assets import member_path, save_json


def complete_prefix(control, expected_bytes):
    """Parse aria2 v1 control metadata; ignore partial/in-flight piece progress."""
    data = Path(control).read_bytes()
    if len(data) < 34:
        raise ValueError("Incomplete aria2 control header")
    version, extension, hash_length = struct.unpack_from(">HII", data)
    if version != 1 or extension != 0 or hash_length != 0:
        raise ValueError("Unsupported aria2 HTTP control format")
    piece, total, _, length = struct.unpack_from(">IQQI", data, 10)
    if total != expected_bytes or not piece:
        raise ValueError("aria2 resource size/piece length mismatch")
    count = math.ceil(total / piece)
    if length != math.ceil(count / 8) or len(data) < 34 + length:
        raise ValueError("Incomplete aria2 piece bitfield")
    bits = data[34:34 + length]
    first_missing = 0
    for value in bits:
        if value == 255:
            first_missing += 8
            continue
        for bit in range(8):
            if not value & (128 >> bit):
                return min(first_missing * piece, total)
            first_missing += 1
        break
    return min(first_missing * piece, total)


class PiecePrefixReader:
    """A forward-only compressed reader that never returns sparse-hole bytes."""
    def __init__(self, partial, final, control, expected_bytes, *, wait_hours=36,
                 poll_seconds=10, on_wait=None):
        self.partial, self.final, self.control = Path(partial), Path(final), Path(control)
        self.total = expected_bytes
        self.deadline = time.monotonic() + wait_hours * 3600
        self.poll_seconds, self.on_wait = poll_seconds, on_wait
        self.path = self.final if self.final.is_file() else self.partial
        # BufferedReader may prefetch sparse-hole zeros beyond the requested
        # completed boundary, then return stale zeros after aria2 fills the hole.
        # Every underlying read must therefore obey our bound exactly.
        self.file = self.path.open("rb", buffering=0)
        info = self.path.stat()
        self.identity = (info.st_dev, info.st_ino)
        self.position = self.limit = 0
        self.digest = hashlib.sha256()
        self.final_seen = False

    def close(self):
        self.file.close()

    def refresh(self):
        path = self.final if self.final.is_file() else self.partial
        info = path.stat()
        if (info.st_dev, info.st_ino) != self.identity:
            raise ValueError("Download file was replaced during streaming")
        if path == self.final:
            if info.st_size != self.total:
                raise ValueError("Final archive has unexpected size")
            self.final_seen = True
            limit = self.total
        else:
            # aria2 normally replaces/saves its control file atomically. Avoid a
            # snapshot whose size/mtime changes while it is being read.
            before = self.control.stat()
            limit = complete_prefix(self.control, self.total)
            after = self.control.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                return
        if limit < self.limit:
            raise ValueError("Completed prefix regressed; refuse potentially rewritten input")
        if info.st_size < limit:
            raise ValueError("Source shorter than its completed piece prefix")
        self.limit = limit

    def read(self, size=-1):
        if size == 0:
            return b""
        if size < 0:
            size = 1024 * 1024
        while self.position >= self.limit:
            try:
                self.refresh()
            except FileNotFoundError:
                # A final-name rename or control-file replacement can briefly
                # leave one of the paths absent. Never interpret this as EOF.
                pass
            if self.position < self.limit:
                break
            if self.position == self.total and self.final_seen:
                return b""
            if self.on_wait:
                self.on_wait(self)
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Streaming wait deadline reached; restart to replay safely")
            time.sleep(self.poll_seconds)
        data = self.file.read(min(size, self.limit - self.position))
        if not data:
            raise ValueError("Unexpected EOF inside an aria2-complete prefix")
        self.position += len(data)
        self.digest.update(data)
        return data


def jpeg_dimensions(path):
    """Read JPEG SOF width/height only; skip metadata and never decode pixels."""
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("Expected JPEG SOI marker")
        while stream.tell() < 4 * 1024 * 1024:
            if stream.read(1) != b"\xff":
                raise ValueError("Invalid JPEG header marker")
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if not marker or marker[0] in (0x00, 0xDA, 0xD9):
                raise ValueError("JPEG has no SOF before scan/end")
            if marker[0] == 0x01 or 0xD0 <= marker[0] <= 0xD7:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError("Truncated JPEG header")
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise ValueError("Invalid JPEG segment length")
            if marker[0] in sof:
                header = stream.read(6)
                if len(header) != 6:
                    raise ValueError("Truncated JPEG SOF")
                precision, height, width, components = struct.unpack(">BHHB", header)
                if not width or not height or not components or length != 8 + 3 * components:
                    raise ValueError("Invalid JPEG SOF dimensions")
                return {"width": width, "height": height, "precision_bits": precision, "components": components}
            stream.seek(length - 2, 1)
    raise ValueError("JPEG header exceeds inspection budget")


def image_identity(relative, tracks):
    match = re.fullmatch(r"([^/]+)/cam_(\d{2})/(\d{5,})_(\d{2})\.(?:jpeg|jpg)", relative)
    if not match:
        return {"recognized_image_path": False}
    sequence, directory_camera, frame, file_camera = match.groups()
    if directory_camera != file_camera:
        raise ValueError("JPEG filename camera differs from directory camera")
    frame, camera = int(frame), int(file_camera)
    rows = tracks.get(sequence, [])
    return {"recognized_image_path": True, "sequence": sequence, "camera_id": camera, "frame_id": frame,
            "sequence_in_annotation_catalog": bool(rows),
            "subjects_with_contact": [row["subject"] for row in rows if frame in row["frames"]],
            "subjects_body_only": [row["subject"] for row in rows if frame in row["body_only_frames"]],
            "camera_in_official_metadata": any(camera in row["camera_ids"] for row in rows)}


def stream_jpg(root, *, expected_bytes=559418993166, wait_hours=36, poll_seconds=10):
    """Stream into an isolated stage, then publish after complete gzip validation.

Caller must hold processing_logs/processing.lock, also used by the full archive
processor. Restarting replays gzip from the beginning and rewrites owned staged
members; DEFLATE state is not assumed seekable. Original archives are untouched.
"""
    root = Path(root)
    logs = root / "processing_logs"; logs.mkdir(parents=True, exist_ok=True)
    state_path = logs / "train_jpg.json"
    staging = root / "extracted/.staging/train_jpg_stream"
    output = root / "extracted/train"
    index_root = root / "processed/images_v1"
    partial, final = root / "raw/train/train.tar.gz.part", root / "raw/train/train.tar.gz"
    control = partial.with_name(partial.name + ".aria2")
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("status") == "complete" and Path(previous["output"]).is_dir():
            return previous
        # Recover the narrow crash window after validated directory publication.
        if previous.get("status") == "publishing" and output.is_dir() and not (staging / "train").exists():
            if previous.get("integrity") != "TAR member sizes and complete gzip CRC/trailer verified":
                raise ValueError("Cannot recover unverified image publication")
            previous.update(status="complete", output=str(output.resolve()), completed=time.time())
            save_json(state_path, previous)
            return previous
    if output.exists():
        raise ValueError("Refusing to overwrite an existing published image tree")
    catalog = json.loads((root / "processed/annotations_v1/catalog.json").read_text())
    tracks = {}
    for row in catalog["tracks"]:
        row = {**row, "frames": set(row["frames"]), "body_only_frames": set(row["body_only_frames"])}
        tracks.setdefault(row["sequence"], []).append(row)
    source_info = (final if final.exists() else partial).stat()
    owner = {"kind": "rich_streaming_jpg_v1", "expected_bytes": expected_bytes,
             "source_device": source_info.st_dev, "source_inode": source_info.st_ino,
             "annotation_catalog_sha256": hashlib.sha256((root / "processed/annotations_v1/catalog.json").read_bytes()).hexdigest()}
    marker = staging / ".source.json"
    if staging.exists() and (not marker.is_file() or json.loads(marker.read_text()) != owner):
        raise ValueError("Unowned or mismatched streaming staging directory")
    staging.mkdir(parents=True, exist_ok=True); save_json(marker, owner)
    index_root.mkdir(parents=True, exist_ok=True)
    state = {"asset": "train_jpg", "mode": "aria2_prefix_stream", "status": "extracting_provisional",
             "started": time.time(), "files": 0, "image_files": 0, "uncompressed_bytes": 0,
             "training_ready": False, "archive_integrity": "pending full gzip CRC/trailer",
             "images_root": str((staging / "train").resolve()),
             "index": str((index_root / "images.provisional.jsonl").resolve()),
             "image_dimensions": {}, "sequences": {}, "images_with_contact_subject": 0,
             "images_outside_metadata_cameras": 0, "unrecognized_paths": 0}
    last_save = 0.0

    def progress(force=False, waiting=False):
        nonlocal last_save
        if force or time.monotonic() - last_save >= 10:
            state.update(updated=time.time(), compressed_bytes_read=reader.position,
                         complete_prefix_bytes=reader.limit,
                         status="waiting_for_download_piece" if waiting else "extracting_provisional")
            save_json(state_path, state); last_save = time.monotonic()

    reader = PiecePrefixReader(partial, final, control, expected_bytes, wait_hours=wait_hours,
                               poll_seconds=poll_seconds, on_wait=lambda _: progress(True, True))
    seen = set()
    try:
        if reader.identity != (owner["source_device"], owner["source_inode"]):
            raise ValueError("Source replaced before extraction started")
        progress(True)
        index_tmp = index_root / "images.provisional.jsonl"
        with index_tmp.open("w") as index, gzip.GzipFile(fileobj=reader, mode="rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                for member in archive:
                    relative = member_path(member.name)
                    if relative.parts[0] != "train" or not (member.isdir() or member.isfile()) or member.sparse or member.size < 0 or (member.isfile() and len(relative.parts) < 2):
                        raise ValueError("Unexpected image archive root, link, or special member")
                    target = staging.joinpath(*relative.parts)
                    for parent in (target, *target.parents):
                        if parent == staging.parent:
                            break
                        if parent.is_symlink():
                            raise ValueError("Symlink in streaming extraction destination")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if relative in seen:
                        raise ValueError("Duplicate file in image archive")
                    seen.add(relative)
                    if state["files"] % 512 == 0:
                        free_budget = shutil.disk_usage(staging).free
                    if member.size > free_budget:
                        raise RuntimeError("Insufficient space for image extraction")
                    free_budget -= member.size
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".extracting")
                    digest = hashlib.sha256(); size = 0
                    with archive.extractfile(member) as source, temporary.open("wb") as dest:
                        while True:
                            data = source.read(1024 * 1024)
                            if not data:
                                break
                            size += len(data); digest.update(data); dest.write(data)
                    if size != member.size:
                        raise ValueError("Truncated TAR member; keep it provisional")
                    record = {"path": str(relative.relative_to("train")), "bytes": size, "sha256": digest.hexdigest()}
                    if target.suffix.lower() in (".jpeg", ".jpg"):
                        record.update(jpeg_dimensions(temporary))
                        identity = image_identity(record["path"], tracks); record.update(identity)
                        state["image_files"] += 1
                        dimensions = f"{record['width']}x{record['height']}"
                        state["image_dimensions"][dimensions] = state["image_dimensions"].get(dimensions, 0) + 1
                        if identity["recognized_image_path"]:
                            seq = identity["sequence"]
                            state["sequences"][seq] = state["sequences"].get(seq, 0) + 1
                            state["images_with_contact_subject"] += bool(identity["subjects_with_contact"])
                            state["images_outside_metadata_cameras"] += not identity["camera_in_official_metadata"]
                        else:
                            state["unrecognized_paths"] += 1
                    temporary.replace(target)
                    index.write(json.dumps(record) + "\n"); index.flush()
                    state["files"] += 1; state["uncompressed_bytes"] += size
                    progress()
            while compressed.read(4 * 1024 * 1024):
                pass
        if reader.position != expected_bytes or not reader.final_seen:
            raise ValueError("Full final archive has not been consumed and verified")
        info = final.stat()
        if (info.st_dev, info.st_ino) != reader.identity or info.st_size != expected_bytes:
            raise ValueError("Final archive changed during verification")
        index_final = index_root / "images.jsonl"
        index_tmp.replace(index_final)
        state.update(status="publishing", source=str(final.resolve()),
                     source_fingerprint={"bytes": info.st_size, "mtime_ns": info.st_mtime_ns},
                     archive_sha256=reader.digest.hexdigest(), archive_integrity="verified",
                     integrity="TAR member sizes and complete gzip CRC/trailer verified",
                     compressed_bytes_read=reader.position, complete_prefix_bytes=reader.limit,
                     index=str(index_final.resolve()), images_root=str(output.resolve()), updated=time.time())
        save_json(state_path, state)
        (staging / "train").replace(output)
        state.update(status="complete", output=str(output.resolve()), completed=time.time(), updated=time.time())
        save_json(state_path, state)
        marker.unlink(); staging.rmdir()
        return state
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated=time.time())
        save_json(state_path, state)
        raise
    finally:
        reader.close()
