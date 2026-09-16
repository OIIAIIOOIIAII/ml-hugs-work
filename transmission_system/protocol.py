"""Wire protocol shared by file and localhost TCP transport modes."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, BinaryIO


MAGIC = b"HUGS"
VERSION = 1
_PREFIX = struct.Struct("!4sBI")  # magic, version, json-header length


def pack_message(header: dict[str, Any], payload: bytes = b"") -> bytes:
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _PREFIX.pack(MAGIC, VERSION, len(body)) + body + payload


def send_message(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    sock.sendall(pack_message(header, payload))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("unexpected end of transmission")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(stream: BinaryIO) -> tuple[dict[str, Any], bytes]:
    prefix = _read_exact(stream, _PREFIX.size)
    magic, version, header_size = _PREFIX.unpack(prefix)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"unsupported packet prefix: magic={magic!r}, version={version}")
    header = json.loads(_read_exact(stream, header_size).decode("utf-8"))
    payload_size = int(header.get("payload_size", 0))
    return header, _read_exact(stream, payload_size) if payload_size else b""

