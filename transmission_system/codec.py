"""Causal delta_mu codec used by both file and TCP experiments."""

from __future__ import annotations

import io
import json
import zlib
from typing import Any

import numpy as np


def _array_to_npy(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def _npy_to_array(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def encode_frame(delta: np.ndarray, previous: np.ndarray | None) -> tuple[dict[str, Any], bytes, np.ndarray]:
    delta = np.asarray(delta, dtype=np.float32)
    if previous is None:
        signal = delta
        keyframe = True
    else:
        signal = delta - previous
        keyframe = False

    peak = float(np.max(np.abs(signal)))
    scale = max(peak / 127.0, 1e-12)
    quantized = np.clip(np.rint(signal / scale), -127, 127).astype(np.int8)
    compressed = zlib.compress(_array_to_npy(quantized), level=6)
    header = {
        "kind": "frame",
        "keyframe": keyframe,
        "shape": list(delta.shape),
        "dtype": "float32",
        "quant_scale": scale,
        "raw_bytes": int(quantized.nbytes),
        "payload_size": len(compressed),
    }
    return header, compressed, delta


def decode_frame(header: dict[str, Any], payload: bytes, previous: np.ndarray | None) -> np.ndarray:
    quantized = _npy_to_array(zlib.decompress(payload)).astype(np.float32)
    signal = quantized * float(header["quant_scale"])
    if bool(header["keyframe"]):
        result = signal
    else:
        if previous is None:
            raise ValueError("delta frame received before a keyframe")
        result = previous + signal
    return result.reshape(tuple(header["shape"])).astype(np.float32, copy=False)


def write_json(path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

