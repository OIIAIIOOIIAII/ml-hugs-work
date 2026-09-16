"""File-mode package/unpackage for the first end-to-end validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from codec import encode_frame, decode_frame, write_json


def pack(input_path: Path, out_dir: Path) -> None:
    sequence = np.load(input_path, allow_pickle=False)
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError("input must have shape [T, N, 3]")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    previous = None
    total_payload = 0
    for frame_id, delta in enumerate(sequence):
        header, payload, previous = encode_frame(delta, previous)
        header["frame_id"] = frame_id
        (frames_dir / f"{frame_id:06d}.bin").write_bytes(payload)
        write_json(frames_dir / f"{frame_id:06d}.json", header)
        total_payload += len(payload)
    write_json(out_dir / "manifest.json", {
        "kind": "hugs_transmission_package",
        "version": 1,
        "frames": int(sequence.shape[0]),
        "delta_shape": list(sequence.shape[1:]),
        "source": str(input_path),
        "payload_bytes": total_payload,
    })
    print(f"packed {len(sequence)} frames -> {out_dir} ({total_payload} payload bytes)")


def unpack(package_dir: Path, output_path: Path) -> None:
    manifest = json.loads((package_dir / "manifest.json").read_text())
    restored = []
    previous = None
    for frame_id in range(int(manifest["frames"])):
        frame_dir = package_dir / "frames"
        header = json.loads((frame_dir / f"{frame_id:06d}.json").read_text())
        payload = (frame_dir / f"{frame_id:06d}.bin").read_bytes()
        previous = decode_frame(header, payload, previous)
        restored.append(previous.copy())
    result = np.stack(restored, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result, allow_pickle=False)
    print(f"unpacked {len(restored)} frames -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--input", type=Path, required=True)
    p_pack.add_argument("--out", type=Path, required=True)
    p_unpack = sub.add_parser("unpack")
    p_unpack.add_argument("--package", type=Path, required=True)
    p_unpack.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.input, args.out)
    else:
        unpack(args.package, args.out)


if __name__ == "__main__":
    main()

