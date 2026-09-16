from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def frame_paths(sequence_dir: str | Path) -> List[Path]:
    sequence_dir = Path(sequence_dir)
    manifest_path = sequence_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        paths = [sequence_dir / record["file"] for record in manifest.get("frames", [])]
        if paths:
            return paths
    return sorted((sequence_dir / "frames").glob("*.npz"))


def load_frame(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz(path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def finite_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    return np.isfinite(array).all(axis=-1)


def stack_records(records: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    records = list(records)
    if not records:
        raise ValueError("Cannot stack an empty record list")
    keys = records[0].keys()
    return {key: np.stack([record[key] for record in records]) for key in keys}
