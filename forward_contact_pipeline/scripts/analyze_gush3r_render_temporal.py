#!/usr/bin/env python3
"""Numerically audit GUSH3R rendered frames without producing visual output.

The script is intentionally renderer-agnostic.  It checks a complete numbered
PNG sequence, computes simple artifact proxies and separates transitions at
declared chunk boundaries from normal temporal transitions.  Image arrays stay
inside this process; only aggregate JSON statistics are written.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _frame_path(render_dir: Path, index: int) -> Path:
    return render_dir / f"frame_{index:06d}.png"


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode render frame: {path}")
    return image[..., ::-1].astype(np.float32) / 255.0


def _chunk_boundaries(manifest: Path | None) -> list[int]:
    if manifest is None:
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [
        int(chunk["start"])
        for chunk in payload.get("chunks", [])
        if int(chunk["start"]) > 0
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--num-frames", required=True, type=int)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--reference-render-dir",
        type=Path,
        default=None,
        help="Optional same-index render directory for aggregate frame difference only.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_dir = args.render_dir / "merged_render" if (args.render_dir / "merged_render").is_dir() else args.render_dir
    reference_dir = None
    if args.reference_render_dir is not None:
        reference_dir = args.reference_render_dir / "merged_render"
        if not reference_dir.is_dir():
            reference_dir = args.reference_render_dir

    boundaries = set(_chunk_boundaries(args.manifest))
    missing = []
    white_rates, black_rates, luminance_means, luminance_stds = [], [], [], []
    transition_l1, boundary_l1, interior_l1, reference_l1 = [], [], [], []
    previous = None
    shape = None

    for index in range(args.num_frames):
        path = _frame_path(render_dir, index)
        if not path.is_file():
            missing.append(index)
            continue
        current = _load_rgb(path)
        if shape is None:
            shape = list(current.shape)
        elif list(current.shape) != shape:
            raise RuntimeError(f"Frame shape changed at {index}: {current.shape} != {tuple(shape)}")

        luminance = 0.2126 * current[..., 0] + 0.7152 * current[..., 1] + 0.0722 * current[..., 2]
        white_rates.append(float(np.all(current >= 0.98, axis=-1).mean()))
        black_rates.append(float(np.all(current <= 0.05, axis=-1).mean()))
        luminance_means.append(float(luminance.mean()))
        luminance_stds.append(float(luminance.std()))

        if previous is not None:
            value = float(np.abs(current - previous).mean())
            transition_l1.append(value)
            (boundary_l1 if index in boundaries else interior_l1).append(value)
        previous = current

        if reference_dir is not None:
            reference_path = _frame_path(reference_dir, index)
            if reference_path.is_file():
                reference = _load_rgb(reference_path)
                if reference.shape != current.shape:
                    raise RuntimeError(f"Reference shape mismatch at {index}: {reference.shape} != {current.shape}")
                reference_l1.append(float(np.abs(current - reference).mean()))

    payload = {
        "protocol": "aggregate pixel statistics only; no visual output",
        "render_dir": str(render_dir),
        "num_frames_requested": args.num_frames,
        "num_frames_found": args.num_frames - len(missing),
        "missing_frame_indices": missing,
        "frame_shape_hwc": shape,
        "chunk_boundary_start_indices": sorted(boundaries),
        "white_pixel_rate": _summarize(white_rates),
        "black_pixel_rate": _summarize(black_rates),
        "luminance_mean": _summarize(luminance_means),
        "luminance_std": _summarize(luminance_stds),
        "temporal_l1_all": _summarize(transition_l1),
        "temporal_l1_chunk_boundaries": _summarize(boundary_l1),
        "temporal_l1_interior": _summarize(interior_l1),
        "same_index_reference_l1": _summarize(reference_l1),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
