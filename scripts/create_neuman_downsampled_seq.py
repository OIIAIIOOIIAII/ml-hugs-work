#!/usr/bin/env python3
"""Create a downsampled NeuMan sequence while preserving the original layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


IMAGE_DIRS = {"images"}
MASK_DIRS = {"masks", "sam_segmentations", "segmentations"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Source NeuMan sequence directory.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination sequence directory.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def symlink_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def resize_image_dir(src: Path, dst: Path, size: tuple[int, int], is_mask: bool) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    count = 0
    for img_path in sorted(src.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            symlink_path(img_path, dst / img_path.name)
            continue
        with Image.open(img_path) as img:
            if is_mask:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = img.resize(size, resample=resample)
            img.save(dst / img_path.name)
        count += 1
    return count


def scale_camera_params(model: str, params: list[float], sx: float, sy: float) -> list[float]:
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return [f * (sx + sy) * 0.5, cx * sx, cy * sy, *params[3:]]
    if model == "SIMPLE_RADIAL":
        f, cx, cy, *rest = params
        return [f * (sx + sy) * 0.5, cx * sx, cy * sy, *rest]
    if model in {"PINHOLE", "OPENCV"}:
        fx, fy, cx, cy, *rest = params
        return [fx * sx, fy * sy, cx * sx, cy * sy, *rest]
    raise ValueError(f"Unsupported camera model: {model}")


def fmt_float(value: float) -> str:
    return f"{value:.10g}"


def scale_cameras_txt(src: Path, dst: Path, width: int, height: int, sx: float, sy: float) -> None:
    out_lines: list[str] = []
    for line in src.read_text().splitlines():
        if not line or line.startswith("#"):
            out_lines.append(line)
            continue
        elems = line.split()
        camera_id, model = elems[0], elems[1]
        params = [float(x) for x in elems[4:]]
        scaled = scale_camera_params(model, params, sx, sy)
        out_lines.append(
            " ".join([camera_id, model, str(width), str(height), *[fmt_float(x) for x in scaled]])
        )
    dst.write_text("\n".join(out_lines) + "\n")


def scale_images_txt(src: Path, dst: Path, sx: float, sy: float) -> None:
    out_lines: list[str] = []
    expect_points_line = False
    for line in src.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            out_lines.append(line)
            continue

        elems = line.split()
        if expect_points_line:
            if len(elems) % 3 != 0:
                raise ValueError(f"Unexpected POINTS2D line in {src}: {line[:120]}")
            scaled: list[str] = []
            for i in range(0, len(elems), 3):
                scaled.append(fmt_float(float(elems[i]) * sx))
                scaled.append(fmt_float(float(elems[i + 1]) * sy))
                scaled.append(elems[i + 2])
            out_lines.append(" ".join(scaled))
            expect_points_line = False
        else:
            if len(elems) != 10:
                raise ValueError(f"Unexpected image metadata line in {src}: {line[:120]}")
            out_lines.append(line)
            expect_points_line = True
    dst.write_text("\n".join(out_lines) + "\n")


def build_sparse(src: Path, dst: Path, width: int, height: int, sx: float, sy: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        target = dst / item.name
        if item.name == "cameras.txt":
            scale_cameras_txt(item, target, width, height, sx, sy)
        elif item.name == "images.txt":
            scale_images_txt(item, target, sx, sy)
        else:
            symlink_path(item, target)


def build_4d_humans(src: Path, dst: Path, size: tuple[int, int]) -> dict[str, int]:
    dst.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for item in sorted(src.iterdir()):
        target = dst / item.name
        if item.is_dir() and item.name in MASK_DIRS:
            counts[item.name] = resize_image_dir(item, target, size, is_mask=True)
        elif item.is_dir() and item.name in IMAGE_DIRS:
            counts[item.name] = resize_image_dir(item, target, size, is_mask=False)
        else:
            symlink_path(item, target)
    return counts


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    size = (args.width, args.height)

    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    with Image.open(next(iter(sorted((src / "images").glob("*"))))) as first:
        src_width, src_height = first.size
    sx = args.width / src_width
    sy = args.height / src_height

    stats: dict[str, object] = {
        "src": str(src),
        "dst": str(dst),
        "src_size": [src_width, src_height],
        "dst_size": [args.width, args.height],
        "scale": [sx, sy],
    }

    for item in sorted(src.iterdir()):
        target = dst / item.name
        if item.name == "images":
            stats["images"] = resize_image_dir(item, target, size, is_mask=False)
        elif item.name == "sparse":
            build_sparse(item, target, args.width, args.height, sx, sy)
            stats["sparse"] = "scaled cameras.txt/images.txt; linked other files"
        elif item.name == "4d_humans":
            stats["4d_humans"] = build_4d_humans(item, target, size)
        else:
            symlink_path(item, target)

    (dst / "downsample_metadata.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
