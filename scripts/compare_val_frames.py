#!/usr/bin/env python3
"""
Compare GT vs rendered output for a given training run.

Produces a single stitched image: N rows × 2 columns (GT | Rendered).
The N val frames are selected evenly from those saved at a given step.

Usage:
    python scripts/compare_val_frames.py \
        --run_dir output/human_scene/neuman/seattle/hugs_trimlp/depth_sup12k.../2026-05-29... \
        --n_frames 4 --step final --output comparison_seattle.png

    # Auto-find the latest timestamp dir under an exp_name dir:
    python scripts/compare_val_frames.py \
        --run_dir output/human_scene/neuman/seattle/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_seattle_20260529 \
        --n_frames 4 --output comparison_seattle.png

    # Human-crop mode (variable bbox size per frame, resized to common height):
    python scripts/compare_val_frames.py \
        --run_dir ... --n_frames 4 --mode human --output comparison_human.png
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# torchvision.utils.make_grid default padding (px on each edge and between cells)
MAKE_GRID_PADDING = 2


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_run_dir(path: str) -> str:
    """Return a directory that contains a val/ subdir.

    If *path* already has val/, return it.  Otherwise look for the latest
    YYYY-MM-DD_HH-MM-SS timestamp subdir inside *path*.
    """
    if os.path.isdir(os.path.join(path, 'val')):
        return path
    pattern = os.path.join(path, '????-??-??_??-??-??')
    subdirs = sorted(glob.glob(pattern))
    if subdirs:
        return subdirs[-1]
    raise FileNotFoundError(
        f"Cannot find a run directory with val/ under: {path}\n"
        f"Searched pattern: {pattern}"
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def split_grid_image(img: Image.Image):
    """Split a make_grid([gt, render], nrow=2, padding=2) image.

    Returns (gt_img, rendered_img) as PIL images with the border stripped.
    """
    W, H = img.size
    p = MAKE_GRID_PADDING
    # Grid layout: p | img_w | p | img_w | p  (width)
    #              p | img_h | p              (height, 1 row)
    img_w = (W - 3 * p) // 2
    img_h = H - 2 * p
    gt = img.crop((p, p, p + img_w, p + img_h))
    rendered = img.crop((p + img_w + p, p, p + img_w + p + img_w, p + img_h))
    return gt, rendered


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize preserving aspect ratio to target height."""
    W, H = img.size
    if H == target_h:
        return img
    new_w = round(W * target_h / H)
    return img.resize((new_w, target_h), Image.LANCZOS)


def select_evenly(lst: list, n: int) -> list:
    """Pick *n* evenly-spaced items from *lst* (endpoint-inclusive)."""
    if n >= len(lst):
        return lst
    if n == 1:
        return [lst[len(lst) // 2]]
    step = (len(lst) - 1) / (n - 1)
    return [lst[round(i * step)] for i in range(n)]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _load_font(size: int = 22):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, text: str,
               x: int, y: int, w: int, h: int,
               font, fg=(255, 255, 255), bg=(60, 60, 60)):
    """Fill a rectangle and center text inside it."""
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=bg)
    try:
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = len(text) * 8, 14
    tx = x + max(0, (w - tw) // 2)
    ty = y + max(0, (h - th) // 2)
    draw.text((tx, ty), text, fill=fg, font=font)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Create a GT vs Rendered comparison grid for a training run.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--run_dir', required=True,
                        help='Timestamp run dir (or parent exp_name dir, auto-picks latest)')
    parser.add_argument('--n_frames', type=int, default=4,
                        help='Number of val frames to show (= number of rows) [default: 4]')
    parser.add_argument('--step', default='final',
                        help='Checkpoint step: "final" or e.g. "001000" [default: final]')
    parser.add_argument('--mode', choices=['full', 'human'], default='full',
                        help='"full" uses full-scene images; "human" uses human-crop images [default: full]')
    parser.add_argument('--output', default='comparison.png',
                        help='Output image path [default: comparison.png]')
    parser.add_argument('--row_gap', type=int, default=6,
                        help='Gap in pixels between rows [default: 6]')
    parser.add_argument('--col_gap', type=int, default=6,
                        help='Gap in pixels between GT and Rendered columns [default: 6]')
    parser.add_argument('--header_height', type=int, default=52,
                        help='Column header row height in pixels; 0 to disable [default: 52]')
    parser.add_argument('--human_height', type=int, default=400,
                        help='Common height for human-crop frames (--mode human) [default: 400]')
    args = parser.parse_args()

    # --- locate run directory ---
    try:
        run_dir = find_run_dir(args.run_dir)
    except FileNotFoundError as e:
        sys.exit(f'ERROR: {e}')

    val_dir = os.path.join(run_dir, 'val')
    prefix = 'full' if args.mode == 'full' else 'human'
    pattern = os.path.join(val_dir, f'{prefix}_{args.step}_*.png')
    files = sorted(glob.glob(pattern))

    if not files:
        sys.exit(
            f'ERROR: No files found matching: {pattern}\n'
            f'Available steps: {set(os.path.basename(f).split("_")[1] for f in glob.glob(os.path.join(val_dir, f"{prefix}_*.png")))}'
        )

    selected = select_evenly(files, args.n_frames)
    n = len(selected)
    print(f'Run dir  : {run_dir}')
    print(f'Mode     : {args.mode}  |  Step: {args.step}  |  Frames: {n}/{len(files)} available')

    # --- load and split images ---
    pairs = []  # list of (gt_img, rendered_img)
    for f in selected:
        img = Image.open(f).convert('RGB')
        if args.mode == 'full':
            gt, rendered = split_grid_image(img)
        else:
            # human_*.png: same make_grid layout but variable size
            gt, rendered = split_grid_image(img)
            gt = resize_to_height(gt, args.human_height)
            rendered = resize_to_height(rendered, args.human_height)
        pairs.append((gt, rendered))
        frame_idx = os.path.basename(f).split('_')[-1].replace('.png', '')
        print(f'  frame {frame_idx}: GT {gt.size[0]}×{gt.size[1]}, Rendered {rendered.size[0]}×{rendered.size[1]}')

    # --- determine column width ---
    # For full mode all frames are the same size; for human mode we padded to same height
    if args.mode == 'full':
        col_w = pairs[0][0].size[0]
        col_h = pairs[0][0].size[1]
        # all rows same height → simple grid
        row_heights = [col_h] * n
    else:
        # all GT/Rendered already resized to same height; widths may differ per frame
        col_w = max(max(gt.size[0], rd.size[0]) for gt, rd in pairs)
        col_h = args.human_height
        row_heights = [col_h] * n

    header_h = args.header_height
    row_gap = args.row_gap
    col_gap = args.col_gap
    bg_color = (230, 230, 230)

    total_w = col_w * 2 + col_gap
    total_h = header_h + sum(row_heights) + row_gap * (n - 1)

    canvas = Image.new('RGB', (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(24)

    # --- column headers ---
    if header_h > 0:
        draw_label(draw, 'GT', 0, 0, col_w, header_h, font)
        draw_label(draw, 'Rendered', col_w + col_gap, 0, col_w, header_h, font)

    # --- paste frames ---
    y = header_h
    for i, (gt, rendered) in enumerate(pairs):
        rh = row_heights[i]
        # center horizontally within col_w (for human mode where widths vary)
        gt_x = (col_w - gt.size[0]) // 2
        rd_x = col_w + col_gap + (col_w - rendered.size[0]) // 2
        canvas.paste(gt, (gt_x, y))
        canvas.paste(rendered, (rd_x, y))
        y += rh + (row_gap if i < n - 1 else 0)

    # --- save ---
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f'\nSaved → {args.output}  ({total_w}×{total_h} px)')


if __name__ == '__main__':
    main()
