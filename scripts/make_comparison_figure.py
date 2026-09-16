#!/usr/bin/env python3
"""
Qualitative comparison figure: N rows × 3 columns.

    Column 1: GT
    Column 2: HUGS baseline rendered
    Column 3: Our method rendered
    Row i:    i-th val viewpoint (evenly sampled)

The output image is saved inside <our_run_dir>/ as:
    comparison_3way_<mode>_<step>.png   (default name)

Usage
-----
Full-scene mode (default):

    python scripts/make_comparison_figure.py \\
        --run-dir  output/.../our_exp/2026-06-01_xx-xx-xx \\
        --baseline-dir  output/.../hugs_baseline/2026-05-27_xx-xx-xx \\
        --n-frames 4 --step final

Human-crop mode:

    python scripts/make_comparison_figure.py \\
        --run-dir  ... --baseline-dir  ... \\
        --mode human --n-frames 4

Auto-resolve latest timestamp subdir:

    python scripts/make_comparison_figure.py \\
        --run-dir  output/.../our_exp_name \\
        --baseline-dir  output/.../hugs_baseline_exp_name

Custom column headers and baseline step:

    python scripts/make_comparison_figure.py \\
        --run-dir  ... --baseline-dir  ... \\
        --col-labels "GT,HUGS (15K),Ours" \\
        --baseline-step final --step final
"""
import argparse
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

MAKE_GRID_PADDING = 2   # torchvision make_grid default padding


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_run_dir(path: str) -> str:
    """Return the directory that contains a val/ subdir.

    If *path* already has val/, return it. Otherwise look for the latest
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
        f"  Searched pattern: {pattern}"
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def split_grid_image(img: Image.Image):
    """Split a make_grid([gt, rendered], nrow=2, padding=2) image.

    Returns (gt_img, rendered_img) as PIL images.
    """
    W, H = img.size
    p = MAKE_GRID_PADDING
    img_w = (W - 3 * p) // 2
    img_h = H - 2 * p
    gt       = img.crop((p,             p, p + img_w,             p + img_h))
    rendered = img.crop((p + img_w + p, p, p + img_w + p + img_w, p + img_h))
    return gt, rendered


def resize_to_height(img: Image.Image, h: int) -> Image.Image:
    W, H = img.size
    if H == h:
        return img
    return img.resize((round(W * h / H), h), Image.LANCZOS)


def select_evenly(lst: list, n: int) -> list:
    """Pick *n* evenly-spaced items from *lst* (endpoint-inclusive)."""
    if n >= len(lst):
        return lst
    if n == 1:
        return [lst[len(lst) // 2]]
    step = (len(lst) - 1) / (n - 1)
    return [lst[round(i * step)] for i in range(n)]


# ---------------------------------------------------------------------------
# Font / drawing helpers
# ---------------------------------------------------------------------------

def _load_font(size: int = 26):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, text: str,
               x: int, y: int, w: int, h: int, font,
               fg=(255, 255, 255), bg=(50, 50, 50)):
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=bg)
    try:
        bb = font.getbbox(text)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw, th = len(text) * 8, 14
    draw.text(
        (x + max(0, (w - tw) // 2), y + max(0, (h - th) // 2)),
        text, fill=fg, font=font,
    )


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_val_frames(run_dir: str, prefix: str, step: str) -> dict:
    """Return {frame_idx_str: (gt_img, rendered_img)} from val/*.png."""
    val_dir = os.path.join(run_dir, 'val')
    pattern = os.path.join(val_dir, f'{prefix}_{step}_*.png')
    files = sorted(glob.glob(pattern))
    if not files:
        avail = set()
        for f in glob.glob(os.path.join(val_dir, f'{prefix}_*.png')):
            parts = os.path.basename(f).split('_')
            if len(parts) >= 2:
                avail.add(parts[1])
        raise FileNotFoundError(
            f"No files match: {pattern}\n"
            f"  Available steps in val/: {sorted(avail) or '(none)'}"
        )
    frames = {}
    for f in files:
        idx = os.path.basename(f).replace('.png', '').split('_')[-1]
        img = Image.open(f).convert('RGB')
        gt, rendered = split_grid_image(img)
        frames[idx] = (gt, rendered)
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='3-column qualitative comparison: GT | HUGS | Ours, N rows.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--run-dir', required=True,
                    help='Our method run dir (or parent exp dir, auto-picks latest)')
    ap.add_argument('--baseline-dir', required=True,
                    help='HUGS baseline run dir (or parent exp dir)')
    ap.add_argument('--n-frames', type=int, default=4,
                    help='Number of val frames to show (= number of rows) [default: 4]')
    ap.add_argument('--step', default='final',
                    help='Our checkpoint step, e.g. "final" or "006000" [default: final]')
    ap.add_argument('--baseline-step', default=None,
                    help='Baseline checkpoint step [default: same as --step]')
    ap.add_argument('--mode', choices=['full', 'human'], default='full',
                    help='"full": full scene images; "human": human-crop [default: full]')
    ap.add_argument('--col-labels', default='GT,HUGS,Ours',
                    help='Comma-separated column header labels [default: "GT,HUGS,Ours"]')
    ap.add_argument('--header-height', type=int, default=52,
                    help='Column header row height in px; 0 to disable [default: 52]')
    ap.add_argument('--row-gap', type=int, default=6,
                    help='Vertical gap between rows in px [default: 6]')
    ap.add_argument('--col-gap', type=int, default=6,
                    help='Horizontal gap between columns in px [default: 6]')
    ap.add_argument('--human-height', type=int, default=400,
                    help='Common height for human-crop mode [default: 400]')
    ap.add_argument('--output-name', default=None,
                    help='Output filename (saved in our run dir). '
                         'Default: comparison_3way_{mode}_{step}.png')
    args = ap.parse_args()

    baseline_step = args.baseline_step or args.step

    # ---- resolve run dirs ----
    try:
        our_run = find_run_dir(args.run_dir)
    except FileNotFoundError as e:
        sys.exit(f'ERROR (--run-dir): {e}')
    try:
        baseline_run = find_run_dir(args.baseline_dir)
    except FileNotFoundError as e:
        sys.exit(f'ERROR (--baseline-dir): {e}')

    prefix = 'full' if args.mode == 'full' else 'human'

    print(f'Our run:        {our_run}')
    print(f'Baseline run:   {baseline_run}')
    print(f'Mode: {args.mode}  |  Our step: {args.step}  |  Baseline step: {baseline_step}')

    # ---- load frames ----
    try:
        our_frames = load_val_frames(our_run, prefix, args.step)
    except FileNotFoundError as e:
        sys.exit(f'ERROR loading our frames: {e}')
    try:
        base_frames = load_val_frames(baseline_run, prefix, baseline_step)
    except FileNotFoundError as e:
        sys.exit(f'ERROR loading baseline frames: {e}')

    # ---- intersect frame indices ----
    common = sorted(set(our_frames) & set(base_frames))
    if not common:
        sys.exit(
            f'ERROR: No common frame indices between runs.\n'
            f'  Ours:     {sorted(our_frames)}\n'
            f'  Baseline: {sorted(base_frames)}'
        )
    selected = select_evenly(common, args.n_frames)
    n = len(selected)
    print(f'Common frames: {len(common)}  |  Selected: {selected}')

    # ---- build triplets: (gt, hugs_rendered, our_rendered) per row ----
    triplets = []
    for idx in selected:
        gt, _          = our_frames[idx]     # GT is same in both; use ours
        _, hugs_render = base_frames[idx]
        _, our_render  = our_frames[idx]

        if args.mode == 'human':
            h = args.human_height
            gt          = resize_to_height(gt,          h)
            hugs_render = resize_to_height(hugs_render, h)
            our_render  = resize_to_height(our_render,  h)

        triplets.append((gt, hugs_render, our_render))
        print(f'  frame {idx}: {gt.size[0]}×{gt.size[1]}')

    # ---- layout dimensions ----
    # column widths: max image width across all rows for that column
    col_widths = [
        max(triplets[r][c].size[0] for r in range(n))
        for c in range(3)
    ]
    row_height = triplets[0][0].size[1]   # all same height

    col_labels  = [s.strip() for s in args.col_labels.split(',')]
    while len(col_labels) < 3:
        col_labels.append('')

    header_h = args.header_height
    row_gap  = args.row_gap
    col_gap  = args.col_gap
    bg_color = (220, 220, 220)
    font     = _load_font(26)

    total_w = sum(col_widths) + col_gap * 2
    total_h = header_h + row_height * n + row_gap * (n - 1)

    canvas = Image.new('RGB', (total_w, total_h), bg_color)
    draw   = ImageDraw.Draw(canvas)

    # ---- column headers ----
    if header_h > 0:
        x = 0
        for c, (label, cw) in enumerate(zip(col_labels, col_widths)):
            draw_label(draw, label, x, 0, cw, header_h, font)
            x += cw + (col_gap if c < 2 else 0)

    # ---- paste rows ----
    for r, (gt, hugs_r, our_r) in enumerate(triplets):
        y = header_h + r * (row_height + row_gap)
        x = 0
        for c, (img, cw) in enumerate(zip([gt, hugs_r, our_r], col_widths)):
            # center horizontally if image narrower than col width
            offset_x = (cw - img.size[0]) // 2
            canvas.paste(img, (x + offset_x, y))
            x += cw + (col_gap if c < 2 else 0)

    # ---- save ----
    out_name = args.output_name or f'comparison_3way_{args.mode}_{args.step}.png'
    out_path = os.path.join(our_run, out_name)
    canvas.save(out_path, quality=95)
    print(f'\nSaved → {out_path}  ({total_w}×{total_h} px)')


if __name__ == '__main__':
    main()
