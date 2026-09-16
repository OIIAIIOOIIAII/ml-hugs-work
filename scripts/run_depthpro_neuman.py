"""
Run DepthPro on all NeuMan sequences to generate metric depth maps.

Output per frame (in data/neuman/dataset/<seq>/depth_pro/):
  <frame>.npy   - float32 metric depth in meters
  <frame>.png   - uint16 PNG (depth_m / max_depth * 65535) for visualization
  metadata.json - per-sequence: f_px used, depth stats
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# DepthPro repo path
DEPTH_PRO_ROOT = Path(__file__).parent.parent.parent / "ml-depth-pro"
sys.path.insert(0, str(DEPTH_PRO_ROOT / "src"))

import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig

CHECKPOINT_PATH = DEPTH_PRO_ROOT / "checkpoints" / "depth_pro.pt"
DATA_ROOT = Path(__file__).parent.parent / "data/neuman/dataset"
ALL_SEQS = ["bike", "citron", "jogging", "lab", "parkinglot", "seattle"]


def load_colmap_intrinsics(seq_dir: Path):
    """Read focal length from COLMAP cameras.txt."""
    cameras_txt = seq_dir / "sparse" / "cameras.txt"
    if not cameras_txt.exists():
        return None
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            # PINHOLE: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
            if len(parts) >= 5 and parts[1] in ("PINHOLE", "SIMPLE_PINHOLE", "OPENCV"):
                fx = float(parts[4])
                return fx
    return None


def run_depthpro_on_seq(model, transform, seq: str, data_root: Path, device, args):
    seq_dir = data_root / seq
    img_dir = seq_dir / "images"
    out_dir = seq_dir / "depth_pro"
    out_dir.mkdir(exist_ok=True)

    img_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not img_files:
        print(f"  [skip] {seq}: no images found")
        return

    # Get focal length from COLMAP if available
    f_px = load_colmap_intrinsics(seq_dir)
    if f_px is not None:
        print(f"  focal length from COLMAP: {f_px:.2f} px")
        f_px_tensor = torch.tensor(f_px, device=device)
    else:
        print(f"  no COLMAP intrinsics, using DepthPro auto-estimate")
        f_px_tensor = None

    depth_stats = []

    for img_path in tqdm(img_files, desc=f"  {seq}", ncols=80):
        stem = img_path.stem
        npy_path = out_dir / f"{stem}.npy"
        png_path = out_dir / f"{stem}.png"

        if npy_path.exists() and png_path.exists() and not args.overwrite:
            continue

        # Load and preprocess image
        image_pil = Image.open(img_path).convert("RGB")
        image_tensor = transform(image_pil).to(device)

        with torch.no_grad():
            pred = model.infer(image_tensor, f_px=f_px_tensor)

        depth_m = pred["depth"].squeeze().cpu().float().numpy()  # (H, W) float32 meters

        # Save float32 npy (metric depth)
        np.save(npy_path, depth_m)

        # Save uint16 PNG: normalize by 98th percentile to preserve detail
        p98 = np.percentile(depth_m, 98)
        depth_uint16 = np.clip(depth_m / max(p98, 1e-6) * 60000, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_uint16).save(png_path)

        depth_stats.append({
            "frame": stem,
            "min_m": float(depth_m.min()),
            "max_m": float(depth_m.max()),
            "mean_m": float(depth_m.mean()),
        })

    # Save metadata
    meta = {
        "seq": seq,
        "f_px_used": float(f_px) if f_px is not None else None,
        "n_frames": len(depth_stats),
        "depth_stats": depth_stats,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  done: {len(depth_stats)} frames -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="+", default=ALL_SEQS,
                        help="sequences to process (default: all)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run even if output already exists")
    parser.add_argument("--fp32", action="store_true",
                        help="use fp32 instead of fp16 (slower, more VRAM)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.float32 if args.fp32 else torch.float16
    print(f"Device: {device}, precision: {precision}")

    print("Loading DepthPro model...")
    config = DepthProConfig(
        patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
        image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
        decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
        use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
        fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
        checkpoint_uri=str(CHECKPOINT_PATH),
    )
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=device,
        precision=precision,
    )
    model.eval()
    print("Model loaded.\n")

    for seq in args.seqs:
        print(f"[{seq}]")
        run_depthpro_on_seq(model, transform, seq, args.data_root, device, args)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
