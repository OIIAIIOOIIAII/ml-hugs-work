#!/usr/bin/env python3
"""Render a full-sequence video from a HUGS run directory.

Renders every frame in the all_dataset (all 103 frames for NeuMan lab), applies
anchor-attention correction when a checkpoint is found, and writes an mp4.

Usage:
    python scripts/render_full_video.py --run-dir <path/to/run> [--fps 10] [--keep-frames]

The output video is written to:
    <run-dir>/render_all_<dataset>_<seq>_final.mp4  (or _<iter>.mp4 when --iter is given)
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="HUGS training output directory (contains config_train.yaml and ckpt/).")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second of the output video. Default 10 (103 frames ≈ 10 s).")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep per-frame PNG images under <run-dir>/render_all/ after video is written.")
    parser.add_argument("--iter", type=int, default=None,
                        help="Override iteration number used for checkpoint selection and filename suffix. "
                             "Defaults to the latest (final) checkpoint.")
    parser.add_argument("--apply-anchor-attention", choices=("auto", "yes", "no"), default="auto",
                        help="Apply trained anchor-attention correction. "
                             "auto applies it when an anchor checkpoint exists.")
    parser.add_argument("--anchor-ckpt", type=Path, default=None,
                        help="Optional anchor_attention checkpoint override.")
    return parser.parse_args()


def natural_ckpt_key(path: str):
    name = Path(path).stem
    if name.endswith("final"):
        return (10**12, name)
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else -1, name)


def find_latest_ckpt(run_dir: Path, pattern: str) -> str | None:
    files = glob.glob(str(run_dir / "ckpt" / pattern))
    files += glob.glob(str(run_dir / pattern))
    files = sorted(set(files), key=natural_ckpt_key)
    return files[-1] if files else None


def configure_trainer(args: argparse.Namespace):
    from hugs.cfg.config import cfg as default_cfg

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config_train.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config_path))
    cfg.eval = True
    cfg.train.anim_interval = -1
    cfg.train.save_progress_images = False
    if str(getattr(cfg.human, "name", "")) == "hugs_triplane":
        cfg.human.name = "hugs_trimlp"
    cfg.logdir = str(run_dir)
    cfg.logdir_ckpt = str(run_dir / "ckpt")

    human_ckpt = find_latest_ckpt(run_dir, "human_*.pth")
    scene_ckpt = find_latest_ckpt(run_dir, "scene_*.pth")
    if not human_ckpt:
        raise FileNotFoundError(f"Missing human checkpoint under {run_dir / 'ckpt'}")
    if not scene_ckpt:
        raise FileNotFoundError(f"Missing scene checkpoint under {run_dir / 'ckpt'}")
    cfg.human.ckpt = human_ckpt
    cfg.scene.ckpt = scene_ckpt

    found_anchor_ckpt = (
        str(args.anchor_ckpt.resolve()) if args.anchor_ckpt
        else find_latest_ckpt(run_dir, "anchor_attention_*.pth")
    )
    if args.apply_anchor_attention == "yes":
        apply_anchor = True
    elif args.apply_anchor_attention == "no":
        apply_anchor = False
    else:
        apply_anchor = bool(found_anchor_ckpt)

    if apply_anchor:
        if not found_anchor_ckpt or not Path(found_anchor_ckpt).is_file():
            raise FileNotFoundError("Anchor attention requested but no checkpoint found.")
        cfg.anchor_attention.use_anchors = True
        cfg.anchor_attention.use_anchor_token_encoder = True
        cfg.anchor_attention.use_scene_query = True
        cfg.anchor_attention.use_cross_attention = True
        cfg.anchor_attention.ckpt = ""

    print(f"Human ckpt : {human_ckpt}")
    print(f"Scene ckpt : {scene_ckpt}")
    if apply_anchor:
        print(f"Anchor ckpt: {found_anchor_ckpt}")
    else:
        print("Anchor attention: disabled")

    return cfg, apply_anchor, found_anchor_ckpt if apply_anchor else None


def main() -> None:
    args = parse_args()

    cfg, apply_anchor, anchor_ckpt_path = configure_trainer(args)

    from hugs.trainer import GaussianTrainer

    trainer = GaussianTrainer(cfg)

    if apply_anchor and anchor_ckpt_path:
        state = torch.load(anchor_ckpt_path)
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing:
            print(f"Anchor ckpt: {len(missing)} missing keys (optional heads)")
        if unexpected:
            print(f"Anchor ckpt: {len(unexpected)} unexpected keys")

    video_path = trainer.render_full_sequence(
        iter=args.iter,
        keep_images=args.keep_frames,
        fps=args.fps,
    )

    print(f"\nVideo written to: {video_path}")
    print(f"FPS: {args.fps}  |  Frames: {len(trainer.all_dataset)}")
    duration = len(trainer.all_dataset) / args.fps
    print(f"Duration: {duration:.1f} s")


if __name__ == "__main__":
    main()
