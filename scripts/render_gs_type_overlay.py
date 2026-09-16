#!/usr/bin/env python3
"""Render diagnostic images that visually separate scene GS vs human GS.

For each validation frame, saves:
  seg_{idx:03d}.png       -- human GS in red, scene GS in blue (flat colors, same alpha compositing)
  human_only_{idx:03d}.png -- human GS rendered alone on black background
  scene_only_{idx:03d}.png -- scene GS rendered alone on black background
  compare_{idx:03d}.png   -- 4-panel: GT | combined | seg | human-only

Usage:
  python scripts/render_gs_type_overlay.py --run-dir <training_output_dir>
  python scripts/render_gs_type_overlay.py --run-dir <path> --out-dir <path> --frames 0 1 2
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch
import torchvision

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Checkpoint / config helpers (same pattern as export_human_scene_ply.py)
# ---------------------------------------------------------------------------

def natural_ckpt_key(path: str) -> tuple[int, str]:
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


def configure_from_run(args: argparse.Namespace):
    from omegaconf import OmegaConf
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
        raise FileNotFoundError(f"No human checkpoint found under {run_dir / 'ckpt'}")
    if not scene_ckpt:
        raise FileNotFoundError(f"No scene checkpoint found under {run_dir / 'ckpt'}")
    cfg.human.ckpt = human_ckpt
    cfg.scene.ckpt = scene_ckpt

    found_anchor_ckpt = (
        str(args.anchor_ckpt.resolve()) if args.anchor_ckpt
        else find_latest_ckpt(run_dir, "anchor_attention_*.pth")
    )
    if args.no_anchor:
        apply_anchor = False
    else:
        apply_anchor = bool(found_anchor_ckpt)

    if apply_anchor:
        if not found_anchor_ckpt or not Path(found_anchor_ckpt).is_file():
            raise FileNotFoundError("Anchor checkpoint not found; pass --no-anchor to skip.")
        cfg.anchor_attention.use_anchors = True
        cfg.anchor_attention.use_anchor_token_encoder = True
        cfg.anchor_attention.use_scene_query = True
        cfg.anchor_attention.use_cross_attention = True
        cfg.anchor_attention.ckpt = ""

    print(f"human ckpt : {human_ckpt}")
    print(f"scene ckpt : {scene_ckpt}")
    if apply_anchor:
        print(f"anchor ckpt: {found_anchor_ckpt}")
    else:
        print("anchor    : skipped")

    return cfg, apply_anchor, found_anchor_ckpt if apply_anchor else None


# ---------------------------------------------------------------------------
# Colored-type render helper
# ---------------------------------------------------------------------------

def _flat_color_dict(gs_out: dict, color_rgb: list[float]) -> dict:
    """Return a shallow copy of gs_out with shs replaced by a flat (N,3) RGB tensor."""
    n = gs_out["xyz"].shape[0]
    device = gs_out["xyz"].device
    flat = torch.tensor(color_rgb, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)
    return {**gs_out, "shs": flat}


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    cfg, apply_anchor, anchor_ckpt_path = configure_from_run(args)

    from hugs.trainer import GaussianTrainer
    from hugs.renderer.gs_renderer import render_human_scene

    trainer = GaussianTrainer(cfg)
    trainer.human_gs.eval()

    if apply_anchor and anchor_ckpt_path:
        state = torch.load(anchor_ckpt_path)
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing:
            print(f"anchor ckpt missing keys: {len(missing)}")
        if unexpected:
            print(f"anchor ckpt unexpected keys: {len(unexpected)}")

    out_dir = args.out_dir if args.out_dir else args.run_dir / "val_overlay"
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    bg_black = torch.zeros(3, dtype=torch.float32, device="cuda")

    val_dataset = trainer.val_dataset
    frame_indices = args.frames if args.frames else list(range(len(val_dataset)))
    eval_iter = cfg.train.num_steps

    human_color = args.human_color   # default [1.0, 0.2, 0.1]  red
    scene_color = args.scene_color   # default [0.1, 0.5, 1.0]  blue

    for idx in frame_indices:
        if idx >= len(val_dataset):
            print(f"Frame {idx} out of range (dataset size {len(val_dataset)}), skipping.")
            continue

        data = val_dataset[idx]

        human_gs_out = trainer.human_gs.forward(
            global_orient=data["global_orient"],
            body_pose=data["body_pose"],
            betas=data["betas"],
            transl=data["transl"],
            smpl_scale=data["smpl_scale"][None],
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
        )
        scene_gs_out = trainer.scene_gs.forward()

        human_gs_out, _ = trainer.maybe_apply_anchor_attention(
            human_gs_out, scene_gs_out, cfg.mode, eval_iter,
        )

        # 1. Normal combined render
        combined_pkg = render_human_scene(
            data=data,
            human_gs_out=human_gs_out,
            scene_gs_out=scene_gs_out,
            bg_color=bg_black,
            render_mode=cfg.mode,
        )
        combined_img = combined_pkg["render"]

        # 2. Colored-type overlay (human=red, scene=blue)
        seg_pkg = render_human_scene(
            data=data,
            human_gs_out=_flat_color_dict(human_gs_out, human_color),
            scene_gs_out=_flat_color_dict(scene_gs_out, scene_color),
            bg_color=bg_black,
            render_mode=cfg.mode,
        )
        seg_img = seg_pkg["render"]

        # 3. Human-only render
        human_only_pkg = render_human_scene(
            data=data,
            human_gs_out=human_gs_out,
            scene_gs_out=scene_gs_out,
            bg_color=bg_black,
            render_mode="human",
        )
        human_only_img = human_only_pkg["render"]

        # 4. Scene-only render
        scene_only_pkg = render_human_scene(
            data=data,
            human_gs_out=human_gs_out,
            scene_gs_out=scene_gs_out,
            bg_color=bg_black,
            render_mode="scene",
        )
        scene_only_img = scene_only_pkg["render"]

        # Save individual images
        torchvision.utils.save_image(seg_img,         out_dir / f"seg_{idx:03d}.png")
        torchvision.utils.save_image(human_only_img,  out_dir / f"human_only_{idx:03d}.png")
        torchvision.utils.save_image(scene_only_img,  out_dir / f"scene_only_{idx:03d}.png")

        # 4-panel comparison: GT | combined | seg | human-only
        gt_img = data["rgb"]
        compare_grid = torchvision.utils.make_grid(
            [gt_img, combined_img, seg_img, human_only_img],
            nrow=4, pad_value=1,
        )
        torchvision.utils.save_image(compare_grid, out_dir / f"compare_{idx:03d}.png")

        print(f"  [{idx:03d}] saved seg / human_only / scene_only / compare")

    print(f"\nDone. {len(frame_indices)} frames written to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="HUGS training output directory (contains config_train.yaml and ckpt/).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write output images. Defaults to <run-dir>/val_overlay/.")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="Frame indices to process (default: all val frames).")
    parser.add_argument("--anchor-ckpt", type=Path, default=None,
                        help="Override anchor-attention checkpoint path.")
    parser.add_argument("--no-anchor", action="store_true",
                        help="Skip anchor-attention even if a checkpoint exists.")
    parser.add_argument("--human-color", type=float, nargs=3, default=[1.0, 0.2, 0.1],
                        metavar=("R", "G", "B"),
                        help="Flat RGB color for human GS in seg render (default: red 1.0 0.2 0.1).")
    parser.add_argument("--scene-color", type=float, nargs=3, default=[0.1, 0.5, 1.0],
                        metavar=("R", "G", "B"),
                        help="Flat RGB color for scene GS in seg render (default: blue 0.1 0.5 1.0).")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
