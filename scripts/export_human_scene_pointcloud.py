#!/usr/bin/env python3
"""Export a plain point-cloud PLY for checking human/scene relative position.

This intentionally does not write 3DGS splat attributes.  It exports ordinary
PLY vertices with RGB colors so common point-cloud viewers can show whether the
posed human gaussians line up with the scene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hugs.cfg.config import cfg as default_cfg
from hugs.trainer import GaussianTrainer
from hugs.utils.spherical_harmonics import SH2RGB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="HUGS training output directory.")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index in all_dataset / render_full_sequence.")
    parser.add_argument("--out", type=Path, default=None, help="Output point-cloud PLY path.")
    parser.add_argument(
        "--scene-stride",
        type=int,
        default=1,
        help="Keep every Nth scene point. Human points are never subsampled by default.",
    )
    parser.add_argument(
        "--human-stride",
        type=int,
        default=1,
        help="Keep every Nth human point.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("actual", "label"),
        default="label",
        help="actual uses learned colors; label colors scene gray and human red for position debugging.",
    )
    parser.add_argument(
        "--crop-scene-to-human-margin",
        type=float,
        default=None,
        help="If set, keep only scene points inside the human bbox expanded by this margin.",
    )
    parser.add_argument("--save-human-only", action="store_true", help="Also save a human-only point cloud.")
    return parser.parse_args()


def to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def scene_rgb_from_sh(features_dc: torch.Tensor) -> np.ndarray:
    dc = features_dc.detach()[:, 0, :]
    return torch.clamp(SH2RGB(dc), 0.0, 1.0).cpu().numpy()


def human_rgb_from_sh(shs: torch.Tensor) -> np.ndarray:
    return torch.clamp(SH2RGB(shs.detach()[:, 0]), 0.0, 1.0).cpu().numpy()


def write_pointcloud_ply(xyz: np.ndarray, rgb: np.ndarray, label: np.ndarray, out_path: Path) -> None:
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[0] != label.shape[0]:
        raise RuntimeError("xyz/rgb/label length mismatch")

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("label", "u1"),
    ]
    vertices = np.empty(xyz.shape[0], dtype=dtype)
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    vertices["label"] = label.astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(out_path)


@torch.no_grad()
def export_pointcloud(
    trainer: GaussianTrainer,
    frame_idx: int,
    out_path: Path,
    scene_stride: int,
    human_stride: int,
    color_mode: str,
    crop_scene_to_human_margin: float | None,
    save_human_only: bool,
) -> None:
    if trainer.scene_gs is None or trainer.human_gs is None:
        raise RuntimeError("This exporter requires both human_gs and scene_gs.")
    if trainer.all_dataset is None:
        raise RuntimeError("This exporter requires all_dataset.")
    if frame_idx < 0 or frame_idx >= len(trainer.all_dataset):
        raise IndexError(f"frame_idx={frame_idx} outside all_dataset length {len(trainer.all_dataset)}")
    if scene_stride < 1 or human_stride < 1:
        raise ValueError("Strides must be >= 1.")

    data = trainer.all_dataset[frame_idx]

    # Match GaussianTrainer.render_full_sequence exactly so the exported human
    # point positions correspond to the rendered video frame.
    human_out = trainer.human_gs.forward(
        global_orient=data["global_orient"],
        body_pose=data["body_pose"],
        betas=data["betas"],
        transl=data["transl"],
        smpl_scale=data["smpl_scale"][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )
    scene_out = trainer.scene_gs.forward()

    xyz_scene_all = scene_out["xyz"].detach().cpu().numpy()
    xyz_human = human_out["xyz"].detach().cpu().numpy()[::human_stride]
    scene_keep = np.arange(xyz_scene_all.shape[0])[::scene_stride]

    if crop_scene_to_human_margin is not None:
        human_min = xyz_human.min(axis=0) - crop_scene_to_human_margin
        human_max = xyz_human.max(axis=0) + crop_scene_to_human_margin
        in_crop = np.all((xyz_scene_all >= human_min) & (xyz_scene_all <= human_max), axis=1)
        scene_keep = scene_keep[in_crop[scene_keep]]

    xyz_scene = xyz_scene_all[scene_keep]

    if color_mode == "actual":
        rgb_scene = to_uint8_rgb(scene_rgb_from_sh(scene_out["shs"])[scene_keep])
        rgb_human = to_uint8_rgb(human_rgb_from_sh(human_out["shs"])[::human_stride])
    else:
        rgb_scene = np.full((xyz_scene.shape[0], 3), (150, 150, 150), dtype=np.uint8)
        rgb_human = np.full((xyz_human.shape[0], 3), (255, 40, 40), dtype=np.uint8)

    label_scene = np.zeros(xyz_scene.shape[0], dtype=np.uint8)
    label_human = np.ones(xyz_human.shape[0], dtype=np.uint8)

    xyz = np.concatenate([xyz_scene, xyz_human], axis=0)
    rgb = np.concatenate([rgb_scene, rgb_human], axis=0)
    label = np.concatenate([label_scene, label_human], axis=0)
    write_pointcloud_ply(xyz, rgb, label, out_path)

    print(f"Wrote point cloud: {out_path}")
    print(f"  frame_idx: {frame_idx}")
    print(f"  scene points: {xyz_scene.shape[0]} (stride {scene_stride})")
    print(f"  human points: {xyz_human.shape[0]} (stride {human_stride})")
    print(f"  color_mode: {color_mode}")
    if crop_scene_to_human_margin is not None:
        print(f"  scene crop margin around human bbox: {crop_scene_to_human_margin}")
    if xyz_scene.shape[0] > 0:
        print(f"  scene xyz min/max: {xyz_scene.min(axis=0)} / {xyz_scene.max(axis=0)}")
    else:
        print("  scene xyz min/max: no scene points after crop")
    print(f"  human xyz min/max: {xyz_human.min(axis=0)} / {xyz_human.max(axis=0)}")

    if save_human_only:
        human_path = out_path.with_name(out_path.stem.replace("human_scene", "human_only") + out_path.suffix)
        write_pointcloud_ply(xyz_human, rgb_human, label_human, human_path)
        print(f"Wrote human-only point cloud: {human_path}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config_train.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config_path))
    cfg.eval = True
    cfg.train.anim_interval = -1
    cfg.human.ckpt = str(run_dir / "ckpt" / "human_final.pth")
    cfg.scene.ckpt = str(run_dir / "ckpt" / "scene_final.pth")
    cfg.logdir = str(run_dir)
    cfg.logdir_ckpt = str(run_dir / "ckpt")

    if not Path(cfg.human.ckpt).is_file():
        raise FileNotFoundError(cfg.human.ckpt)
    if not Path(cfg.scene.ckpt).is_file():
        raise FileNotFoundError(cfg.scene.ckpt)

    out = args.out
    if out is None:
        out = run_dir / "meshes" / f"human_scene_final_frame_{args.frame_idx:05d}_{args.color_mode}_pointcloud.ply"

    trainer = GaussianTrainer(cfg)
    export_pointcloud(
        trainer=trainer,
        frame_idx=args.frame_idx,
        out_path=out.resolve(),
        scene_stride=args.scene_stride,
        human_stride=args.human_stride,
        color_mode=args.color_mode,
        crop_scene_to_human_margin=args.crop_scene_to_human_margin,
        save_human_only=args.save_human_only,
    )


if __name__ == "__main__":
    main()
