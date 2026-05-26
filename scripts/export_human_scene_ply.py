#!/usr/bin/env python3
"""Export a static posed human+scene Gaussian Splatting PLY from a HUGS run.

The exported human splats are posed in the selected frame's world coordinates,
then optionally passed through the trained anchor-attention correction. This
matches the tensors used by validation/render_all instead of the canonical human
PLY written by hugs.utils.vis.save_ply().
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement

sys.path.append(str(Path(__file__).resolve().parents[1]))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="HUGS training output directory.")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame used to pose the human splats.")
    parser.add_argument("--out", type=Path, default=None, help="Output combined human+scene PLY path.")
    parser.add_argument(
        "--human-color-mode",
        choices=("raw_sh", "baked_dc"),
        default="raw_sh",
        help="raw_sh preserves renderer SHs; baked_dc is easier in generic PLY viewers.",
    )
    parser.add_argument(
        "--apply-anchor-attention",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Apply trained anchor-attention correction. auto applies it when an anchor checkpoint exists.",
    )
    parser.add_argument("--anchor-ckpt", type=Path, default=None, help="Optional anchor_attention checkpoint override.")
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration passed to anchor-attention gating. Defaults to cfg.train.num_steps like validation/render_all.",
    )
    parser.add_argument("--save-human-only", action="store_true", help="Also save posed human-only splat PLY.")
    parser.add_argument("--save-scene-only", action="store_true", help="Also save scene-only splat PLY in the same format.")
    parser.add_argument(
        "--save-render-check",
        action="store_true",
        help="Save a PNG rendered from the exact tensors exported to the PLY.",
    )
    return parser.parse_args()


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


def construct_attributes(max_sh_degree: int = 3) -> list[str]:
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(3)]
    attrs += [f"f_rest_{i}" for i in range(3 * ((max_sh_degree + 1) ** 2 - 1))]
    attrs += ["opacity"]
    attrs += [f"scale_{i}" for i in range(3)]
    attrs += [f"rot_{i}" for i in range(4)]
    return attrs


def shs_to_ply_features(shs: torch.Tensor, color_mode: str = "raw_sh") -> tuple[np.ndarray, np.ndarray]:
    if shs.dim() != 3 or shs.shape[1] < 16 or shs.shape[2] != 3:
        raise RuntimeError(f"Expected SH tensor [N,16,3], got {tuple(shs.shape)}")

    if color_mode == "baked_dc":
        from hugs.utils.spherical_harmonics import RGB2SH, SH2RGB

        rgb = torch.clamp(SH2RGB(shs[:, 0]), 0.0, 1.0)
        f_dc = RGB2SH(rgb).detach().cpu().numpy()
        f_rest = np.zeros((shs.shape[0], 45), dtype=np.float32)
        return f_dc, f_rest

    f_dc = shs[:, :1].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = shs[:, 1:16].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    return f_dc, f_rest


def gs_out_to_3dgs_attrs(gs_out: dict, color_mode: str = "raw_sh") -> np.ndarray:
    """Convert renderer-ready Gaussian tensors to GraphDECO 3DGS PLY attributes."""
    xyz = gs_out["xyz"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc, f_rest = shs_to_ply_features(gs_out["shs"], color_mode=color_mode)
    from hugs.utils.general import inverse_sigmoid

    opacity = inverse_sigmoid(torch.clamp(gs_out["opacity"], 1e-6, 1.0 - 1e-6)).detach().cpu().numpy()
    scale = torch.log(torch.clamp(gs_out["scales"], min=1e-8)).detach().cpu().numpy()
    rot = gs_out["rotq"].detach().cpu().numpy()
    attrs = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rot], axis=1)
    return attrs.astype(np.float32)


def write_3dgs_ply(attributes: np.ndarray, out_path: Path) -> None:
    dtype_full = [(name, "f4") for name in construct_attributes(max_sh_degree=3)]
    if len(dtype_full) != attributes.shape[1]:
        raise RuntimeError(f"Attribute name count {len(dtype_full)} != data width {attributes.shape[1]}")

    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.astype(np.float32)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)


def derive_part_path(out_path: Path, part: str) -> Path:
    if part not in {"human_only", "scene_only"}:
        raise ValueError(part)
    stem = out_path.stem
    if "human_scene" in stem:
        stem = stem.replace("human_scene", part)
    else:
        stem = f"{stem}_{part}"
    return out_path.with_name(stem + out_path.suffix)


@torch.no_grad()
def forward_world_frame(
    trainer: GaussianTrainer,
    frame_idx: int,
    apply_anchor_attention: bool,
    iteration: int | None,
) -> tuple[dict, dict, dict, dict]:
    if trainer.scene_gs is None or trainer.human_gs is None:
        raise RuntimeError("This exporter requires both human_gs and scene_gs.")
    if trainer.all_dataset is None:
        raise RuntimeError("This exporter requires an all split dataset.")
    if frame_idx < 0 or frame_idx >= len(trainer.all_dataset):
        raise IndexError(f"frame_idx={frame_idx} outside all_dataset length {len(trainer.all_dataset)}")

    data = trainer.all_dataset[frame_idx]
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

    stats = {}
    export_iter = trainer.cfg.train.num_steps if iteration is None else int(iteration)
    if apply_anchor_attention:
        human_out, stats = trainer.maybe_apply_anchor_attention(
            human_out,
            scene_out,
            trainer.cfg.mode,
            export_iter,
        )
    return data, human_out, scene_out, stats


@torch.no_grad()
def export_combined_ply(
    trainer: GaussianTrainer,
    frame_idx: int,
    out_path: Path,
    human_color_mode: str,
    apply_anchor_attention: bool,
    iteration: int | None,
    save_human_only: bool,
    save_scene_only: bool,
    save_render_check: bool,
) -> None:
    data, human_out, scene_out, anchor_stats = forward_world_frame(
        trainer,
        frame_idx,
        apply_anchor_attention,
        iteration,
    )
    export_iter = trainer.cfg.train.num_steps if iteration is None else int(iteration)

    attrs_human = gs_out_to_3dgs_attrs(human_out, color_mode=human_color_mode)
    attrs_scene = gs_out_to_3dgs_attrs(scene_out, color_mode="raw_sh")
    if attrs_scene.shape[1] != attrs_human.shape[1]:
        raise RuntimeError(
            f"Scene/human PLY attribute width mismatch: {attrs_scene.shape[1]} vs {attrs_human.shape[1]}"
        )

    # Match render_human_scene concatenation order: human splats first, scene splats second.
    attributes = np.concatenate([attrs_human, attrs_scene], axis=0).astype(np.float32)
    write_3dgs_ply(attributes, out_path)

    human_path = None
    scene_path = None
    if save_human_only:
        human_path = derive_part_path(out_path, "human_only")
        write_3dgs_ply(attrs_human.astype(np.float32), human_path)
    if save_scene_only:
        scene_path = derive_part_path(out_path, "scene_only")
        write_3dgs_ply(attrs_scene.astype(np.float32), scene_path)

    render_path = None
    if save_render_check:
        import torchvision
        from hugs.renderer.gs_renderer import render_human_scene

        render = render_human_scene(
            data=data,
            human_gs_out=human_out,
            scene_gs_out=scene_out,
            bg_color=trainer.bg_color,
            render_mode=trainer.cfg.mode,
        )["render"].clamp(0.0, 1.0)
        render_path = out_path.with_suffix(".png")
        torchvision.utils.save_image(render, render_path)

    meta = {
        "frame_idx": int(frame_idx),
        "human_first": True,
        "num_human_splats": int(attrs_human.shape[0]),
        "num_scene_splats": int(attrs_scene.shape[0]),
        "combined_ply": str(out_path),
        "human_only_ply": str(human_path) if human_path else None,
        "scene_only_ply": str(scene_path) if scene_path else None,
        "render_check_png": str(render_path) if render_path else None,
        "human_color_mode": human_color_mode,
        "anchor_attention_requested": bool(apply_anchor_attention),
        "anchor_attention_applied": bool(anchor_stats),
        "anchor_iteration": int(export_iter),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {attributes.shape[0]} splats to {out_path}")
    print(f"  human splats: {attrs_human.shape[0]}")
    print(f"  scene splats: {attrs_scene.shape[0]}")
    print(f"  human first: true")
    print(f"  human color mode: {human_color_mode}")
    print(f"  anchor attention requested: {apply_anchor_attention}")
    print(f"  anchor attention applied: {bool(anchor_stats)}")
    print(f"  anchor iteration: {export_iter}")
    if human_path:
        print(f"Wrote human-only splats to {human_path}")
    if scene_path:
        print(f"Wrote scene-only splats to {scene_path}")
    if render_path:
        print(f"Wrote render check image to {render_path}")


def configure_from_run(args: argparse.Namespace) -> tuple[object, bool, str | None]:
    from hugs.cfg.config import cfg as default_cfg

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config_train.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config_path))
    cfg.eval = True
    cfg.train.anim_interval = -1
    cfg.train.save_progress_images = False
    # Older released checkpoints used this name for the current TRIMLP model.
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

    found_anchor_ckpt = str(args.anchor_ckpt.resolve()) if args.anchor_ckpt else find_latest_ckpt(run_dir, "anchor_attention_*.pth")
    if args.apply_anchor_attention == "yes":
        apply_anchor_attention = True
    elif args.apply_anchor_attention == "no":
        apply_anchor_attention = False
    else:
        apply_anchor_attention = bool(found_anchor_ckpt)

    if apply_anchor_attention:
        if not found_anchor_ckpt or not Path(found_anchor_ckpt).is_file():
            raise FileNotFoundError("Anchor attention export requested, but no anchor_attention checkpoint was found.")
        cfg.anchor_attention.use_anchors = True
        cfg.anchor_attention.use_anchor_token_encoder = True
        cfg.anchor_attention.use_scene_query = True
        cfg.anchor_attention.use_cross_attention = True
        # Let the trainer instantiate the module, then load below with strict=False
        # for old checkpoints that lack newer optional correction heads.
        cfg.anchor_attention.ckpt = ""

    print(f"Using human checkpoint: {cfg.human.ckpt}")
    print(f"Using scene checkpoint: {cfg.scene.ckpt}")
    if apply_anchor_attention:
        print(f"Using anchor checkpoint: {found_anchor_ckpt}")

    return cfg, apply_anchor_attention, found_anchor_ckpt if apply_anchor_attention else None


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg, apply_anchor_attention, anchor_ckpt = configure_from_run(args)

    out = args.out
    if out is None:
        out = run_dir / "meshes" / f"human_scene_final_frame_{args.frame_idx:05d}_{args.human_color_mode}_splat.ply"
    out = out.resolve()

    from hugs.trainer import GaussianTrainer

    trainer = GaussianTrainer(cfg)
    if apply_anchor_attention and anchor_ckpt:
        state = torch.load(anchor_ckpt)
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing:
            print(f"Anchor checkpoint missing optional/current keys: {len(missing)}")
        if unexpected:
            print(f"Anchor checkpoint unexpected keys: {len(unexpected)}")
    export_combined_ply(
        trainer,
        args.frame_idx,
        out,
        args.human_color_mode,
        apply_anchor_attention,
        args.iteration,
        args.save_human_only,
        args.save_scene_only,
        args.save_render_check,
    )


if __name__ == "__main__":
    main()
