#!/usr/bin/env python3
"""Render temporally upsampled GT-view and/or orbit-view HUGS videos.

Unlike simply writing a low-rate image sequence with a larger video FPS, this
script synthesizes intermediate SMPL states.  Axis-angle joints are converted
to quaternions and SLERPed; translation, betas and scale are linearly
interpolated.  The human Gaussian is then re-deformed by its existing LBS
model at every synthesized time.  The scene Gaussian remains static.

The orbit camera is a look-at camera centered on the posed human each frame,
so it does not use the dataset/GT camera trajectory.

Examples:
  # Preserve a 10 FPS source clip's duration while producing true 30 FPS.
  python scripts/render_smooth_orbit_video.py --run-dir <RUN> --view gt \
      --source-fps 10 --fps 30

  # One full orbit around the moving person, at the same interpolated poses.
  python scripts/render_smooth_orbit_video.py --run-dir <RUN> --view orbit \
      --source-fps 10 --fps 30 --orbit-radius 3.0 --orbit-elevation-deg 10
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import torch
import torchvision
from torch import nn
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hugs.renderer.gs_renderer import render_human_scene
from hugs.utils.general import create_video
from hugs.utils.graphics import get_projection_matrix
from hugs.utils.rotations import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("gt", "orbit", "both"), default="both")
    parser.add_argument("--source-fps", type=float, default=10.0,
                        help="Temporal rate represented by adjacent NeuMan frames (default: 10).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output rate. It must be an integer multiple of --source-fps (default: 30).")
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--apply-anchor-attention", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--anchor-ckpt", type=Path, default=None)
    parser.add_argument("--orbit-radius", type=float, default=3.0,
                        help="Orbit radius in the scene coordinate system.")
    parser.add_argument("--orbit-elevation-deg", type=float, default=10.0)
    parser.add_argument("--orbit-start-deg", type=float, default=0.0)
    parser.add_argument("--orbit-turns", type=float, default=1.0)
    parser.add_argument("--orbit-swing-deg", type=float, default=None,
                        help="If set, use a sinusoidal +/- azimuth sway instead of a full orbit.")
    parser.add_argument("--orbit-swing-cycles", type=float, default=1.0,
                        help="Number of back-and-forth sway cycles across the video.")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--max-source-frames", type=int, default=None,
                        help="Render only the first N source frames (smoke-test/debug helper).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only this script's matching output directories/files.")
    return parser.parse_args()


def slerp_quaternion(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Shortest-path SLERP for tensors with a trailing quaternion dimension."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.clamp((q0 * q1).sum(dim=-1, keepdim=True), -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = (1.0 - alpha) * q0 + alpha * q1
    spherical = (torch.sin((1.0 - alpha) * theta) / sin_theta) * q0 + (torch.sin(alpha * theta) / sin_theta) * q1
    q = torch.where(sin_theta.abs() < 1e-6, linear, spherical)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    return q


def slerp_axis_angle(a0: torch.Tensor, a1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Shortest-path quaternion SLERP for an arbitrary trailing (..., 3) shape."""
    q0 = axis_angle_to_quaternion(a0.reshape(-1, 3))
    q1 = axis_angle_to_quaternion(a1.reshape(-1, 3))
    return quaternion_to_axis_angle(slerp_quaternion(q0, q1, alpha)).reshape_as(a0)


def interpolate_camera(d0: dict, d1: dict, alpha: float, out: dict) -> None:
    """Interpolate GT camera center + rotation so static scene motion is smooth."""
    if "c2w" not in d0 or "c2w" not in d1:
        raise KeyError("NeuMan data must contain c2w for GT-view temporal interpolation")
    c2w0, c2w1 = d0["c2w"], d1["c2w"]
    rot = quaternion_to_matrix(slerp_quaternion(
        matrix_to_quaternion(c2w0[:3, :3]), matrix_to_quaternion(c2w1[:3, :3]), alpha
    ))
    center = torch.lerp(c2w0[:3, 3], c2w1[:3, 3], alpha)
    c2w = torch.eye(4, dtype=c2w0.dtype, device=c2w0.device)
    c2w[:3, :3] = rot
    c2w[:3, 3] = center
    w2c = torch.linalg.inv(c2w)
    world_view = w2c.T.contiguous()
    proj = get_projection_matrix(out["near"], out["far"], out["fovx"], out["fovy"]).T.to(
        device=world_view.device, dtype=world_view.dtype
    )
    out["c2w"] = c2w
    out["world_view_transform"] = world_view
    out["full_proj_transform"] = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    out["camera_center"] = center
    out["cam_ext"] = world_view


def interpolate_data(d0: dict, d1: dict, alpha: float) -> dict:
    """Keep camera/image metadata from d0 and synthesize only SMPL parameters."""
    if alpha == 0.0:
        return d0
    out = dict(d0)
    out["global_orient"] = slerp_axis_angle(d0["global_orient"], d1["global_orient"], alpha)
    out["body_pose"] = slerp_axis_angle(d0["body_pose"], d1["body_pose"], alpha)
    for key in ("transl", "betas", "smpl_scale"):
        out[key] = torch.lerp(d0[key], d1[key], alpha)
    interpolate_camera(d0, d1, alpha, out)
    out["_anchor_frame_idx0"] = d0["frame_idx"]
    out["_anchor_frame_idx1"] = d1["frame_idx"]
    out["_interp_alpha"] = alpha
    return out


def make_orbit_camera(reference: dict, target: torch.Tensor, azimuth: float, radius: float, elevation_deg: float) -> dict:
    """Build renderer camera data from a world-space target and a look-at orbit."""
    device, dtype = target.device, target.dtype
    elev = torch.deg2rad(torch.tensor(elevation_deg, dtype=dtype, device=device))
    az = torch.tensor(azimuth, dtype=dtype, device=device)
    offset = torch.stack((
        radius * torch.cos(elev) * torch.sin(az),
        radius * torch.sin(elev),
        radius * torch.cos(elev) * torch.cos(az),
    ))
    center = target + offset
    z_axis = (center - target) / torch.linalg.vector_norm(center - target).clamp_min(1e-8)
    up_guess = torch.tensor((0.0, 1.0, 0.0), dtype=dtype, device=device)
    # Avoid the singularity at a vertical view direction.
    if torch.abs(torch.dot(z_axis, up_guess)) > 0.98:
        up_guess = torch.tensor((0.0, 0.0, 1.0), dtype=dtype, device=device)
    x_axis = torch.linalg.cross(up_guess, z_axis)
    x_axis = x_axis / torch.linalg.vector_norm(x_axis).clamp_min(1e-8)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    rot = torch.stack((x_axis, y_axis, z_axis), dim=0)  # world -> camera rotation
    w2c = torch.eye(4, dtype=dtype, device=device)
    w2c[:3, :3] = rot
    w2c[:3, 3] = -rot @ center
    world_view = w2c.T.contiguous()  # renderer uses row-vector convention
    proj = get_projection_matrix(reference["near"], reference["far"], reference["fovx"], reference["fovy"]).T.to(device=device, dtype=dtype)
    out = dict(reference)
    out["world_view_transform"] = world_view
    out["full_proj_transform"] = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    out["camera_center"] = center
    out["cam_ext"] = world_view
    return out


def prepare_trainer(args: argparse.Namespace):
    # Reuse checkpoint/config resolution from the existing full-sequence renderer.
    from render_full_video import configure_trainer
    from hugs.models.anchor_attention import AnchorSceneAttentionBaseline
    from hugs.trainer import GaussianTrainer

    cfg, apply_anchor, anchor_ckpt = configure_trainer(args)
    trainer = GaussianTrainer(cfg)
    if apply_anchor and anchor_ckpt:
        state = torch.load(anchor_ckpt)
        # all_dataset contains every source frame, whereas the trainer builds
        # its module from the train split.  Restore the checkpoint's original
        # frame-embedding cardinality before loading it.
        embed_key = "frame_embed.weight"
        if embed_key in state and hasattr(trainer.anchor_attention, "frame_embed"):
            n_frames, emb_dim = state[embed_key].shape
            current = trainer.anchor_attention.frame_embed
            if current.num_embeddings != n_frames:
                trainer.anchor_attention.frame_embed = nn.Embedding(n_frames, emb_dim).to(
                    device=current.weight.device, dtype=current.weight.dtype
                )
                trainer.anchor_attention.num_frames = n_frames
        elif embed_key not in state and getattr(trainer.anchor_attention, "num_frames", 0) != 0:
            # Early best-pipeline checkpoints predate frame embeddings.  Their
            # delta_transl MLP therefore consumes only the 128-D scene context.
            trainer.anchor_attention = AnchorSceneAttentionBaseline(
                trainer.cfg.anchor_attention,
                num_anchors=state["anchor_embed.weight"].shape[0],
                top_m=int(getattr(trainer.cfg.anchor_attention, "top_m", 2)),
                num_frames=0,
            ).to("cuda")
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Anchor checkpoint non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
    return trainer


def render_one(trainer, data: dict, orbit_azimuth: float | None, args: argparse.Namespace, scene_out: dict):
    human_out = trainer.human_gs.forward(
        global_orient=data["global_orient"], body_pose=data["body_pose"], betas=data["betas"],
        transl=data["transl"], smpl_scale=data["smpl_scale"][None], dataset_idx=-1,
        is_train=False, ext_tfs=None,
    )
    alpha = float(data.get("_interp_alpha", 0.0))
    frame0 = int(data.get("_anchor_frame_idx0", data["frame_idx"]).item())
    frame1 = int(data.get("_anchor_frame_idx1", data["frame_idx"]).item())
    corrected0, _ = trainer.maybe_apply_anchor_attention(
        human_out, scene_out, trainer.cfg.mode, trainer.cfg.train.num_steps, frame_idx=frame0,
    )
    if alpha > 0.0 and frame1 != frame0:
        corrected1, _ = trainer.maybe_apply_anchor_attention(
            human_out, scene_out, trainer.cfg.mode, trainer.cfg.train.num_steps, frame_idx=frame1,
        )
        # Only correction-dependent tensor fields differ; blending them avoids
        # a hard frame-embedding switch at every source-frame boundary.
        human_out = {
            key: torch.lerp(value, corrected1[key], alpha)
            if torch.is_tensor(value) and key in corrected1 and torch.is_tensor(corrected1[key])
            and value.shape == corrected1[key].shape and value.is_floating_point()
            else value
            for key, value in corrected0.items()
        }
    else:
        human_out = corrected0
    if orbit_azimuth is not None:
        target = human_out["xyz"].mean(dim=0)
        data = make_orbit_camera(data, target, orbit_azimuth, args.orbit_radius, args.orbit_elevation_deg)
    return render_human_scene(data, human_out, scene_out, trainer.bg_color, render_mode=trainer.cfg.mode)["render"]


def reset_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite or choose another run directory.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.source_fps <= 0 or args.fps <= 0:
        raise ValueError("--source-fps and --fps must be positive")
    ratio = args.fps / args.source_fps
    upsample = round(ratio)
    if abs(ratio - upsample) > 1e-6 or upsample < 1:
        raise ValueError("--fps must be an integer multiple of --source-fps")

    trainer = prepare_trainer(args)
    if trainer.all_dataset is None or len(trainer.all_dataset) < 2:
        raise RuntimeError("A full sequence with at least two frames is required.")
    trainer.human_gs.eval()
    scene_out = trainer.scene_gs.forward()
    n_source = len(trainer.all_dataset)
    if args.max_source_frames is not None:
        if args.max_source_frames < 2:
            raise ValueError("--max-source-frames must be at least 2")
        n_source = min(n_source, args.max_source_frames)
    n_output = (n_source - 1) * upsample + 1
    tag = f"smooth_{int(args.fps) if args.fps.is_integer() else args.fps}fps_x{upsample}"
    views = ("gt", "orbit") if args.view == "both" else (args.view,)

    for view in views:
        frame_dir = args.run_dir.resolve() / f"render_{tag}_{view}"
        video_path = args.run_dir.resolve() / f"render_{tag}_{view}.mp4"
        reset_output(frame_dir, args.overwrite)
        if video_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{video_path} exists; use --overwrite")
            video_path.unlink()
        for out_idx in tqdm(range(n_output), desc=f"Render {view}"):
            pos = out_idx / upsample
            lo = min(int(pos), n_source - 1)
            hi = min(lo + 1, n_source - 1)
            alpha = pos - lo
            data = interpolate_data(trainer.all_dataset[lo], trainer.all_dataset[hi], alpha)
            azimuth = None
            if view == "orbit":
                progress = out_idx / max(1, n_output - 1)
                center_azimuth = math.radians(args.orbit_start_deg)
                if args.orbit_swing_deg is not None:
                    azimuth = center_azimuth + math.radians(args.orbit_swing_deg) * math.sin(
                        progress * args.orbit_swing_cycles * 2.0 * math.pi
                    )
                else:
                    azimuth = center_azimuth + progress * args.orbit_turns * 2.0 * math.pi
            image = render_one(trainer, data, azimuth, args, scene_out)
            torchvision.utils.save_image(image, frame_dir / f"{out_idx:05d}.png")
        create_video(str(frame_dir), str(video_path), fps=args.fps)
        print(f"Wrote {view}: {video_path} ({n_output} frames at {args.fps:g} FPS)")
        if not args.keep_frames:
            shutil.rmtree(frame_dir)
    print(f"Source frames: {n_source}; temporal upsample: {upsample}x; duration: {n_source / args.source_fps:.2f}s")


if __name__ == "__main__":
    main()
