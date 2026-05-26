#!/usr/bin/env python3
"""Export scaled depth point cloud + final scene/human Gaussian centers in one PLY."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hugs.datasets import NeumanDataset
from hugs.models import SceneGS
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP


def parse_frames(text, nframes):
    if text == "all":
        return list(range(nframes))
    frames = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [int(x) for x in item.split(":")]
            if len(parts) == 2:
                start, stop = parts
                step = 1
            elif len(parts) == 3:
                start, stop, step = parts
            else:
                raise ValueError(f"Bad frame range: {item}")
            frames.extend(range(start, stop, step))
        else:
            frames.append(int(item))
    return [f for f in frames if 0 <= f < nframes]


def load_depth_world_points(dataset, depth_dir, frames, max_points_per_frame, mask_human=True, stride=4):
    depth_dir = Path(depth_dir)
    all_xyz = []
    all_rgb = []
    rng = np.random.default_rng(0)

    seq_dir = Path(dataset.dataset_path)
    mask_dir = seq_dir / "4d_humans" / "sam_segmentations"
    masks = sorted(mask_dir.glob("*.png"))

    for fid in frames:
        data = dataset[fid]
        depth_path = depth_dir / f"{fid:05d}_depth.npy"
        if not depth_path.exists():
            depth_path = depth_dir / f"{fid:06d}.npy"
        if not depth_path.exists():
            print(f"skip missing depth: frame={fid} path={depth_path}")
            continue

        depth = np.load(depth_path).astype(np.float32)
        height = int(data["image_height"])
        width = int(data["image_width"])
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

        valid = np.isfinite(depth) & (depth > 0)
        if mask_human and fid < len(masks):
            mask = cv2.imread(str(masks[fid]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                valid &= mask < 127

        ys, xs = np.where(valid[::stride, ::stride])
        ys = ys * stride
        xs = xs * stride
        if len(xs) == 0:
            continue
        if max_points_per_frame > 0 and len(xs) > max_points_per_frame:
            keep = rng.choice(len(xs), max_points_per_frame, replace=False)
            xs = xs[keep]
            ys = ys[keep]

        z = depth[ys, xs]
        k = data["cam_intrinsics"].detach().cpu().numpy().astype(np.float64)
        x_cam = (xs.astype(np.float64) - k[0, 2]) / k[0, 0] * z
        y_cam = (ys.astype(np.float64) - k[1, 2]) / k[1, 1] * z
        cam = np.stack([x_cam, y_cam, z.astype(np.float64), np.ones_like(z, dtype=np.float64)], axis=1)
        c2w = data["c2w"].detach().cpu().numpy().astype(np.float64)
        world = (c2w @ cam.T).T[:, :3].astype(np.float32)
        all_xyz.append(world)

        if "rgb" in data:
            rgb = data["rgb"].detach().cpu().numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb[ys, xs] * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.full((len(xs), 3), [60, 220, 80], dtype=np.uint8)
        all_rgb.append(rgb)

    if not all_xyz:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(all_xyz, axis=0), np.concatenate(all_rgb, axis=0)


def build_human_model(cfg, betas):
    kwargs = dict(
        sh_degree=cfg.human.sh_degree,
        n_subdivision=cfg.human.n_subdivision,
        use_surface=cfg.human.use_surface,
        init_2d=cfg.human.init_2d,
        rotate_sh=cfg.human.rotate_sh,
        isotropic=cfg.human.isotropic,
        init_scale_multiplier=cfg.human.init_scale_multiplier,
    )
    if cfg.human.name == "hugs_trimlp":
        return HUGS_TRIMLP(
            **kwargs,
            n_features=32,
            use_deformer=cfg.human.use_deformer,
            disable_posedirs=cfg.human.disable_posedirs,
            triplane_res=cfg.human.triplane_res,
            betas=betas,
        )
    if cfg.human.name == "hugs_wo_trimlp":
        model = HUGS_WO_TRIMLP(**kwargs)
        model.create_betas(betas, False)
        model.initialize()
        return model
    raise ValueError(f"Unsupported human model: {cfg.human.name}")


@torch.no_grad()
def load_final_centers(logdir, frame_index):
    logdir = Path(logdir)
    cfg = OmegaConf.load(logdir / "config_train.yaml")
    dataset = NeumanDataset(
        cfg.dataset.seq,
        "all",
        cfg.mode,
        init_pcd_path=getattr(cfg.scene, "init_pcd_path", None),
    )
    data = dataset[frame_index]

    human_ckpt = torch.load(logdir / "ckpt" / "human_final.pth", map_location="cuda", weights_only=False)
    scene_ckpt = torch.load(logdir / "ckpt" / "scene_final.pth", map_location="cuda", weights_only=False)

    human = build_human_model(cfg, data["betas"])
    human.load_state_dict(human_ckpt, cfg.human.lr)
    scene = SceneGS(sh_degree=cfg.scene.sh_degree)
    scene.restore(scene_ckpt, cfg.scene.lr)

    human_out = human.forward(
        global_orient=data["global_orient"],
        body_pose=data["body_pose"],
        betas=data["betas"],
        transl=data["transl"],
        smpl_scale=data["smpl_scale"][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )
    scene_out = scene.forward()
    return (
        dataset,
        scene_out["xyz"].detach().cpu().numpy().astype(np.float32),
        human_out["xyz"].detach().cpu().numpy().astype(np.float32),
    )


def write_ply(out, depth_xyz, depth_rgb, scene_xyz, human_xyz, depth_use_rgb):
    n_depth = len(depth_xyz)
    n_scene = len(scene_xyz)
    n_human = len(human_xyz)
    xyz = np.concatenate([depth_xyz, scene_xyz, human_xyz], axis=0).astype(np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("part", "u1"),
    ]
    verts = np.empty(len(xyz), dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if depth_use_rgb:
        verts["red"][:n_depth] = depth_rgb[:, 0]
        verts["green"][:n_depth] = depth_rgb[:, 1]
        verts["blue"][:n_depth] = depth_rgb[:, 2]
    else:
        verts["red"][:n_depth] = 60
        verts["green"][:n_depth] = 220
        verts["blue"][:n_depth] = 80
    verts["part"][:n_depth] = 2

    s0, s1 = n_depth, n_depth + n_scene
    verts["red"][s0:s1] = 70
    verts["green"][s0:s1] = 150
    verts["blue"][s0:s1] = 255
    verts["part"][s0:s1] = 0

    h0 = s1
    verts["red"][h0:] = 255
    verts["green"][h0:] = 80
    verts["blue"][h0:] = 40
    verts["part"][h0:] = 1

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(out))
    print(f"Wrote {out}")
    print(f"depth_points={n_depth} scene_points={n_scene} human_points={n_human}")
    print("part: 0=scene_blue, 1=human_red, 2=scaled_depth_green_or_rgb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--human-frame", type=int, default=0)
    parser.add_argument("--depth-frames", default="0:103:5")
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-depth-points-per-frame", type=int, default=8000)
    parser.add_argument("--keep-human-depth", action="store_true")
    parser.add_argument("--depth-use-rgb", action="store_true")
    args = parser.parse_args()

    dataset, scene_xyz, human_xyz = load_final_centers(args.logdir, args.human_frame)
    frames = parse_frames(args.depth_frames, dataset.num_frames)
    depth_xyz, depth_rgb = load_depth_world_points(
        dataset,
        args.depth_dir,
        frames,
        args.max_depth_points_per_frame,
        mask_human=not args.keep_human_depth,
        stride=args.depth_stride,
    )
    write_ply(args.out, depth_xyz, depth_rgb, scene_xyz, human_xyz, args.depth_use_rgb)


if __name__ == "__main__":
    main()
