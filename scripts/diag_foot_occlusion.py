"""Diagnose scene Gaussians near the feet by coloring them red in the render.

For each validation frame:
  - Run SMPL forward pass to get foot anchor world positions
  - Find scene GS whose centers are within --radius of any foot anchor
  - Color those GS solid red (zero all SH, set DC to red)
  - Save side-by-side: left = original render, right = foot-colored render

Usage:
  python scripts/diag_foot_occlusion.py \\
      --run-dir output/human_scene/neuman/lab/hugs_trimlp/<exp>/<timestamp> \\
      --radius 0.5 --frames 0 1 2 3 4
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FOOT_ANCHORS = ['left_sole', 'right_sole', 'left_toe', 'right_toe', 'left_heel', 'right_heel']
SH_C0 = 0.28209479177387814  # zeroth-order SH coefficient


# ---------------------------------------------------------------------------
# Trainer setup (mirrors render_gs_structure.py)
# ---------------------------------------------------------------------------

def natural_ckpt_key(path: str):
    name = Path(path).stem
    if name.endswith("final"):
        return (10**12, name)
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else -1, name)


def find_latest_ckpt(run_dir: Path, pattern: str):
    files = glob.glob(str(run_dir / "ckpt" / pattern))
    files += glob.glob(str(run_dir / pattern))
    return sorted(set(files), key=natural_ckpt_key)[-1] if files else None


def configure_from_run(args):
    from omegaconf import OmegaConf
    from hugs.cfg.config import cfg as default_cfg

    run_dir = Path(args.run_dir).resolve()
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
        raise FileNotFoundError(f"No human checkpoint under {run_dir / 'ckpt'}")
    if not scene_ckpt:
        raise FileNotFoundError(f"No scene checkpoint under {run_dir / 'ckpt'}")
    cfg.human.ckpt = human_ckpt
    cfg.scene.ckpt = scene_ckpt

    anchor_ckpt = (
        str(Path(args.anchor_ckpt).resolve()) if getattr(args, 'anchor_ckpt', None)
        else find_latest_ckpt(run_dir, "anchor_attention_*.pth")
    )
    apply_anchor = not getattr(args, 'no_anchor', False) and bool(anchor_ckpt)
    if apply_anchor:
        cfg.anchor_attention.use_anchors = True
        cfg.anchor_attention.use_anchor_token_encoder = True
        cfg.anchor_attention.use_scene_query = True
        cfg.anchor_attention.use_cross_attention = True
        cfg.anchor_attention.ckpt = ""

    print(f"human ckpt : {human_ckpt}")
    print(f"scene ckpt : {scene_ckpt}")
    print(f"anchor     : {'skipped' if not apply_anchor else anchor_ckpt}")
    return cfg, apply_anchor, anchor_ckpt if apply_anchor else None


# ---------------------------------------------------------------------------
# Foot anchor world positions
# ---------------------------------------------------------------------------

@torch.no_grad()
def foot_anchor_world(trainer, data):
    """Return (F, 3) world-space positions for foot anchors, plus their names."""
    from hugs.utils.rotations import rotation_6d_to_axis_angle

    hg = trainer.human_gs
    frame_idx = int(data['frame_idx'].detach().cpu())

    # Use optimized pose parameters when available
    if hasattr(hg, 'global_orient'):
        global_orient = rotation_6d_to_axis_angle(
            hg.global_orient[frame_idx].reshape(-1, 6)).reshape(3)
    else:
        global_orient = data['global_orient']

    if hasattr(hg, 'body_pose'):
        body_pose = rotation_6d_to_axis_angle(
            hg.body_pose[frame_idx].reshape(-1, 6)).reshape(23 * 3)
    else:
        body_pose = data['body_pose']

    betas = hg.betas if hasattr(hg, 'betas') else data['betas']
    transl = hg.transl[frame_idx] if hasattr(hg, 'transl') else data['transl']
    scale = data['smpl_scale']
    scale_val = scale.reshape(-1)[0] if scale.dim() > 0 else scale

    smpl_out = hg.smpl_template(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient.unsqueeze(0),
        disable_posedirs=False,
    )
    verts = smpl_out.vertices[0] * scale_val + transl.reshape(1, 3)

    # Fallback when no anchor_data: use SMPL joint positions for feet
    # SMPL joints: 7=left_ankle, 8=right_ankle, 10=left_foot, 11=right_foot
    if trainer.anchor_data is None:
        joints = smpl_out.joints[0] * scale_val + transl.reshape(1, 3)
        SMPL_FOOT_JOINTS = {
            'left_ankle': 7, 'right_ankle': 8,
            'left_foot': 10, 'right_foot': 11,
        }
        pts = [joints[idx] for idx in SMPL_FOOT_JOINTS.values()]
        found_names = list(SMPL_FOOT_JOINTS.keys())
        return torch.stack(pts, dim=0), found_names

    anchor_names = list(trainer.anchor_data['anchor_names'])
    vertex_ids = trainer.anchor_data['anchors']['vertex_ids']

    pts = []
    found_names = []
    for fa in FOOT_ANCHORS:
        if fa not in anchor_names:
            continue
        a = anchor_names.index(fa)
        ids = torch.as_tensor(vertex_ids[a], dtype=torch.long, device=verts.device)
        pts.append(verts[ids].mean(dim=0))
        found_names.append(fa)

    if not pts:
        raise RuntimeError(f"None of {FOOT_ANCHORS} found in anchor_names={anchor_names}")
    return torch.stack(pts, dim=0), found_names


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_frame(trainer, data, cfg, eval_iter, args, out_dir, frame_label):
    from hugs.renderer.gs_renderer import render_human_scene

    device = "cuda"
    bg_white = torch.ones(3, dtype=torch.float32, device=device)

    # Human forward
    human_gs_out = trainer.human_gs.forward(
        global_orient=data['global_orient'],
        body_pose=data['body_pose'],
        betas=data['betas'],
        transl=data['transl'],
        smpl_scale=data['smpl_scale'][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )
    scene_gs_out = trainer.scene_gs.forward()
    human_gs_out, _ = trainer.maybe_apply_anchor_attention(
        human_gs_out, scene_gs_out, cfg.mode, eval_iter,
    )

    # Foot positions
    foot_pts, foot_names = foot_anchor_world(trainer, data)  # (F, 3)
    foot_y = [f"{p[1].item():.6f}" for p in foot_pts]
    print(f"  foot_world_Y = [{', '.join(foot_y)}]")

    # Find scene GS near feet
    s_xyz = scene_gs_out['xyz']  # (N, 3)
    dist_sq = torch.cdist(s_xyz, foot_pts).min(dim=1).values  # (N,)
    near_foot_mask = dist_sq < args.radius
    n_near = int(near_foot_mask.sum())
    print(f"  Scene GS within R={args.radius}: {n_near} / {len(s_xyz)}")

    # Original render
    orig_pkg = render_human_scene(
        data=data,
        human_gs_out=human_gs_out,
        scene_gs_out=scene_gs_out,
        bg_color=bg_white,
        render_mode=cfg.mode,
    )
    orig_np = (orig_pkg['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Red-colored render for near-foot scene GS
    shs_modified = scene_gs_out['shs'].clone()
    if n_near > 0:
        red_dc = torch.tensor(
            [(1.0 - 0.5) / SH_C0, (0.0 - 0.5) / SH_C0, (0.0 - 0.5) / SH_C0],
            dtype=shs_modified.dtype, device=shs_modified.device,
        )
        shs_modified[near_foot_mask] = 0.0
        if shs_modified.dim() == 3:  # (N, num_coeff, 3)
            shs_modified[near_foot_mask, 0, :] = red_dc
        else:  # (N, 3) only_rgb
            shs_modified[near_foot_mask] = red_dc

    scene_gs_out_colored = dict(scene_gs_out)
    scene_gs_out_colored['shs'] = shs_modified

    colored_pkg = render_human_scene(
        data=data,
        human_gs_out=human_gs_out,
        scene_gs_out=scene_gs_out_colored,
        bg_color=bg_white,
        render_mode=cfg.mode,
    )
    colored_np = (colored_pkg['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    gt_np = (data['rgb'].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Side-by-side: GT | original render | foot-colored render
    comp = np.concatenate([gt_np, orig_np, colored_np], axis=1)
    comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
    W = gt_np.shape[1]
    for x, label in [(4, "GT"), (W + 4, "render"), (2 * W + 4, f"foot R={args.radius} ({n_near} GS)")]:
        cv2.putText(comp_bgr, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(comp_bgr, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_comp = out_dir / f"frame_{frame_label}_comp.png"
    out_foot = out_dir / f"frame_{frame_label}_foot.png"
    cv2.imwrite(str(out_comp), comp_bgr)
    cv2.imwrite(str(out_foot), cv2.cvtColor(colored_np, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_comp.name}, {out_foot.name}")
    return n_near


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='HUGS training output directory.')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Output directory (default: <run-dir>/diag_foot_occlusion/val/).')
    parser.add_argument('--frames', type=int, nargs='*', default=None,
                        help='Val frame indices to process (default: all).')
    parser.add_argument('--radius', type=float, default=1.0,
                        help='3D radius (m) around foot anchors to select scene GS (default: 1.0).')
    parser.add_argument('--anchor-ckpt', type=str, default=None,
                        help='Override anchor-attention checkpoint path.')
    parser.add_argument('--no-anchor', action='store_true',
                        help='Skip anchor-attention even if a checkpoint exists.')
    args = parser.parse_args()

    cfg, apply_anchor, anchor_ckpt_path = configure_from_run(args)

    from hugs.trainer import GaussianTrainer
    trainer = GaussianTrainer(cfg)
    trainer.human_gs.eval()

    if apply_anchor and anchor_ckpt_path:
        state = torch.load(anchor_ckpt_path)
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing:
            print(f"anchor ckpt missing keys: {len(missing)}")
        if unexpected:
            print(f"anchor ckpt unexpected keys: {len(unexpected)}")

    if trainer.anchor_data is None:
        print("No anchor_data found — using SMPL joint positions as foot fallback.")

    out_dir = args.out_dir or (Path(args.run_dir).resolve() / "diag_foot_occlusion" / "val")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {out_dir}")
    print(f"Radius     : {args.radius} m")
    print(f"Foot anchors: {FOOT_ANCHORS}")

    val_dataset = trainer.val_dataset
    frame_indices = args.frames if args.frames is not None else list(range(len(val_dataset)))
    eval_iter = cfg.train.num_steps

    total_near = 0
    for local_i, idx in enumerate(frame_indices):
        if idx >= len(val_dataset):
            print(f"Frame {idx} out of range ({len(val_dataset)} val frames), skipping.")
            continue
        data = val_dataset[idx]
        frame_idx = int(data['frame_idx'].detach().cpu())
        label = f"{local_i:02d}"
        print(f"Frame {label} (dataset idx={idx}, frame_idx={frame_idx}):")
        n = process_frame(trainer, data, cfg, eval_iter, args, out_dir, label)
        total_near += n

    print(f"\nDone. {len(frame_indices)} frames → {out_dir}")
    print(f"Total near-foot scene GS events: {total_near}")


if __name__ == '__main__':
    main()
