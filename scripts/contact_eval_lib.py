# Shared utilities for contact/probe evaluation scripts.

import glob
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf

sys.path.append('.')

import hugs.trainer.gs_trainer as gst
from hugs.cfg.config import cfg as default_cfg
from hugs.losses.utils import ssim
from hugs.renderer.gs_renderer import render_human_scene
from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state


def natural_ckpt_key(path):
    name = Path(path).stem
    if name.endswith('final'):
        return (10**12, name)
    digits = ''.join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else -1, name)


def find_ckpt(output_dir, pattern):
    output_dir = Path(output_dir)
    files = glob.glob(str(output_dir / 'ckpt' / pattern))
    files += glob.glob(str(output_dir / pattern))
    files = sorted(set(files), key=natural_ckpt_key)
    return files[-1] if files else None


def configure_from_output(output_dir, extras=None, enable_anchors=False, enable_attention=False, anchor_ckpt=None):
    output_dir = Path(output_dir)
    cfg_path = output_dir / 'config_train.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing config_train.yaml under {output_dir}')
    extras = extras or []
    cfg_file = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(default_cfg, cfg_file, OmegaConf.from_cli(extras))
    cfg.eval = True
    cfg.logdir = str(output_dir)
    cfg.logdir_ckpt = str(output_dir / 'ckpt')
    cfg.train.anim_interval = -1
    cfg.train.save_progress_images = False

    human_ckpt = find_ckpt(output_dir, 'human_*.pth')
    scene_ckpt = find_ckpt(output_dir, 'scene_*.pth')
    found_anchor_ckpt = find_ckpt(output_dir, 'anchor_attention_*.pth')
    if cfg.mode in ['human', 'human_scene']:
        if human_ckpt is None:
            raise FileNotFoundError(f'Missing human checkpoint under {output_dir}/ckpt')
        cfg.human.ckpt = human_ckpt
    if cfg.mode in ['scene', 'human_scene']:
        if scene_ckpt is None:
            raise FileNotFoundError(f'Missing scene checkpoint under {output_dir}/ckpt')
        cfg.scene.ckpt = scene_ckpt

    if enable_anchors:
        cfg.anchor_attention.use_anchors = True
    if enable_attention:
        cfg.anchor_attention.use_anchors = True
        cfg.anchor_attention.use_anchor_token_encoder = True
        cfg.anchor_attention.use_scene_query = True
        cfg.anchor_attention.use_cross_attention = True
        cfg.anchor_attention.use_interaction_correction = False
    if anchor_ckpt:
        cfg.anchor_attention.ckpt = str(anchor_ckpt)
    elif found_anchor_ckpt and enable_attention:
        cfg.anchor_attention.ckpt = found_anchor_ckpt

    ckpts = {'human': human_ckpt, 'scene': scene_ckpt, 'anchor_attention': getattr(cfg.anchor_attention, 'ckpt', '') or None}
    return cfg, ckpts


def make_trainer(output_dir, extras=None, enable_anchors=False, enable_attention=False, anchor_ckpt=None):
    gst.get_anim_dataset = lambda cfg: None
    cfg, ckpts = configure_from_output(output_dir, extras, enable_anchors, enable_attention, anchor_ckpt)
    safe_state(seed=cfg.seed)
    trainer = GaussianTrainer(cfg)
    return trainer, ckpts


def to_cpu_float(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def get_rgb_mask_bbox(dataset, data):
    if 'rgb' in data:
        rgb = data['rgb'].clamp(0, 1)
        mask = data.get('mask')
        bbox = data.get('bbox')
        return rgb, mask, bbox
    frame_idx = int(data.get('frame_idx', torch.tensor(0)).detach().cpu())
    device = data['world_view_transform'].device
    img = dataset.scene.captures[frame_idx].captured_image.image
    img = (img[..., :3] / 255.0).astype(np.float32).transpose(2, 0, 1)
    rgb = torch.from_numpy(img).float().to(device)
    mask = None
    bbox = None
    if hasattr(dataset, 'msk_lists') and frame_idx < len(dataset.msk_lists):
        msk = cv2.imread(dataset.msk_lists[frame_idx], cv2.IMREAD_GRAYSCALE)
        if msk is not None:
            msk = (msk.astype(np.float32) / 255.0)
            mask = torch.from_numpy(msk).float().to(device)
            rows = np.any(msk > 0.5, axis=0)
            cols = np.any(msk > 0.5, axis=1)
            if rows.any() and cols.any():
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=device)
    return rgb, mask, bbox


@torch.no_grad()
def forward_frame(trainer, data, apply_attention=False, iteration=None):
    human_out = None
    scene_out = None
    if trainer.human_gs:
        human_out = trainer.human_gs.forward(
            global_orient=data['global_orient'],
            body_pose=data['body_pose'],
            betas=data['betas'],
            transl=data['transl'],
            smpl_scale=data['smpl_scale'][None],
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
        )
    if trainer.scene_gs:
        scene_out = trainer.scene_gs.forward()
    stats = {}
    if apply_attention:
        human_out, stats = trainer.maybe_apply_anchor_attention(
            human_out, scene_out, trainer.cfg.mode, trainer.cfg.train.num_steps if iteration is None else iteration
        )
    return human_out, scene_out, stats


@torch.no_grad()
def render_frame(trainer, data, human_out, scene_out):
    return render_human_scene(
        data=data,
        human_gs_out=human_out,
        scene_gs_out=scene_out,
        bg_color=trainer.bg_color,
        render_mode=trainer.cfg.mode,
    )['render'].clamp(0, 1)


def anchor_world_from_bindings(human_out, anchor_ids, anchor_weights, num_anchors, top_m=2):
    xyz = human_out['xyz']
    top_m = min(int(top_m), anchor_ids.shape[1])
    device = xyz.device
    dtype = xyz.dtype
    anchor_world = torch.zeros(num_anchors, 3, device=device, dtype=dtype)
    denom = torch.zeros(num_anchors, 1, device=device, dtype=dtype)
    ids = anchor_ids[:, :top_m].reshape(-1).to(device).long()
    weights = anchor_weights[:, :top_m].reshape(-1, 1).to(device=device, dtype=dtype)
    xyz_rep = xyz[:, None, :].expand(-1, top_m, -1).reshape(-1, 3)
    anchor_world.scatter_add_(0, ids[:, None].expand(-1, 3), xyz_rep * weights)
    denom.scatter_add_(0, ids[:, None], weights)
    global_center = xyz.mean(dim=0, keepdim=True)
    return torch.where(denom > 1e-6, anchor_world / denom.clamp_min(1e-6), global_center)



def posed_semantic_anchor_world(trainer, data, frame_idx, use_optimized_pose=False):
    """Return posed semantic anchor centers from the anchor vertex regions.

    This is the ExpE-style visual anchor position: pose the SMPL template and
    average each anchor's semantic vertex set. It is better for overlay/ROI
    visualization than the center of all Gaussians bound to an anchor.
    """
    from hugs.utils.rotations import rotation_6d_to_axis_angle

    human_gs = trainer.human_gs
    if use_optimized_pose and hasattr(human_gs, 'global_orient'):
        global_orient = rotation_6d_to_axis_angle(human_gs.global_orient[int(frame_idx)].reshape(-1, 6)).reshape(3)
    else:
        global_orient = data['global_orient']
    if use_optimized_pose and hasattr(human_gs, 'body_pose'):
        body_pose = rotation_6d_to_axis_angle(human_gs.body_pose[int(frame_idx)].reshape(-1, 6)).reshape(23 * 3)
    else:
        body_pose = data['body_pose']
    betas = human_gs.betas if use_optimized_pose and hasattr(human_gs, 'betas') else data['betas']
    transl = human_gs.transl[int(frame_idx)] if use_optimized_pose and hasattr(human_gs, 'transl') else data['transl']

    smpl_out = human_gs.smpl_template(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient.unsqueeze(0),
        disable_posedirs=False,
    )
    verts = smpl_out.vertices[0]
    smpl_scale = data['smpl_scale']
    verts = verts * (smpl_scale.reshape(-1)[0] if smpl_scale.dim() > 0 else smpl_scale)
    verts = verts + transl.reshape(1, 3)

    vertex_ids = trainer.anchor_data['anchors']['vertex_ids']
    centers = []
    for ids in vertex_ids:
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=verts.device)
        centers.append(verts[ids_t].mean(dim=0))
    return torch.stack(centers, dim=0)

def project_points(data, points):
    if points.numel() == 0:
        return torch.empty(0, 2, device=points.device), torch.empty(0, device=points.device)
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)
    view = data['world_view_transform'].to(points.device, points.dtype)
    cam = pts_h @ view
    z = cam[:, 2]
    if 'cam_intrinsics' in data:
        k = data['cam_intrinsics'].to(points.device, points.dtype)
        z_safe = torch.where(z.abs() > 1e-8, z, torch.full_like(z, 1e-8))
        x = k[0, 0] * (cam[:, 0] / z_safe) + k[0, 2]
        y = k[1, 1] * (cam[:, 1] / z_safe) + k[1, 2]
        return torch.stack([x, y], dim=-1), z

    clip = cam @ data['projection_matrix'].to(points.device, points.dtype) if 'projection_matrix' in data else pts_h @ data['full_proj_transform'].to(points.device, points.dtype)
    w = torch.where(clip[:, 3].abs() > 1e-8, clip[:, 3], torch.full_like(clip[:, 3], 1e-8))
    ndc = clip[:, :3] / w[:, None]
    width = float(data['image_width'])
    height = float(data['image_height'])
    x = (ndc[:, 0] + 1.0) * 0.5 * width
    y = (ndc[:, 1] + 1.0) * 0.5 * height
    return torch.stack([x, y], dim=-1), ndc[:, 2]


def crop_tensor(img, center_xy, size):
    c = center_xy.detach().round().long()
    x, y = int(c[0]), int(c[1])
    half = int(size) // 2
    h, w = img.shape[-2], img.shape[-1]
    x0, x1 = max(0, x - half), min(w, x + half)
    y0, y1 = max(0, y - half), min(h, y + half)
    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0, x1, y1)
    return img[:, y0:y1, x0:x1], (x0, y0, x1, y1)


def psnr_value(pred, gt):
    mse = torch.mean((pred - gt).pow(2)).clamp_min(1e-8)
    return float((-10.0 * torch.log10(mse)).detach().cpu())


def ssim_value(pred, gt):
    if pred.shape[-1] < 8 or pred.shape[-2] < 8:
        return None
    return float(ssim(pred, gt).mean().detach().cpu())


def save_image_tensor(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(tensor.detach().cpu().clamp(0, 1), str(path))


def draw_anchor_overlay(gt, render, anchors_xy, names, out_path, radius=5):
    gt_np = (gt.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    rd_np = (render.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    canvas = np.concatenate([gt_np, rd_np], axis=1)
    width = gt_np.shape[1]
    colors = [(255, 64, 64), (64, 180, 255), (80, 220, 120), (255, 190, 64), (220, 80, 220)]
    for i, (xy, name) in enumerate(zip(anchors_xy, names)):
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if x < 0 or y < 0 or x >= width or y >= gt_np.shape[0]:
            continue
        color = colors[i % len(colors)]
        cv2.circle(canvas, (x, y), radius, color, 2)
        cv2.circle(canvas, (x + width, y), radius, color, 2)
        cv2.putText(canvas, name[:18], (x + 4, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, name[:18], (x + width + 4, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def write_ply(path, points, colors=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.max() <= 1.0:
            colors = (colors * 255.0).clip(0, 255)
        colors = colors.astype(np.uint8).reshape(-1, 3)
    with path.open('w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n')



def read_deco_pred_obj(path, contact_color=(0.0, 1.0, 0.0), gray_color=(130.0 / 255.0, 130.0 / 255.0, 130.0 / 255.0)):
    """Read DECO demo pred.obj and recover binary vertex contact from colors."""
    path = Path(path)
    verts = []
    colors = []
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('v '):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
    verts = np.asarray(verts, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return verts, colors.reshape(0, 3), np.zeros((0,), dtype=np.float32)

    contact_color = np.asarray(contact_color, dtype=np.float32)
    gray_color = np.asarray(gray_color, dtype=np.float32)
    dist_contact = np.linalg.norm(colors - contact_color[None], axis=1)
    dist_gray = np.linalg.norm(colors - gray_color[None], axis=1)
    contact = (dist_contact < dist_gray).astype(np.float32)
    return verts, colors, contact


def deco_contact_obj_path(deco_dir, frame_idx):
    deco_dir = Path(deco_dir)
    frame = f'{int(frame_idx):05d}'
    return deco_dir / 'Preds' / frame / 'pred.obj'


def deco_contact_png_path(deco_dir, frame_idx):
    deco_dir = Path(deco_dir)
    frame = f'{int(frame_idx):05d}'
    return deco_dir / 'Preds' / frame / f'{frame}.png'


def aggregate_deco_contact_to_anchors(contact, anchor_vertex_ids, anchor_names):
    contact = np.asarray(contact, dtype=np.float32).reshape(-1)
    probs = {}
    counts = {}
    for name, ids in zip(anchor_names, anchor_vertex_ids):
        ids = np.asarray(ids, dtype=np.int64)
        valid = ids[(ids >= 0) & (ids < contact.shape[0])]
        if valid.size == 0:
            probs[name] = 0.0
            counts[name] = 0
        else:
            probs[name] = float(contact[valid].mean())
            counts[name] = int(valid.size)
    return probs, counts


def read_colmap_points3d(path):
    pts = []
    cols = []
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)


def summarize_values(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return {'count': 0}
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
