import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hugs.trainer.gs_trainer as gst
from hugs.cfg.config import cfg as default_cfg
from hugs.renderer.gs_renderer import render_human_scene
from hugs.utils.anchor_utils import (
    bind_gaussians_to_anchors,
    compute_binding_local_mask,
    compute_anchors,
    generate_semantic_anchor_vertices,
    load_smpl_template,
    make_template_from_vertices,
)
from hugs.utils.config import get_cfg_items
from hugs.utils.general import safe_state
from hugs.utils.rotations import rotation_6d_to_axis_angle


ANCHOR_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
]


def make_cfg(args):
    cfg_file = OmegaConf.load(args.cfg_file)
    cfg_items, _ = get_cfg_items(cfg_file)
    overrides = [
        f"dataset.seq={args.seq}",
        "mode=human_scene",
        f"exp_name={args.exp_name}",
        f"output_path={args.output_path}",
        "eval=false",
        "detect_anomaly=false",
    ]

    overrides.extend(
        [
            f"train.num_steps={args.steps}",
            f"train.save_ckpt_interval={args.export_interval}",
            "train.anim_interval=-1",
            "train.save_progress_images=false",
        ]
    )

    if args.preset == "smoke":
        overrides.extend(["train.val_interval=10000000", "train.progress_save_interval=10000000"])
        # Force early densification so a small smoke run has visible changing anchors->scene queries.
        scene_from = min(args.scene_densify_from, max(1, args.steps // 3))
        human_from = min(args.human_densify_from, max(1, args.steps // 2))
        overrides.extend(
            [
                f"scene.densify_from_iter={scene_from}",
                f"scene.densification_interval={args.scene_densify_interval}",
                "scene.opacity_reset_interval=10000000",
                f"human.densify_from_iter={human_from}",
                f"human.densification_interval={args.human_densify_interval}",
                "human.opacity_reset_interval=10000000",
                "human.loss.lpips_w=0.0",
                "scene.loss.lpips_w=0.0",
            ]
        )
    else:
        # Normal keeps the HUGS config's loss and densification schedule. We only
        # reduce validation/progress image overhead and save checkpoints often
        # enough to export ExpE overlays.
        overrides.extend(["train.val_interval=10000000", "train.progress_save_interval=10000000"])

    overrides.extend(args.opts)
    cfg = OmegaConf.merge(default_cfg, cfg_items[0], OmegaConf.from_dotlist(overrides))
    cfg.cfg_file = args.cfg_file

    if args.logdir:
        cfg.logdir = args.logdir
    else:
        cfg.logdir = str(Path(args.output_path) / args.exp_name / args.seq)
    cfg.logdir_ckpt = str(Path(cfg.logdir) / "ckpt")
    if args.no_resume:
        cfg.logdir_ckpt = str(Path(cfg.logdir) / "ckpt_no_resume")
    Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logdir, "ckpt").mkdir(exist_ok=True)
    Path(cfg.logdir_ckpt).mkdir(exist_ok=True)
    Path(cfg.logdir, "val").mkdir(exist_ok=True)
    Path(cfg.logdir, "train").mkdir(exist_ok=True)
    return cfg


def tensor_image_to_pil(image):
    arr = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def project_points(points, full_proj_transform, width, height):
    if points.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)
    clip = pts_h @ full_proj_transform.to(points.device)
    w = clip[:, 3].clamp(min=1e-8)
    ndc = clip[:, :3] / w[:, None]
    xy = torch.stack(
        [
            (ndc[:, 0] + 1.0) * 0.5 * (width - 1),
            (ndc[:, 1] + 1.0) * 0.5 * (height - 1),
        ],
        dim=-1,
    )
    visible = (
        (clip[:, 3] > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
        & (ndc[:, 2] > -1.2)
        & (ndc[:, 2] < 1.2)
    )
    return xy.detach().cpu().numpy(), visible.detach().cpu().numpy()


def heat_color(value):
    value = float(np.clip(value, 0.0, 1.0))
    # blue -> cyan -> yellow -> red
    if value < 0.5:
        t = value / 0.5
        return (int(40 * (1 - t)), int(130 + 110 * t), int(255 - 155 * t))
    t = (value - 0.5) / 0.5
    return (int(255), int(240 - 190 * t), int(80 * (1 - t)))


def load_expA_anchor_vertices(anchor_dir, smpl_root, gender):
    anchor_json = Path(anchor_dir) / "expA_template" / "anchor_vertices.json"
    if anchor_json.exists():
        with anchor_json.open("r") as f:
            return json.load(f)

    smpl_out = load_smpl_template(model_path=smpl_root, canonical_pose="template")
    anchors = generate_semantic_anchor_vertices(smpl_out)
    anchor_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [int(v) for v in ids] for name, ids in anchors.items()}
    with anchor_json.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


def compute_anchors_from_hugs_model(human_gs, anchor_vertices):
    template = make_template_from_vertices(
        human_gs.get_vitruvian_verts_template(),
        human_gs.smpl_template.faces,
        human_gs.smpl_template.lbs_weights,
        canonical_pose="hugs_vitruvian",
    )
    return compute_anchors(template, anchor_vertices)


def get_optimized_smpl_params(human_gs, data, frame_idx):
    if hasattr(human_gs, "global_orient"):
        global_orient = rotation_6d_to_axis_angle(human_gs.global_orient[frame_idx].reshape(-1, 6)).reshape(3)
    else:
        global_orient = data["global_orient"]

    if hasattr(human_gs, "body_pose"):
        body_pose = rotation_6d_to_axis_angle(human_gs.body_pose[frame_idx].reshape(-1, 6)).reshape(23 * 3)
    else:
        body_pose = data["body_pose"]

    betas = human_gs.betas if hasattr(human_gs, "betas") else data["betas"]
    transl = human_gs.transl[frame_idx] if hasattr(human_gs, "transl") else data["transl"]
    return global_orient, body_pose, betas, transl


@torch.no_grad()
def compute_posed_anchor_points(human_gs, anchor_vertices, data, frame_idx):
    global_orient, body_pose, betas, transl = get_optimized_smpl_params(human_gs, data, frame_idx)
    smpl_out = human_gs.smpl_template(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient.unsqueeze(0),
        disable_posedirs=False,
    )
    verts = smpl_out.vertices[0]
    smpl_scale = data["smpl_scale"]
    if smpl_scale.dim() == 0:
        verts = verts * smpl_scale
    else:
        verts = verts * smpl_scale.reshape(-1)[0]
    verts = verts + transl.reshape(1, 3)

    anchor_world = []
    for name in anchor_vertices.keys():
        ids = torch.as_tensor(anchor_vertices[name], dtype=torch.long, device=verts.device)
        anchor_world.append(verts[ids].mean(dim=0))
    return torch.stack(anchor_world, dim=0)


@torch.no_grad()
def export_cross_attention_snapshot(
    trainer,
    anchor_vertices,
    iter_id,
    out_dir,
    frame_idx=0,
    top_k=32,
    temperature=0.08,
    opacity_weight=0.25,
    draw_local_human=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = trainer.train_dataset[frame_idx % len(trainer.train_dataset)]
    for key, value in list(data.items()):
        if torch.is_tensor(value):
            data[key] = value.cuda()

    anchors = compute_anchors_from_hugs_model(trainer.human_gs, anchor_vertices)
    human_gs_out = trainer.human_gs.forward(
        smpl_scale=data["smpl_scale"][None] if data["smpl_scale"].dim() == 0 else data["smpl_scale"],
        dataset_idx=frame_idx,
        is_train=False,
    )
    scene_gs_out = trainer.scene_gs.forward()

    render_pkg = render_human_scene(
        data,
        human_gs_out,
        scene_gs_out,
        trainer.bg_color,
        scaling_modifier=1.0,
        render_mode="human_scene",
    )
    image = render_pkg["render"]
    pil = tensor_image_to_pil(image)
    raw_path = out_dir / f"iter{iter_id:06d}_frame{frame_idx:04d}_render.png"
    overlay_path = out_dir / f"iter{iter_id:06d}_frame{frame_idx:04d}_cross_attention_overlay.png"
    pil.save(raw_path)

    xyz_canon = human_gs_out.get("xyz_canon", trainer.human_gs.get_xyz).detach().cpu()
    lbs_weights = human_gs_out.get("lbs_weights", None)
    lbs_cpu = lbs_weights.detach().cpu() if lbs_weights is not None else None
    binding = bind_gaussians_to_anchors(
        xyz_canon,
        lbs_cpu,
        anchors,
        top_m=2,
        lambda_lbs=0.35,
        sigma_scale=1.8,
        min_sigma=0.035,
    )
    anchor_ids = binding["top1_anchor_id"].cuda()
    anchor_w = binding["gaussian_anchor_weights"][:, 0].cuda()
    local_mask = compute_binding_local_mask(binding, vis_radius_factor=2.5).cuda()

    human_xyz = human_gs_out["xyz"].detach()
    scene_xyz = scene_gs_out["xyz"].detach()
    scene_opacity = scene_gs_out["opacity"].detach().flatten().clamp(min=1e-6)
    num_anchors = len(anchors["names"])

    anchor_world = compute_posed_anchor_points(trainer.human_gs, anchor_vertices, data, frame_idx)

    k = min(top_k, scene_xyz.shape[0])
    dists = torch.cdist(anchor_world, scene_xyz)
    knn_dist, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)
    knn_opacity = scene_opacity[knn_idx]
    # ExpE nearest-neighbor probe: compare all anchor-scene KNN distances in
    # one global scale. Red means geometrically closer among all anchors; blue
    # means the anchor's neighbors are far even if they are its own nearest.
    global_min_dist = knn_dist.min()
    global_max_dist = knn_dist.max()
    attn = 1.0 - (knn_dist - global_min_dist) / (global_max_dist - global_min_dist).clamp_min(1e-6)
    attn = attn.clamp(0.0, 1.0)

    scene_strength = torch.zeros(scene_xyz.shape[0], device=scene_xyz.device)
    scene_owner = torch.full((scene_xyz.shape[0],), -1, device=scene_xyz.device, dtype=torch.long)
    for a_idx in range(num_anchors):
        for j in range(k):
            idx = knn_idx[a_idx, j]
            if attn[a_idx, j] > scene_strength[idx]:
                scene_strength[idx] = attn[a_idx, j]
                scene_owner[idx] = a_idx
    scene_strength_norm = scene_strength.clamp(0.0, 1.0)

    height, width = image.shape[1], image.shape[2]
    full_proj = data["full_proj_transform"]
    scene_xy, scene_vis = project_points(scene_xyz, full_proj, width, height)
    anchor_xy, anchor_vis = project_points(anchor_world, full_proj, width, height)
    top_scene_xy, top_scene_vis = project_points(scene_xyz[knn_idx[:, 0]], full_proj, width, height)

    overlay = pil.convert("RGBA")
    dots = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dots)

    queried = torch.unique(knn_idx.flatten()).detach().cpu().numpy()
    for scene_idx in queried:
        if not scene_vis[scene_idx]:
            continue
        s = float(scene_strength_norm[scene_idx].detach().cpu())
        if s <= 0:
            continue
        x, y = scene_xy[scene_idx]
        color = heat_color(s) + (190,)
        radius = 2 + int(4 * s)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    if draw_local_human:
        human_xy, human_vis = project_points(human_xyz, full_proj, width, height)
        local_idx = torch.where(local_mask)[0].detach().cpu().numpy()
        # Draw a sparse but stable subset to avoid hiding the rendered person.
        stride = max(1, len(local_idx) // 1200)
        for idx in local_idx[::stride]:
            if not human_vis[idx]:
                continue
            a = int(anchor_ids[idx].detach().cpu())
            x, y = human_xy[idx]
            color = ANCHOR_COLORS[a % len(ANCHOR_COLORS)] + (80,)
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=color)

    overlay = Image.alpha_composite(overlay, dots)
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        font = None
        small_font = None

    for a_idx, anchor_name in enumerate(anchors["names"]):
        if not anchor_vis[a_idx]:
            continue
        x, y = anchor_xy[a_idx]
        max_a = float(attn[a_idx].max().detach().cpu())
        color = heat_color(max_a)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color + (255,), width=3)
        if top_scene_vis[a_idx]:
            tx, ty = top_scene_xy[a_idx]
            draw.line((x, y, tx, ty), fill=color + (180,), width=2)
        label = f"{a_idx}:{anchor_name} {max_a:.2f}"
        draw.text((x + 8, y - 8), label, fill=(255, 255, 255, 255), font=small_font, stroke_width=2, stroke_fill=(0, 0, 0, 190))

    # Legend
    lx, ly = 16, max(16, height - 50)
    draw.rectangle((lx - 8, ly - 8, lx + 220, ly + 34), fill=(0, 0, 0, 120))
    for i in range(120):
        t = i / 119.0
        draw.line((lx + i, ly, lx + i, ly + 12), fill=heat_color(t) + (255,))
    draw.text((lx, ly + 16), "anchor distance: far -> near", fill=(255, 255, 255, 255), font=font)
    overlay.convert("RGB").save(overlay_path)

    attn_prob = torch.softmax(-knn_dist / max(temperature, 1e-5), dim=1)
    entropy = -(attn_prob * (attn_prob.clamp(min=1e-8).log())).sum(dim=1)
    rows = []
    for a_idx, anchor_name in enumerate(anchors["names"]):
        rows.append(
            {
                "iter": int(iter_id),
                "frame_idx": int(frame_idx),
                "anchor_id": int(a_idx),
                "anchor_name": anchor_name,
                "num_controlled_human_gaussians": int(((anchor_ids == a_idx) & local_mask).sum().detach().cpu()),
                "nearest_scene_distance": float(knn_dist[a_idx, 0].detach().cpu()),
                "distance_weighted_distance": float((knn_dist[a_idx] * attn_prob[a_idx]).sum().detach().cpu()),
                "nearest_neighbor_strength": float(attn[a_idx, 0].detach().cpu()),
                "distance_weight_entropy": float(entropy[a_idx].detach().cpu()),
                "top_scene_opacity": float(knn_opacity[a_idx, 0].detach().cpu()),
            }
        )

    csv_path = out_dir / f"iter{iter_id:06d}_frame{frame_idx:04d}_cross_attention.csv"
    with csv_path.open("w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    torch.save(
        {
            "iter": int(iter_id),
            "frame_idx": int(frame_idx),
            "anchor_names": list(anchors["names"]),
            "anchor_world": anchor_world.detach().cpu(),
            "knn_idx": knn_idx.detach().cpu(),
            "knn_dist": knn_dist.detach().cpu(),
            "attention": attn.detach().cpu(),
            "attention_definition": "global KNN distance strength across all anchor-scene pairs: 1 is globally nearest, 0 is globally farthest",
            "scene_strength": scene_strength.detach().cpu(),
            "scene_owner": scene_owner.detach().cpu(),
        },
        out_dir / f"iter{iter_id:06d}_frame{frame_idx:04d}_cross_attention.pt",
    )
    return {"raw": str(raw_path), "overlay": str(overlay_path), "csv": str(csv_path)}


def load_checkpoint_pair(trainer, cfg, iter_id):
    ckpt_dir = Path(cfg.logdir_ckpt)
    human_ckpt = ckpt_dir / f"human_{iter_id:06d}.pth"
    scene_ckpt = ckpt_dir / f"scene_{iter_id:06d}.pth"
    if not human_ckpt.exists() or not scene_ckpt.exists():
        return False
    trainer.human_gs.load_state_dict(torch.load(human_ckpt), cfg.human.lr)
    trainer.scene_gs.restore(torch.load(scene_ckpt), cfg.scene.lr)
    return True


def main():
    parser = argparse.ArgumentParser("Run real HUGS training and export ExpE anchor->scene attention overlays.")
    parser.add_argument("--seq", default="lab")
    parser.add_argument("--cfg-file", default="cfg_files/release/neuman/hugs_human_scene.yaml")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-path", default="./output")
    parser.add_argument("--exp-name", default="anchor_debug/expE_real_hugs_smoke")
    parser.add_argument("--logdir", default="")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--export-interval", type=int, default=100)
    parser.add_argument("--preset", choices=["smoke", "normal"], default="smoke")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--opacity-weight", type=float, default=0.25)
    parser.add_argument("--scene-densify-from", type=int, default=50)
    parser.add_argument("--scene-densify-interval", type=int, default=50)
    parser.add_argument("--human-densify-from", type=int, default=80)
    parser.add_argument("--human-densify-interval", type=int, default=80)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Only export existing checkpoints from --logdir; skips iter 0.")
    parser.add_argument("--no-resume", action="store_true", help="Do not auto-load checkpoints while constructing the trainer.")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.dataset_path:
        args.opts = [f"dataset_path={args.dataset_path}"] + args.opts

    cfg = make_cfg(args)
    safe_state(seed=cfg.seed)
    print(OmegaConf.to_yaml(cfg))
    gst.get_anim_dataset = lambda cfg: None
    if args.preset == "smoke" or args.skip_train or args.export_only:
        gst.optimize_init = lambda human_gs, num_steps=7000: human_gs

    smpl_root = cfg.dataset.get("smpl_path", "data/smpl")
    anchor_vertices = load_expA_anchor_vertices(
        Path(cfg.output_path) / "anchor_debug",
        smpl_root,
        cfg.human.get("gender", "neutral"),
    )

    trainer = gst.GaussianTrainer(cfg)

    out_dir = Path(cfg.logdir) / "anchor_cross_attention"
    if not args.export_only:
        export_cross_attention_snapshot(
            trainer,
            anchor_vertices,
            iter_id=0,
            out_dir=out_dir,
            frame_idx=args.frame_idx,
            top_k=args.top_k,
            temperature=args.temperature,
            opacity_weight=args.opacity_weight,
        )

    if not args.skip_train and not args.export_only:
        trainer.train()

    export_iters = {args.steps}
    if not args.export_only:
        export_iters.add(0)
    export_iters.update(range(args.export_interval, args.steps + 1, args.export_interval))
    for iter_id in sorted(export_iters):
        if iter_id == 0:
            continue
        loaded = load_checkpoint_pair(trainer, cfg, iter_id)
        if not loaded and iter_id != args.steps:
            print(f"[ExpE] skip missing checkpoint iter {iter_id}")
            continue
        export_cross_attention_snapshot(
            trainer,
            anchor_vertices,
            iter_id=iter_id,
            out_dir=out_dir,
            frame_idx=args.frame_idx,
            top_k=args.top_k,
            temperature=args.temperature,
            opacity_weight=args.opacity_weight,
        )

    readme = out_dir / "README_expE.md"
    readme.write_text(
        f"# ExpE real HUGS {args.preset} run\n\n"
        "This folder contains anchor-to-scene KNN distance diagnostics exported from a real HUGS human_scene training run.\n\n"
        "- `*_render.png`: original full human+scene 3DGS render.\n"
        "- `*_cross_attention_overlay.png`: same render with queried scene Gaussians colored by globally normalized anchor-scene KNN distance. Blue is farther among all queried anchor-scene pairs, red is globally closer. Anchor circles and lines mark each anchor's nearest scene Gaussian.\n"
        "- `*_cross_attention.csv`: per-anchor KNN/distance summary.\n"
        "- `*_cross_attention.pt`: raw tensors for downstream analysis.\n",
        encoding="utf-8",
    )
    print(f"[ExpE] outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
