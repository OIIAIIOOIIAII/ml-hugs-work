#!/usr/bin/env python3
"""Temporal contact quality evaluation for HUGS models.

Evaluates per-frame rendering quality stability and foot-ground contact
stability across all frames in a sequence.

Metrics (Phase 1 — always computed):
  - Temporal PSNR/SSIM stability (mean, std, stability = 1/(1+std))
  - Flickering score (|ΔPSNR| between adjacent frames)
  - Foot Sliding (contact-frame XZ velocity, cm/frame)
  - Contact Jitter (depth std within contact periods, mm)
  - Contact Consistency (1 - state transition rate)

Metrics (Phase 2 — only when anchor attention is present):
  - Contact Attention Stability (cosine similarity of attention weights)

Metrics (Phase 3 — motion quality):
  - Motion Smoothness (foot XYZ acceleration std, cm/frame²)

Usage:
    python scripts/evaluate_temporal_quality.py \\
        --run-dir output/human_scene/neuman/lab/hugs_trimlp/<exp>/<ts> \\
        --seq lab \\
        [--contact-thresh 0.43] \\
        [--foot-topk 300] \\
        [--with-lpips]

Outputs in <run-dir>/temporal_eval/:
    report.json   — all metrics
    report.txt    — human-readable report
    psnr_timeline.png, contact_metrics.png, attention_stability.png
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers: trainer loading (mirrors render_full_video.py)
# ---------------------------------------------------------------------------

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


def load_trainer(run_dir: Path, device: str = "cuda"):
    from hugs.cfg.config import cfg as default_cfg
    from hugs.trainer import GaussianTrainer

    run_dir = run_dir.resolve()
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

    cfg.human.ckpt = find_latest_ckpt(run_dir, "human_*.pth")
    cfg.scene.ckpt = find_latest_ckpt(run_dir, "scene_*.pth")
    anchor_ckpt   = find_latest_ckpt(run_dir, "anchor_attention_*.pth")

    has_anchor = bool(anchor_ckpt and Path(anchor_ckpt).is_file())
    if has_anchor:
        # Keep anchor_attention.use_anchors from saved config (True)
        # but clear the ckpt field so the trainer doesn't try to load it twice
        cfg.anchor_attention.ckpt = ""

    trainer = GaussianTrainer(cfg)

    if has_anchor:
        state = torch.load(anchor_ckpt, map_location=device)
        missing, unexpected = trainer.anchor_attention.load_state_dict(state, strict=False)
        if missing:
            print(f"  Anchor ckpt: {len(missing)} missing keys (new temporal params — expected)")

    print(f"  Human ckpt:  {cfg.human.ckpt}")
    print(f"  Scene ckpt:  {cfg.scene.ckpt}")
    print(f"  Anchor ckpt: {anchor_ckpt if has_anchor else 'None'}")
    return trainer, has_anchor


# ---------------------------------------------------------------------------
# Per-frame forward + render
# ---------------------------------------------------------------------------

def build_full_dataset(trainer) -> list[dict]:
    """Combine train + val datasets sorted by frame_idx.

    all_dataset has camera params but NO GT images.
    train_dataset + val_dataset have GT images and cover all frames.
    Returns a list of data dicts sorted by frame_idx (0, 1, 2, ...).
    """
    sources = []
    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        for i in range(len(trainer.train_dataset)):
            sources.append(trainer.train_dataset[i])
    if hasattr(trainer, "val_dataset") and trainer.val_dataset is not None:
        for i in range(len(trainer.val_dataset)):
            sources.append(trainer.val_dataset[i])

    # Sort by frame_idx
    sources.sort(key=lambda d: int(d["frame_idx"].item()))
    return sources


@torch.no_grad()
def process_all_frames(
    trainer,
    has_anchor: bool,
    foot_topk: int = 300,
    with_lpips: bool = False,
    device: str = "cuda",
):
    """Render every frame (train+val, sorted by frame_idx) and collect data.

    Foot positions are obtained from SMPL joints (joint 10=left foot, 11=right foot)
    transformed to world space via smpl_scale and transl.  This is more reliable than
    averaging top-K Gaussian centers, which conflates both feet and is sensitive to
    which Gaussians happen to be near the ground.

    Returns a dict with keys:
        psnr_seq       : (N,)
        ssim_seq       : (N,)
        lpips_seq      : (N,) or None
        left_foot_xyz  : (N, 3)  — left  foot joint in world space
        right_foot_xyz : (N, 3)  — right foot joint in world space
        attn_weights   : list of Tensors or None
    """
    from hugs.renderer.gs_renderer import render_human_scene
    from hugs.utils.image import psnr as compute_psnr
    from hugs.losses.utils import ssim as compute_ssim

    # SMPL joint indices for feet
    LEFT_FOOT_J  = 10
    RIGHT_FOOT_J = 11

    lpips_fn = None
    if with_lpips:
        from lpips import LPIPS
        lpips_fn = LPIPS(net="alex", pretrained=True).to(device)

    # Build frame list: train + val sorted by frame_idx
    all_frames = build_full_dataset(trainer)
    N = len(all_frames)
    eval_iter = trainer.cfg.train.num_steps

    psnr_seq       = []
    ssim_seq       = []
    lpips_seq      = [] if with_lpips else None
    left_foot_xyz  = []
    right_foot_xyz = []
    attn_weights_seq = [] if has_anchor else None

    # Pre-compute foot Gaussian masks from SMPL LBS weights.
    # We use the model's canonical Gaussian positions to look up the nearest
    # SMPL template vertex (via KDTree) and inherit its LBS weights.
    # This gives model-specific foot tracking that captures transl corrections.
    print("  [Pre-computing foot Gaussian masks]")
    try:
        from scipy.spatial import KDTree
        _template_verts = trainer.human_gs.vitruvian_verts.detach().cpu().numpy()  # (V, 3)
        _smpl_lbs = trainer.human_gs.smpl.lbs_weights.detach().cpu().numpy()      # (V, 24)
        _canon_xyz = trainer.human_gs._xyz.detach().cpu().numpy()                  # (G, 3)
        _tree = KDTree(_template_verts)
        _, _nn_idx = _tree.query(_canon_xyz, k=1)
        _gauss_lbs = _smpl_lbs[_nn_idx]       # (G, 24)
        # SMPL joints: 10 = L_Foot, 11 = R_Foot (sole region)
        # Use threshold 0.3 to capture the foot pad area
        _left_mask  = _gauss_lbs[:, 10] > 0.3   # (G,) bool
        _right_mask = _gauss_lbs[:, 11] > 0.3   # (G,) bool
        _foot_topk  = 30   # take bottom-30% of masked Gaussians by Y for sole contact
        _use_lbs_mask = True
        print(f"    L_Foot mask: {_left_mask.sum()} Gaussians  |  R_Foot mask: {_right_mask.sum()} Gaussians")
    except Exception as e:
        print(f"  [WARN] KDTree foot mask failed ({e}), falling back to top-K by Y")
        _use_lbs_mask = False

    for seq_idx, data in enumerate(all_frames):
        frame_idx = int(data["frame_idx"].item())

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

        anchor_stats = {}
        if has_anchor:
            human_out, anchor_stats = trainer.maybe_apply_anchor_attention(
                human_out, scene_out, trainer.cfg.mode, eval_iter,
                frame_idx=frame_idx,
            )

        render_pkg = render_human_scene(
            data=data,
            human_gs_out=human_out,
            scene_gs_out=scene_out,
            bg_color=trainer.bg_color,
            render_mode=trainer.cfg.mode,
        )
        rendered = render_pkg["render"].clamp(0.0, 1.0)   # (3, H, W)
        gt = data["rgb"].to(device)                        # (3, H, W)

        psnr_val = compute_psnr(rendered[None], gt[None]).mean().item()
        ssim_val = compute_ssim(rendered[None], gt[None]).mean().item()
        psnr_seq.append(psnr_val)
        ssim_seq.append(ssim_val)

        if with_lpips and lpips_fn is not None:
            lp = lpips_fn(rendered[None].clamp(max=1), gt[None]).mean().item()
            lpips_seq.append(lp)

        # Foot positions: use LBS-weighted Gaussian positions from corrected human_out.
        # This is model-specific because it captures transl + xyz corrections.
        xyz_np = human_out["xyz"].detach().cpu().numpy()  # (G, 3) world-space corrected

        if _use_lbs_mask:
            def _foot_pos(mask):
                if mask.sum() < 3:
                    return xyz_np[mask].mean(axis=0) if mask.sum() > 0 else xyz_np.mean(axis=0)
                gauss = xyz_np[mask]  # foot Gaussians
                # Take the bottom-most (max Y in Y-down = closest to ground) 30%
                k = max(3, len(gauss) // 3)
                top_y = np.argpartition(gauss[:, 1], -k)[-k:]
                return gauss[top_y].mean(axis=0)
            lf = _foot_pos(_left_mask)
            rf = _foot_pos(_right_mask)
        else:
            # Fallback: top-K by Y gives approximate foot center (mix of both feet)
            k = min(foot_topk, len(xyz_np))
            top_idx = np.argpartition(xyz_np[:, 1], -k)[-k:]
            lf = rf = xyz_np[top_idx].mean(axis=0)

        left_foot_xyz.append(lf)
        right_foot_xyz.append(rf)

        if has_anchor and "attention_weights" in anchor_stats:
            attn = anchor_stats["attention_weights"].detach().cpu()  # (N_anchors, topk)
            attn_weights_seq.append(attn)

        if (seq_idx + 1) % 10 == 0 or seq_idx == N - 1:
            print(f"  [{seq_idx+1:3d}/{N}] frame={frame_idx:03d}  PSNR={psnr_val:.3f}  SSIM={ssim_val:.4f}")

    return {
        "psnr_seq":       np.array(psnr_seq),
        "ssim_seq":       np.array(ssim_seq),
        "lpips_seq":      np.array(lpips_seq) if lpips_seq else None,
        "left_foot_xyz":  np.stack(left_foot_xyz),   # (N, 3)
        "right_foot_xyz": np.stack(right_foot_xyz),  # (N, 3)
        "attn_weights":   attn_weights_seq,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_rendering_metrics(psnr_seq: np.ndarray, ssim_seq: np.ndarray,
                               lpips_seq: np.ndarray | None) -> dict:
    """Phase 1a: Temporal PSNR/SSIM stability + flickering."""
    delta_psnr = np.abs(np.diff(psnr_seq))
    flicker_thresh = 2.0  # dB

    m = {
        # PSNR
        "mean_psnr":       float(psnr_seq.mean()),
        "psnr_std":        float(psnr_seq.std()),
        "psnr_stability":  float(1.0 / (1.0 + psnr_seq.std())),
        "psnr_range":      float(psnr_seq.max() - psnr_seq.min()),
        # Flickering
        "mean_delta_psnr": float(delta_psnr.mean()),
        "max_delta_psnr":  float(delta_psnr.max()),
        "p95_delta_psnr":  float(np.percentile(delta_psnr, 95)),
        "flickering_ratio":float((delta_psnr > flicker_thresh).mean()),
        "flickering_count":int((delta_psnr > flicker_thresh).sum()),
        # SSIM
        "mean_ssim":      float(ssim_seq.mean()),
        "ssim_std":       float(ssim_seq.std()),
        "ssim_stability": float(1.0 / (1.0 + ssim_seq.std())),
    }
    if lpips_seq is not None:
        m["mean_lpips"]      = float(lpips_seq.mean())
        m["lpips_std"]       = float(lpips_seq.std())
        m["lpips_stability"] = float(1.0 / (1.0 + lpips_seq.std()))
    return m


def _contact_metrics_one_foot(
    foot_xyz: np.ndarray,            # (N, 3) world coords for ONE foot
    normal: np.ndarray,              # (3,)
    d: float,
    contact_thresh: float,
    foot_sliding_thresh_cm: float,
    world_to_cm: float,
) -> dict:
    """Compute contact metrics for a single foot trajectory."""
    signed_dist = foot_xyz @ normal + d  # (N,)
    contact = (signed_dist < contact_thresh).astype(int)

    # Foot Sliding (horizontal velocity during contact)
    xz_pos = foot_xyz[:, [0, 2]]
    xz_vel = np.linalg.norm(np.diff(xz_pos, axis=0), axis=1)
    xz_vel_cm = xz_vel * world_to_cm
    contact_pairs = (contact[:-1] == 1) & (contact[1:] == 1)
    if contact_pairs.sum() > 0:
        sc = xz_vel_cm[contact_pairs]
        mean_sliding = float(sc.mean())
        max_sliding  = float(sc.max())
        sliding_ratio = float((sc > foot_sliding_thresh_cm).mean())
    else:
        mean_sliding = max_sliding = sliding_ratio = 0.0

    # Contact Jitter (depth std within each contact period)
    periods = []
    in_contact = False
    start = 0
    for i, c in enumerate(contact):
        if c and not in_contact:
            start = i; in_contact = True
        elif not c and in_contact:
            periods.append((start, i)); in_contact = False
    if in_contact:
        periods.append((start, len(contact)))

    jitter_mm = []
    for s, e in periods:
        if e - s >= 2:
            depths = signed_dist[s:e] * world_to_cm * 10  # mm
            jitter_mm.append(float(np.std(depths)))

    mean_jitter = float(np.mean(jitter_mm)) if jitter_mm else 0.0
    max_jitter  = float(np.max(jitter_mm))  if jitter_mm else 0.0

    # Contact Consistency
    transitions = int(np.abs(np.diff(contact)).sum())
    consistency = 1.0 - transitions / max(len(contact) - 1, 1)

    false_contacts = sum(1 for s, e in periods if (e - s) == 1)
    n_contact = int(contact.sum())
    false_rate = false_contacts / max(n_contact, 1)

    return {
        "contact_frames":   n_contact,
        "n_periods":        len(periods),
        "mean_sliding_cm":  mean_sliding,
        "max_sliding_cm":   max_sliding,
        "sliding_ratio":    sliding_ratio,
        "mean_jitter_mm":   mean_jitter,
        "max_jitter_mm":    max_jitter,
        "consistency":      float(consistency),
        "transitions":      transitions,
        "false_rate":       float(false_rate),
        "signed_dist_seq":  signed_dist.tolist(),
        "contact_seq":      contact.tolist(),
        "xz_vel_cm_seq":    xz_vel_cm.tolist(),
    }


def compute_contact_metrics(
    left_foot_xyz: np.ndarray,     # (N, 3) world coords
    right_foot_xyz: np.ndarray,    # (N, 3) world coords
    ground_npz: dict,
    contact_thresh: float = 0.43,  # world units ≈5 cm
    foot_sliding_thresh_cm: float = 1.0,
) -> dict:
    """Phase 1b: Foot sliding, contact jitter, contact consistency.

    Computes metrics per foot (left and right) and averages them.
    Ground plane: normal·x + d = 0 (signed distance = normal·foot + d).
    World units: 1 wu ≈ 11.6 cm (scale ≈ 8.62).
    """
    normal = ground_npz["normal"].astype(float)
    d      = float(ground_npz["d"])
    world_to_cm = 11.6

    left  = _contact_metrics_one_foot(left_foot_xyz,  normal, d, contact_thresh, foot_sliding_thresh_cm, world_to_cm)
    right = _contact_metrics_one_foot(right_foot_xyz, normal, d, contact_thresh, foot_sliding_thresh_cm, world_to_cm)

    def avg(key): return (left[key] + right[key]) / 2.0
    def mx(key):  return max(left[key], right[key])

    return {
        "contact_frames":         left["contact_frames"] + right["contact_frames"],
        "n_contact_periods":      left["n_periods"] + right["n_periods"],
        "mean_foot_sliding_cm":   avg("mean_sliding_cm"),
        "max_foot_sliding_cm":    mx("max_sliding_cm"),
        "sliding_ratio":          avg("sliding_ratio"),
        "mean_contact_jitter_mm": avg("mean_jitter_mm"),
        "max_contact_jitter_mm":  mx("max_jitter_mm"),
        "contact_consistency":    avg("consistency"),
        "total_transitions":      left["transitions"] + right["transitions"],
        "false_contact_rate":     avg("false_rate"),
        # For visualization: use average foot (midpoint) as proxy
        "signed_dist_seq":        ((np.array(left["signed_dist_seq"]) + np.array(right["signed_dist_seq"])) / 2).tolist(),
        "contact_seq":            ((np.array(left["contact_seq"]) | np.array(right["contact_seq"])).astype(int)).tolist(),
        "xz_vel_cm_seq":          ((np.array(left["xz_vel_cm_seq"]) + np.array(right["xz_vel_cm_seq"])) / 2).tolist(),
        "_left":  left,
        "_right": right,
    }


def compute_attention_stability(attn_weights_seq: list) -> dict:
    """Phase 2: Cosine similarity of attention weights between adjacent frames.

    Foot anchors: indices 0-5 (left/right sole/toe/heel).
    Non-foot anchors: indices 6-15.
    """
    if len(attn_weights_seq) < 2:
        return {}

    FOOT_IDX = list(range(6))
    NONFOO_IDX = list(range(6, 16))

    cos_sims_all    = []
    cos_sims_foot   = []
    cos_sims_nonfoot= []

    for t in range(1, len(attn_weights_seq)):
        a_prev = attn_weights_seq[t-1].float()   # (N_anchors, topk)
        a_curr = attn_weights_seq[t].float()

        # Cosine similarity per anchor
        sim = F.cosine_similarity(a_curr, a_prev, dim=-1)  # (N_anchors,)
        cos_sims_all.append(sim.mean().item())
        cos_sims_foot.append(sim[FOOT_IDX].mean().item())
        cos_sims_nonfoot.append(sim[NONFOO_IDX].mean().item())

    return {
        "contact_attn_stability":     float(np.mean(cos_sims_foot)),
        "non_contact_attn_stability": float(np.mean(cos_sims_nonfoot)),
        "overall_attn_stability":     float(np.mean(cos_sims_all)),
        "attn_cos_sim_seq":           cos_sims_all,
        "foot_cos_sim_seq":           cos_sims_foot,
    }


def compute_motion_smoothness(foot_xyz: np.ndarray) -> dict:
    """Phase 3: Foot acceleration (proxy for motion smoothness)."""
    world_to_cm = 11.6
    vel = np.diff(foot_xyz, axis=0) * world_to_cm      # (N-1, 3)  cm/frame
    acc = np.diff(vel, axis=0)                           # (N-2, 3)  cm/frame²
    acc_mag = np.linalg.norm(acc, axis=1)                # (N-2,)
    return {
        "mean_acceleration_cm_f2": float(acc_mag.mean()),
        "max_acceleration_cm_f2":  float(acc_mag.max()),
        "velocity_std_cm":         float(np.linalg.norm(vel, axis=1).std()),
    }


# ---------------------------------------------------------------------------
# Grade helper
# ---------------------------------------------------------------------------

def grade(value, thresholds_good_ok, higher_is_better=True):
    """Return '✓✓ 优秀', '✓ 良好', or '✗ 较差'."""
    g, ok = thresholds_good_ok
    if higher_is_better:
        if value >= g:  return "✓✓ 优秀"
        if value >= ok: return "✓ 良好"
        return "✗ 较差"
    else:
        if value <= g:  return "✓✓ 优秀"
        if value <= ok: return "✓ 良好"
        return "✗ 较差"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(metrics: dict, model_name: str, out_dir: Path, n_frames: int) -> str:
    rm  = metrics.get("rendering", {})
    cm  = metrics.get("contact", {})
    am  = metrics.get("attention", {})
    mm  = metrics.get("motion", {})

    lines = [
        "=" * 56,
        "时序接触质量评估报告",
        "=" * 56,
        f"模型：{model_name}",
        f"帧数：{n_frames}",
        "",
        "-" * 56,
        "1. 整体时序渲染质量",
        "-" * 56,
        f"  Mean PSNR      : {rm.get('mean_psnr', 0):7.4f} dB",
        f"  PSNR Std       : {rm.get('psnr_std', 0):7.4f} dB   {grade(rm.get('psnr_std',1), (0.3, 0.5), higher_is_better=False)}",
        f"  PSNR Stability : {rm.get('psnr_stability', 0):7.4f}     {grade(rm.get('psnr_stability',0), (0.95, 0.90))}",
        f"  PSNR Range     : {rm.get('psnr_range', 0):7.4f} dB",
        f"  Flickering Ratio: {rm.get('flickering_ratio', 0)*100:5.1f}%   {grade(rm.get('flickering_ratio',1), (0.05, 0.10), higher_is_better=False)}",
        f"  Mean ΔPSNR     : {rm.get('mean_delta_psnr', 0):7.4f} dB   {grade(rm.get('mean_delta_psnr',1), (0.3, 0.5), higher_is_better=False)}",
        f"  Mean SSIM      : {rm.get('mean_ssim', 0):7.4f}",
        f"  SSIM Stability : {rm.get('ssim_stability', 0):7.4f}",
    ]
    if "mean_lpips" in rm:
        lines.append(f"  Mean LPIPS     : {rm.get('mean_lpips', 0):7.4f}")
        lines.append(f"  LPIPS Stability: {rm.get('lpips_stability', 0):7.4f}")

    lines += [
        "",
        "-" * 56,
        "2. 接触时序质量",
        "-" * 56,
        f"  Contact Frames      : {cm.get('contact_frames', 0)} / {n_frames}",
        f"  Contact Periods     : {cm.get('n_contact_periods', 0)}",
        f"  Mean Foot Sliding   : {cm.get('mean_foot_sliding_cm', 0):7.4f} cm/frame  {grade(cm.get('mean_foot_sliding_cm',99), (0.5, 1.0), higher_is_better=False)}",
        f"  Max  Foot Sliding   : {cm.get('max_foot_sliding_cm', 0):7.4f} cm/frame  {grade(cm.get('max_foot_sliding_cm',99), (1.0, 2.0), higher_is_better=False)}",
        f"  Sliding Ratio (>1cm): {cm.get('sliding_ratio', 0)*100:5.1f}%",
        f"  Mean Contact Jitter : {cm.get('mean_contact_jitter_mm', 0):7.4f} mm   {grade(cm.get('mean_contact_jitter_mm',99), (1.0, 2.0), higher_is_better=False)}",
        f"  Max  Contact Jitter : {cm.get('max_contact_jitter_mm', 0):7.4f} mm   {grade(cm.get('max_contact_jitter_mm',99), (2.0, 4.0), higher_is_better=False)}",
        f"  Contact Consistency : {cm.get('contact_consistency', 0)*100:5.1f}%   {grade(cm.get('contact_consistency',0), (0.95, 0.90))}",
        f"  Total Transitions   : {cm.get('total_transitions', 0)}",
        f"  False Contact Rate  : {cm.get('false_contact_rate', 0)*100:5.1f}%   {grade(cm.get('false_contact_rate',1), (0.03, 0.10), higher_is_better=False)}",
    ]

    if am:
        lines += [
            "",
            "-" * 56,
            "3. 接触注意力稳定性",
            "-" * 56,
            f"  Foot Attn Stability    : {am.get('contact_attn_stability', 0):7.4f}  {grade(am.get('contact_attn_stability',0), (0.90, 0.80))}",
            f"  Non-Foot Attn Stability: {am.get('non_contact_attn_stability', 0):7.4f}  {grade(am.get('non_contact_attn_stability',0), (0.80, 0.70))}",
            f"  Overall Attn Stability : {am.get('overall_attn_stability', 0):7.4f}",
        ]

    if mm:
        lines += [
            "",
            "-" * 56,
            "4. 运动平滑性",
            "-" * 56,
            f"  Mean Foot Acceleration: {mm.get('mean_acceleration_cm_f2', 0):7.4f} cm/frame²  {grade(mm.get('mean_acceleration_cm_f2',99), (1.0, 5.0), higher_is_better=False)}",
            f"  Max  Foot Acceleration: {mm.get('max_acceleration_cm_f2', 0):7.4f} cm/frame²",
            f"  Velocity Std          : {mm.get('velocity_std_cm', 0):7.4f} cm/frame",
        ]

    lines += ["", "=" * 56]
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def visualize_all(frame_data: dict, metrics: dict, model_name: str, out_dir: Path):
    psnr_seq = frame_data["psnr_seq"]
    ssim_seq = frame_data["ssim_seq"]
    N = len(psnr_seq)
    frames = np.arange(N)

    # --- Plot 1: PSNR timeline + flickering ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    rm = metrics["rendering"]

    ax = axes[0]
    ax.plot(frames, psnr_seq, lw=1.5, color="#2196F3", label="PSNR")
    mean_p = rm["mean_psnr"]
    std_p  = rm["psnr_std"]
    ax.axhline(mean_p, color="red", ls="--", lw=1.2, label=f"Mean={mean_p:.3f} dB")
    ax.fill_between(frames, mean_p - std_p, mean_p + std_p, alpha=0.15, color="red",
                    label=f"±Std={std_p:.3f} dB")
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title(f"Temporal PSNR — {model_name}\nStability={rm['psnr_stability']:.3f}  "
                 f"Flickering={rm['flickering_ratio']*100:.1f}%", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    delta_p = np.abs(np.diff(psnr_seq))
    ax.bar(frames[1:], delta_p, color=np.where(delta_p > 2.0, "#F44336", "#4CAF50"),
           width=1.0, label="|ΔPSNR|")
    ax.axhline(2.0, color="red", ls="--", lw=1.0, label="Flicker threshold (2 dB)")
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("|ΔPSNR| (dB)", fontsize=12)
    ax.set_title(f"Flickering Score — Mean ΔPSNR={rm['mean_delta_psnr']:.3f} dB", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "psnr_timeline.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Contact metrics ---
    cm = metrics["contact"]
    if cm:
        signed_dist = np.array(cm["signed_dist_seq"])
        contact_seq = np.array(cm["contact_seq"])
        xz_vel_cm   = np.array(cm["xz_vel_cm_seq"])
        world_to_mm = 11.6 * 10  # world units → mm

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax = axes[0]
        ax.plot(frames, signed_dist * 11.6 * 10, lw=1.5, color="#9C27B0", label="Foot-Ground dist (mm)")
        ax.axhline(0, color="brown", ls="--", lw=1.0, label="Ground plane")
        thresh_mm = 0.43 * 11.6 * 10
        ax.axhline(thresh_mm, color="orange", ls=":", lw=1.0, label=f"Contact thresh ({thresh_mm:.0f} mm)")
        ax.set_ylabel("Distance (mm)", fontsize=12)
        ax.set_title(f"Foot-Ground Signed Distance — {model_name}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.fill_between(frames, contact_seq, alpha=0.4, color="#4CAF50", step="post", label="Contact")
        ax.set_ylim(-0.1, 1.5)
        ax.set_ylabel("Contact State", fontsize=12)
        ax.set_title(f"Contact State — Consistency={cm['contact_consistency']*100:.1f}%  "
                     f"Transitions={cm['total_transitions']}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        colors = np.where(np.array(cm["contact_seq"][:-1]) == 1, "#F44336", "#90A4AE")
        ax.bar(frames[1:], xz_vel_cm, color=colors, width=1.0)
        ax.axhline(1.0, color="red", ls="--", lw=1.0, label="Sliding threshold (1 cm)")
        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("XZ Velocity (cm/frame)", fontsize=12)
        ax.set_title(f"Foot XZ Velocity (red = contact frame) — "
                     f"MeanSlide={cm['mean_foot_sliding_cm']:.3f} cm  "
                     f"Jitter={cm['mean_contact_jitter_mm']:.2f} mm", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "contact_metrics.png", dpi=150)
        plt.close(fig)

    # --- Plot 3: Attention stability (if available) ---
    am = metrics.get("attention", {})
    if am and "foot_cos_sim_seq" in am:
        foot_sim  = np.array(am["foot_cos_sim_seq"])
        all_sim   = np.array(am["attn_cos_sim_seq"])
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(np.arange(1, len(all_sim)+1), all_sim, lw=1.5, color="#2196F3",
                label=f"Overall Sim (mean={am['overall_attn_stability']:.3f})")
        ax.plot(np.arange(1, len(foot_sim)+1), foot_sim, lw=1.5, color="#F44336",
                label=f"Foot Anchors Sim (mean={am['contact_attn_stability']:.3f})")
        ax.axhline(0.9, color="green", ls="--", lw=1.0, label="Excellent threshold (0.90)")
        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Cosine Similarity", fontsize=12)
        ax.set_title(f"Attention Stability — {model_name}", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "attention_stability.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seq",    default="lab")
    p.add_argument("--contact-thresh", type=float, default=0.43,
                   help="Foot-ground contact threshold in world units (default 0.43 ≈ 5 cm)")
    p.add_argument("--foot-topk", type=int, default=300,
                   help="Top-K human Gaussians by Y (foot bottom) to average for foot position")
    p.add_argument("--with-lpips", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ground-npz", type=Path, default=None,
                   help="Path to ground_plane.npz (default: eval_contact/{seq}/ground_plane.npz)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "temporal_eval"
    out_dir.mkdir(exist_ok=True)

    model_name = run_dir.parent.name  # e.g. exp0_hugs_original_15000_20260527

    ground_npz_path = args.ground_npz or (Path("eval_contact") / args.seq / "ground_plane.npz")
    if not ground_npz_path.is_file():
        raise FileNotFoundError(f"Ground plane not found: {ground_npz_path}. "
                                "Run scripts/eval_ground_contact.py first.")
    ground_data = dict(np.load(ground_npz_path))
    print(f"Ground plane: normal={ground_data['normal'].round(3).tolist()}  d={float(ground_data['d']):.4f}")

    print(f"\n[Loading trainer] {run_dir}")
    trainer, has_anchor = load_trainer(run_dir, args.device)
    n_frames = len(trainer.all_dataset)
    print(f"Frames in all_dataset: {n_frames}  |  has_anchor: {has_anchor}")

    print(f"\n[Rendering {n_frames} frames]")
    frame_data = process_all_frames(
        trainer, has_anchor,
        foot_topk=args.foot_topk,
        with_lpips=args.with_lpips,
        device=args.device,
    )

    print("\n[Computing metrics]")
    rendering_metrics = compute_rendering_metrics(
        frame_data["psnr_seq"], frame_data["ssim_seq"], frame_data["lpips_seq"]
    )
    contact_metrics = compute_contact_metrics(
        frame_data["left_foot_xyz"], frame_data["right_foot_xyz"],
        ground_data, contact_thresh=args.contact_thresh
    )
    attention_metrics = {}
    if has_anchor and frame_data["attn_weights"] and len(frame_data["attn_weights"]) >= 2:
        attention_metrics = compute_attention_stability(frame_data["attn_weights"])
    avg_foot_xyz = (frame_data["left_foot_xyz"] + frame_data["right_foot_xyz"]) / 2.0
    motion_metrics = compute_motion_smoothness(avg_foot_xyz)

    metrics = {
        "rendering":  rendering_metrics,
        "contact":    contact_metrics,
        "attention":  attention_metrics,
        "motion":     motion_metrics,
    }

    # Save JSON (exclude long sequences and internal sub-dicts for compactness)
    def _strip_seqs(d):
        return {k: v for k, v in d.items()
                if not k.startswith("_") and (not isinstance(v, list) or len(v) <= 10)}

    json_metrics = {
        "model_name": model_name,
        "run_dir": str(run_dir),
        "n_frames": n_frames,
        "rendering":  _strip_seqs(rendering_metrics),
        "contact":    _strip_seqs(contact_metrics),
        "attention":  _strip_seqs(attention_metrics),
        "motion":     motion_metrics,
    }
    (out_dir / "report.json").write_text(
        json.dumps(json_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[Generating report]")
    report_text = generate_report(metrics, model_name, out_dir, n_frames)
    print(report_text)

    print("\n[Generating visualizations]")
    visualize_all(frame_data, metrics, model_name, out_dir)

    print(f"\nDone. Outputs: {out_dir}/")
    return json_metrics


if __name__ == "__main__":
    main()
