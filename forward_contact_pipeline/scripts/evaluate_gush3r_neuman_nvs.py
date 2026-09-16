#!/usr/bin/env python3
"""Independent NeuMan target-view evaluation for the local GUSH3R checkpoint.

This is deliberately *not* claimed to be the paper's Table 2 evaluator:
the authors released neither their NeuMan/EMDB view splits nor evaluation
code.  The available local ``lab`` subset contains calibrated RGB/cameras and
person masks but no target-frame SMPL-X.  Consequently this script measures a
strictly held-out, person-masked **scene** NVS protocol.  It is useful for
checking the checkpoint, renderer, camera alignment, and long-video fixes;
it must not be compared as a full-image Table 2 reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity


REPO_ROOT = Path(__file__).resolve().parents[2]
GUSH_ROOT = REPO_ROOT / "GUSH3R"
DEFAULT_DATA = REPO_ROOT / "forward_contact_pipeline/datasets/neuman_lab_h3r_hugs_p0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-input-views", type=int, choices=(4, 16), required=True)
    parser.add_argument("--num-targets", type=int, default=8,
                        help="Uniform held-out target count; 0 evaluates every non-input frame.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--bg-gaussian-max", type=int, default=2_000_000)
    parser.add_argument("--bg-mask-threshold", type=float, default=0.02)
    parser.add_argument("--bg-mask-dilation", type=int, default=3)
    parser.add_argument("--bg-voxel-size", type=float, default=0.005)
    return parser.parse_args()


def uniform_indices(count: int, num: int) -> list[int]:
    if num > count:
        raise ValueError(f"Requested {num} views but only {count} frames are available")
    return sorted(np.linspace(0, count - 1, num=num, dtype=int).tolist())


def umeyama_gt_to_pred(gt: np.ndarray, pred: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return s,R,t such that pred ~= s * (R @ gt) + t."""
    if len(gt) < 3:
        raise ValueError("At least three input views are required for camera Sim(3) alignment")
    mu_x, mu_y = gt.mean(0), pred.mean(0)
    x, y = gt - mu_x, pred - mu_y
    cov = y.T @ x / len(gt)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rot = u @ sign @ vt
    var_x = np.mean(np.sum(x * x, axis=1))
    scale = float(np.trace(np.diag(d) @ sign) / max(var_x, 1e-12))
    trans = mu_y - scale * (rot @ mu_x)
    return scale, rot, trans


def target_pose_in_pred_world(gt_c2w: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Map a calibrated NeuMan camera pose into GUSH3R's predicted world."""
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rot @ gt_c2w[:3, :3]
    result[:3, 3] = scale * (rot @ gt_c2w[:3, 3]) + trans
    return result


def masked_metrics(render: np.ndarray, target: np.ndarray, person_mask: np.ndarray, perceptual: torch.nn.Module) -> dict[str, float]:
    """Evaluate background pixels only; masked pixels become white for SSIM/LPIPS."""
    bg = person_mask <= 127
    if bg.sum() == 0:
        raise ValueError("Mask excludes all pixels")
    render = np.clip(render.astype(np.float32), 0.0, 1.0)
    target = np.clip(target.astype(np.float32), 0.0, 1.0)
    mse = float(np.mean((render[bg] - target[bg]) ** 2))
    psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
    r_vis, t_vis = render.copy(), target.copy()
    r_vis[~bg], t_vis[~bg] = 1.0, 1.0
    ssim = float(structural_similarity(t_vis, r_vis, channel_axis=2, data_range=1.0))
    with torch.no_grad():
        r_tensor = torch.from_numpy(r_vis).permute(2, 0, 1).unsqueeze(0).cuda() * 2.0 - 1.0
        t_tensor = torch.from_numpy(t_vis).permute(2, 0, 1).unsqueeze(0).cuda() * 2.0 - 1.0
        lpips_value = float(perceptual(r_tensor, t_tensor).item())
    return {"psnr": psnr, "ssim": ssim, "lpips_alex": lpips_value, "background_pixels": int(bg.sum())}


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("GUSH3R evaluation requires CUDA")
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "renders").mkdir(exist_ok=True)

    # The released inference code keeps checkpoint/model asset paths relative
    # to its repository root.
    os.chdir(GUSH_ROOT)
    sys.path.insert(0, str(GUSH_ROOT))
    import infer  # local, already patched only for inference/render compatibility

    camera_data = np.load(args.data_root / "cameras.npz")
    c2w_gt, intrinsics, names = camera_data["c2w"], camera_data["intrinsics"], camera_data["frame_names"]
    image_paths = [args.data_root / "images" / str(name) for name in names]
    n_frames = len(image_paths)
    input_indices = uniform_indices(n_frames, args.num_input_views)
    candidates = [idx for idx in range(n_frames) if idx not in set(input_indices)]
    if args.num_targets == 0:
        target_indices = candidates
    else:
        target_indices = [candidates[idx] for idx in uniform_indices(len(candidates), min(args.num_targets, len(candidates)))]

    class ModelArgs:
        gs_conf_threshold = 1.0
        bg_mask_threshold = args.bg_mask_threshold
        bg_mask_dilation = args.bg_mask_dilation
        bg_voxel_size = args.bg_voxel_size
        bg_gaussian_max = args.bg_gaussian_max

    model = infer.load_model("cuda", ModelArgs())
    # ``load_model`` restores the checkpoint's dust3r import path.
    from dust3r.inference import inference_recurrent_lighter
    input_paths = [str(image_paths[idx]) for idx in input_indices]
    views = infer.prepare_input(input_paths, size=args.size, img_res=getattr(model, "mhmr_img_res", None))
    # The locally calibrated pixels are already 512x288; pass their true K to
    # HumanGS instead of the demo script's synthetic central-camera K.
    for local_idx, view in enumerate(views):
        view["camera_intrinsics"] = torch.from_numpy(intrinsics[input_indices[local_idx]][None]).float()

    start = time.perf_counter()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, "cuda", use_ttt3r=False, save_all_bg_gaussians=False)
    inference_seconds = time.perf_counter() - start
    state = infer.build_render_state(outputs)
    background = infer.select_background_gaussians(state["gaussians"])
    if background is None:
        raise RuntimeError("No background Gaussians were predicted")
    tensors = infer.background_tensors(background)

    pred_centres = state["t"]
    gt_centres = c2w_gt[input_indices, :3, 3]
    scale, rot, trans = umeyama_gt_to_pred(gt_centres, pred_centres)
    alignment_error = np.linalg.norm(
        pred_centres - np.stack([scale * (rot @ c) + trans for c in gt_centres]), axis=1
    )
    perceptual = lpips.LPIPS(net="alex").cuda().eval()

    records: list[dict[str, object]] = []
    for target_idx in target_indices:
        pose = target_pose_in_pred_world(c2w_gt[target_idx], scale, rot, trans)
        render = infer.render_frame(tensors, intrinsics[target_idx], pose[:3, :3], pose[:3, 3], state["image_hw"], "cuda")
        target_bgr = cv2.imread(str(image_paths[target_idx]), cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = cv2.imread(str(args.data_root / "masks" / str(names[target_idx])), cv2.IMREAD_GRAYSCALE)
        score = masked_metrics(render, target, mask, perceptual)
        score["frame_index"] = int(target_idx)
        score["frame_name"] = str(names[target_idx])
        records.append(score)
        canvas = np.concatenate([(render * 255).astype(np.uint8), (target * 255).astype(np.uint8)], axis=1)
        cv2.imwrite(str(args.output_dir / "renders" / f"{target_idx:06d}_render_target.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    summary = {
        "protocol": "independent_neuman_lab_person_masked_scene_nvs_v1",
        "scope_limit": "Background-only held-out NVS. Full Table-2 reproduction requires target SMPL-X, paper view splits, and EMDB.",
        "checkpoint": str(GUSH_ROOT / "checkpoints/gush3r.pth"),
        "dataset_root": str(args.data_root),
        "input_indices": input_indices,
        "target_indices": target_indices,
        "num_input_views": args.num_input_views,
        "inference_seconds": inference_seconds,
        "input_fps": args.num_input_views / inference_seconds,
        "camera_sim3_gt_to_pred": {"scale": scale, "rotation": rot.tolist(), "translation": trans.tolist(),
                                    "input_camera_error_mean": float(alignment_error.mean()), "input_camera_error_max": float(alignment_error.max())},
        "metrics_mean": {key: float(np.mean([row[key] for row in records])) for key in ("psnr", "ssim", "lpips_alex")},
        "per_target": records,
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["metrics_mean"], indent=2))
    print(f"Saved independent evaluation: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
