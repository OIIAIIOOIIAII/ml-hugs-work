"""Global-aware voxelized adaptive density control for SceneGS.

This module is deliberately scene-only.  It changes the densification decision
for the static background Gaussians while leaving the HUGS human branch,
SuGaR-style regularization, and anchor-attention modules untouched.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger

from hugs.utils.general import build_rotation


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _cfg_float(cfg, name, default):
    return float(_cfg_get(cfg, name, default) or default)


def _cfg_int(cfg, name, default):
    return int(_cfg_get(cfg, name, default) or default)


def scene_global_aware_adc_enabled(cfg) -> bool:
    return bool(_cfg_get(cfg, "enabled", False))


def scene_global_aware_adc_active(cfg, iteration: int) -> bool:
    if not scene_global_aware_adc_enabled(cfg):
        return False
    start_iter = _cfg_int(cfg, "start_iter", 0)
    densify_until_iter = _cfg_int(cfg, "densify_until_iter", 15_000)
    return start_iter <= iteration <= densify_until_iter


def scene_global_aware_adc_should_densify(cfg, iteration: int) -> bool:
    if not scene_global_aware_adc_active(cfg, iteration):
        return False
    interval = _cfg_int(cfg, "densification_interval", 100)
    return interval > 0 and iteration % interval == 0


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3 or image.shape[0] not in (1, 3):
        raise ValueError(f"Expected image as [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 1:
        return image[:1]
    weight = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (image * weight).sum(dim=0, keepdim=True)


def _gaussian_kernel1d(window_size: int, sigma: Optional[float], device, dtype) -> torch.Tensor:
    if window_size % 2 == 0:
        window_size += 1
    if sigma is None or sigma <= 0:
        sigma = max(float(window_size) / 6.0, 1e-6)
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    return kernel / torch.clamp(kernel.sum(), min=1e-8)


def gaussian_blur2d(image: torch.Tensor, window_size: int = 11, sigma: Optional[float] = None) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[1] != 1:
        raise ValueError(f"Expected grayscale image as [1,1,H,W], got {tuple(image.shape)}")
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    kernel = _gaussian_kernel1d(window_size, sigma, image.device, image.dtype)
    kernel_x = kernel.view(1, 1, 1, window_size)
    kernel_y = kernel.view(1, 1, window_size, 1)
    image = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    image = F.conv2d(image, kernel_x)
    image = F.conv2d(image, kernel_y)
    return image


def _highpass_gaussian_approx(gray: torch.Tensor, window_size: int) -> torch.Tensor:
    low = gaussian_blur2d(gray, window_size=window_size)
    return torch.abs(gray.unsqueeze(0) - low).squeeze(0).squeeze(0)


def _highpass_local_fft(gray: torch.Tensor, window_size: int, cutoff_ratio: float) -> torch.Tensor:
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    gray4 = F.pad(gray.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    patches = F.unfold(gray4, kernel_size=window_size).transpose(1, 2)
    patches = patches.reshape(-1, window_size, window_size)
    patches = patches - patches.mean(dim=(1, 2), keepdim=True)
    fft = torch.fft.fftshift(torch.fft.fft2(patches), dim=(-2, -1))

    coords = torch.arange(window_size, device=gray.device, dtype=gray.dtype) - window_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(xx.pow(2) + yy.pow(2))
    cutoff = float(cutoff_ratio) * float(radius.max().item())
    high_mask = radius >= cutoff
    energy = fft.abs().pow(2)[:, high_mask].mean(dim=1)
    return energy.reshape(gray.shape[-2], gray.shape[-1])


def compute_relative_blur_map(
    rendered_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    window_size: int = 11,
    highpass_cutoff_ratio: float = 0.5,
    backend: str = "gaussian_highpass_approx",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a [H,W] map where GT has more high-frequency energy than render."""
    render_gray = rgb_to_grayscale(rendered_rgb.detach()).clamp(0.0, 1.0)
    gt_gray = rgb_to_grayscale(gt_rgb.detach()).clamp(0.0, 1.0)

    if backend == "local_fft":
        e_render = _highpass_local_fft(render_gray, window_size, highpass_cutoff_ratio)
        e_gt = _highpass_local_fft(gt_gray, window_size, highpass_cutoff_ratio)
    elif backend in ("gaussian_highpass_approx", "gaussian"):
        e_render = _highpass_gaussian_approx(render_gray, window_size)
        e_gt = _highpass_gaussian_approx(gt_gray, window_size)
    else:
        raise ValueError(f"Unknown relative blur backend: {backend}")

    return torch.relu(e_gt - e_render + eps * 0.0)


def compute_global_aware_weights(
    relative_blur: torch.Tensor,
    percentile: float = 95.0,
    lower: float = 0.5,
    upper: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    flat = relative_blur.detach().flatten()
    if flat.numel() == 0:
        return torch.ones_like(relative_blur)
    q = torch.quantile(flat.float(), float(percentile) / 100.0).to(relative_blur.dtype)
    weights = relative_blur.detach() / torch.clamp(q, min=eps)
    return torch.clamp(weights, min=float(lower), max=float(upper))


def compute_weighted_adc_loss(
    rendered_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    relative_blur = compute_relative_blur_map(
        rendered_rgb,
        gt_rgb,
        window_size=_cfg_int(cfg, "blur_window_size", 11),
        highpass_cutoff_ratio=_cfg_float(cfg, "highpass_cutoff_ratio", 0.5),
        backend=str(_cfg_get(cfg, "relative_blur_backend", "gaussian_highpass_approx")),
        eps=_cfg_float(cfg, "eps", 1e-6),
    )
    weights = compute_global_aware_weights(
        relative_blur,
        percentile=_cfg_float(cfg, "blur_percentile", 95.0),
        lower=_cfg_float(cfg, "weight_lower_bound", 0.5),
        upper=_cfg_float(cfg, "weight_upper_bound", 1.0),
        eps=_cfg_float(cfg, "eps", 1e-6),
    )
    pixel_loss = torch.abs(rendered_rgb - gt_rgb).mean(dim=0)
    adc_loss = (weights.detach() * pixel_loss).mean()
    logs = {
        "scene_adc_blur_mean": relative_blur.detach().mean(),
        "scene_adc_blur_max": relative_blur.detach().max(),
        "scene_adc_weight_mean": weights.detach().mean(),
        "scene_adc_weight_max": weights.detach().max(),
        "scene_adc_loss": adc_loss.detach(),
    }
    return adc_loss, logs


class SceneMaxGradientAccumulator:
    def __init__(self, num_scene_gaussians: int, device):
        self.max_grad = torch.zeros((int(num_scene_gaussians),), device=device)

    @torch.no_grad()
    def sync(self, num_scene_gaussians: int, device=None):
        num_scene_gaussians = int(num_scene_gaussians)
        if device is None:
            device = self.max_grad.device
        if self.max_grad.device != device:
            self.max_grad = self.max_grad.to(device)
        old_n = int(self.max_grad.shape[0])
        if old_n == num_scene_gaussians:
            return
        if num_scene_gaussians < old_n:
            self.max_grad = self.max_grad[:num_scene_gaussians]
            return
        extra = torch.zeros((num_scene_gaussians - old_n,), device=device, dtype=self.max_grad.dtype)
        self.max_grad = torch.cat([self.max_grad, extra], dim=0)

    @torch.no_grad()
    def update(self, scene_viewspace_grad: torch.Tensor, visibility_filter: Optional[torch.Tensor] = None):
        if scene_viewspace_grad is None or scene_viewspace_grad.numel() == 0:
            return
        grad_norm = torch.linalg.norm(scene_viewspace_grad[..., :2], dim=-1)
        self.sync(grad_norm.shape[0], device=grad_norm.device)
        if visibility_filter is not None:
            visible = visibility_filter.to(device=grad_norm.device, dtype=torch.bool)
            if visible.shape[0] != grad_norm.shape[0]:
                visible = visible[:grad_norm.shape[0]]
            grad_norm = torch.where(visible, grad_norm, torch.zeros_like(grad_norm))
        self.max_grad = torch.maximum(self.max_grad, grad_norm.detach())

    @torch.no_grad()
    def reset(self, num_scene_gaussians: Optional[int] = None, device=None):
        if num_scene_gaussians is None:
            self.max_grad.zero_()
            return
        if device is None:
            device = self.max_grad.device
        self.max_grad = torch.zeros((int(num_scene_gaussians),), device=device)


@torch.no_grad()
def _half_opacity_logits(scene_gs, opacity_logits: torch.Tensor) -> torch.Tensor:
    alpha = torch.clamp(scene_gs.opacity_activation(opacity_logits), min=1e-6, max=1.0 - 1e-6)
    child_alpha = torch.clamp(alpha * 0.5, min=1e-6, max=1.0 - 1e-6)
    return scene_gs.inverse_opacity_activation(child_alpha)


@torch.no_grad()
def _select_one_gaussian_per_voxel(
    scene_xyz: torch.Tensor,
    scene_gmax: torch.Tensor,
    voxel_size: float,
    tau_pos: float,
    selection_policy: str = "importance_sampling",
    max_slots: int = -1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    n_points = int(scene_xyz.shape[0])
    device = scene_xyz.device
    selected_mask = torch.zeros((n_points,), dtype=torch.bool, device=device)
    if n_points == 0 or scene_gmax.numel() == 0 or voxel_size <= 0:
        return selected_mask, {"num_voxels": 0.0, "selected_voxels": 0.0}

    gmax = scene_gmax.reshape(-1).to(device=device)
    gmax = torch.nan_to_num(gmax, nan=0.0, posinf=0.0, neginf=0.0)
    coords = torch.floor(scene_xyz / float(voxel_size)).to(torch.int64)
    unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
    num_voxels = int(unique_coords.shape[0])
    sums = torch.zeros((num_voxels,), device=device, dtype=gmax.dtype)
    counts = torch.zeros((num_voxels,), device=device, dtype=gmax.dtype)
    sums.scatter_add_(0, inverse, gmax)
    counts.scatter_add_(0, inverse, torch.ones_like(gmax))
    voxel_score = sums / torch.clamp(counts, min=1.0)
    candidate_voxels = torch.where(voxel_score >= float(tau_pos))[0]
    if candidate_voxels.numel() == 0:
        return selected_mask, {"num_voxels": float(num_voxels), "selected_voxels": 0.0}

    if max_slots is not None and max_slots > 0 and candidate_voxels.numel() > max_slots:
        top = torch.topk(voxel_score[candidate_voxels], k=int(max_slots), largest=True).indices
        candidate_voxels = candidate_voxels[top]

    for voxel_id in candidate_voxels.tolist():
        point_idx = torch.where(inverse == voxel_id)[0]
        if point_idx.numel() == 0:
            continue
        point_scores = torch.clamp(gmax[point_idx], min=0.0)
        if selection_policy == "importance_sampling" and point_scores.sum() > 0:
            local_idx = torch.multinomial(point_scores / point_scores.sum(), 1).item()
        else:
            local_idx = torch.argmax(point_scores).item()
        selected_mask[point_idx[local_idx]] = True

    return selected_mask, {
        "num_voxels": float(num_voxels),
        "selected_voxels": float(candidate_voxels.numel()),
        "selected_gaussians": float(selected_mask.sum().item()),
        "voxel_score_mean": float(voxel_score.mean().item()) if voxel_score.numel() else 0.0,
        "voxel_score_max": float(voxel_score.max().item()) if voxel_score.numel() else 0.0,
    }


@torch.no_grad()
def voxel_based_scene_densify_and_prune(
    scene_gs,
    scene_gmax: torch.Tensor,
    cfg,
    extent: float,
    min_opacity: float,
    max_screen_size: Optional[float],
    max_n_gs: Optional[int] = None,
) -> Dict[str, float]:
    assert bool(_cfg_get(cfg, "apply_to_scene_only", True)), "Global-aware ADC must remain scene-only"
    n_before = int(scene_gs.get_xyz.shape[0])
    device = scene_gs.get_xyz.device
    if n_before == 0:
        return {"num_before": 0.0, "num_after": 0.0, "selected_gaussians": 0.0}

    scene_gmax = scene_gmax.reshape(-1).to(device=device)
    if scene_gmax.shape[0] != n_before:
        fixed = torch.zeros((n_before,), device=device, dtype=scene_gmax.dtype)
        fixed[: min(n_before, scene_gmax.shape[0])] = scene_gmax[: min(n_before, scene_gmax.shape[0])]
        scene_gmax = fixed

    max_n_gs = int(max_n_gs) if max_n_gs is not None and max_n_gs > 0 else None
    remaining_slots = -1 if max_n_gs is None else max(0, max_n_gs - n_before)
    if remaining_slots == 0:
        return {"num_before": float(n_before), "num_after": float(n_before), "selected_gaussians": 0.0}

    user_max_slots = _cfg_int(cfg, "max_densify_per_call", -1)
    max_slots = remaining_slots
    if user_max_slots > 0:
        max_slots = user_max_slots if max_slots < 0 else min(max_slots, user_max_slots)

    selected_mask, stats = _select_one_gaussian_per_voxel(
        scene_gs.get_xyz.detach(),
        scene_gmax,
        voxel_size=_cfg_float(cfg, "voxel_size", 0.005),
        tau_pos=_cfg_float(cfg, "tau_pos", 0.0002),
        selection_policy=str(_cfg_get(cfg, "selection_inside_voxel", "importance_sampling")),
        max_slots=max_slots,
    )
    if selected_mask.sum().item() == 0:
        stats.update({"num_before": float(n_before), "num_after": float(n_before), "pruned_count": 0.0})
        return stats

    scale_limit = scene_gs.percent_dense * float(extent)
    max_scale = scene_gs.get_scaling.max(dim=1).values
    clone_mask = selected_mask & (max_scale <= scale_limit)
    split_mask = selected_mask & (max_scale > scale_limit)
    new_xyz_parts = []
    new_fdc_parts = []
    new_frest_parts = []
    new_opacity_parts = []
    new_scaling_parts = []
    new_rotation_parts = []

    if clone_mask.any():
        idx = torch.where(clone_mask)[0]
        new_xyz_parts.append(scene_gs._xyz[idx])
        new_fdc_parts.append(scene_gs._features_dc[idx])
        new_frest_parts.append(scene_gs._features_rest[idx])
        new_opacity_parts.append(_half_opacity_logits(scene_gs, scene_gs._opacity[idx]))
        new_scaling_parts.append(scene_gs._scaling[idx])
        new_rotation_parts.append(scene_gs._rotation[idx])

    if split_mask.any():
        idx = torch.where(split_mask)[0]
        n_children = _cfg_int(cfg, "split_children", 2)
        stds = scene_gs.get_scaling[idx].repeat(n_children, 1)
        means = torch.zeros((stds.shape[0], 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(scene_gs._rotation[idx]).repeat(n_children, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + scene_gs.get_xyz[idx].repeat(n_children, 1)
        new_xyz_parts.append(new_xyz)
        new_fdc_parts.append(scene_gs._features_dc[idx].repeat(n_children, 1, 1))
        new_frest_parts.append(scene_gs._features_rest[idx].repeat(n_children, 1, 1))
        new_opacity_parts.append(_half_opacity_logits(scene_gs, scene_gs._opacity[idx]).repeat(n_children, 1))
        new_scaling_parts.append(scene_gs.scaling_inverse_activation(scene_gs.get_scaling[idx].repeat(n_children, 1) / (0.8 * n_children)))
        new_rotation_parts.append(scene_gs._rotation[idx].repeat(n_children, 1))

    if new_xyz_parts:
        scene_gs.densification_postfix(
            torch.cat(new_xyz_parts, dim=0),
            torch.cat(new_fdc_parts, dim=0),
            torch.cat(new_frest_parts, dim=0),
            torch.cat(new_opacity_parts, dim=0),
            torch.cat(new_scaling_parts, dim=0),
            torch.cat(new_rotation_parts, dim=0),
        )

    if split_mask.any():
        n_added = int(scene_gs.get_xyz.shape[0] - n_before)
        prune_filter = torch.cat([
            split_mask,
            torch.zeros((n_added,), dtype=torch.bool, device=device),
        ])
    else:
        prune_filter = torch.zeros((scene_gs.get_xyz.shape[0],), dtype=torch.bool, device=device)

    prune_mask = (scene_gs.get_opacity < float(min_opacity)).squeeze()
    if max_screen_size:
        big_points_vs = scene_gs.max_radii2D > max_screen_size
        big_points_ws = scene_gs.get_scaling.max(dim=1).values > 0.1 * float(extent)
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    prune_mask = torch.logical_or(prune_mask, prune_filter)
    min_scene_gaussians = _cfg_int(cfg, "min_scene_gaussians", 1000)
    pruned_count = int(prune_mask.sum().item())
    if pruned_count > 0 and scene_gs.get_xyz.shape[0] - pruned_count >= min_scene_gaussians:
        scene_gs.prune_points(prune_mask)
    else:
        pruned_count = 0

    n_after = int(scene_gs.get_xyz.shape[0])
    stats.update({
        "num_before": float(n_before),
        "num_after": float(n_after),
        "cloned_count": float(clone_mask.sum().item()),
        "split_count": float(split_mask.sum().item()),
        "pruned_count": float(pruned_count),
    })
    logger.info(
        "Global-aware voxel ADC densify: "
        f"selected={int(selected_mask.sum().item())}, clone={int(clone_mask.sum().item())}, "
        f"split={int(split_mask.sum().item())}, prune={pruned_count}, "
        f"scene={n_before}->{n_after}, voxels={int(stats.get('num_voxels', 0.0))}"
    )
    torch.cuda.empty_cache()
    return stats
