"""Scene-only SuGaR-style regularization for HUGS background Gaussians.

This module intentionally only touches the static SceneGS branch. It borrows the
lightweight parts of SuGaR that make scene Gaussians cleaner surface elements:
opacity binarization, low-opacity cleanup, compact scale volume, and anisotropic
flattening. Mesh extraction and mesh-bound refinement are deliberately out of
scope for this HUGS integration.
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
from loguru import logger
from plyfile import PlyData, PlyElement


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _cfg_float(cfg, name, default):
    return float(_cfg_get(cfg, name, default) or default)


def _cfg_int(cfg, name, default):
    return int(_cfg_get(cfg, name, default) or default)


def scene_sugar_enabled(cfg) -> bool:
    return bool(_cfg_get(cfg, 'enabled', False))


def scene_sugar_active(cfg, iteration: int) -> bool:
    if not scene_sugar_enabled(cfg):
        return False
    start_iter = _cfg_int(cfg, 'start_iter', 0)
    stop_after_iter = _cfg_int(cfg, 'stop_after_iter', -1)
    if iteration < start_iter:
        return False
    if stop_after_iter >= 0 and iteration > stop_after_iter:
        return False
    return True


def _zero_like_scene(scene_gs):
    return scene_gs.get_xyz.sum() * 0.0


def scene_opacity_entropy_loss(scene_gs, eps: float = 1e-6) -> torch.Tensor:
    alpha = torch.clamp(scene_gs.get_opacity, eps, 1.0 - eps)
    entropy = -alpha * torch.log(alpha) - (1.0 - alpha) * torch.log(1.0 - alpha)
    return entropy.mean()


def scene_scale_volume_loss(scene_gs) -> torch.Tensor:
    scale = scene_gs.get_scaling
    if scale.numel() == 0:
        return _zero_like_scene(scene_gs)
    return torch.prod(scale, dim=-1).mean()


def scene_flatten_loss(scene_gs, eps: float = 1e-6) -> torch.Tensor:
    scale = scene_gs.get_scaling
    if scale.numel() == 0:
        return _zero_like_scene(scene_gs)
    sorted_scale, _ = torch.sort(scale, dim=-1)
    return (sorted_scale[:, 0] / (sorted_scale[:, 1] + eps)).mean()


def scene_sugar_stats(scene_gs, low_opacity_threshold: float = 0.03) -> Dict[str, torch.Tensor]:
    opacity = scene_gs.get_opacity.detach().squeeze(-1)
    scale = scene_gs.get_scaling.detach()
    if opacity.numel() == 0:
        zero = torch.tensor(0.0, device=scene_gs.get_xyz.device)
        return {
            'num_scene_gaussians': zero,
            'mean_opacity': zero,
            'median_opacity': zero,
            'low_opacity_ratio': zero,
            'mean_scale_volume': zero,
            'mean_flatten_ratio': zero,
        }

    scale_volume = torch.prod(scale, dim=-1)
    sorted_scale, _ = torch.sort(scale, dim=-1)
    flatten_ratio = sorted_scale[:, 0] / (sorted_scale[:, 1] + 1e-6)
    return {
        'num_scene_gaussians': torch.tensor(float(opacity.numel()), device=opacity.device),
        'mean_opacity': opacity.mean(),
        'median_opacity': torch.median(opacity),
        'low_opacity_ratio': (opacity < low_opacity_threshold).float().mean(),
        'mean_scale_volume': scale_volume.mean(),
        'mean_flatten_ratio': flatten_ratio.mean(),
    }


def compute_scene_sugar_loss(scene_gs, cfg, iteration: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total = _zero_like_scene(scene_gs)
    logs: Dict[str, torch.Tensor] = {}
    if not scene_sugar_active(cfg, iteration):
        return total, logs

    eps = _cfg_float(cfg, 'eps', 1e-6)

    opacity_start = _cfg_int(cfg, 'opacity_entropy_start_iter', _cfg_int(cfg, 'start_iter', 0))
    lambda_opacity = _cfg_float(cfg, 'lambda_opacity_entropy', 0.0)
    if lambda_opacity > 0.0 and iteration >= opacity_start:
        loss_opacity = scene_opacity_entropy_loss(scene_gs, eps=eps)
        total = total + lambda_opacity * loss_opacity
        logs['scene_sugar_opacity_entropy'] = loss_opacity.detach()

    surface_start = _cfg_int(cfg, 'surface_reg_start_iter', _cfg_int(cfg, 'start_iter', 0))
    if iteration >= surface_start:
        lambda_volume = _cfg_float(cfg, 'lambda_scale_volume', 0.0)
        if lambda_volume > 0.0:
            loss_volume = scene_scale_volume_loss(scene_gs)
            total = total + lambda_volume * loss_volume
            logs['scene_sugar_scale_volume'] = loss_volume.detach()

        lambda_flatten = _cfg_float(cfg, 'lambda_flatten', 0.0)
        if lambda_flatten > 0.0:
            loss_flatten = scene_flatten_loss(scene_gs, eps=eps)
            total = total + lambda_flatten * loss_flatten
            logs['scene_sugar_flatten'] = loss_flatten.detach()

        # First HUGS integration uses the document's lightweight replacement for
        # the heavier SuGaR SDF/density alignment: scale volume + flattening.
        lambda_surface = _cfg_float(cfg, 'lambda_surface_alignment', 0.0)
        if lambda_surface > 0.0:
            loss_surface = scene_flatten_loss(scene_gs, eps=eps) + scene_scale_volume_loss(scene_gs)
            total = total + lambda_surface * loss_surface
            logs['scene_sugar_surface_alignment'] = loss_surface.detach()

    if logs:
        logs['scene_sugar_total'] = total.detach()
    return total, logs


def should_prune_scene_sugar(cfg, iteration: int) -> bool:
    if not scene_sugar_enabled(cfg):
        return False
    prune_iter = _cfg_int(cfg, 'prune_iter', -1)
    if prune_iter < 0 or iteration < prune_iter:
        return False
    stop_after_iter = _cfg_int(cfg, 'stop_after_iter', -1)
    if stop_after_iter >= 0 and iteration > stop_after_iter:
        return False
    prune_interval = _cfg_int(cfg, 'prune_interval', 0)
    if iteration == prune_iter:
        return True
    return prune_interval > 0 and (iteration - prune_iter) % prune_interval == 0


@torch.no_grad()
def prune_scene_gaussians_by_sugar(scene_gs, cfg, iteration: int) -> Dict[str, float]:
    if not should_prune_scene_sugar(cfg, iteration):
        return {'pruned_count': 0.0, 'num_before': float(scene_gs.get_xyz.shape[0]), 'num_after': float(scene_gs.get_xyz.shape[0])}

    opacity_threshold = _cfg_float(cfg, 'opacity_prune_threshold', 0.03)
    max_prune_frac = _cfg_float(cfg, 'max_prune_frac', 0.25)
    min_remaining = _cfg_int(cfg, 'min_remaining', 1000)

    opacity = scene_gs.get_opacity.detach().squeeze(-1)
    n_before = int(opacity.numel())
    if n_before <= min_remaining:
        return {'pruned_count': 0.0, 'num_before': float(n_before), 'num_after': float(n_before)}

    prune_mask = opacity < opacity_threshold
    candidate_idx = torch.where(prune_mask)[0]
    if candidate_idx.numel() == 0:
        return {'pruned_count': 0.0, 'num_before': float(n_before), 'num_after': float(n_before)}

    max_prune = max(1, int(n_before * max_prune_frac))
    max_prune = min(max_prune, max(0, n_before - min_remaining))
    if candidate_idx.numel() > max_prune:
        candidate_opacity = opacity[candidate_idx]
        selected = torch.topk(candidate_opacity, k=max_prune, largest=False).indices
        candidate_idx = candidate_idx[selected]

    if candidate_idx.numel() == 0 or n_before - candidate_idx.numel() < min_remaining:
        return {'pruned_count': 0.0, 'num_before': float(n_before), 'num_after': float(n_before)}

    prune_mask = torch.zeros(n_before, dtype=torch.bool, device=opacity.device)
    prune_mask[candidate_idx] = True
    scene_gs.prune_points(prune_mask)

    n_after = int(scene_gs.get_xyz.shape[0])
    logger.info(
        f"[{iteration:06d}] Scene-SuGaR pruned {candidate_idx.numel()}/{n_before} scene gaussians "
        f"(opacity_thr={opacity_threshold}, max_frac={max_prune_frac})"
    )
    return {'pruned_count': float(candidate_idx.numel()), 'num_before': float(n_before), 'num_after': float(n_after)}


def _normalize_to_u8(values: torch.Tensor, inverse: bool = False) -> np.ndarray:
    if values.numel() == 0:
        return np.zeros((0,), dtype=np.uint8)
    v = values.detach().float().cpu()
    lo = torch.quantile(v, 0.02)
    hi = torch.quantile(v, 0.98)
    denom = torch.clamp(hi - lo, min=1e-8)
    v = torch.clamp((v - lo) / denom, 0.0, 1.0)
    if inverse:
        v = 1.0 - v
    return (v.numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def save_scene_sugar_debug_ply(scene_gs, path: str, color_by: str = 'opacity') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = scene_gs.get_xyz.detach().cpu().numpy()
    opacity = scene_gs.get_opacity.detach().squeeze(-1)
    scale = scene_gs.get_scaling.detach()
    scale_volume = torch.prod(scale, dim=-1) if scale.numel() else torch.empty_like(opacity)

    if color_by == 'scale_volume':
        red = _normalize_to_u8(scale_volume)
        green = _normalize_to_u8(scale_volume, inverse=True)
        blue = np.full_like(red, 80, dtype=np.uint8)
    else:
        red = _normalize_to_u8(opacity)
        green = _normalize_to_u8(opacity)
        blue = _normalize_to_u8(opacity)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('opacity', 'f4'), ('scale_volume', 'f4'),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = red
    elements['green'] = green
    elements['blue'] = blue
    elements['opacity'] = opacity.detach().cpu().numpy().astype(np.float32)
    elements['scale_volume'] = scale_volume.detach().cpu().numpy().astype(np.float32)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
