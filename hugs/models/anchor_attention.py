#
# Minimal anchor-scene attention baseline modules.
#

import csv
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, zero_init_last=False):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(max(num_layers - 1, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)
        if zero_init_last:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class AnchorSceneAttentionBaseline(nn.Module):
    """Small, replaceable interaction module for HUGS human_scene training.

    The module intentionally keeps a narrow contract: it consumes the regular
    HUGS human/scene forward dictionaries and optional human anchor bindings,
    then returns a shallow-copied human dictionary with optional mu/opacity
    corrections plus debug tensors.
    """

    def __init__(self, cfg, num_anchors, top_m=2):
        super().__init__()
        self.cfg = cfg
        self.num_anchors = int(num_anchors)
        self.top_m = int(top_m)
        self.hidden_dim = int(getattr(cfg, "hidden_dim", 128))
        self.human_feature_dim = 16
        self.scene_feature_dim = 11

        self.anchor_embed = nn.Embedding(self.num_anchors, self.hidden_dim)
        self.human_encoder = MLP(self.human_feature_dim, self.hidden_dim, self.hidden_dim)
        self.scene_encoder = MLP(self.scene_feature_dim, self.hidden_dim, self.hidden_dim)
        self.human_attn_bias = MLP(4, 32, 1)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        zero_delta = bool(getattr(cfg, "zero_init_delta", True))
        self.delta_mu = MLP(self.human_feature_dim + self.hidden_dim, self.hidden_dim, 3, zero_init_last=zero_delta)
        self.delta_transl = MLP(self.hidden_dim, self.hidden_dim, 3, zero_init_last=zero_delta)
        self.delta_opacity = MLP(1 + self.hidden_dim, max(self.hidden_dim // 2, 32), 1, zero_init_last=zero_delta)
        self.delta_scale = MLP(self.human_feature_dim + self.hidden_dim, self.hidden_dim, 3, zero_init_last=zero_delta)
        self.delta_feature_dc = MLP(self.human_feature_dim + self.hidden_dim, self.hidden_dim, 3, zero_init_last=zero_delta)

    @staticmethod
    def _safe_log_scale(scales):
        return torch.log(torch.clamp(scales, min=1e-6))

    def _anchor_world_from_bindings(self, human_out, anchor_ids, anchor_weights):
        xyz = human_out["xyz"]
        device = xyz.device
        dtype = xyz.dtype
        anchor_world = torch.zeros(self.num_anchors, 3, device=device, dtype=dtype)
        denom = torch.zeros(self.num_anchors, 1, device=device, dtype=dtype)
        ids = anchor_ids[:, : self.top_m].reshape(-1)
        weights = anchor_weights[:, : self.top_m].reshape(-1, 1)
        xyz_rep = xyz[:, None, :].expand(-1, min(self.top_m, anchor_ids.shape[1]), -1).reshape(-1, 3)
        anchor_world.scatter_add_(0, ids[:, None].expand(-1, 3), xyz_rep * weights)
        denom.scatter_add_(0, ids[:, None], weights)
        global_center = xyz.mean(dim=0, keepdim=True)
        anchor_world = torch.where(denom > 1e-6, anchor_world / denom.clamp_min(1e-6), global_center)
        return anchor_world

    def _human_features(self, human_out, anchor_world, anchor_ids, anchor_weights):
        xyz = human_out["xyz"]
        top_ids = anchor_ids[:, 0].long()
        top_weight = anchor_weights[:, 0:1]
        rel = xyz - anchor_world[top_ids]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        normals = human_out.get("normals", torch.zeros_like(xyz))
        normal_dist = (rel * F.normalize(normals, dim=-1, eps=1e-6)).sum(dim=-1, keepdim=True)
        opacity = human_out["opacity"]
        log_scale = self._safe_log_scale(human_out["scales"])
        canon = human_out.get("xyz_canon", xyz)
        rel_canon = canon - canon.mean(dim=0, keepdim=True)
        return torch.cat([rel, dist, normal_dist, opacity, log_scale, normals, rel_canon, top_weight], dim=-1)

    def _pool_human_tokens(self, human_features, anchor_ids, anchor_weights):
        device = human_features.device
        encoded = self.human_encoder(human_features)
        tokens = []
        pooling = str(getattr(self.cfg, "human_pooling", "attention"))
        for anchor_id in range(self.num_anchors):
            mask = anchor_ids[:, 0] == anchor_id
            if not torch.any(mask):
                tokens.append(torch.zeros(self.hidden_dim, device=device, dtype=encoded.dtype))
                continue
            if pooling == "mean":
                tokens.append(encoded[mask].mean(dim=0))
                continue
            q = self.anchor_embed(torch.tensor(anchor_id, device=device))
            local = encoded[mask]
            f = human_features[mask]
            bias_in = torch.cat([f[:, 3:5], f[:, 5:6], f[:, -1:]], dim=-1)
            logits = (local @ q) / math.sqrt(self.hidden_dim)
            logits = logits + self.human_attn_bias(bias_in).squeeze(-1)
            alpha = torch.softmax(logits, dim=0)
            tokens.append((alpha[:, None] * local).sum(dim=0))
        return torch.stack(tokens, dim=0)

    def _query_scene_tokens(self, scene_out, anchor_world):
        scene_xyz = scene_out["xyz"]
        scene_opacity = scene_out["opacity"].reshape(-1)
        min_opacity = float(getattr(self.cfg, "scene_opacity_threshold", 0.01))
        max_candidates = int(getattr(self.cfg, "max_scene_candidates", 200000))
        valid = scene_opacity > min_opacity
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if valid_idx.numel() == 0:
            valid_idx = torch.arange(scene_xyz.shape[0], device=scene_xyz.device)
        if valid_idx.numel() > max_candidates > 0:
            perm = torch.randperm(valid_idx.numel(), device=scene_xyz.device)[:max_candidates]
            valid_idx = valid_idx[perm]

        xyz = scene_xyz[valid_idx]
        k = min(int(getattr(self.cfg, "scene_topk", 32)), xyz.shape[0])
        d = torch.cdist(anchor_world, xyz)
        knn_dist, local_idx = torch.topk(d, k=k, dim=1, largest=False)
        knn_idx = valid_idx[local_idx]

        rel = scene_out["xyz"][knn_idx] - anchor_world[:, None, :]
        opacity = scene_out["opacity"][knn_idx]
        log_scale = self._safe_log_scale(scene_out["scales"][knn_idx])
        shs = scene_out["shs"]
        if shs.dim() == 3:
            color = shs[knn_idx][:, :, 0, :]
        else:
            color = shs[knn_idx][..., :3]
        features = torch.cat([rel, knn_dist[..., None], opacity, log_scale, color], dim=-1)
        return features, knn_idx, knn_dist

    def _cross_attention(self, human_tokens, scene_features):
        scene_tokens = self.scene_encoder(scene_features)
        q = self.query(human_tokens)[:, None, :]
        k = self.key(scene_tokens)
        v = self.value(scene_tokens)
        logits = (q * k).sum(dim=-1) / math.sqrt(self.hidden_dim)
        weights = torch.softmax(logits, dim=-1)
        context = (weights[..., None] * v).sum(dim=1)
        return self.context_norm(context), weights

    def _context_per_gaussian(self, context, anchor_ids, anchor_weights):
        ids = anchor_ids[:, : self.top_m]
        weights = anchor_weights[:, : self.top_m]
        ctx = context[ids]
        return (ctx * weights[..., None]).sum(dim=1)

    def forward(self, human_out, scene_out, anchor_ids, anchor_weights, iteration=0):
        enabled = bool(getattr(self.cfg, "use_cross_attention", True))
        module_start = int(getattr(self.cfg, "module_start_iter", getattr(self.cfg, "correction_start_iter", 3000)))
        if scene_out is None or not enabled or int(iteration) < module_start:
            return human_out, {}

        anchor_ids = anchor_ids.to(human_out["xyz"].device).long()
        anchor_weights = anchor_weights.to(human_out["xyz"].device, dtype=human_out["xyz"].dtype)
        anchor_world = self._anchor_world_from_bindings(human_out, anchor_ids, anchor_weights)
        human_features = self._human_features(human_out, anchor_world, anchor_ids, anchor_weights)
        human_tokens = self._pool_human_tokens(human_features, anchor_ids, anchor_weights)
        scene_features, knn_idx, knn_dist = self._query_scene_tokens(scene_out, anchor_world)
        context, attn = self._cross_attention(human_tokens, scene_features)

        corrected = dict(human_out)
        stats = {
            "anchor_world": anchor_world.detach(),
            "human_tokens": human_tokens.detach(),
            "scene_knn_idx": knn_idx.detach(),
            "scene_knn_dist": knn_dist.detach(),
            "attention_weights": attn.detach(),
            "context": context.detach(),
        }

        correction_start = int(getattr(self.cfg, "correction_start_iter", 3000))
        use_correction = bool(getattr(self.cfg, "use_interaction_correction", False)) and iteration >= correction_start
        if not use_correction:
            stats["delta_loss"] = human_out["xyz"].new_tensor(0.0)
            return corrected, stats

        ctx_g = self._context_per_gaussian(context, anchor_ids, anchor_weights)
        delta_mu = self.delta_mu(torch.cat([human_features, ctx_g], dim=-1))
        delta_transl = self.delta_transl(context.mean(dim=0, keepdim=True)).reshape(3)
        gamma_mu = float(getattr(self.cfg, "gamma_mu", 0.02))
        warmup_iters = int(getattr(self.cfg, "correction_warmup_iters", 0) or 0)
        if warmup_iters > 0:
            warm = min(1.0, max(0.0, float(iteration - correction_start + 1) / float(warmup_iters)))
        else:
            warm = 1.0
        gamma_mu_eff = gamma_mu * warm
        if bool(getattr(self.cfg, "correct_xyz", True)):
            corrected["xyz"] = human_out["xyz"] + gamma_mu_eff * delta_mu
        else:
            corrected["xyz"] = human_out["xyz"]

        if bool(getattr(self.cfg, "correct_transl", False)):
            transl_delta_clamp = float(getattr(self.cfg, "transl_delta_clamp", 0.05) or 0.0)
            if transl_delta_clamp > 0.0:
                delta_transl = torch.clamp(delta_transl, -transl_delta_clamp, transl_delta_clamp)
            gamma_transl_eff = float(getattr(self.cfg, "gamma_transl", 0.005)) * warm
            corrected["xyz"] = corrected["xyz"] + gamma_transl_eff * delta_transl.reshape(1, 3)
        else:
            gamma_transl_eff = 0.0
            delta_transl = torch.zeros_like(delta_transl)

        if bool(getattr(self.cfg, "correct_opacity", False)):
            delta_opacity = self.delta_opacity(torch.cat([human_out["opacity"], ctx_g], dim=-1))
            opacity_delta_clamp = float(getattr(self.cfg, "opacity_delta_clamp", 0.0) or 0.0)
            if opacity_delta_clamp > 0.0:
                delta_opacity = torch.clamp(delta_opacity, -opacity_delta_clamp, opacity_delta_clamp)
            gamma_opacity = float(getattr(self.cfg, "gamma_opacity", 0.05)) * warm
            corrected["opacity"] = torch.clamp(human_out["opacity"] + gamma_opacity * delta_opacity, 1e-4, 0.999)
        else:
            delta_opacity = torch.zeros_like(human_out["opacity"])

        if bool(getattr(self.cfg, "correct_scale", False)):
            delta_scale = self.delta_scale(torch.cat([human_features, ctx_g], dim=-1))
            scale_delta_clamp = float(getattr(self.cfg, "scale_delta_clamp", 0.05) or 0.0)
            if scale_delta_clamp > 0.0:
                delta_scale = torch.clamp(delta_scale, -scale_delta_clamp, scale_delta_clamp)
            gamma_scale = float(getattr(self.cfg, "gamma_scale", 0.01)) * warm
            corrected["scales"] = torch.clamp(
                human_out["scales"] * torch.exp(gamma_scale * delta_scale),
                min=1e-6,
                max=float(getattr(self.cfg, "scale_max", 1.0) or 1.0),
            )
        else:
            delta_scale = torch.zeros_like(human_out["scales"])

        if bool(getattr(self.cfg, "correct_feature_dc", False)):
            delta_feature_dc = self.delta_feature_dc(torch.cat([human_features, ctx_g], dim=-1))
            feature_delta_clamp = float(getattr(self.cfg, "feature_delta_clamp", 0.05) or 0.0)
            if feature_delta_clamp > 0.0:
                delta_feature_dc = torch.clamp(delta_feature_dc, -feature_delta_clamp, feature_delta_clamp)
            gamma_feature_dc = float(getattr(self.cfg, "gamma_feature_dc", 0.01)) * warm
            shs = human_out["shs"].clone()
            shs[:, 0, :] = shs[:, 0, :] + gamma_feature_dc * delta_feature_dc
            corrected["shs"] = shs
        else:
            shs = human_out.get("shs", None)
            if shs is not None and shs.dim() >= 3:
                delta_feature_dc = torch.zeros_like(shs[:, 0, :])
            else:
                delta_feature_dc = torch.zeros_like(human_out["xyz"])

        stats["delta_mu"] = delta_mu.detach()
        stats["delta_transl"] = delta_transl.detach()
        stats["gamma_mu_eff"] = human_out["xyz"].new_tensor(gamma_mu_eff).detach()
        stats["gamma_transl_eff"] = human_out["xyz"].new_tensor(gamma_transl_eff).detach()
        stats["correction_warmup"] = human_out["xyz"].new_tensor(warm).detach()
        stats["delta_opacity"] = delta_opacity.detach()
        stats["delta_scale"] = delta_scale.detach()
        stats["delta_feature_dc"] = delta_feature_dc.detach()
        xyz_delta_loss = delta_mu.pow(2).mean() if bool(getattr(self.cfg, "correct_xyz", True)) else delta_mu.new_tensor(0.0)
        stats["delta_loss"] = (
            xyz_delta_loss
            + delta_transl.pow(2).mean()
            + delta_opacity.pow(2).mean()
            + delta_scale.pow(2).mean()
            + delta_feature_dc.pow(2).mean()
        )
        return corrected, stats


def save_anchor_attention_debug(out_dir, iteration, stats, anchor_names=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iter_s = f"{int(iteration):06d}"
    payload = {
        "anchor_names": anchor_names,
        "anchor_world": stats.get("anchor_world"),
        "human_tokens": stats.get("human_tokens"),
        "scene_knn_idx": stats.get("scene_knn_idx"),
        "scene_knn_dist": stats.get("scene_knn_dist"),
        "attention_weights": stats.get("attention_weights"),
        "context": stats.get("context"),
        "delta_mu": stats.get("delta_mu"),
        "delta_transl": stats.get("delta_transl"),
        "gamma_mu_eff": stats.get("gamma_mu_eff"),
        "gamma_transl_eff": stats.get("gamma_transl_eff"),
        "delta_opacity": stats.get("delta_opacity"),
        "delta_scale": stats.get("delta_scale"),
        "delta_feature_dc": stats.get("delta_feature_dc"),
    }
    torch.save(payload, out_dir / f"anchor_attention_iter{iter_s}.pt")

    knn_dist = payload["scene_knn_dist"]
    attn = payload["attention_weights"]
    if knn_dist is None or attn is None:
        return
    knn_dist = knn_dist.detach().cpu()
    attn = attn.detach().cpu()
    names = anchor_names or [f"anchor_{idx}" for idx in range(knn_dist.shape[0])]
    with (out_dir / f"anchor_attention_iter{iter_s}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter",
            "anchor_id",
            "anchor_name",
            "nearest_scene_distance",
            "attention_entropy",
            "max_attention",
            "attention_weighted_distance",
        ])
        for anchor_id in range(knn_dist.shape[0]):
            w = attn[anchor_id].clamp(min=1e-8)
            entropy = float(-(w * torch.log(w)).sum())
            weighted_dist = float((attn[anchor_id] * knn_dist[anchor_id]).sum())
            writer.writerow([
                int(iteration),
                anchor_id,
                names[anchor_id],
                float(knn_dist[anchor_id].min()),
                entropy,
                float(attn[anchor_id].max()),
                weighted_dist,
            ])
