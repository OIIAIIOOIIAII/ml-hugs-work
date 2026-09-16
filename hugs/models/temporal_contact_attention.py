#
# Temporal extension for AnchorSceneAttentionBaseline.
#
# Adds two mechanisms:
#   1. Frame-keyed temporal memory + learned temporal bias on attention logits.
#   2. Temporal smoothness loss between current and historical context vectors.
#
# Compatible with random frame sampling: memory is indexed by frame_idx
# and neighbor lookup uses index proximity, not chronological order.
#
# NOT implemented (design doc section 4.3):
#   HierarchicalTemporalAccelerator — cache reuse breaks gradient computation
#   and cannot be used in gradient-based training.
#

import math

import torch
import torch.nn.functional as F
from torch import nn

from hugs.models.anchor_attention import AnchorSceneAttentionBaseline


class TemporalAnchorAttention(AnchorSceneAttentionBaseline):
    """AnchorSceneAttentionBaseline with optional frame-keyed temporal memory.

    When ``temporal.enabled=True`` in the config, two things change:

    Temporal bias:
        Before the softmax step of cross-attention, a per-anchor learnable scalar
        bias is added that is derived from the historical context of neighboring
        frames.  The bias MLP is zero-initialized so it has no effect at the
        start of training and must be learned.

    Temporal smoothness loss:
        If a neighboring frame's context is found in memory, an MSE loss between
        the current context vectors and the historical context is added to
        ``delta_loss`` (weighted by ``temporal.smooth_loss_w``).

    Memory details:
        - Stored on CPU to avoid GPU memory pressure (~820 KB for 100 frames).
        - Keyed by ``frame_idx``; a neighbor is any frame within
          ``temporal.temporal_window`` index distance.
        - Capacity capped at ``temporal.memory_size`` (FIFO eviction).

    Usage:
        Set ``anchor_attention.temporal.enabled: true`` in your YAML.
        The trainer automatically instantiates this class instead of the
        baseline when that flag is set.
    """

    def __init__(self, cfg, t_cfg, num_anchors, top_m=2):
        super().__init__(cfg, num_anchors, top_m)

        self.temporal_enabled = bool(getattr(t_cfg, "enabled", False))

        if not self.temporal_enabled:
            # All temporal attributes are absent; forward falls back to parent.
            return

        self.t_memory_size = int(getattr(t_cfg, "memory_size", 100))
        self.t_decay = float(getattr(t_cfg, "decay", 0.85))
        self.t_window = int(getattr(t_cfg, "temporal_window", 5))
        self.t_smooth_w = float(getattr(t_cfg, "smooth_loss_w", 0.01))
        self.t_bias_scale = float(getattr(t_cfg, "temporal_bias_scale", 1.0))

        # Zero-init: no temporal effect at start; must be learned.
        self.temporal_bias_mlp = nn.Linear(self.hidden_dim, 1, bias=True)
        nn.init.zeros_(self.temporal_bias_mlp.weight)
        nn.init.zeros_(self.temporal_bias_mlp.bias)

        # Frame memory: {frame_idx: Tensor[N_anchors, hidden_dim]} on CPU
        self._frame_memory: dict = {}
        self._frame_memory_order: list = []  # FIFO eviction order

    # ------------------------------------------------------------------ #
    # Memory helpers                                                       #
    # ------------------------------------------------------------------ #

    def _query_temporal_context(self, frame_idx: int):
        """Decay-weighted average context from frames within t_window, or None."""
        neighbors = [
            (self.t_decay ** abs(frame_idx - fid), ctx)
            for fid, ctx in self._frame_memory.items()
            if 0 < abs(frame_idx - fid) <= self.t_window
        ]
        if not neighbors:
            return None
        total_w = sum(w for w, _ in neighbors)
        result = None
        for w, ctx in neighbors:
            contrib = (w / total_w) * ctx
            result = contrib if result is None else result + contrib
        return result  # CPU tensor

    def _store_context(self, frame_idx: int, context: torch.Tensor):
        """Store current frame context in memory; evict oldest if over capacity."""
        if frame_idx not in self._frame_memory:
            if len(self._frame_memory) >= self.t_memory_size:
                evict = self._frame_memory_order.pop(0)
                self._frame_memory.pop(evict, None)
            self._frame_memory_order.append(frame_idx)
        self._frame_memory[frame_idx] = context.detach().cpu()

    def clear_memory(self):
        """Reset temporal memory (call between independent sequences if needed)."""
        self._frame_memory.clear()
        self._frame_memory_order.clear()

    # ------------------------------------------------------------------ #
    # Modified cross-attention                                             #
    # ------------------------------------------------------------------ #

    def _cross_attention_temporal(self, human_tokens, scene_features, temporal_context=None):
        """Cross-attention with optional per-anchor temporal bias on logits."""
        scene_tokens = self.scene_encoder(scene_features)
        q = self.query(human_tokens)[:, None, :]   # [N_anchors, 1, D]
        k = self.key(scene_tokens)                  # [N_anchors, topk, D]
        v = self.value(scene_tokens)                # [N_anchors, topk, D]
        logits = (q * k).sum(dim=-1) / math.sqrt(self.hidden_dim)  # [N_anchors, topk]

        if temporal_context is not None:
            # [N_anchors, D] → [N_anchors, 1] broadcast over topk
            t_bias = self.t_bias_scale * self.temporal_bias_mlp(temporal_context)
            logits = logits + t_bias

        weights = torch.softmax(logits, dim=-1)
        context = (weights[..., None] * v).sum(dim=1)  # [N_anchors, D]
        return self.context_norm(context), weights

    # ------------------------------------------------------------------ #
    # Forward override                                                     #
    # ------------------------------------------------------------------ #

    def forward(self, human_out, scene_out, anchor_ids, anchor_weights,
                iteration=0, frame_idx=None):
        """Forward pass; identical to parent when temporal is disabled.

        Args:
            frame_idx: Dataset index of the current frame.  Required for temporal
                memory lookup/storage.  When None, temporal context is skipped.
        """
        if not self.temporal_enabled or frame_idx is None:
            # Delegate entirely to parent (no overhead).
            return super().forward(human_out, scene_out, anchor_ids, anchor_weights, iteration)

        # --- standard early-exit gates (same as parent) ---
        enabled = bool(getattr(self.cfg, "use_cross_attention", True))
        module_start = int(getattr(self.cfg, "module_start_iter",
                                   getattr(self.cfg, "correction_start_iter", 3000)))
        if scene_out is None or not enabled or int(iteration) < module_start:
            return human_out, {}

        device = human_out["xyz"].device
        dtype = human_out["xyz"].dtype

        anchor_ids = anchor_ids.to(device).long()
        anchor_weights = anchor_weights.to(device, dtype=dtype)

        anchor_world = self._anchor_world_from_bindings(human_out, anchor_ids, anchor_weights)
        human_features = self._human_features(human_out, anchor_world, anchor_ids, anchor_weights)
        human_tokens = self._pool_human_tokens(human_features, anchor_ids, anchor_weights)
        scene_features, knn_idx, knn_dist = self._query_scene_tokens(scene_out, anchor_world)

        # --- temporal context ---
        temporal_context = None
        if self._frame_memory:
            ctx_cpu = self._query_temporal_context(frame_idx)
            if ctx_cpu is not None:
                temporal_context = ctx_cpu.to(device, dtype=dtype)

        context, attn = self._cross_attention_temporal(human_tokens, scene_features, temporal_context)

        # Store current frame for future frames
        self._store_context(frame_idx, context)

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
        use_correction = (
            bool(getattr(self.cfg, "use_interaction_correction", False))
            and iteration >= correction_start
        )

        # Temporal smooth loss (computed whether or not correction is active)
        smooth_loss = human_out["xyz"].new_tensor(0.0)
        if temporal_context is not None and self.t_smooth_w > 0.0:
            smooth_loss = F.mse_loss(context, temporal_context.detach()) * self.t_smooth_w
        stats["temporal_smooth_loss"] = smooth_loss.detach()

        if not use_correction:
            stats["delta_loss"] = smooth_loss
            return corrected, stats

        # --- corrections (identical logic to parent) ---
        ctx_g = self._context_per_gaussian(context, anchor_ids, anchor_weights)

        delta_mu = self.delta_mu(torch.cat([human_features, ctx_g], dim=-1))
        delta_transl = self.delta_transl(context.mean(dim=0, keepdim=True)).reshape(3)

        gamma_mu = float(getattr(self.cfg, "gamma_mu", 0.02))
        warmup_iters = int(getattr(self.cfg, "correction_warmup_iters", 0) or 0)
        if warmup_iters > 0:
            warm = min(1.0, max(0.0,
                                float(iteration - correction_start + 1) / float(warmup_iters)))
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
            delta_opacity = self.delta_opacity(
                torch.cat([human_out["opacity"], ctx_g], dim=-1)
            )
            opacity_delta_clamp = float(getattr(self.cfg, "opacity_delta_clamp", 0.0) or 0.0)
            if opacity_delta_clamp > 0.0:
                delta_opacity = torch.clamp(delta_opacity, -opacity_delta_clamp, opacity_delta_clamp)
            gamma_opacity = float(getattr(self.cfg, "gamma_opacity", 0.05)) * warm
            corrected["opacity"] = torch.clamp(
                human_out["opacity"] + gamma_opacity * delta_opacity, 1e-4, 0.999
            )
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
            delta_feature_dc = self.delta_feature_dc(
                torch.cat([human_features, ctx_g], dim=-1)
            )
            feature_delta_clamp = float(getattr(self.cfg, "feature_delta_clamp", 0.05) or 0.0)
            if feature_delta_clamp > 0.0:
                delta_feature_dc = torch.clamp(
                    delta_feature_dc, -feature_delta_clamp, feature_delta_clamp
                )
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

        xyz_delta_loss = (
            delta_mu.pow(2).mean()
            if bool(getattr(self.cfg, "correct_xyz", True))
            else delta_mu.new_tensor(0.0)
        )
        delta_loss = (
            xyz_delta_loss
            + delta_transl.pow(2).mean()
            + delta_opacity.pow(2).mean()
            + delta_scale.pow(2).mean()
            + delta_feature_dc.pow(2).mean()
            + smooth_loss
        )
        stats["delta_loss"] = delta_loss
        return corrected, stats
