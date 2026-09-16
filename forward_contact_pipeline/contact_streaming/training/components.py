"""Replaceable Stage-A model and objective factories; no raw image decoding."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from contact_streaming.stagea import LocalRelationEncoder


class GeometryModel(nn.Module):
    required_inputs = ()

    def __init__(self, hidden=96):
        super().__init__()
        self.encoder = LocalRelationEncoder(hidden)

    def forward(self, inputs):
        return self.encoder(inputs["vertex_local"], inputs["vertex_normal"], inputs["point_local"], inputs["point_normal"], inputs["point_valid"])


class FusionModel(nn.Module):
    """Local geometry plus optional cached RGB attention and Gaussian attributes.

    Cache exporters determine token semantics/dimensions; those versions belong
    in the manifest. This model neither estimates nor claims calibrated safety.
    """
    def __init__(self, hidden=96, rgb_dim=0, gaussian_dim=0, heads=4):
        super().__init__()
        if hidden % heads or min(rgb_dim, gaussian_dim) < 0:
            raise ValueError("Invalid fusion dimensions")
        self.rgb_dim, self.gaussian_dim = rgb_dim, gaussian_dim
        self.required_inputs = tuple(k for k, d in (("rgb_tokens", rgb_dim), ("gaussian_features", gaussian_dim)) if d)
        self.point = nn.Sequential(nn.Linear(7 + gaussian_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.vertex = nn.Sequential(nn.Linear(hidden + 6, hidden), nn.LayerNorm(hidden), nn.GELU())
        if rgb_dim:
            self.rgb = nn.Linear(rgb_dim, hidden)
            self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
            self.norm = nn.LayerNorm(hidden)
        self.contact = nn.Linear(hidden, 1)
        self.proximity = nn.Linear(hidden, 1)

    def forward(self, inputs):
        valid = inputs["point_valid"].unsqueeze(-1)
        pieces = [inputs["point_local"], inputs["point_normal"], valid]
        if self.gaussian_dim:
            if inputs["gaussian_features"].shape[-1] != self.gaussian_dim:
                raise ValueError("gaussian_dim does not match cache")
            pieces.append(inputs["gaussian_features"])
        point = self.point(torch.cat(pieces, dim=-1))
        pooled = (point * valid).sum(-2) / valid.sum(-2).clamp_min(1)
        vertex = self.vertex(torch.cat([inputs["vertex_local"], inputs["vertex_normal"], pooled], dim=-1))
        if self.rgb_dim:
            if inputs["rgb_tokens"].shape[-1] != self.rgb_dim:
                raise ValueError("rgb_dim does not match cache")
            rgb = self.rgb(inputs["rgb_tokens"])
            update, _ = self.attention(vertex, rgb, rgb, need_weights=False)
            vertex = self.norm(vertex + update)
        return {"contact_logits": self.contact(vertex).squeeze(-1), "proximity": self.proximity(vertex).squeeze(-1)}


class ContactObjective(nn.Module):
    def __init__(self, contact_weight=1.0, proximity_weight=0.0, positive_weight=1.0, proximity_beta=0.01):
        super().__init__()
        if contact_weight <= 0 or proximity_weight < 0 or positive_weight <= 0 or proximity_beta <= 0:
            raise ValueError("Invalid objective weights")
        self.contact_weight, self.proximity_weight = contact_weight, proximity_weight
        self.positive_weight, self.proximity_beta = positive_weight, proximity_beta

    def forward(self, outputs, targets):
        logits = outputs["contact_logits"]
        mask = targets["contact_valid"]
        if logits.shape != targets["contact"].shape:
            raise ValueError("Contact prediction and target shape mismatch")
        bce = F.binary_cross_entropy_with_logits(logits, targets["contact"], reduction="none", pos_weight=logits.new_tensor(self.positive_weight))
        loss = self.contact_weight * (bce * mask).sum() / mask.sum().clamp_min(1)
        if self.proximity_weight:
            if not {"proximity", "proximity_valid"} <= targets.keys():
                raise ValueError("Proximity objective requires explicit proximity and validity targets")
            valid = targets["proximity_valid"]
            error = F.smooth_l1_loss(outputs["proximity"], targets["proximity"], reduction="none", beta=self.proximity_beta)
            loss = loss + self.proximity_weight * (error * valid).sum() / valid.sum().clamp_min(1)
        return loss


class ContactMetrics:
    """Bounded-memory per-vertex metrics; AP is explicitly a histogram estimate."""
    def __init__(self, threshold=0.5, bins=1024, proximity=False):
        if not 0 < threshold < 1 or bins < 2:
            raise ValueError("Invalid metric threshold/bins")
        self.threshold, self.bins = threshold, bins
        self.proximity = proximity
        self.pos = torch.zeros(bins, dtype=torch.float64)
        self.neg = self.pos.clone()
        self.tp = self.fp = self.fn = self.count = self.brier = 0
        self.distance = self.distance_count = 0

    def update(self, outputs, targets):
        mask = targets["contact_valid"].bool()
        p = outputs["contact_logits"].detach().sigmoid()[mask].cpu().double()
        y = targets["contact"][mask].detach().cpu().double()
        if not torch.isfinite(p).all():
            raise ValueError("Non-finite contact predictions")
        pred, label = p >= self.threshold, y > .5
        self.tp += int((pred & label).sum())
        self.fp += int((pred & ~label).sum())
        self.fn += int((~pred & label).sum())
        self.count += p.numel()
        self.brier += float(((p - y) ** 2).sum())
        idx = (p * self.bins).long().clamp_max(self.bins - 1)
        self.pos += torch.bincount(idx[label], minlength=self.bins)
        self.neg += torch.bincount(idx[~label], minlength=self.bins)
        if self.proximity and "proximity" in targets and "proximity" in outputs:
            valid = targets["proximity_valid"].bool()
            error = (outputs["proximity"].detach()[valid] - targets["proximity"][valid]).abs()
            if not torch.isfinite(error).all():
                raise ValueError("Non-finite proximity predictions")
            self.distance += float(error.sum())
            self.distance_count += error.numel()

    def compute(self):
        if not self.count:
            raise ValueError("No valid metric samples")
        positives = self.pos.flip(0)
        tp = positives.cumsum(0)
        fp = self.neg.flip(0).cumsum(0)
        ap = ((tp / (tp + fp).clamp_min(1)) * positives).sum() / tp[-1].clamp_min(1)
        return {"contact_f1": 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1),
                "precision": self.tp / max(self.tp + self.fp, 1), "recall": self.tp / max(self.tp + self.fn, 1),
                "average_precision_histogram": float(ap), "histogram_bins": self.bins,
                "brier": self.brier / self.count, "valid_vertices": self.count,
                "proximity_mae_m": self.distance / self.distance_count if self.distance_count else None}
