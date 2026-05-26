#
# Lightweight anchor token pooling modules for debug experiments.
#

import math

import torch
from torch import nn
import torch.nn.functional as F


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AnchorQueryMLP(nn.Module):
    def __init__(self, num_anchors, hidden_dim=128):
        super().__init__()
        self.anchor_embed = nn.Embedding(num_anchors, hidden_dim)
        self.part_embed = nn.Embedding(num_anchors, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, anchor_ids):
        x = torch.cat([self.anchor_embed(anchor_ids), self.part_embed(anchor_ids)], dim=-1)
        return self.net(x)


class BiasMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AnchorTokenPooler(nn.Module):
    def __init__(self, input_dim, num_anchors, hidden_dim=128):
        super().__init__()
        self.encoder = GaussianEncoder(input_dim, hidden_dim)
        self.query = AnchorQueryMLP(num_anchors, hidden_dim)
        self.bias = BiasMLP()
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward_anchor(self, features, bias_features, anchor_id):
        if features.numel() == 0:
            empty = torch.zeros(self.hidden_dim, dtype=torch.float32, device=features.device)
            return empty, torch.empty(0, dtype=torch.float32, device=features.device)

        encoded = self.encoder(features)
        keys = self.key(encoded)
        values = self.value(encoded)
        query = self.query(torch.tensor([anchor_id], dtype=torch.long, device=features.device))[0]
        logits = (keys @ query) / math.sqrt(self.hidden_dim)
        logits = logits + self.bias(bias_features)
        alpha = torch.softmax(logits, dim=0)
        token = (alpha[:, None] * values).sum(dim=0)
        return token, alpha


def mean_pool_token(features, out_dim=128):
    if features.numel() == 0:
        return torch.zeros(out_dim, dtype=torch.float32, device=features.device)
    pooled = features.mean(dim=0)
    if pooled.numel() >= out_dim:
        return pooled[:out_dim]
    return F.pad(pooled, (0, out_dim - pooled.numel()))


def attention_entropy(weights):
    if weights.numel() == 0:
        return torch.tensor(float("nan"))
    w = weights.clamp(min=1e-8)
    return -(w * torch.log(w)).sum()
