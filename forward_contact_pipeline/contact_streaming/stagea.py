"""Small local mesh--point relation encoder for the Stage-A ablation."""
from __future__ import annotations
import torch
from torch import nn

class LocalRelationEncoder(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(7, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.vertex = nn.Sequential(nn.Linear(hidden + 6, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.contact = nn.Linear(hidden, 1)
        self.proximity = nn.Linear(hidden, 1)
        self.reliability = nn.Linear(hidden, 1)
    def forward(self, vertex_local, vertex_normal_local, point_local, point_normal_local, point_valid):
        point_input=torch.cat([point_local, point_normal_local, point_valid[...,None]],dim=-1)
        encoded=self.point(point_input)
        mask=point_valid[...,None]
        pooled=(encoded*mask).sum(-2)/mask.sum(-2).clamp_min(1.)
        token=self.vertex(torch.cat([vertex_local,vertex_normal_local,pooled],dim=-1))
        return {"contact_logits":self.contact(token).squeeze(-1),"proximity":self.proximity(token).squeeze(-1),"reliability_logits":self.reliability(token).squeeze(-1)}
