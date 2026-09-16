from __future__ import annotations

import torch
from torch import nn

from .schema import OUTPUT_DIM, split_model_output


class ContactGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.temporal = nn.GRU(
            hidden_dim, hidden_dim, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, OUTPUT_DIM))

    def forward(self, features: torch.Tensor):
        encoded = self.input_proj(features)
        hidden, _ = self.temporal(encoded)
        return split_model_output(self.head(hidden[:, -1]))


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv(value)
        if self.padding:
            value = value[..., :-self.padding]
        return residual + self.dropout(torch.nn.functional.gelu(self.norm(value)))


class ContactTCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 3, kernel_size: int = 3, dropout: float = 0.05):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.Sequential(
            *[CausalBlock(hidden_dim, kernel_size, 2**index, dropout) for index in range(layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, OUTPUT_DIM))

    def forward(self, features: torch.Tensor):
        value = self.input_proj(features.transpose(1, 2))
        value = self.blocks(value)[..., -1]
        return split_model_output(self.head(value))


def build_model(name: str, input_dim: int, hidden_dim: int = 128, layers: int = 1, dropout: float = 0.0) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return ContactGRU(input_dim, hidden_dim, layers, dropout)
    if name == "tcn":
        return ContactTCN(input_dim, hidden_dim, layers, dropout=dropout)
    raise ValueError(f"Unknown model type: {name}")
