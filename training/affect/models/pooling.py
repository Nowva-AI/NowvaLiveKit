"""Layer selection and temporal pooling: learnable softmax layer weights, masked mean⊕std, attentive stats."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.affect.config import PoolingConfig

EPSILON = 1e-5


def masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D], mask: [B, T] bool -> [B, 2D]
    weights = mask.to(x.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    mean = (x * weights).sum(dim=1) / counts
    var = (((x - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / counts
    std = torch.sqrt(var + EPSILON)
    return torch.cat([mean, std], dim=-1)


class LayerSelector(nn.Module):
    def __init__(self, config: PoolingConfig, num_hidden_states: int) -> None:
        super().__init__()
        self.mode = config.layer_select
        self.num_hidden_states = num_hidden_states
        if self.mode == "weighted":
            self.layer_logits = nn.Parameter(torch.zeros(num_hidden_states))

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        if self.mode == "last":
            return hidden_states[-1]
        if self.mode == "mean_last4":
            return torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
        stacked = torch.stack(hidden_states, dim=0)
        weights = torch.softmax(self.layer_logits, dim=0).view(-1, 1, 1, 1).to(stacked.dtype)
        return (stacked * weights).sum(dim=0)

    def layer_weights(self) -> list[float]:
        if self.mode != "weighted":
            return []
        return torch.softmax(self.layer_logits.detach(), dim=0).tolist()


class AttentiveStatsPooling(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (alpha * x).sum(dim=1)
        var = (alpha * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = torch.sqrt(var + EPSILON)
        return torch.cat([mean, std], dim=-1)


class TemporalPooling(nn.Module):
    def __init__(self, config: PoolingConfig, dim: int) -> None:
        super().__init__()
        self.mode = config.pooling
        self.output_dim = 2 * dim
        if self.mode == "attentive_stats":
            self.attentive = AttentiveStatsPooling(dim, config.attention_hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "attentive_stats":
            return self.attentive(x, mask)
        return masked_mean_std(x, mask)
