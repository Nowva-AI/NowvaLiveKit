"""Task heads on the pooled embedding: categorical logits, sigmoid arousal/dominance/valence, exertion logit."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CategoricalHead(MLPHead):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float) -> None:
        super().__init__(in_dim, hidden, num_classes, dropout)


class AVDHead(MLPHead):
    """Outputs arousal, dominance, valence in [0, 1] (sigmoid), in that order."""

    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__(in_dim, hidden, 3, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))


class ExertionHead(MLPHead):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__(in_dim, hidden, 1, dropout)
