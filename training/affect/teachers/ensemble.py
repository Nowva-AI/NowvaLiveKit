"""Teacher ensembling: average dimensional outputs, temperature-calibrate and average categorical distributions."""

from __future__ import annotations

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, target_probs: np.ndarray, iterations: int = 200) -> float:
    """Single scalar temperature minimizing KL(target || softmax(logits / T)) on a held-out set."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(target_probs, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iterations)

    def _closure():
        optimizer.zero_grad()
        log_probs = torch.log_softmax(logits_t / log_t.exp(), dim=-1)
        loss = (targets * (torch.log(targets.clamp(min=1e-8)) - log_probs)).sum(dim=-1).mean()
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(log_t.exp().item())


def average_avd(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    weights = weights or [1.0] * len(predictions)
    total = sum(weights)
    return sum(w * p for w, p in zip(weights, predictions)) / total


def average_probs(prob_sets: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    weights = weights or [1.0] * len(prob_sets)
    total = sum(weights)
    mixed = sum(w * p for w, p in zip(weights, prob_sets)) / total
    return mixed / mixed.sum(axis=-1, keepdims=True).clip(min=1e-8)


def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, 1e-8, 1.0))
