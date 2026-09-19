"""Losses: soft-label KL with class masks, 1−CCC for dimensions, distillation CCC + quadrant agreement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

CCC_EPSILON = 1e-8
NEG_INF = float("-inf")


def concordance_cc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    return 2.0 * covariance / (pred_var + target_var + (pred_mean - target_mean) ** 2 + CCC_EPSILON)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over dimensions of 1 − CCC, computed over the rows where mask is true."""
    if mask is None:
        mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    if mask.sum() < 2:
        return pred.sum() * 0.0
    losses = []
    for dim in range(pred.shape[1]):
        losses.append(1.0 - concordance_cc(pred[mask, dim], target[mask, dim]))
    return torch.stack(losses).mean()


def soft_label_kl(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    class_mask: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """KL(target || softmax(logits)); masked classes are removed from both sides and the target renormalized."""
    if class_mask is not None:
        logits = logits.masked_fill(~class_mask, NEG_INF)
        target_probs = target_probs * class_mask.to(target_probs.dtype)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    target = target_probs.float()
    per_class = target * (torch.log(target.clamp(min=1e-8)) - log_probs)
    if focal_gamma > 0.0:
        per_class = per_class * (1.0 - log_probs.exp()) ** focal_gamma
    if class_weights is not None:
        per_class = per_class * class_weights.to(per_class.dtype)[None, :]
    return per_class.sum(dim=-1).mean()


def exertion_bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    if mask.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[mask].float(), target[mask].float())


def quadrant_disagreement_l1(student: torch.Tensor, teacher: torch.Tensor, midpoint: float = 0.5) -> torch.Tensor:
    """L1 penalty only where the student and teacher disagree on which side of the midpoint a value falls."""
    disagree = (torch.sign(student - midpoint) != torch.sign(teacher - midpoint)).to(student.dtype)
    return ((student - teacher).abs() * disagree).mean()


def distillation_loss(
    student_avd: torch.Tensor,
    teacher_avd: torch.Tensor,
    w_ccc: float = 1.0,
    w_quadrant: float = 0.5,
    student_cat_logits: torch.Tensor | None = None,
    teacher_cat_probs: torch.Tensor | None = None,
    w_cat_kl: float = 1.0,
) -> torch.Tensor:
    loss = w_ccc * ccc_loss(student_avd, teacher_avd) + w_quadrant * quadrant_disagreement_l1(student_avd, teacher_avd)
    if student_cat_logits is not None and teacher_cat_probs is not None:
        loss = loss + w_cat_kl * soft_label_kl(student_cat_logits, teacher_cat_probs)
    return loss


def inverse_frequency_weights(class_counts: torch.Tensor) -> torch.Tensor:
    counts = class_counts.float().clamp(min=1.0)
    weights = counts.sum() / (counts.numel() * counts)
    return weights / weights.mean()
