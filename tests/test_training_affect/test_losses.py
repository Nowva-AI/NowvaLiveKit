"""Tests for the training losses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.affect.losses import (  # noqa: E402
    ccc_loss,
    concordance_cc,
    distillation_loss,
    inverse_frequency_weights,
    quadrant_disagreement_l1,
    soft_label_kl,
)

TOLERANCE = 1e-5


class TestCCC:
    def test_perfect_agreement_is_one(self) -> None:
        x = torch.linspace(0.1, 0.9, 20)
        assert concordance_cc(x, x).item() == pytest.approx(1.0, abs=1e-4)
        assert ccc_loss(x[:, None], x[:, None]).item() == pytest.approx(0.0, abs=1e-4)

    def test_mask_excludes_rows(self) -> None:
        pred = torch.tensor([[0.1], [0.9], [0.5], [0.5]])
        target = torch.tensor([[0.1], [0.9], [0.0], [1.0]])
        full = ccc_loss(pred, target)
        masked = ccc_loss(pred, target, torch.tensor([True, True, False, False]))
        assert masked.item() < full.item()

    def test_too_few_rows_is_zero(self) -> None:
        pred = torch.tensor([[0.3]], requires_grad=True)
        loss = ccc_loss(pred, torch.tensor([[0.7]]))
        assert loss.item() == 0.0


class TestSoftLabelKL:
    def test_matching_distribution_is_zero(self) -> None:
        probs = torch.tensor([[0.7, 0.2, 0.1]])
        logits = torch.log(probs)
        assert soft_label_kl(logits, probs).item() == pytest.approx(0.0, abs=TOLERANCE)

    def test_masked_class_ignored(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -5.0]])
        target = torch.tensor([[0.5, 0.5, 0.0]])
        mask = torch.tensor([[True, True, False]])
        loss = soft_label_kl(logits, target, class_mask=mask)
        logits_changed = torch.tensor([[2.0, 0.0, 50.0]])
        assert soft_label_kl(logits_changed, target, class_mask=mask).item() == pytest.approx(loss.item(), abs=TOLERANCE)

    def test_class_weights_scale(self) -> None:
        logits = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0]])
        base = soft_label_kl(logits, target).item()
        weighted = soft_label_kl(logits, target, class_weights=torch.tensor([2.0, 1.0])).item()
        assert weighted == pytest.approx(2 * base, abs=TOLERANCE)


class TestDistillation:
    def test_quadrant_penalty_only_on_disagreement(self) -> None:
        student = torch.tensor([[0.6, 0.6, 0.4]])
        teacher = torch.tensor([[0.7, 0.4, 0.3]])
        penalty = quadrant_disagreement_l1(student, teacher)
        assert penalty.item() == pytest.approx(0.2 / 3, abs=TOLERANCE)

    def test_distillation_combines(self) -> None:
        student = torch.rand(8, 3)
        teacher = student.clone()
        assert distillation_loss(student, teacher).item() == pytest.approx(0.0, abs=1e-4)

    def test_inverse_frequency_weights_mean_one(self) -> None:
        weights = inverse_frequency_weights(torch.tensor([100, 10, 1]))
        assert weights.mean().item() == pytest.approx(1.0, abs=TOLERANCE)
        assert weights[2] > weights[0]
