"""Tests for layer selection and temporal pooling."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.affect.config import PoolingConfig  # noqa: E402
from training.affect.models.pooling import LayerSelector, TemporalPooling, masked_mean_std  # noqa: E402

TOLERANCE = 1e-5


class TestMaskedStats:
    def test_mask_excludes_padding(self) -> None:
        x = torch.tensor([[[1.0], [3.0], [100.0]]])
        mask = torch.tensor([[True, True, False]])
        out = masked_mean_std(x, mask)
        assert out[0, 0].item() == pytest.approx(2.0, abs=TOLERANCE)
        assert out[0, 1].item() == pytest.approx(1.0, abs=1e-3)

    def test_output_dim_doubles(self) -> None:
        x = torch.randn(2, 5, 8)
        assert masked_mean_std(x, torch.ones(2, 5, dtype=torch.bool)).shape == (2, 16)


class TestLayerSelector:
    def test_weighted_starts_uniform(self) -> None:
        selector = LayerSelector(PoolingConfig(layer_select="weighted"), 3)
        weights = selector.layer_weights()
        assert weights == pytest.approx([1 / 3] * 3, abs=TOLERANCE)
        states = [torch.full((1, 2, 4), float(i)) for i in range(3)]
        assert selector(states).mean().item() == pytest.approx(1.0, abs=TOLERANCE)

    def test_last_and_mean_last4(self) -> None:
        states = [torch.full((1, 2, 4), float(i)) for i in range(6)]
        assert LayerSelector(PoolingConfig(layer_select="last"), 6)(states).mean().item() == 5.0
        assert LayerSelector(PoolingConfig(layer_select="mean_last4"), 6)(states).mean().item() == pytest.approx(3.5)


class TestTemporalPooling:
    def test_attentive_stats_shape_and_mask(self) -> None:
        pooling = TemporalPooling(PoolingConfig(pooling="attentive_stats", attention_hidden=8), 4)
        x = torch.randn(2, 6, 4)
        mask = torch.tensor([[True] * 6, [True] * 3 + [False] * 3])
        out = pooling(x, mask)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()
