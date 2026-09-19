"""Tests for the per-speaker baseline: neutral gating, shrinkage, test-exclusive stats, persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from affect.baseline import SpeakerBaseline
from affect.config import BaselineConfig

POP_MEAN = np.array([0.5, 0.5, 0.5])
POP_STD = np.array([0.15, 0.15, 0.15])
TOLERANCE = 1e-6


def _baseline(**overrides) -> SpeakerBaseline:
    config = BaselineConfig(**overrides)
    return SpeakerBaseline(config, POP_MEAN, POP_STD, user_id="u1")


class TestEnrollment:
    def test_rejects_non_neutral_context(self) -> None:
        baseline = _baseline()
        assert baseline.observe(POP_MEAN, 2.0, neutral_context=False) is False
        assert baseline.sample_count == 0

    def test_first_samples_enroll_even_far_from_population_norms(self) -> None:
        baseline = _baseline(enrollment_target_seconds=1e9)
        assert baseline.observe(np.array([0.2, 0.35, 0.3]), 2.0, neutral_context=True) is True

    def test_rejects_outliers_against_own_stats_after_min_samples(self) -> None:
        baseline = _baseline(enrollment_target_seconds=1e9, min_samples_before_gate=3, neutral_z_max=2.5)
        for i in range(3):
            assert baseline.observe(np.array([0.25, 0.35, 0.3]) + 0.005 * i, 2.0, neutral_context=True, utterance_id=f"u{i}") is True
        assert baseline.observe(np.array([0.95, 0.35, 0.3]), 2.0, neutral_context=True) is False
        assert baseline.observe(np.array([0.27, 0.36, 0.31]), 2.0, neutral_context=True) is True

    def test_accepts_neutral_and_freezes_at_target(self) -> None:
        baseline = _baseline(enrollment_target_seconds=10.0, enrollment_floor_seconds=5.0)
        for i in range(4):
            accepted = baseline.observe(POP_MEAN + 0.01 * i, 3.0, neutral_context=True, utterance_id=f"u{i}")
            assert accepted is True
        assert baseline.ready
        assert baseline.frozen
        assert baseline.observe(POP_MEAN, 3.0, neutral_context=True) is False


class TestStatistics:
    def test_no_samples_uses_prior(self) -> None:
        baseline = _baseline()
        mean, std, n = baseline.stats()
        assert n == 0
        assert np.allclose(mean, POP_MEAN) and np.allclose(std, POP_STD)
        assert np.allclose(baseline.z_scores(POP_MEAN), 0.0)

    def test_shrinkage_moves_toward_user(self) -> None:
        baseline = _baseline(shrinkage_prior_n=20.0, enrollment_target_seconds=1e9)
        rng = np.random.default_rng(0)
        for i in range(40):
            avd = np.array([0.6, 0.5, 0.45]) + rng.normal(0, 0.01, 3)
            baseline.observe(avd, 1.0, neutral_context=True, utterance_id=f"u{i}")
        mean, std, n = baseline.stats()
        assert n == 40
        assert 0.5 < mean[0] < 0.6
        assert mean[0] == pytest.approx((40 * 0.6 + 20 * 0.5) / 60, abs=0.012)
        assert np.all(std >= 0.5 * POP_STD - TOLERANCE)

    def test_exclude_ids_is_test_exclusive(self) -> None:
        baseline = _baseline(enrollment_target_seconds=1e9)
        baseline.observe(np.array([0.55, 0.5, 0.5]), 1.0, neutral_context=True, utterance_id="a")
        baseline.observe(np.array([0.45, 0.5, 0.5]), 1.0, neutral_context=True, utterance_id="b")
        with_all = baseline.stats()[0]
        without_a = baseline.stats(exclude_ids={"a"})[0]
        assert not np.allclose(with_all, without_a)
        assert baseline.stats(exclude_ids={"a", "b"})[2] == 0

    def test_std_floor(self) -> None:
        baseline = _baseline(shrinkage_prior_n=0.001, enrollment_target_seconds=1e9)
        for i in range(30):
            baseline.observe(POP_MEAN, 1.0, neutral_context=True, utterance_id=f"u{i}")
        _, std, _ = baseline.stats()
        assert np.allclose(std, 0.5 * POP_STD, atol=1e-6)


class TestPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        baseline = _baseline(enrollment_target_seconds=4.0)
        baseline.observe(np.array([0.52, 0.5, 0.48]), 2.0, neutral_context=True, utterance_id="x", embedding=np.ones(4))
        baseline.observe(np.array([0.51, 0.5, 0.49]), 2.0, neutral_context=True, utterance_id="y", embedding=np.zeros(4))
        path = baseline.save(tmp_path / "u1.npz")
        loaded = SpeakerBaseline.load(path, BaselineConfig(), POP_MEAN, POP_STD)
        assert loaded.user_id == "u1"
        assert loaded.frozen is True
        assert loaded.sample_count == 2
        assert np.allclose(loaded.stats()[0], baseline.stats()[0])

    def test_end_session_moves_prior(self) -> None:
        baseline = _baseline(cross_session_ema_alpha=0.5, enrollment_target_seconds=1e9)
        for i in range(10):
            baseline.observe(np.array([0.6, 0.5, 0.5]), 1.0, neutral_context=True, utterance_id=f"u{i}")
        baseline.end_session()
        assert baseline.prior_mean[0] > POP_MEAN[0]
        assert baseline.session_count == 1

    def test_summary_keys(self) -> None:
        summary = _baseline().summary()
        assert {"user_id", "enrollment_seconds", "samples", "frozen", "ready", "mean", "std"} <= set(summary)
