"""Tests for numpy prosody features."""

from __future__ import annotations

import numpy as np
import pytest

from affect.prosody import estimate_f0_track, prosody_features

from .conftest import SAMPLE_RATE, make_noise, make_tone

F0_TOLERANCE_HZ = 6.0


class TestF0:
    def test_tone_pitch_recovered(self) -> None:
        wave = make_tone(1.0, freq_hz=140.0)
        f0, rms = estimate_f0_track(wave, SAMPLE_RATE)
        voiced = f0[~np.isnan(f0)]
        assert voiced.size > 0.8 * f0.size
        assert np.median(voiced) == pytest.approx(140.0, abs=F0_TOLERANCE_HZ)
        assert rms.shape == f0.shape

    def test_noise_mostly_unvoiced(self) -> None:
        wave = make_noise(1.0, amplitude=0.2)
        f0, _ = estimate_f0_track(wave, SAMPLE_RATE)
        assert np.isnan(f0).mean() > 0.5

    def test_empty_input(self) -> None:
        f0, rms = estimate_f0_track(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
        assert f0.size == 0 and rms.size == 0


class TestFeatures:
    def test_feature_vector_shape(self) -> None:
        features = prosody_features(make_tone(1.5, freq_hz=200.0), SAMPLE_RATE)
        assert features.as_vector().shape == (6,)
        assert features.f0_median_hz == pytest.approx(200.0, abs=F0_TOLERANCE_HZ)
        assert 0.0 <= features.voiced_ratio <= 1.0

    def test_pause_ratio_from_probabilities(self) -> None:
        probs = np.array([0.9, 0.9, 0.1, 0.1])
        features = prosody_features(make_tone(1.0), SAMPLE_RATE, speech_probabilities=probs)
        assert features.pause_ratio == pytest.approx(0.5, abs=1e-6)

    def test_pause_ratio_from_energy_when_no_probabilities(self) -> None:
        wave = np.concatenate([make_tone(0.5), np.zeros(SAMPLE_RATE // 2, dtype=np.float32)])
        features = prosody_features(wave, SAMPLE_RATE)
        assert features.pause_ratio > 0.3

    def test_silence_has_no_pitch(self) -> None:
        features = prosody_features(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        assert features.f0_median_hz == 0.0
        assert features.rms_dbfs < -100.0
