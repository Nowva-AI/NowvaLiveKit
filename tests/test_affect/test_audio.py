"""Tests for affect audio preparation: resampling, trimming, cropping, tiling, buckets."""

from __future__ import annotations

import numpy as np
import pytest

from affect.audio import (
    StreamingResampler,
    bucket_samples,
    crop_last_seconds,
    int16_to_float32,
    resample,
    rms_dbfs,
    speech_span,
    tile_pad,
    trim_to_speech,
    voiced_seconds,
)

from .conftest import SAMPLE_RATE, make_tone

WINDOW_S = 0.032
TOLERANCE = 1e-6


class TestConversion:
    def test_int16_scaling(self) -> None:
        pcm = np.array([0, 16384, -32768], dtype=np.int16)
        out = int16_to_float32(pcm)
        assert out.dtype == np.float32
        assert out[1] == pytest.approx(0.5, abs=TOLERANCE)
        assert out[2] == pytest.approx(-1.0, abs=TOLERANCE)

    def test_resample_length(self) -> None:
        wave = make_tone(1.0, sample_rate=24000)
        out = resample(wave, 24000, 16000)
        assert abs(out.shape[0] - 16000) <= 2
        assert out.dtype == np.float32

    def test_resample_noop_same_rate(self) -> None:
        wave = make_tone(0.5)
        assert resample(wave, SAMPLE_RATE, SAMPLE_RATE) is wave or np.array_equal(resample(wave, SAMPLE_RATE, SAMPLE_RATE), wave)

    def test_streaming_resampler_matches_offline(self) -> None:
        wave = make_tone(2.0, sample_rate=24000)
        offline = resample(wave, 24000, 16000)
        streamer = StreamingResampler(24000, 16000)
        chunks = [streamer.push(part) for part in np.array_split(wave, 20)]
        chunks.append(streamer.push(np.zeros(0, dtype=np.float32), last=True))
        streamed = np.concatenate(chunks)
        assert abs(streamed.shape[0] - offline.shape[0]) <= 4
        n = min(streamed.shape[0], offline.shape[0]) - 200
        assert np.max(np.abs(streamed[100:n] - offline[100:n])) < 0.02


class TestTrim:
    def test_speech_span_finds_bounds(self) -> None:
        probs = np.array([0.1, 0.1, 0.9, 0.95, 0.8, 0.2, 0.1])
        span = speech_span(probs, WINDOW_S, 0.5)
        assert span == pytest.approx((2 * WINDOW_S, 5 * WINDOW_S), abs=TOLERANCE)

    def test_speech_span_none_without_speech(self) -> None:
        assert speech_span(np.array([0.1, 0.2]), WINDOW_S, 0.5) is None
        assert speech_span(np.zeros(0), WINDOW_S, 0.5) is None

    def test_trim_removes_padding_and_silence(self) -> None:
        sr = SAMPLE_RATE
        n_windows = int(3.0 / WINDOW_S)
        probs = np.zeros(n_windows)
        probs[int(0.5 / WINDOW_S) : int(2.0 / WINDOW_S)] = 0.9
        wave = make_tone(3.0)
        trimmed = trim_to_speech(wave, sr, probs, WINDOW_S, 0.5, pad_seconds=0.1)
        expected = 1.5 + 0.2
        assert trimmed.shape[0] == pytest.approx(expected * sr, abs=sr * WINDOW_S * 2)

    def test_trim_without_probabilities_returns_input(self) -> None:
        wave = make_tone(1.0)
        assert trim_to_speech(wave, SAMPLE_RATE, None, WINDOW_S, 0.5, 0.1) is wave

    def test_voiced_seconds_counts_windows(self) -> None:
        probs = np.array([0.9, 0.9, 0.1, 0.9])
        assert voiced_seconds(probs, WINDOW_S, 0.5) == pytest.approx(3 * WINDOW_S, abs=TOLERANCE)


class TestCropAndTile:
    def test_crop_keeps_tail(self) -> None:
        wave = np.arange(10 * SAMPLE_RATE, dtype=np.float32)
        cropped = crop_last_seconds(wave, SAMPLE_RATE, 8.0)
        assert cropped.shape[0] == 8 * SAMPLE_RATE
        assert cropped[-1] == wave[-1]

    def test_crop_noop_when_short(self) -> None:
        wave = make_tone(2.0)
        assert crop_last_seconds(wave, SAMPLE_RATE, 8.0) is wave

    def test_tile_pad_repeats_without_zeros(self) -> None:
        wave = make_tone(1.0)
        tiled = tile_pad(wave, 3 * SAMPLE_RATE)
        assert tiled.shape[0] == 3 * SAMPLE_RATE
        assert np.array_equal(tiled[:SAMPLE_RATE], wave)
        assert np.array_equal(tiled[SAMPLE_RATE : 2 * SAMPLE_RATE], wave)

    def test_tile_pad_truncates(self) -> None:
        wave = make_tone(4.0)
        assert tile_pad(wave, SAMPLE_RATE).shape[0] == SAMPLE_RATE

    def test_tile_pad_empty(self) -> None:
        assert tile_pad(np.zeros(0, dtype=np.float32), 100).shape[0] == 100

    def test_bucket_selection(self) -> None:
        buckets = [2.0, 4.0, 8.0]
        assert bucket_samples(int(1.5 * SAMPLE_RATE), SAMPLE_RATE, buckets) == 2 * SAMPLE_RATE
        assert bucket_samples(int(4.0 * SAMPLE_RATE), SAMPLE_RATE, buckets) == 4 * SAMPLE_RATE
        assert bucket_samples(int(20.0 * SAMPLE_RATE), SAMPLE_RATE, buckets) == 8 * SAMPLE_RATE

    def test_rms_dbfs(self) -> None:
        assert rms_dbfs(np.zeros(0)) == -120.0
        assert rms_dbfs(np.ones(100, dtype=np.float32)) == pytest.approx(0.0, abs=TOLERANCE)
