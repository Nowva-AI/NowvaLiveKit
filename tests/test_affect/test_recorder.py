"""Tests for the opt-in utterance recorder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from affect.config import RecorderConfig
from affect.recorder import META_FILENAME, UtteranceRecorder, read_recording

from .conftest import SAMPLE_RATE, make_tone


class TestRecorder:
    def test_disabled_writes_nothing(self, tmp_path: Path) -> None:
        recorder = UtteranceRecorder(RecorderConfig(enabled=False), tmp_path, "u1", "s1")
        assert recorder.record(make_tone(1.0), SAMPLE_RATE, {"mode": "main_menu"}) is None
        assert not (tmp_path / "u1").exists()

    def test_round_trip_and_metadata(self, tmp_path: Path) -> None:
        recorder = UtteranceRecorder(RecorderConfig(enabled=True), tmp_path, "u1", "s1")
        wave = make_tone(1.0)
        path = recorder.record(wave, SAMPLE_RATE, {"mode": "workout", "arousal": np.float32(0.7), "z": np.array([1.0, 2.0])})
        assert path is not None and path.exists()
        loaded, rate = read_recording(path)
        assert rate == SAMPLE_RATE
        assert loaded.shape == wave.shape
        assert np.max(np.abs(loaded - wave)) < 1e-3
        rows = [json.loads(line) for line in (recorder.directory / META_FILENAME).read_text().splitlines()]
        assert rows[0]["file"] == path.name
        assert rows[0]["mode"] == "workout"
        assert rows[0]["z"] == [1.0, 2.0]

    def test_anonymous_user(self, tmp_path: Path) -> None:
        recorder = UtteranceRecorder(RecorderConfig(enabled=True), tmp_path, None, "s2")
        recorder.record(make_tone(0.5), SAMPLE_RATE, {})
        assert (tmp_path / "anonymous" / "s2").exists()
