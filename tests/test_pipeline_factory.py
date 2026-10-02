"""Tests that the factory defaults are the tuned voice stack the product ships, with env overrides."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core import pipeline_factory

TUNED_ENV_VARS = (
    "TTS_BACKEND", "TTS_MODEL", "TTS_VOICE_ID", "ELEVENLABS_VOICE_ID", "STT_BACKEND",
    "STT_MODEL", "STT_BASE_URL", "VAD_BACKEND", "VAD_MIN_SILENCE", "ENDPOINTING_MIN_DELAY",
    "ENDPOINTING_MAX_DELAY", "REPLY_START_VAD_WINDOW", "TURN_DETECTOR",
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(**kwargs)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in TUNED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TURN_DETECTOR", "vad")
    return monkeypatch


class TestTunedDefaults:
    def test_tts_defaults_to_elevenlabs_flash(self, clean_env: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        clean_env.setattr(pipeline_factory, "elevenlabs", SimpleNamespace(TTS=recorder))
        pipeline_factory.build_tts()
        assert recorder.calls == [{
            "model": "eleven_flash_v2_5",
            "voice_id": pipeline_factory.DEFAULT_ELEVENLABS_VOICE_ID,
        }]

    def test_stt_defaults_to_deepgram_eu(self, clean_env: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        clean_env.setattr(pipeline_factory, "deepgram", SimpleNamespace(STT=recorder))
        pipeline_factory.build_stt()
        assert recorder.calls[0]["base_url"] == pipeline_factory.DEEPGRAM_EU_URL

    def test_vad_defaults_to_tuned_silence(self, clean_env: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        clean_env.setattr(pipeline_factory, "silero", SimpleNamespace(VAD=SimpleNamespace(load=recorder)))
        pipeline_factory.build_vad()
        assert recorder.calls[0]["min_silence_duration"] == pytest.approx(0.35)

    def test_turn_handling_defaults(self, clean_env: pytest.MonkeyPatch) -> None:
        options = pipeline_factory.build_turn_handling()
        assert options["endpointing"]["min_delay"] == pytest.approx(0.2)
        assert options["interruption"]["backchannel_boundary"] == (0.0, 1.0)


class TestEnvOverrides:
    def test_tts_backend_env_still_selects_cartesia(self, clean_env: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        clean_env.setattr(pipeline_factory, "inference", SimpleNamespace(TTS=recorder))
        clean_env.setenv("TTS_BACKEND", "inference")
        pipeline_factory.build_tts()
        assert recorder.calls[0]["model"] == "cartesia/sonic-3"

    def test_elevenlabs_voice_env_overrides_default(self, clean_env: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        clean_env.setattr(pipeline_factory, "elevenlabs", SimpleNamespace(TTS=recorder))
        clean_env.setenv("ELEVENLABS_VOICE_ID", "voice-from-env")
        pipeline_factory.build_tts()
        assert recorder.calls[0]["voice_id"] == "voice-from-env"

    def test_stt_base_url_env_overrides_default(self, clean_env: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        clean_env.setattr(pipeline_factory, "deepgram", SimpleNamespace(STT=recorder))
        clean_env.setenv("STT_BASE_URL", pipeline_factory.DEEPGRAM_US_URL)
        pipeline_factory.build_stt()
        assert recorder.calls[0]["base_url"] == pipeline_factory.DEEPGRAM_US_URL

    def test_reply_start_window_env_restores_livekit_default(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("REPLY_START_VAD_WINDOW", "1.0")
        options = pipeline_factory.build_turn_handling()
        assert options["interruption"]["backchannel_boundary"] == (1.0, 1.0)

    def test_endpointing_env_overrides_default(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("ENDPOINTING_MIN_DELAY", "0.5")
        options = pipeline_factory.build_turn_handling()
        assert options["endpointing"]["min_delay"] == pytest.approx(0.5)
