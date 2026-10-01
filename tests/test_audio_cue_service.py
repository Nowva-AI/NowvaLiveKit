"""Tests for AudioCueService — on-disk cue clips and the runtime TTS fallback."""

import asyncio
import sys
import time
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent.services.audio_cue_service as audio_cue_module
from agent.services.audio_cue_service import (
    CUE_TEXT_MAP,
    SAMPLE_RATE,
    AudioCueService,
)
from agent.services.coaching_constants import CUE_DISPLAY_LABELS
from biomechanics.coaching.cue_cache import SQUAT_CUES

# Sample cue dict as sent by the biomechanics IPC bridge
SAMPLE_SQUAT_CUES = {
    "knees_out": "knees_out",
    "knees_out_left": "knees_out_left",
    "chest_up": "chest_up",
    "deeper": "deeper",
    "good_rep": "good_rep",
    "rep_1": "rep_1",
    "rep_2": "rep_2",
}
SAMPLE_TTS_KEYS = {key for key in SAMPLE_SQUAT_CUES if not key.startswith("rep_")}

FAKE_PCM = b"\x00\x01" * 1200  # 2400 bytes of fake 16-bit PCM
JARGON_WORDS = ["valgus", "eccentric", "concentric", "dorsiflexion"]


def _make_speech_response(audio_bytes: bytes = FAKE_PCM) -> MagicMock:
    """Create a mock response for client.audio.speech.create()."""
    response = MagicMock()
    response.read.return_value = audio_bytes
    return response


def _write_cue_wav(path: Path, sample_count: int = 2400) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"\x00\x00" * sample_count)


@pytest.fixture
def cue_wav_dir(tmp_path, monkeypatch) -> Path:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    monkeypatch.setattr(audio_cue_module, "CUES_WAV_DIR", wav_dir)
    monkeypatch.setattr(audio_cue_module, "REP_SOUND_PATH", tmp_path / "no_rep_sound.wav")
    return wav_dir


@pytest.fixture
def service(cue_wav_dir) -> AudioCueService:
    return AudioCueService(session=None)


class TestCacheCuesFallback:
    def test_generates_tts_for_cues_missing_on_disk(self, service):
        mock_client = AsyncMock()
        mock_client.audio.speech.create = AsyncMock(return_value=_make_speech_response())
        service._client = mock_client

        asyncio.run(service.cache_cues(SAMPLE_SQUAT_CUES))

        assert set(service._fallback_cache) == SAMPLE_TTS_KEYS
        assert all(service.has_cue(key) for key in SAMPLE_TTS_KEYS)

    def test_skips_keys_without_text(self, service):
        mock_client = AsyncMock()
        mock_client.audio.speech.create = AsyncMock(return_value=_make_speech_response())
        service._client = mock_client

        asyncio.run(service.cache_cues({"knees_out": "knees_out", "mystery_cue": "mystery_cue"}))

        assert set(service._fallback_cache) == {"knees_out"}

    def test_skips_cues_already_on_disk(self, cue_wav_dir):
        _write_cue_wav(cue_wav_dir / "knees_out_0.wav")
        service = AudioCueService(session=None)
        mock_client = AsyncMock()
        mock_client.audio.speech.create = AsyncMock(return_value=_make_speech_response())
        service._client = mock_client

        asyncio.run(service.cache_cues({"knees_out": "knees_out"}))

        mock_client.audio.speech.create.assert_not_awaited()
        assert service.has_cue("knees_out")

    def test_generates_concurrently(self, service):
        async def _slow_create(**kwargs):
            await asyncio.sleep(0.05)
            return _make_speech_response()

        mock_client = AsyncMock()
        mock_client.audio.speech.create = _slow_create
        service._client = mock_client

        started = time.monotonic()
        asyncio.run(service.cache_cues(SAMPLE_SQUAT_CUES))
        assert time.monotonic() - started < 0.2

    def test_partial_failure_keeps_the_rest(self, service):
        call_count = 0

        async def _flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Simulated TTS API error")
            return _make_speech_response()

        mock_client = AsyncMock()
        mock_client.audio.speech.create = _flaky_create
        service._client = mock_client

        asyncio.run(service.cache_cues(SAMPLE_SQUAT_CUES))

        assert 0 < len(service._fallback_cache) < len(SAMPLE_TTS_KEYS)


class TestDiskIndex:
    def test_side_variant_clips_index_under_their_own_key(self, cue_wav_dir):
        _write_cue_wav(cue_wav_dir / "knees_out_0.wav")
        _write_cue_wav(cue_wav_dir / "knees_out_left_0.wav")
        _write_cue_wav(cue_wav_dir / "knees_out_left_1.wav")
        service = AudioCueService(session=None)

        assert len(service._memory_cache["knees_out"]) == 1
        assert len(service._memory_cache["knees_out_left"]) == 2
        assert not service.has_cue("knees_out_right")

    def test_files_without_a_variant_number_are_ignored(self, cue_wav_dir):
        _write_cue_wav(cue_wav_dir / "review.wav")
        _write_cue_wav(cue_wav_dir / "knees_out_final.wav")
        service = AudioCueService(session=None)

        assert service._disk_cache == {}
        assert not service.is_cache_valid()


class TestCueText:
    def test_get_cue_text_returns_text_for_known_key(self):
        assert AudioCueService.get_cue_text("knees_out") == "Knees out!"
        assert AudioCueService.get_cue_text("rep_5") == "Five!"
        assert AudioCueService.get_cue_text("good_rep") == "Good rep!"

    def test_get_cue_text_returns_none_for_unknown_key(self):
        assert AudioCueService.get_cue_text("nonexistent") is None

    def test_every_squat_cue_has_text(self):
        missing = [key for key in SQUAT_CUES if key not in CUE_TEXT_MAP]
        assert missing == []

    def test_every_squat_correction_has_a_report_label(self):
        missing = [
            key for key in SQUAT_CUES
            if not key.startswith("rep_") and key not in CUE_DISPLAY_LABELS
        ]
        assert missing == []

    def test_cue_text_has_no_jargon(self):
        for key, text in CUE_TEXT_MAP.items():
            assert not any(word in text.lower() for word in JARGON_WORDS), key

    def test_dead_cue_keys_have_no_text(self):
        for key in ("hips_through", "flat_back"):
            assert key not in CUE_TEXT_MAP
            assert key not in CUE_DISPLAY_LABELS
