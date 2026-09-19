"""Tests for AffectService: early trigger, cancel-on-resume, reuse at end of speech, snapshot budget, hooks."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit import rtc  # noqa: E402
from livekit.agents import vad as agents_vad  # noqa: E402

from affect.config import AffectConfig, EngineConfig  # noqa: E402
from affect.engine import AffectEngine  # noqa: E402
from affect.manifest import ModelManifest  # noqa: E402
from agent.services.affect_service import AffectService  # noqa: E402

INPUT_RATE = 24000
FRAME_SAMPLES = 240
WINDOW_S = 0.032
TOLERANCE = 1e-6


class _SlowFakeSession:
    def __init__(self, delay_s: float = 0.0, arousal: float = 0.5, valence: float = 0.5) -> None:
        self.delay_s = delay_s
        self.arousal = arousal
        self.valence = valence
        self.calls = 0

    def get_inputs(self):
        return [type("IO", (), {"name": "waveform"})()]

    def get_outputs(self):
        return [type("IO", (), {"name": n})() for n in ("embedding", "avd")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feeds):
        import time

        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return [np.zeros((1, 4), dtype=np.float32), np.array([[self.arousal, 0.5, self.valence]], dtype=np.float32)]


class _State:
    def __init__(self, mode: str = "main_menu", user_id: str | None = "u1") -> None:
        self.mode = mode
        self.user_id = user_id

    def get_mode(self) -> str:
        return self.mode

    def get_user(self) -> dict:
        return {"id": self.user_id}


def _service(tmp_path: Path, session=None, mode: str = "main_menu", **config_overrides) -> AffectService:
    config = AffectConfig(**config_overrides)
    config.baseline.profile_dir = str(tmp_path / "profiles")
    config.recorder.output_dir = str(tmp_path / "rec")
    config.engine.warmup = False
    manifest = ModelManifest(version="t", family="fake", embedding_dim=4)
    engine = AffectEngine(tmp_path, EngineConfig(warmup=False), manifest=manifest, session=session or _SlowFakeSession())
    engine.initialize()
    return AffectService(config, engine, state=_State(mode=mode), session_id="s1")


def _frame(value: int = 3000) -> rtc.AudioFrame:
    data = (np.full(FRAME_SAMPLES, value, dtype=np.int16) * np.sign(np.sin(np.arange(FRAME_SAMPLES) * 0.3))).astype(np.int16)
    return rtc.AudioFrame(data=data.tobytes(), sample_rate=INPUT_RATE, num_channels=1, samples_per_channel=FRAME_SAMPLES)


def _window(prob: float, silence: float = 0.0, speaking: bool = True) -> agents_vad.VADEvent:
    frame = rtc.AudioFrame(data=bytes(2 * 768), sample_rate=INPUT_RATE, num_channels=1, samples_per_channel=768)
    return agents_vad.VADEvent(
        type=agents_vad.VADEventType.INFERENCE_DONE, samples_index=0, timestamp=0.0, speech_duration=0.0,
        silence_duration=0.0, probability=prob, speaking=speaking, raw_accumulated_silence=silence, frames=[frame],
    )


def _end(speech_duration: float = 2.0) -> agents_vad.VADEvent:
    return agents_vad.VADEvent(
        type=agents_vad.VADEventType.END_OF_SPEECH, samples_index=0, timestamp=0.0,
        speech_duration=speech_duration, silence_duration=0.55,
    )


async def _speak(service: AffectService, voiced_windows: int, frames_per_window: int = 3) -> None:
    service.on_speech_start(0.0)
    for _ in range(voiced_windows):
        for _ in range(frames_per_window):
            service.push_audio(_frame())
        service.on_speech_window(_window(0.95))


class TestTriggering:
    def test_early_trigger_then_reuse_at_end(self, tmp_path: Path) -> None:
        session = _SlowFakeSession()
        service = _service(tmp_path, session)

        async def _run() -> None:
            await _speak(service, voiced_windows=40)
            for silence in (0.1, 0.2):
                service.push_audio(_frame(0))
                service.on_speech_window(_window(0.05, silence=silence))
            assert service._early_task is None
            service.push_audio(_frame(0))
            service.on_speech_window(_window(0.05, silence=0.26))
            assert service._early_task is not None
            service.on_speech_end(_end())
            await asyncio.sleep(0.2)

        asyncio.run(_run())
        assert session.calls == 1
        assert service.snapshot_dict()["stats"]["early_reused"] == 1
        assert service.snapshot_dict()["stats"]["final_runs"] == 0

    def test_resumed_speech_cancels_early(self, tmp_path: Path) -> None:
        session = _SlowFakeSession(delay_s=0.3)
        service = _service(tmp_path, session)

        async def _run() -> None:
            await _speak(service, voiced_windows=40)
            service.on_speech_window(_window(0.05, silence=0.3))
            early = service._early_task
            assert early is not None
            service.push_audio(_frame())
            service.on_speech_window(_window(0.95, silence=0.0))
            await asyncio.sleep(0)
            assert early.cancelled() or early.done()
            assert service._early_task is None
            for _ in range(10):
                service.push_audio(_frame())
                service.on_speech_window(_window(0.95))
            service.on_speech_end(_end())
            await asyncio.sleep(0.6)

        asyncio.run(_run())
        assert service.snapshot_dict()["stats"]["final_runs"] == 1
        assert service.snapshot_dict()["stats"]["early_reused"] == 0

    def test_short_utterance_skipped(self, tmp_path: Path) -> None:
        session = _SlowFakeSession()
        service = _service(tmp_path, session)

        async def _run() -> None:
            await _speak(service, voiced_windows=10)
            service.on_speech_end(_end(0.3))
            await asyncio.sleep(0.1)

        asyncio.run(_run())
        assert session.calls == 0
        assert service.snapshot_dict()["stats"]["skipped_short"] == 1


class TestSnapshot:
    def test_snapshot_respects_budget(self, tmp_path: Path) -> None:
        service = _service(tmp_path, _SlowFakeSession(delay_s=0.5))

        async def _run() -> tuple[float, bool]:
            await _speak(service, voiced_windows=40)
            service.on_speech_end(_end())
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            state = await service.snapshot(max_wait_s=0.04)
            elapsed = loop.time() - t0
            await asyncio.sleep(0.7)
            return elapsed, state.fresh

        elapsed, fresh = asyncio.run(_run())
        assert elapsed < 0.2
        assert fresh is False

    def test_snapshot_fresh_only_once(self, tmp_path: Path) -> None:
        service = _service(tmp_path, _SlowFakeSession())

        async def _run() -> tuple[bool, bool]:
            await _speak(service, voiced_windows=40)
            service.on_speech_end(_end())
            await asyncio.sleep(0.2)
            first = (await service.snapshot(0.0)).fresh
            second = (await service.snapshot(0.0)).fresh
            return first, second

        assert asyncio.run(_run()) == (True, False)


class TestBaselineAndHooks:
    def test_neutral_context_enrolls_and_workout_does_not(self, tmp_path: Path) -> None:
        neutral = _service(tmp_path, mode="main_menu")
        workout = _service(tmp_path, mode="workout")

        async def _run(service: AffectService) -> int:
            await _speak(service, voiced_windows=40)
            service.on_speech_end(_end())
            await asyncio.sleep(0.2)
            return service.baseline.sample_count

        assert asyncio.run(_run(neutral)) == 1
        assert asyncio.run(_run(workout)) == 0

    def test_not_confident_before_enrollment_floor(self, tmp_path: Path) -> None:
        service = _service(tmp_path)

        async def _run() -> bool:
            await _speak(service, voiced_windows=40)
            service.on_speech_end(_end())
            await asyncio.sleep(0.2)
            return (await service.snapshot(0.0)).confident

        assert asyncio.run(_run()) is False
        assert service.current_style().is_neutral()

    def test_rep_effort_and_set_reset(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        service.on_rep_effort(1.0)
        state = service.on_rep_effort(1.5)
        assert state.effort == "near_limit"
        service.on_set_reset()
        assert service.last_state.effort == "fresh"

    def test_stop_saves_profile(self, tmp_path: Path) -> None:
        service = _service(tmp_path)

        async def _run() -> None:
            await _speak(service, voiced_windows=40)
            service.on_speech_end(_end())
            await asyncio.sleep(0.2)
            await service.stop()

        asyncio.run(_run())
        assert (tmp_path / "profiles" / "u1.npz").exists()

    def test_disabled_without_engine(self) -> None:
        service = AffectService(AffectConfig(), engine=None)
        assert service.enabled is False
        service.push_audio(_frame())
        service.on_speech_start(0.0)
        assert service.last_state.affect == "flat"


class TestDescribeForLLM:
    def test_no_reading_yet(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        assert "No voice reading yet" in service.describe_for_llm()

    def test_disabled(self) -> None:
        assert "off" in AffectService(AffectConfig(), engine=None).describe_for_llm()

    def test_after_reading_mentions_dimensions_and_confidence(self, tmp_path: Path) -> None:
        service = _service(tmp_path, _SlowFakeSession(arousal=0.2, valence=0.8))

        async def _run() -> str:
            await _speak(service, voiced_windows=40)
            service.on_speech_end(_end())
            await asyncio.sleep(0.2)
            return service.describe_for_llm()

        text = asyncio.run(_run())
        assert "Arousal (energy) low" in text
        assert "Valence (positivity) high" in text
        assert "Confidence: low" in text
        assert "seconds ago" in text
