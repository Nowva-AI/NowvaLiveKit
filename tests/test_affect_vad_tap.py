"""Tests for TappedVAD: transparent event forwarding, audio tee, and the real Silero event contract."""

from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit import rtc  # noqa: E402
from livekit.agents import vad as agents_vad  # noqa: E402

from agent.services.affect_vad_tap import TappedVAD, TappedVADStream  # noqa: E402

CUE_WAV_DIR = Path(__file__).parent.parent / "src" / "assets" / "cues" / "wav"
FRAME_SAMPLES_24K = 240


class _RecordingSink:
    def __init__(self) -> None:
        self.frames: list[rtc.AudioFrame] = []
        self.starts: list[float] = []
        self.windows: list[agents_vad.VADEvent] = []
        self.ends: list[agents_vad.VADEvent] = []

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)

    def on_speech_start(self, timestamp: float) -> None:
        self.starts.append(timestamp)

    def on_speech_window(self, event: agents_vad.VADEvent) -> None:
        self.windows.append(event)

    def on_speech_end(self, event: agents_vad.VADEvent) -> None:
        self.ends.append(event)


class _ScriptedStream(agents_vad.VADStream):
    """Emits one scripted event per pushed frame so the tap's ordering can be checked exactly."""

    def __init__(self, vad: agents_vad.VAD, script: list[agents_vad.VADEvent]) -> None:
        self._script = list(script)
        self.pushed = 0
        self.flushed = 0
        super().__init__(vad)

    async def _main_task(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, agents_vad.VADStream._FlushSentinel):
                self.flushed += 1
                continue
            self.pushed += 1
            if self._script:
                self._event_ch.send_nowait(self._script.pop(0))


class _ScriptedVAD(agents_vad.VAD):
    def __init__(self, script: list[agents_vad.VADEvent]) -> None:
        super().__init__(capabilities=agents_vad.VADCapabilities(update_interval=0.032))
        self._script = script
        self.streams: list[_ScriptedStream] = []

    def stream(self) -> _ScriptedStream:
        stream = _ScriptedStream(self, self._script)
        self.streams.append(stream)
        return stream


def _frame(sample_rate: int = 24000, samples: int = FRAME_SAMPLES_24K, value: int = 1000) -> rtc.AudioFrame:
    data = np.full(samples, value, dtype=np.int16).tobytes()
    return rtc.AudioFrame(data=data, sample_rate=sample_rate, num_channels=1, samples_per_channel=samples)


def _event(kind: agents_vad.VADEventType, **kwargs) -> agents_vad.VADEvent:
    base = dict(type=kind, samples_index=0, timestamp=0.0, speech_duration=0.0, silence_duration=0.0)
    base.update(kwargs)
    return agents_vad.VADEvent(**base)


class TestScriptedTap:
    def test_events_forwarded_and_sink_notified(self) -> None:
        script = [
            _event(agents_vad.VADEventType.START_OF_SPEECH, timestamp=1.0, speaking=True),
            _event(agents_vad.VADEventType.INFERENCE_DONE, probability=0.9, speaking=True),
            _event(agents_vad.VADEventType.INFERENCE_DONE, probability=0.1, speaking=True, raw_accumulated_silence=0.3),
            _event(agents_vad.VADEventType.END_OF_SPEECH, speech_duration=1.2),
        ]
        sink = _RecordingSink()
        tapped = TappedVAD(_ScriptedVAD(script), sink)

        async def _run() -> list[agents_vad.VADEvent]:
            stream = tapped.stream()
            assert isinstance(stream, TappedVADStream)
            received: list[agents_vad.VADEvent] = []

            async def _consume() -> None:
                async for ev in stream:
                    received.append(ev)

            consumer = asyncio.create_task(_consume())
            for _ in range(4):
                stream.push_frame(_frame())
            stream.flush()
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.wait_for(consumer, timeout=2.0)
            await stream.aclose()
            return received

        received = asyncio.run(_run())
        assert [ev.type for ev in received] == [ev.type for ev in script]
        assert len(sink.frames) == 4
        assert sink.starts == [1.0]
        assert len(sink.windows) == 2 and sink.windows[1].raw_accumulated_silence == pytest.approx(0.3)
        assert len(sink.ends) == 1 and sink.ends[0].speech_duration == pytest.approx(1.2)
        # The explicit flush is forwarded; end_input() adds its own flush sentinels on both layers.
        assert tapped.inner.streams[0].flushed >= 1

    def test_sink_failure_does_not_break_forwarding(self) -> None:
        class _BrokenSink(_RecordingSink):
            def push_audio(self, frame: rtc.AudioFrame) -> None:
                raise RuntimeError("boom")

        script = [_event(agents_vad.VADEventType.INFERENCE_DONE, probability=0.2)]
        tapped = TappedVAD(_ScriptedVAD(script), _BrokenSink())

        async def _run() -> int:
            stream = tapped.stream()
            stream.push_frame(_frame())
            await asyncio.sleep(0.02)
            stream.end_input()
            count = 0
            async for _ in stream:
                count += 1
            await stream.aclose()
            return count

        assert asyncio.run(_run()) == 1

    def test_metrics_still_emitted(self) -> None:
        script = [_event(agents_vad.VADEventType.INFERENCE_DONE, probability=0.2, inference_duration=0.001) for _ in range(40)]
        sink = _RecordingSink()
        tapped = TappedVAD(_ScriptedVAD(script), sink)
        collected: list = []
        tapped.on("metrics_collected", collected.append)

        async def _run() -> None:
            stream = tapped.stream()
            for _ in range(40):
                stream.push_frame(_frame())
            await asyncio.sleep(0.05)
            stream.end_input()
            async for _ in stream:
                pass
            await asyncio.sleep(0.02)
            await stream.aclose()

        asyncio.run(_run())
        assert collected, "TappedVAD should re-emit metrics_collected from forwarded events"

    def test_capabilities_and_labels_delegate(self) -> None:
        inner = _ScriptedVAD([])
        tapped = TappedVAD(inner, _RecordingSink())
        assert tapped.capabilities.update_interval == pytest.approx(0.032)
        assert tapped.model == inner.model and tapped.provider == inner.provider


def _load_cue_wav() -> tuple[np.ndarray, int] | None:
    if not CUE_WAV_DIR.exists():
        return None
    for path in sorted(CUE_WAV_DIR.glob("*.wav"))[:1]:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        return pcm, rate
    return None


class TestRealSilero:
    def test_silero_events_through_tap(self) -> None:
        silero = pytest.importorskip("livekit.plugins.silero")
        loaded = _load_cue_wav()
        if loaded is None:
            pytest.skip("no cached cue WAV available for a real speech sample")
        pcm, rate = loaded
        gap = np.zeros(rate, dtype=np.int16)
        signal = np.concatenate([pcm, gap, pcm, np.zeros(rate, dtype=np.int16)])
        sink = _RecordingSink()

        async def _run() -> list[agents_vad.VADEvent]:
            inner = silero.VAD.load()
            tapped = TappedVAD(inner, sink)
            stream = tapped.stream()
            received: list[agents_vad.VADEvent] = []

            async def _consume() -> None:
                async for ev in stream:
                    received.append(ev)

            consumer = asyncio.create_task(_consume())
            frame_len = rate // 100
            for start in range(0, len(signal) - frame_len, frame_len):
                chunk = signal[start : start + frame_len]
                stream.push_frame(rtc.AudioFrame(data=chunk.tobytes(), sample_rate=rate, num_channels=1, samples_per_channel=frame_len))
                await asyncio.sleep(0)
            await asyncio.sleep(0.5)
            stream.end_input()
            await asyncio.wait_for(consumer, timeout=30.0)
            await stream.aclose()
            return received

        received = asyncio.run(_run())
        types = [ev.type for ev in received]
        assert types.count(agents_vad.VADEventType.START_OF_SPEECH) >= 2
        assert types.count(agents_vad.VADEventType.END_OF_SPEECH) >= 2
        assert len(sink.starts) == types.count(agents_vad.VADEventType.START_OF_SPEECH)
        end = sink.ends[0]
        assert end.frames and end.frames[0].sample_rate == rate
        assert end.frames[0].duration >= end.speech_duration
