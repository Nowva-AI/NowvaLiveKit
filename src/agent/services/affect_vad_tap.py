"""TappedVAD: a transparent proxy over the session VAD that also feeds audio and speech events to the affect service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Protocol

from livekit import rtc
from livekit.agents import vad as agents_vad

logger = logging.getLogger(__name__)


class UtteranceSink(Protocol):
    """What the affect service exposes to the tap. All calls happen on the event loop."""

    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    def on_speech_start(self, timestamp: float) -> None: ...

    def on_speech_window(self, event: agents_vad.VADEvent) -> None: ...

    def on_speech_end(self, event: agents_vad.VADEvent) -> None: ...


class TappedVAD(agents_vad.VAD):
    """Wraps another VAD (Silero) so the session sees identical events while the sink sees the audio."""

    def __init__(self, inner: agents_vad.VAD, sink: UtteranceSink) -> None:
        super().__init__(capabilities=inner.capabilities)
        self._inner = inner
        self._sink = sink
        self._label = getattr(inner, "_label", self._label)

    @property
    def inner(self) -> agents_vad.VAD:
        return self._inner

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    def stream(self) -> TappedVADStream:
        return TappedVADStream(self, self._inner.stream(), self._sink)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this proxy does not define (e.g. Silero-specific helpers).
        return getattr(self._inner, name)


class TappedVADStream(agents_vad.VADStream):
    def __init__(self, vad: TappedVAD, inner: agents_vad.VADStream, sink: UtteranceSink) -> None:
        self._inner = inner
        self._sink = sink
        super().__init__(vad)

    def update_options(self, **kwargs: Any) -> None:
        update = getattr(self._inner, "update_options", None)
        if callable(update):
            update(**kwargs)

    async def _forward_input(self) -> None:
        try:
            async for item in self._input_ch:
                if isinstance(item, agents_vad.VADStream._FlushSentinel):
                    self._inner.flush()
                    continue
                try:
                    self._sink.push_audio(item)
                except Exception:  # noqa: BLE001 — the sink must never break the session VAD
                    logger.exception("[AFFECT] sink.push_audio failed")
                self._inner.push_frame(item)
        finally:
            with contextlib.suppress(Exception):
                self._inner.end_input()

    def _observe(self, event: agents_vad.VADEvent) -> None:
        try:
            if event.type == agents_vad.VADEventType.START_OF_SPEECH:
                self._sink.on_speech_start(event.timestamp)
            elif event.type == agents_vad.VADEventType.INFERENCE_DONE:
                self._sink.on_speech_window(event)
            elif event.type == agents_vad.VADEventType.END_OF_SPEECH:
                self._sink.on_speech_end(event)
        except Exception:  # noqa: BLE001
            logger.exception("[AFFECT] sink event handler failed")

    async def _main_task(self) -> None:
        forward = asyncio.create_task(self._forward_input(), name="TappedVADStream._forward_input")
        try:
            async for event in self._inner:
                self._observe(event)
                self._event_ch.send_nowait(event)
        finally:
            forward.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await forward
            with contextlib.suppress(Exception):
                await self._inner.aclose()
