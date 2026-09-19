"""TTS style adapters: realize a VoiceStyle on Cartesia sonic-3 (inline tags or per-stream options) or Qwen3-TTS."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, Protocol

from affect.config import StyleConfig
from affect.voice_style import CartesiaControls, VoiceStyle, qwen3_instruct, to_cartesia

logger = logging.getLogger(__name__)

# LLM-written style tags are stripped; <break .../> and [laughter] are deliberate and pass through.
STYLE_TAG_RE = re.compile(r"<\s*/?\s*(?:emotion|speed|volume)\b[^>]*>", re.IGNORECASE)
STYLE_KEYS = ("emotion", "speed", "volume")


def strip_style_tags(text: str) -> str:
    return STYLE_TAG_RE.sub("", text)


def cartesia_inline_tags(controls: CartesiaControls) -> str:
    parts: list[str] = []
    if controls.emotion:
        parts.append(f'<emotion value="{controls.emotion.lower()}"/>')
    if abs(controls.speed - 1.0) > 1e-6:
        parts.append(f'<speed ratio="{controls.speed:g}"/>')
    if abs(controls.volume - 1.0) > 1e-6:
        parts.append(f'<volume ratio="{controls.volume:g}"/>')
    return "".join(parts)


class TTSStyleAdapter(Protocol):
    name: str

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None: ...

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]: ...


async def _strip_stream(text: AsyncIterable[str]) -> AsyncIterator[str]:
    async for chunk in text:
        yield strip_style_tags(chunk)


class NullAdapter:
    name = "none"

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        return None

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        return _strip_stream(text)


class CartesiaInlineTagAdapter:
    """Prepends sonic-3 SSML tags as one chunk so the style belongs to this generation only."""

    name = "cartesia_inline"

    def __init__(self, config: StyleConfig) -> None:
        self.config = config

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        return None

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        prefix = cartesia_inline_tags(to_cartesia(style, self.config))
        return self._wrap(text, prefix)

    @staticmethod
    async def _wrap(text: AsyncIterable[str], prefix: str) -> AsyncIterator[str]:
        first = True
        async for chunk in text:
            cleaned = strip_style_tags(chunk)
            if first and prefix:
                yield prefix + cleaned
                first = False
                continue
            first = False
            yield cleaned


class CartesiaExtraKwargsAdapter:
    """Sets per-stream generation options by swapping in a NEW extra_kwargs dict; never mutates the shared one."""

    name = "cartesia_extra"

    def __init__(self, config: StyleConfig) -> None:
        self.config = config

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        opts = getattr(tts, "_opts", None)
        if opts is None or not hasattr(opts, "extra_kwargs"):
            return None
        controls = to_cartesia(style, self.config)
        base = {k: v for k, v in dict(opts.extra_kwargs or {}).items() if k not in STYLE_KEYS}
        if controls.emotion:
            base["emotion"] = controls.emotion.lower()
        base["speed"] = controls.speed
        base["volume"] = controls.volume
        opts.extra_kwargs = base

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        return _strip_stream(text)


class Qwen3InstructAdapter:
    """Interface for the local Qwen3-TTS backend: the instruct string is exposed for the future TTS plugin."""

    name = "qwen3"

    def __init__(self, config: StyleConfig) -> None:
        self.config = config
        self.last_instruct: str | None = None

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        self.last_instruct = qwen3_instruct(style)
        setter = getattr(tts, "set_instruct", None)
        if callable(setter):
            setter(self.last_instruct)

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        return _strip_stream(text)


def build_adapter(config: StyleConfig) -> TTSStyleAdapter:
    if config.adapter == "cartesia_inline":
        return CartesiaInlineTagAdapter(config)
    if config.adapter == "cartesia_extra":
        return CartesiaExtraKwargsAdapter(config)
    if config.adapter == "qwen3":
        return Qwen3InstructAdapter(config)
    return NullAdapter()
