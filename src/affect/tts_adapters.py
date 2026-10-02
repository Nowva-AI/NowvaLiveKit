"""TTS style adapters: realize a VoiceStyle on Cartesia sonic-3, ElevenLabs or Qwen3-TTS.

Each engine takes style a different way and speaks another engine's markup aloud,
so the "auto" adapter resolves from the TTS actually in use.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, Protocol

from affect.config import StyleConfig
from affect.voice_style import (
    ELEVENLABS_DEFAULT_SPEED,
    ELEVENLABS_DEFAULT_STABILITY,
    CartesiaControls,
    VoiceStyle,
    qwen3_instruct,
    to_cartesia,
    to_elevenlabs,
)

logger = logging.getLogger(__name__)

# LLM-written style tags are stripped; <break .../> and [laughter] are deliberate and pass through.
STYLE_TAG_RE = re.compile(r"<\s*/?\s*(?:emotion|speed|volume)\b[^>]*>", re.IGNORECASE)
STYLE_KEYS = ("emotion", "speed", "volume")

# sonic-3 performs [laughter]; every other engine here reads the word out.
AUDIO_TAG_RE = re.compile(r"\[\s*(?:laughter|laughs|sigh|breath)\s*\]\s?", re.IGNORECASE)
ELEVENLABS_DEFAULT_SIMILARITY = 0.75


def strip_style_tags(text: str) -> str:
    return STYLE_TAG_RE.sub("", text)


def tts_parses_cartesia_tags(tts: Any) -> bool:
    """Whether this TTS consumes sonic-3 inline tags instead of speaking them.

    Covers both the direct Cartesia plugin and Cartesia served over LiveKit
    Inference, whose model id is "cartesia/<model>".
    """
    if "cartesia" in type(tts).__module__:
        return True
    model = getattr(getattr(tts, "_opts", None), "model", None)
    return str(model or "").lower().startswith("cartesia")


def tts_is_elevenlabs(tts: Any) -> bool:
    return "elevenlabs" in type(tts).__module__


def strip_spoken_markup(text: str) -> str:
    return AUDIO_TAG_RE.sub("", strip_style_tags(text))


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


async def _strip_markup_stream(text: AsyncIterable[str]) -> AsyncIterator[str]:
    async for chunk in text:
        cleaned = strip_spoken_markup(chunk)
        if cleaned:
            yield cleaned


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
        # None until a TTS has been seen; unknown keeps the tags so a session
        # that never exposes its TTS still styles Cartesia as before.
        self._tags_understood: bool | None = None
        self._warned = False

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        understood = tts_parses_cartesia_tags(tts)
        self._tags_understood = understood
        if not understood and not self._warned:
            self._warned = True
            logger.warning(
                "[AFFECT] %s is configured but the TTS is %s, which speaks sonic-3 "
                "tags aloud instead of parsing them. Dropping inline style tags. "
                "Set AFFECT_STYLE_ADAPTER=none to silence this, or use a Cartesia TTS.",
                self.name,
                type(tts).__name__,
            )

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        if self._tags_understood is False:
            return _strip_stream(text)
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


class ElevenLabsSettingsAdapter:
    """Sets per-utterance voice settings by swapping in a NEW object on the shared options.

    ElevenLabs sends voice_settings when it opens each synthesis context, reading them
    from the options object the live websocket already holds. Assigning here restyles
    the next utterance on the warm connection; TTS.update_options() would instead drop
    the socket and put a fresh handshake in front of every reply.
    """

    name = "elevenlabs_settings"

    def __init__(self, config: StyleConfig) -> None:
        self.config = config
        self._baselines: dict[int, Any] = {}

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        opts = getattr(tts, "_opts", None)
        if opts is None or not hasattr(opts, "voice_settings") or not tts_is_elevenlabs(tts):
            return None
        baseline = self._baselines.setdefault(id(tts), opts.voice_settings)
        if style.is_neutral():
            # Hand back exactly what was configured; when that is "not given" the
            # voice falls back to the settings stored with it on ElevenLabs.
            opts.voice_settings = baseline
            return None

        configured = baseline if dataclasses.is_dataclass(baseline) else None
        controls = to_elevenlabs(
            style,
            self.config,
            base_stability=_given_float(configured, "stability", ELEVENLABS_DEFAULT_STABILITY),
            base_speed=_given_float(configured, "speed", ELEVENLABS_DEFAULT_SPEED),
        )
        if configured is not None:
            opts.voice_settings = dataclasses.replace(
                configured, stability=controls.stability, speed=controls.speed
            )
            return None

        # The plugin module is already loaded whenever one of its TTS objects exists,
        # so this import never triggers LiveKit's main-thread-only plugin registration.
        from livekit.plugins.elevenlabs import VoiceSettings

        opts.voice_settings = VoiceSettings(
            stability=controls.stability,
            similarity_boost=ELEVENLABS_DEFAULT_SIMILARITY,
            speed=controls.speed,
        )

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        return _strip_markup_stream(text)


def _given_float(settings: Any, field: str, default: float) -> float:
    value = getattr(settings, field, None) if settings is not None else None
    return float(value) if isinstance(value, (int, float)) else default


class AutoAdapter:
    """Resolves to the adapter that fits the TTS in use, re-checking if the TTS changes."""

    name = "auto"

    def __init__(self, config: StyleConfig) -> None:
        self._cartesia = CartesiaInlineTagAdapter(config)
        self._elevenlabs = ElevenLabsSettingsAdapter(config)
        self._fallback = MarkupFreeAdapter()
        self._active: TTSStyleAdapter = self._fallback

    @property
    def active(self) -> TTSStyleAdapter:
        return self._active

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        if tts_parses_cartesia_tags(tts):
            resolved: TTSStyleAdapter = self._cartesia
        elif tts_is_elevenlabs(tts):
            resolved = self._elevenlabs
        else:
            resolved = self._fallback
        if resolved is not self._active:
            logger.info("[AFFECT] voice style adapter: %s (tts=%s)", resolved.name, type(tts).__module__)
            self._active = resolved
        resolved.prepare_tts(tts, style)

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        return self._active.wrap_text(text, style)


class MarkupFreeAdapter:
    """For an engine with no known style controls: never let markup reach the voice."""

    name = "markup_free"

    def prepare_tts(self, tts: Any, style: VoiceStyle) -> None:
        return None

    def wrap_text(self, text: AsyncIterable[str], style: VoiceStyle) -> AsyncIterator[str]:
        return _strip_markup_stream(text)


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
    if config.adapter == "auto":
        return AutoAdapter(config)
    if config.adapter == "elevenlabs_settings":
        return ElevenLabsSettingsAdapter(config)
    if config.adapter == "cartesia_inline":
        return CartesiaInlineTagAdapter(config)
    if config.adapter == "cartesia_extra":
        return CartesiaExtraKwargsAdapter(config)
    if config.adapter == "qwen3":
        return Qwen3InstructAdapter(config)
    return NullAdapter()
