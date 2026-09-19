"""Tests for TTS style adapters: tag prefixing, tag stripping, per-stream option swapping, tokenizer safety."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from affect.config import StyleConfig
from affect.tts_adapters import (
    CartesiaExtraKwargsAdapter,
    CartesiaInlineTagAdapter,
    NullAdapter,
    Qwen3InstructAdapter,
    build_adapter,
    cartesia_inline_tags,
    strip_style_tags,
)
from affect.voice_style import NEUTRAL_STYLE, VoiceStyle, to_cartesia


async def _stream(*chunks: str):
    for chunk in chunks:
        yield chunk


def _collect(agen) -> list[str]:
    async def _run() -> list[str]:
        return [chunk async for chunk in agen]

    return asyncio.run(_run())


class TestStripping:
    def test_removes_style_tags_only(self) -> None:
        text = 'Hi <emotion value="calm"/> there <speed ratio="0.9"/> <break time="400ms"/> [laughter] ok</volume>'
        assert strip_style_tags(text) == 'Hi  there  <break time="400ms"/> [laughter] ok'


class TestInlineAdapter:
    def test_prefix_on_first_chunk_only(self) -> None:
        adapter = CartesiaInlineTagAdapter(StyleConfig())
        style = VoiceStyle(energy=-1, pace=-1, warmth=1)
        out = _collect(adapter.wrap_text(_stream("Hello.", " Take a breath."), style))
        expected_prefix = cartesia_inline_tags(to_cartesia(style, StyleConfig()))
        assert out[0].startswith(expected_prefix)
        assert "emotion" in expected_prefix and "speed" in expected_prefix
        assert out[1] == " Take a breath."

    def test_neutral_style_adds_nothing(self) -> None:
        adapter = CartesiaInlineTagAdapter(StyleConfig())
        out = _collect(adapter.wrap_text(_stream("Hello."), NEUTRAL_STYLE))
        assert out == ["Hello."]

    def test_llm_tags_are_stripped(self) -> None:
        adapter = CartesiaInlineTagAdapter(StyleConfig())
        out = _collect(adapter.wrap_text(_stream('<emotion value="angry"/>Hello.'), NEUTRAL_STYLE))
        assert out == ["Hello."]

    def test_blingfire_keeps_tag_with_first_sentence(self) -> None:
        tokenize = pytest.importorskip("livekit.agents.tokenize")
        tokenizer = tokenize.blingfire.SentenceTokenizer(retain_format=True)
        prefix = cartesia_inline_tags(to_cartesia(VoiceStyle(energy=-1, pace=-1, warmth=1), StyleConfig()))
        sentences = tokenizer.tokenize(prefix + "Rack it. You are done.")
        assert sentences[0].startswith(prefix)
        assert "0.9" in sentences[0]


class TestExtraKwargsAdapter:
    def test_swaps_new_dict_without_mutating_shared(self) -> None:
        shared = {"add_timestamps": True, "emotion": "old"}
        tts = SimpleNamespace(_opts=SimpleNamespace(extra_kwargs=shared))
        adapter = CartesiaExtraKwargsAdapter(StyleConfig())
        adapter.prepare_tts(tts, VoiceStyle(energy=1, pace=1, warmth=0))
        assert tts._opts.extra_kwargs is not shared
        assert shared == {"add_timestamps": True, "emotion": "old"}
        assert tts._opts.extra_kwargs["emotion"] == "enthusiastic"
        assert tts._opts.extra_kwargs["speed"] == pytest.approx(1.1)
        assert tts._opts.extra_kwargs["add_timestamps"] is True

    def test_tolerates_foreign_tts(self) -> None:
        CartesiaExtraKwargsAdapter(StyleConfig()).prepare_tts(object(), NEUTRAL_STYLE)


class TestOtherAdapters:
    def test_qwen3_exposes_instruct(self) -> None:
        adapter = Qwen3InstructAdapter(StyleConfig())
        received: list[str] = []
        tts = SimpleNamespace(set_instruct=received.append)
        adapter.prepare_tts(tts, VoiceStyle(energy=1, pace=0, warmth=1))
        assert received and received[0] == adapter.last_instruct

    def test_null_adapter_strips(self) -> None:
        out = _collect(NullAdapter().wrap_text(_stream('<speed ratio="2"/>Go.'), NEUTRAL_STYLE))
        assert out == ["Go."]

    def test_build_adapter(self) -> None:
        assert build_adapter(StyleConfig(adapter="cartesia_inline")).name == "cartesia_inline"
        assert build_adapter(StyleConfig(adapter="cartesia_extra")).name == "cartesia_extra"
        assert build_adapter(StyleConfig(adapter="qwen3")).name == "qwen3"
        assert build_adapter(StyleConfig(adapter="none")).name == "none"
