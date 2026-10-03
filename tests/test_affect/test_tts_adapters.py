"""Tests for TTS style adapters: tag prefixing, tag stripping, per-stream option swapping, tokenizer safety."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from affect.config import StyleConfig
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.elevenlabs import VoiceSettings

from affect.tts_adapters import (
    AutoAdapter,
    CartesiaExtraKwargsAdapter,
    ElevenLabsSettingsAdapter,
    CartesiaInlineTagAdapter,
    NullAdapter,
    Qwen3InstructAdapter,
    build_adapter,
    cartesia_inline_tags,
    strip_style_tags,
    tts_parses_cartesia_tags,
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


class TestNonCartesiaTTS:
    """sonic-3 tags are Cartesia syntax; any other engine speaks them aloud."""

    def test_detects_cartesia_by_module(self) -> None:
        cartesia_like = type("TTS", (), {})()
        cartesia_like.__class__.__module__ = "livekit.plugins.cartesia.tts"
        assert tts_parses_cartesia_tags(cartesia_like) is True

    def test_detects_cartesia_over_inference_by_model_id(self) -> None:
        gateway = SimpleNamespace(_opts=SimpleNamespace(model="cartesia/sonic-3"))
        assert tts_parses_cartesia_tags(gateway) is True

    def test_detects_non_cartesia(self) -> None:
        eleven = SimpleNamespace(_opts=SimpleNamespace(model="eleven_flash_v2_5"))
        assert tts_parses_cartesia_tags(eleven) is False

    def test_tags_dropped_for_non_cartesia_tts(self) -> None:
        adapter = CartesiaInlineTagAdapter(StyleConfig())
        style = VoiceStyle(energy=-1, pace=-1, warmth=1)
        adapter.prepare_tts(SimpleNamespace(_opts=SimpleNamespace(model="eleven_flash_v2_5")), style)
        out = _collect(adapter.wrap_text(_stream("Take it easy today."), style))
        assert out == ["Take it easy today."]

    def test_tags_kept_for_cartesia_tts(self) -> None:
        adapter = CartesiaInlineTagAdapter(StyleConfig())
        style = VoiceStyle(energy=-1, pace=-1, warmth=1)
        adapter.prepare_tts(SimpleNamespace(_opts=SimpleNamespace(model="cartesia/sonic-3")), style)
        out = _collect(adapter.wrap_text(_stream("Take it easy today."), style))
        assert out[0].startswith(cartesia_inline_tags(to_cartesia(style, StyleConfig())))

    def test_unknown_tts_keeps_existing_behaviour(self) -> None:
        adapter = CartesiaInlineTagAdapter(StyleConfig())
        style = VoiceStyle(energy=-1, pace=-1, warmth=1)
        out = _collect(adapter.wrap_text(_stream("Take it easy today."), style))
        assert out[0].startswith(cartesia_inline_tags(to_cartesia(style, StyleConfig())))


def _fake_tts(module: str, **opts: object) -> object:
    tts_class = type("TTS", (), {})
    tts_class.__module__ = module
    tts = tts_class()
    tts._opts = SimpleNamespace(**opts)
    return tts


def _elevenlabs_tts(voice_settings: object = NOT_GIVEN) -> object:
    return _fake_tts("livekit.plugins.elevenlabs.tts", voice_settings=voice_settings, model="eleven_flash_v2_5")


CALM_SLOW = VoiceStyle(energy=-1, pace=-1, warmth=1)


class TestElevenLabsAdapter:
    def test_style_becomes_voice_settings(self) -> None:
        tts = _elevenlabs_tts()
        ElevenLabsSettingsAdapter(StyleConfig()).prepare_tts(tts, CALM_SLOW)
        assert tts._opts.voice_settings.stability == pytest.approx(0.65)
        assert tts._opts.voice_settings.speed == pytest.approx(0.9)
        assert tts._opts.voice_settings.similarity_boost == pytest.approx(0.75)

    def test_neutral_restores_exactly_what_was_configured(self) -> None:
        tts = _elevenlabs_tts()
        adapter = ElevenLabsSettingsAdapter(StyleConfig())
        adapter.prepare_tts(tts, CALM_SLOW)
        adapter.prepare_tts(tts, NEUTRAL_STYLE)
        assert tts._opts.voice_settings is NOT_GIVEN

    def test_configured_settings_are_the_baseline_and_never_mutated(self) -> None:
        configured = VoiceSettings(stability=0.4, similarity_boost=0.9, use_speaker_boost=False)
        tts = _elevenlabs_tts(configured)
        ElevenLabsSettingsAdapter(StyleConfig()).prepare_tts(tts, CALM_SLOW)
        styled = tts._opts.voice_settings
        assert styled is not configured
        assert configured.stability == pytest.approx(0.4)
        assert styled.stability == pytest.approx(0.55)
        assert styled.similarity_boost == pytest.approx(0.9)
        assert styled.use_speaker_boost is False

    def test_style_does_not_compound_across_turns(self) -> None:
        tts = _elevenlabs_tts()
        adapter = ElevenLabsSettingsAdapter(StyleConfig())
        for _ in range(5):
            adapter.prepare_tts(tts, CALM_SLOW)
        assert tts._opts.voice_settings.stability == pytest.approx(0.65)

    def test_other_engines_are_left_alone(self) -> None:
        tts = _fake_tts("livekit.plugins.cartesia.tts", voice_settings="untouched")
        ElevenLabsSettingsAdapter(StyleConfig()).prepare_tts(tts, CALM_SLOW)
        assert tts._opts.voice_settings == "untouched"

    def test_laughter_and_style_tags_never_reach_the_voice(self) -> None:
        adapter = ElevenLabsSettingsAdapter(StyleConfig())
        out = _collect(adapter.wrap_text(_stream("Ha [laughter] fair.", ' <speed ratio="0.9"/>Reset.'), CALM_SLOW))
        assert "".join(out) == "Ha fair. Reset."


class TestAutoAdapter:
    def test_cartesia_gets_inline_tags(self) -> None:
        adapter = AutoAdapter(StyleConfig())
        adapter.prepare_tts(_fake_tts("livekit.plugins.cartesia.tts"), CALM_SLOW)
        out = _collect(adapter.wrap_text(_stream("Easy."), CALM_SLOW))
        assert adapter.active.name == "cartesia_inline"
        assert out[0].startswith("<emotion")

    def test_cartesia_over_inference_gets_inline_tags(self) -> None:
        adapter = AutoAdapter(StyleConfig())
        adapter.prepare_tts(_fake_tts("livekit.agents.inference.tts", model="cartesia/sonic-3"), CALM_SLOW)
        assert adapter.active.name == "cartesia_inline"

    def test_elevenlabs_gets_voice_settings_and_no_markup(self) -> None:
        tts = _elevenlabs_tts()
        adapter = AutoAdapter(StyleConfig())
        adapter.prepare_tts(tts, CALM_SLOW)
        out = _collect(adapter.wrap_text(_stream("Easy."), CALM_SLOW))
        assert adapter.active.name == "elevenlabs_settings"
        assert out == ["Easy."]
        assert tts._opts.voice_settings.speed == pytest.approx(0.9)

    def test_unknown_engine_gets_no_markup(self) -> None:
        adapter = AutoAdapter(StyleConfig())
        adapter.prepare_tts(_fake_tts("some.other.tts", model="mystery"), CALM_SLOW)
        out = _collect(adapter.wrap_text(_stream('<emotion value="calm"/>Easy. [laughter]'), CALM_SLOW))
        assert "".join(out).strip() == "Easy."

    def test_before_any_tts_is_seen_nothing_is_spoken_as_markup(self) -> None:
        out = _collect(AutoAdapter(StyleConfig()).wrap_text(_stream("Easy."), CALM_SLOW))
        assert out == ["Easy."]

    def test_follows_a_tts_swap(self) -> None:
        adapter = AutoAdapter(StyleConfig())
        adapter.prepare_tts(_fake_tts("livekit.plugins.cartesia.tts"), CALM_SLOW)
        adapter.prepare_tts(_elevenlabs_tts(), CALM_SLOW)
        assert adapter.active.name == "elevenlabs_settings"

    def test_is_the_default(self) -> None:
        assert build_adapter(StyleConfig()).name == "auto"


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
