"""Tests for the voice-style policy and its Cartesia and Qwen3 mappings."""

from __future__ import annotations

import pytest

from affect.config import StyleConfig
from affect.state import AthleteState
from affect.voice_style import NEUTRAL_STYLE, VoiceStyle, qwen3_instruct, style_for, to_cartesia


class TestPolicy:
    def test_unknown_state_is_neutral(self) -> None:
        assert style_for(None) == NEUTRAL_STYLE
        assert style_for(AthleteState(affect="frustrated", confident=False)) == NEUTRAL_STYLE

    def test_frustration_is_never_mirrored(self) -> None:
        style = style_for(AthleteState(affect="frustrated", confident=True))
        assert style.energy == -1 and style.warmth == 1

    def test_engaged_conversation_vs_coaching(self) -> None:
        state = AthleteState(affect="engaged", confident=True)
        assert style_for(state, "conversation").pace == 0
        assert style_for(state, "coaching").pace == 1

    def test_near_limit_coaching_is_calm_and_slow(self) -> None:
        state = AthleteState(effort="near_limit", affect="flat", confident=True)
        style = style_for(state, "coaching")
        assert style.pace == -1 and style.warmth == 1


class TestCartesiaMapping:
    def test_neutral_style_has_no_controls(self) -> None:
        controls = to_cartesia(NEUTRAL_STYLE, StyleConfig())
        assert controls.is_neutral()

    def test_clamps_speed_and_volume(self) -> None:
        config = StyleConfig(pace_step=0.5, energy_volume_step=0.5)
        controls = to_cartesia(VoiceStyle(energy=1, pace=1, warmth=0), config)
        assert controls.speed == pytest.approx(config.speed_max)
        assert controls.volume == pytest.approx(config.volume_max)

    def test_emotion_names(self) -> None:
        config = StyleConfig()
        assert to_cartesia(VoiceStyle(energy=-1, pace=-1, warmth=1), config).emotion == "Sympathetic"
        assert to_cartesia(VoiceStyle(energy=-1, pace=0, warmth=0), config).emotion == "Calm"
        assert to_cartesia(VoiceStyle(energy=1, pace=0, warmth=0), config).emotion == "Enthusiastic"


class TestQwen3:
    def test_27_distinct_strings(self) -> None:
        strings = {qwen3_instruct(VoiceStyle(energy=e, pace=p, warmth=w)) for e in (-1, 0, 1) for p in (-1, 0, 1) for w in (-1, 0, 1)}
        assert len(strings) == 27
