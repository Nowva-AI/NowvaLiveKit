"""Tests for the voice-style policy and its Cartesia and Qwen3 mappings."""

from __future__ import annotations

import pytest

from affect.config import StyleConfig
from affect.state import AthleteState
from affect.voice_style import (
    ELEVENLABS_DEFAULT_SPEED,
    ELEVENLABS_DEFAULT_STABILITY,
    ELEVENLABS_STABILITY_MAX,
    ELEVENLABS_STABILITY_MIN,
    NEUTRAL_STYLE,
    VoiceStyle,
    qwen3_instruct,
    set_coaching_moment,
    style_for,
    to_cartesia,
    to_elevenlabs,
)


@pytest.fixture(autouse=True)
def no_coaching_moment():
    set_coaching_moment(None)
    yield
    set_coaching_moment(None)


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


class TestCoachingMoments:
    def test_last_rep_is_firmer_for_coaching(self) -> None:
        set_coaching_moment("last_rep")
        assert style_for(None, "coaching").energy == 1

    def test_recap_is_calm(self) -> None:
        set_coaching_moment("recap")
        style = style_for(AthleteState(affect="engaged", confident=True), "coaching")
        assert style.energy == -1 and style.pace == 0

    def test_moment_only_shapes_coaching_speech(self) -> None:
        set_coaching_moment("last_rep")
        assert style_for(None, "conversation") == NEUTRAL_STYLE

    def test_frustration_still_wins_over_moment(self) -> None:
        set_coaching_moment("last_rep")
        style = style_for(AthleteState(affect="frustrated", confident=True), "coaching")
        assert style.energy == -1 and style.warmth == 1

    def test_cleared_moment_falls_back_to_affect(self) -> None:
        set_coaching_moment("recap")
        set_coaching_moment(None)
        assert style_for(None, "coaching") == NEUTRAL_STYLE

    def test_unknown_moment_rejected(self) -> None:
        with pytest.raises(ValueError):
            set_coaching_moment("warmup")

    def test_last_rep_moves_elevenlabs_settings(self) -> None:
        set_coaching_moment("last_rep")
        controls = to_elevenlabs(style_for(None, "coaching"), StyleConfig())
        assert controls.stability < ELEVENLABS_DEFAULT_STABILITY


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


class TestElevenLabsMapping:
    def test_neutral_style_is_the_voice_baseline(self) -> None:
        controls = to_elevenlabs(NEUTRAL_STYLE, StyleConfig())
        assert controls.stability == pytest.approx(ELEVENLABS_DEFAULT_STABILITY)
        assert controls.speed == pytest.approx(ELEVENLABS_DEFAULT_SPEED)

    def test_low_energy_is_steadier_high_energy_more_dynamic(self) -> None:
        calm = to_elevenlabs(VoiceStyle(energy=-1), StyleConfig())
        driven = to_elevenlabs(VoiceStyle(energy=1), StyleConfig())
        assert calm.stability > ELEVENLABS_DEFAULT_STABILITY > driven.stability

    def test_pace_moves_speed_within_config_bounds(self) -> None:
        config = StyleConfig()
        assert to_elevenlabs(VoiceStyle(pace=-1), config).speed == pytest.approx(config.speed_min)
        assert to_elevenlabs(VoiceStyle(pace=1), config).speed == pytest.approx(config.speed_max)

    def test_shifts_from_a_custom_voice_baseline(self) -> None:
        controls = to_elevenlabs(VoiceStyle(energy=-1), StyleConfig(), base_stability=0.4)
        assert controls.stability == pytest.approx(0.55)

    def test_stability_is_clamped(self) -> None:
        high = to_elevenlabs(VoiceStyle(energy=-1), StyleConfig(), base_stability=0.95)
        low = to_elevenlabs(VoiceStyle(energy=1), StyleConfig(), base_stability=0.05)
        assert high.stability == pytest.approx(ELEVENLABS_STABILITY_MAX)
        assert low.stability == pytest.approx(ELEVENLABS_STABILITY_MIN)

    def test_warmth_alone_changes_nothing(self) -> None:
        assert to_elevenlabs(VoiceStyle(warmth=1), StyleConfig()) == to_elevenlabs(NEUTRAL_STYLE, StyleConfig())


class TestQwen3:
    def test_27_distinct_strings(self) -> None:
        strings = {qwen3_instruct(VoiceStyle(energy=e, pace=p, warmth=w)) for e in (-1, 0, 1) for p in (-1, 0, 1) for w in (-1, 0, 1)}
        assert len(strings) == 27
