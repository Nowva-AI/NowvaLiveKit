"""Deterministic voice-style policy: athlete state + speech kind → (energy, pace, warmth), and the per-engine mappings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from affect.config import StyleConfig
from affect.state import AthleteState

SpeechKind = Literal["conversation", "coaching"]
CoachingMoment = Literal["last_rep", "recap"]
Level = Literal[-1, 0, 1]

CARTESIA_EMOTION_BY_STYLE: dict[tuple[int, int], str] = {
    (-1, 1): "Sympathetic",
    (-1, 0): "Calm",
    (0, 1): "Content",
    (1, 0): "Enthusiastic",
    (1, 1): "Enthusiastic",
}


# ElevenLabs realizes a style through per-utterance voice settings. Stability is its
# expressiveness dial — low is dynamic, high is steady — so energy moves it inversely.
# Warmth has no mapping: its only lever, style exaggeration, adds synthesis latency.
ELEVENLABS_DEFAULT_STABILITY = 0.5
ELEVENLABS_DEFAULT_SPEED = 1.0
ELEVENLABS_STABILITY_STEP = 0.15
ELEVENLABS_STABILITY_MIN = 0.3
ELEVENLABS_STABILITY_MAX = 0.8
ELEVENLABS_SPEED_MIN = 0.7
ELEVENLABS_SPEED_MAX = 1.2


class VoiceStyle(BaseModel):
    energy: Level = 0
    pace: Level = 0
    warmth: Level = 0

    def is_neutral(self) -> bool:
        return self.energy == 0 and self.pace == 0 and self.warmth == 0

    def key(self) -> tuple[int, int, int]:
        return (self.energy, self.pace, self.warmth)


NEUTRAL_STYLE = VoiceStyle()

# Negative affect is never mirrored: a frustrated athlete gets calm and warm, never agitated.
_AFFECT_STYLES: dict[str, dict[SpeechKind, VoiceStyle]] = {
    "frustrated": {
        "conversation": VoiceStyle(energy=-1, pace=-1, warmth=1),
        "coaching": VoiceStyle(energy=-1, pace=-1, warmth=1),
    },
    "strained": {
        "conversation": VoiceStyle(energy=-1, pace=0, warmth=1),
        "coaching": VoiceStyle(energy=0, pace=-1, warmth=1),
    },
    "engaged": {
        "conversation": VoiceStyle(energy=1, pace=0, warmth=0),
        "coaching": VoiceStyle(energy=1, pace=1, warmth=0),
    },
    "flat": {
        "conversation": NEUTRAL_STYLE,
        "coaching": NEUTRAL_STYLE,
    },
}


# Where the workout is shapes how coaching speech sounds: firmer near the last rep,
# calm for the recap. A frustrated or strained athlete still gets the calm, warm style
# below — negative affect is never met with intensity.
_MOMENT_STYLES: dict[str, VoiceStyle] = {
    "last_rep": VoiceStyle(energy=1, pace=0, warmth=0),
    "recap": VoiceStyle(energy=-1, pace=0, warmth=0),
}
_NEVER_OVERRIDDEN_AFFECTS = ("frustrated", "strained")
_coaching_moment: CoachingMoment | None = None


def set_coaching_moment(moment: CoachingMoment | None) -> None:
    """The coaching orchestrator calls this as a workout moves between moments; None clears it."""
    global _coaching_moment
    if moment is not None and moment not in _MOMENT_STYLES:
        raise ValueError(f"Unknown coaching moment: {moment!r}")
    _coaching_moment = moment


def style_for(state: AthleteState | None, speech_kind: SpeechKind = "conversation") -> VoiceStyle:
    moment_style = _MOMENT_STYLES.get(_coaching_moment) if speech_kind == "coaching" else None
    if state is None or not state.confident:
        return moment_style or NEUTRAL_STYLE
    if moment_style is not None and state.affect not in _NEVER_OVERRIDDEN_AFFECTS:
        return moment_style
    style = _AFFECT_STYLES.get(state.affect, _AFFECT_STYLES["flat"])[speech_kind]
    if state.effort == "near_limit" and speech_kind == "coaching" and state.affect in ("flat", "engaged"):
        return VoiceStyle(energy=0, pace=-1, warmth=1)
    return style


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class CartesiaControls(BaseModel):
    emotion: str | None = None
    speed: float = 1.0
    volume: float = 1.0

    def is_neutral(self) -> bool:
        return self.emotion is None and abs(self.speed - 1.0) < 1e-6 and abs(self.volume - 1.0) < 1e-6


def to_cartesia(style: VoiceStyle, config: StyleConfig) -> CartesiaControls:
    emotion = CARTESIA_EMOTION_BY_STYLE.get((style.energy, style.warmth))
    speed = _clamp(1.0 + config.pace_step * style.pace, config.speed_min, config.speed_max)
    volume = _clamp(1.0 + config.energy_volume_step * style.energy, config.volume_min, config.volume_max)
    return CartesiaControls(emotion=emotion, speed=round(speed, 3), volume=round(volume, 3))


class ElevenLabsControls(BaseModel):
    stability: float = ELEVENLABS_DEFAULT_STABILITY
    speed: float = ELEVENLABS_DEFAULT_SPEED


def to_elevenlabs(
    style: VoiceStyle,
    config: StyleConfig,
    base_stability: float = ELEVENLABS_DEFAULT_STABILITY,
    base_speed: float = ELEVENLABS_DEFAULT_SPEED,
) -> ElevenLabsControls:
    """Shift the voice's own baseline, so a voice tuned away from the defaults keeps its character."""
    stability = _clamp(
        base_stability - ELEVENLABS_STABILITY_STEP * style.energy,
        ELEVENLABS_STABILITY_MIN,
        ELEVENLABS_STABILITY_MAX,
    )
    speed = _clamp(
        base_speed + config.pace_step * style.pace,
        max(config.speed_min, ELEVENLABS_SPEED_MIN),
        min(config.speed_max, ELEVENLABS_SPEED_MAX),
    )
    return ElevenLabsControls(stability=round(stability, 3), speed=round(speed, 3))


def qwen3_instruct(style: VoiceStyle) -> str:
    """One of 27 deterministic instruct strings for the Qwen3-TTS adapter (interface only in V1)."""
    energy = {-1: "gently", 0: "evenly", 1: "with real drive"}[style.energy]
    pace = {-1: "slowly and deliberately", 0: "at a steady pace", 1: "briskly"}[style.pace]
    warmth = {-1: "matter-of-fact", 0: "friendly", 1: "warm and encouraging"}[style.warmth]
    return f"Speak {energy}, {pace}, in a {warmth} tone."
