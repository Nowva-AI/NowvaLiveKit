"""Deterministic voice-style policy: athlete state + speech kind → (energy, pace, warmth), and the Cartesia mapping."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from affect.config import StyleConfig
from affect.state import AthleteState

SpeechKind = Literal["conversation", "coaching"]
Level = Literal[-1, 0, 1]

CARTESIA_EMOTION_BY_STYLE: dict[tuple[int, int], str] = {
    (-1, 1): "Sympathetic",
    (-1, 0): "Calm",
    (0, 1): "Content",
    (1, 0): "Enthusiastic",
    (1, 1): "Enthusiastic",
}


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


def style_for(state: AthleteState | None, speech_kind: SpeechKind = "conversation") -> VoiceStyle:
    if state is None or not state.confident:
        return NEUTRAL_STYLE
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


def qwen3_instruct(style: VoiceStyle) -> str:
    """One of 27 deterministic instruct strings for the Qwen3-TTS adapter (interface only in V1)."""
    energy = {-1: "gently", 0: "evenly", 1: "with real drive"}[style.energy]
    pace = {-1: "slowly and deliberately", 0: "at a steady pace", 1: "briskly"}[style.pace]
    warmth = {-1: "matter-of-fact", 0: "friendly", 1: "warm and encouraging"}[style.warmth]
    return f"Speak {energy}, {pace}, in a {warmth} tone."
