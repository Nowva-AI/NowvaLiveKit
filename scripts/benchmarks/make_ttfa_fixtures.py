#!/usr/bin/env python3
"""Synthesize the user-utterance WAVs the TTFA benchmark replays.

Fixtures are written once to benchmarks/fixtures/voice/ and reused by every
benchmark run so latency numbers are comparable across iterations. A voice
distinct from Nova's is used so nothing in the pipeline can confuse the
synthetic user with the agent.

Usage:
    python scripts/benchmarks/make_ttfa_fixtures.py
    python scripts/benchmarks/make_ttfa_fixtures.py --force
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

FIXTURES_DIR = PROJECT_ROOT / "benchmarks" / "fixtures" / "voice"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "ash"
SAMPLE_RATE = 24000
FRAME_MS = 20
# Synthesised speech inserts pauses a person would not: gpt-4o-mini-tts leaves
# 400-800 ms between clauses. Left in, every pause past the VAD silence window
# splits one utterance into several turns and the benchmark measures
# fragmentation instead of latency. Cap internal pauses at a natural length and
# trim the lead/tail so "end of speech" is the real end of speech.
MAX_INTERNAL_SILENCE_MS = 150
LEAD_SILENCE_MS = 100
TAIL_SILENCE_MS = 40
SILENCE_RMS = 150
INSTRUCTIONS = (
    "You are a gym-goer talking to a voice assistant on a squat rack. "
    "Speak casually and at a normal conversational pace, as if mid-workout. "
    "Do not trail off; finish the sentence cleanly."
)

# Conversational turns that keep MainMenuAgent in place: no tool call, no agent
# handoff, so every trial measures the same pipeline. Tool-preamble latency is
# covered separately by TOOL_UTTERANCES.
UTTERANCES: dict[str, str] = {
    "chitchat": "Man, my legs are still wrecked from Tuesday.",
    "depth_question": "Should I be going all the way down on my squats?",
    "rest_question": "How long should I be resting between sets?",
    "greeting": "Hey Nova, how's it going?",
    "knee_check": "My knees felt a bit off last session. What do you think?",
    "long_ask": "I've been training four days a week for about three months now and I feel like my squat has completely stalled out. What would you change?",
}

TOOL_UTTERANCES: dict[str, str] = {
    "tool_progress": "How have I been doing lately?",
    "tool_schedule": "Can you show me what's on my schedule this week?",
}



def _voiced_frames(samples: np.ndarray) -> np.ndarray:
    frame_len = SAMPLE_RATE * FRAME_MS // 1000
    usable = len(samples) // frame_len * frame_len
    frames = samples[:usable].astype(np.float32).reshape(-1, frame_len)
    rms = np.sqrt((frames**2).mean(axis=1))
    return rms > max(SILENCE_RMS, rms.max() * 0.03)


def normalize_pauses(samples: np.ndarray) -> np.ndarray:
    frame_len = SAMPLE_RATE * FRAME_MS // 1000
    voiced = _voiced_frames(samples)
    idx = np.nonzero(voiced)[0]
    if len(idx) == 0:
        return samples

    max_gap_frames = MAX_INTERNAL_SILENCE_MS // FRAME_MS
    kept: list[np.ndarray] = [np.zeros(SAMPLE_RATE * LEAD_SILENCE_MS // 1000, dtype=np.int16)]
    gap = 0
    for i in range(idx[0], idx[-1] + 1):
        frame = samples[i * frame_len : (i + 1) * frame_len]
        if voiced[i]:
            gap = 0
            kept.append(frame)
        else:
            gap += 1
            if gap <= max_gap_frames:
                kept.append(frame)
    kept.append(np.zeros(SAMPLE_RATE * TAIL_SILENCE_MS // 1000, dtype=np.int16))
    return np.concatenate(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate existing fixtures")
    args = parser.parse_args()

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    for name, text in {**UTTERANCES, **TOOL_UTTERANCES}.items():
        path = FIXTURES_DIR / f"{name}.wav"
        if path.exists() and not args.force:
            print(f"  skip   {path.name} (exists)")
            continue
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            instructions=INSTRUCTIONS,
            response_format="pcm",
        )
        raw = np.frombuffer(response.content, dtype=np.int16)
        pcm = normalize_pauses(raw).tobytes()
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        before = len(raw) / SAMPLE_RATE
        after = len(pcm) / 2 / SAMPLE_RATE
        print(f"  write  {path.name:22s} {before:.2f}s -> {after:.2f}s  {text!r}")

    print(f"\nFixtures in {FIXTURES_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
