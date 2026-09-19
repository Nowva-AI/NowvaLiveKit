#!/usr/bin/env python3
"""Probe which Cartesia sonic-3 style-delivery method actually changes the audio.

Renders one line neutral and in three styles through up to three delivery methods
(inline SSML tags via LiveKit inference, per-stream extra_kwargs via LiveKit inference,
and the standalone cartesia plugin), then measures duration and RMS. Optionally
transcribes each render with Deepgram to catch spoken tag text.

Usage:
    PYTHONPATH=src ./venv/bin/python scripts/tools/affect_tts_probe.py [--methods inline,extra,plugin] [--transcribe] [--out-dir DIR]

Pass criteria per method: speed 0.9 vs 1.1 shifts duration by >= 8%, volume 1.15 vs 0.9
shifts RMS by >= 1.5 dB, and no tag words appear in the transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from affect.config import StyleConfig  # noqa: E402
from affect.tts_adapters import cartesia_inline_tags  # noqa: E402
from affect.voice_style import CartesiaControls  # noqa: E402

LINE = "Rack it, you are done for today. Take your time, breathe, and we go again on Thursday."
MIN_SPEED_DURATION_RATIO = 1.08
MIN_VOLUME_DB_DELTA = 1.5
TAG_WORDS = ("emotion", "ratio", "speed", "volume", "value")

VARIANTS: dict[str, CartesiaControls] = {
    "neutral": CartesiaControls(),
    "slow": CartesiaControls(speed=0.9),
    "fast": CartesiaControls(speed=1.1),
    "quiet": CartesiaControls(volume=0.9),
    "loud": CartesiaControls(volume=1.15),
    "sympathetic_slow": CartesiaControls(emotion="Sympathetic", speed=0.9, volume=1.0),
    "enthusiastic": CartesiaControls(emotion="Enthusiastic", speed=1.05, volume=1.075),
}


def _voice_id() -> str:
    return os.getenv("CARTESIA_VOICE_ID", "3e39e9a5-585c-4f5f-bac6-5e4905c51095")


async def _collect(stream) -> tuple[np.ndarray, int]:
    frames = []
    sample_rate = 24000
    async for ev in stream:
        frame = ev.frame
        sample_rate = frame.sample_rate
        frames.append(np.frombuffer(frame.data, dtype=np.int16))
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
    return audio, sample_rate


async def _render_inline(controls: CartesiaControls) -> tuple[np.ndarray, int]:
    from livekit.agents import inference

    tts = inference.TTS(model="cartesia/sonic-3", voice=_voice_id(), language="en")
    text = cartesia_inline_tags(controls) + LINE
    async with tts.synthesize(text) as stream:
        return await _collect(stream)


async def _render_extra(controls: CartesiaControls) -> tuple[np.ndarray, int]:
    from livekit.agents import inference

    extra: dict = {}
    if controls.emotion:
        extra["emotion"] = controls.emotion.lower()
    if abs(controls.speed - 1.0) > 1e-6:
        extra["speed"] = controls.speed
    if abs(controls.volume - 1.0) > 1e-6:
        extra["volume"] = controls.volume
    tts = inference.TTS(model="cartesia/sonic-3", voice=_voice_id(), language="en", extra_kwargs=extra)
    async with tts.synthesize(LINE) as stream:
        return await _collect(stream)


async def _render_plugin(controls: CartesiaControls) -> tuple[np.ndarray, int]:
    from livekit.plugins import cartesia

    kwargs: dict = {"model": "sonic-3", "voice": _voice_id(), "language": "en", "speed": controls.speed, "volume": controls.volume}
    if controls.emotion:
        kwargs["emotion"] = [controls.emotion.lower()]
    tts = cartesia.TTS(**kwargs)
    async with tts.synthesize(LINE) as stream:
        return await _collect(stream)


RENDERERS = {"inline": _render_inline, "extra": _render_extra, "plugin": _render_plugin}


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    return float(20.0 * np.log10(max(rms, 1e-9) / 32768.0))


def _save(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


async def _transcribe(audio: np.ndarray, sample_rate: int) -> str:
    try:
        from livekit import rtc
        from livekit.plugins import deepgram
    except ImportError:
        return ""
    stt = deepgram.STT(model="nova-3", language="en")
    frame = rtc.AudioFrame(data=audio.tobytes(), sample_rate=sample_rate, num_channels=1, samples_per_channel=audio.shape[0])
    event = await stt.recognize(frame)
    return event.alternatives[0].text if event.alternatives else ""


async def _probe_method(name: str, out_dir: Path, transcribe: bool) -> dict:
    renderer = RENDERERS[name]
    results: dict[str, dict] = {}
    for variant, controls in VARIANTS.items():
        try:
            audio, sr = await renderer(controls)
        except Exception as exc:  # noqa: BLE001
            results[variant] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name:7s} {variant:17s} ERROR {type(exc).__name__}: {str(exc)[:120]}")
            continue
        seconds = audio.shape[0] / sr
        _save(out_dir / f"{name}_{variant}.wav", audio, sr)
        row = {"seconds": round(seconds, 3), "rms_db": round(_rms_db(audio), 2)}
        if transcribe:
            text = await _transcribe(audio, sr)
            row["transcript"] = text
            row["tag_words_spoken"] = any(word in text.lower() for word in TAG_WORDS)
        results[variant] = row
        print(f"  {name:7s} {variant:17s} {seconds:6.2f}s  {row['rms_db']:7.2f} dB" + (f"  '{row.get('transcript', '')[:60]}'" if transcribe else ""))
    verdict = _verdict(results)
    return {"variants": results, "verdict": verdict}


def _verdict(results: dict[str, dict]) -> dict:
    def _get(variant: str, key: str) -> float | None:
        row = results.get(variant, {})
        return row.get(key) if "error" not in row else None

    slow, fast = _get("slow", "seconds"), _get("fast", "seconds")
    quiet, loud = _get("quiet", "rms_db"), _get("loud", "rms_db")
    speed_ok = slow is not None and fast is not None and fast > 0 and (slow / fast) >= MIN_SPEED_DURATION_RATIO
    volume_ok = quiet is not None and loud is not None and (loud - quiet) >= MIN_VOLUME_DB_DELTA
    spoken = [v for v, row in results.items() if row.get("tag_words_spoken")]
    return {
        "speed_effect_ratio": round(slow / fast, 3) if slow and fast else None,
        "volume_effect_db": round(loud - quiet, 2) if quiet is not None and loud is not None else None,
        "speed_ok": speed_ok,
        "volume_ok": volume_ok,
        "tag_words_spoken_in": spoken,
        "pass": speed_ok and volume_ok and not spoken,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default="inline,extra,plugin")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe renders with Deepgram to catch spoken tags")
    parser.add_argument("--out-dir", default="data/affect/tts_probe")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in RENDERERS]
    if "plugin" in methods and not os.getenv("CARTESIA_API_KEY"):
        print("CARTESIA_API_KEY not set; skipping the plugin method")
        methods.remove("plugin")

    # LiveKit plugins expect a job-scoped aiohttp session; create one since this runs outside a worker.
    from livekit.agents.utils import http_context

    http_context._new_session_ctx()
    report: dict[str, dict] = {}
    try:
        for name in methods:
            print(f"\n== {name} ==")
            report[name] = await _probe_method(name, out_dir, args.transcribe)
            print(f"  verdict: {json.dumps(report[name]['verdict'])}")
    finally:
        await http_context._close_http_ctx()

    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    passing = [m for m, r in report.items() if r["verdict"]["pass"]]
    print("\nPASSING METHODS:", passing or "none")
    if "inline" in passing:
        print("Recommended adapter: cartesia_inline (config/affect.yaml style.adapter)")
    elif "extra" in passing:
        print("Recommended adapter: cartesia_extra")
    elif "plugin" in passing:
        print("Only the standalone cartesia plugin honours controls; switch voice_agent.py to cartesia.TTS + cartesia_extra")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
