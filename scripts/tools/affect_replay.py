#!/usr/bin/env python3
"""Replay WAV files through the real TappedVAD + AffectService offline and print per-utterance state.

Usage:
    PYTHONPATH=src ./venv/bin/python scripts/tools/affect_replay.py --wav-dir data/affect/recordings/<user>/<session>
    PYTHONPATH=src ./venv/bin/python scripts/tools/affect_replay.py --wav a.wav b.wav --mode workout --no-baseline

Each WAV is streamed as 10 ms frames at its native rate (as the console mic would) so the
Silero VAD, trimming, early trigger and the encoder see exactly what they see live.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from livekit import rtc  # noqa: E402
from livekit.plugins import silero  # noqa: E402

from affect.config import load_affect_config  # noqa: E402
from affect.engine import AffectEngine  # noqa: E402
from agent.services.affect_service import AffectService  # noqa: E402
from agent.services.affect_vad_tap import TappedVAD  # noqa: E402

FRAME_MS = 10
TAIL_SILENCE_S = 1.0


class _State:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def get_mode(self) -> str:
        return self.mode

    def get_user(self) -> dict:
        return {"id": "replay"}


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]
    return pcm, rate


async def _stream_file(stream, pcm: np.ndarray, rate: int) -> None:
    frame_len = rate * FRAME_MS // 1000
    signal = np.concatenate([pcm, np.zeros(int(TAIL_SILENCE_S * rate), dtype=np.int16)])
    for start in range(0, len(signal) - frame_len, frame_len):
        chunk = signal[start : start + frame_len]
        stream.push_frame(rtc.AudioFrame(data=chunk.tobytes(), sample_rate=rate, num_channels=1, samples_per_channel=frame_len))
        await asyncio.sleep(0)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-dir", type=Path, default=None)
    parser.add_argument("--wav", type=Path, nargs="*", default=[])
    parser.add_argument("--mode", default="main_menu", help="Agent mode to simulate (main_menu = neutral context)")
    parser.add_argument("--no-baseline", action="store_true", help="Treat every utterance as confident (population z-scores)")
    parser.add_argument("--json", action="store_true", help="Print one JSON line per utterance")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    files = list(args.wav)
    if args.wav_dir:
        files.extend(sorted(args.wav_dir.glob("*.wav")))
    if not files:
        parser.error("give --wav-dir or --wav")

    config = load_affect_config()
    config.recorder.enabled = False
    config.baseline.profile_dir = str(Path("data/affect/replay_profiles"))
    if args.no_baseline:
        config.baseline.require_ready_for_confidence = False
    engine = AffectEngine(config.resolve_model_dir(), config.engine)
    engine.initialize()
    service = AffectService(config, engine, state=_State(args.mode), user_id=None, session_id="replay")
    tapped = TappedVAD(silero.VAD.load(), service)
    stream = tapped.stream()

    async def _drain() -> None:
        async for _ in stream:
            pass

    drain = asyncio.create_task(_drain())
    print(f"model={engine.manifest.version} provider={engine.provider} adapter={service.style_adapter.name}")
    for path in files:
        pcm, rate = _read_wav(path)
        before = service.snapshot_dict()["stats"]["utterances"]
        await _stream_file(stream, pcm, rate)
        await asyncio.sleep(0.8)
        while service._final_task is not None and not service._final_task.done():
            await asyncio.sleep(0.05)
        snap = service.snapshot_dict()
        state = snap["state"]
        avd = snap["last_avd"]
        row = {
            "file": path.name, "utterances": snap["stats"]["utterances"] - before,
            "avd": [round(v, 3) for v in avd] if avd else None,
            "z": [round(state["arousal_z"], 2), round(state["dominance_z"], 2), round(state["valence_z"], 2)],
            "effort": state["effort"], "affect": state["affect"], "confident": state["confident"],
            "prompt_line": snap["prompt_line"], "style": snap["style"], "infer_ms": snap["last_infer_ms"],
        }
        if args.json:
            print(json.dumps(row))
        else:
            print(
                f"{path.name:40s} utt={row['utterances']} avd={row['avd']} z={row['z']} "
                f"→ {row['effort']}/{row['affect']}{'' if row['confident'] else ' (unsure)'} "
                f"style={tuple(row['style'].values())} {row['infer_ms']}ms"
            )
    stream.end_input()
    await drain
    await stream.aclose()
    print("baseline:", json.dumps(service.baseline.summary()))
    print("stats:", json.dumps(service.snapshot_dict()["stats"]))
    engine.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
