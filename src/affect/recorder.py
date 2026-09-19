"""Opt-in utterance recorder: 16 kHz WAV after the audio-processing chain plus a jsonl row of outputs and context."""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from affect.config import RecorderConfig

INT16_MAX = 32767.0
META_FILENAME = "utterances.jsonl"


class UtteranceRecorder:
    def __init__(self, config: RecorderConfig, root: Path, user_id: str | None, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self._dir = root / (user_id or "anonymous") / session_id
        self._count = 0

    @property
    def directory(self) -> Path:
        return self._dir

    def record(self, wave16k: np.ndarray, sample_rate: int, meta: dict[str, Any]) -> Path | None:
        if not self.config.enabled:
            return None
        self._dir.mkdir(parents=True, exist_ok=True)
        self._count += 1
        stem = f"utt_{self._count:04d}_{int(time.time())}"
        wav_path = self._dir / f"{stem}.wav"
        pcm = np.clip(wave16k.astype(np.float64) * INT16_MAX, -INT16_MAX, INT16_MAX).astype(np.int16)
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        row = {"file": wav_path.name, "sample_rate": sample_rate, "seconds": round(len(pcm) / sample_rate, 3), **meta}
        with (self._dir / META_FILENAME).open("a") as handle:
            handle.write(json.dumps(row, default=_json_default) + "\n")
        return wav_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def read_recording(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / (INT16_MAX + 1.0)).astype(np.float32), sample_rate
