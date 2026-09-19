"""Audio preparation for the affect encoder: int16→float, streaming resample, speech-span trim, crop, tiling."""

from __future__ import annotations

import numpy as np
import soxr

INT16_SCALE = 32768.0
RESAMPLE_QUALITY = "HQ"


def int16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return (pcm.astype(np.float32) / INT16_SCALE).astype(np.float32)


def resample(wave: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return wave.astype(np.float32, copy=False)
    return soxr.resample(wave.astype(np.float32, copy=False), source_rate, target_rate, quality=RESAMPLE_QUALITY).astype(np.float32)


class StreamingResampler:
    """Chunk-wise resampler so a live 24 kHz frame stream becomes a 16 kHz buffer without seams."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._stream = soxr.ResampleStream(source_rate, target_rate, 1, dtype="float32", quality=RESAMPLE_QUALITY)

    def push(self, chunk: np.ndarray, last: bool = False) -> np.ndarray:
        if self.source_rate == self.target_rate:
            return chunk.astype(np.float32, copy=False)
        return self._stream.resample_chunk(chunk.astype(np.float32, copy=False), last=last)

    def reset(self) -> None:
        self._stream = soxr.ResampleStream(self.source_rate, self.target_rate, 1, dtype="float32", quality=RESAMPLE_QUALITY)


def speech_span(
    probabilities: np.ndarray,
    window_seconds: float,
    threshold: float,
) -> tuple[float, float] | None:
    """First and last time (seconds) where the VAD probability reaches the threshold."""
    if probabilities.size == 0:
        return None
    voiced = np.flatnonzero(probabilities >= threshold)
    if voiced.size == 0:
        return None
    start_s = float(voiced[0]) * window_seconds
    end_s = float(voiced[-1] + 1) * window_seconds
    return start_s, end_s


def voiced_seconds(probabilities: np.ndarray, window_seconds: float, threshold: float) -> float:
    if probabilities.size == 0:
        return 0.0
    return float(np.count_nonzero(probabilities >= threshold)) * window_seconds


def trim_to_speech(
    wave: np.ndarray,
    sample_rate: int,
    probabilities: np.ndarray | None,
    window_seconds: float,
    threshold: float,
    pad_seconds: float,
) -> np.ndarray:
    """Cut the waveform to the VAD speech span plus padding; returns the input when probabilities are absent."""
    if probabilities is None or probabilities.size == 0:
        return wave
    span = speech_span(probabilities, window_seconds, threshold)
    if span is None:
        return wave[:0]
    start_s, end_s = span
    start = max(0, int((start_s - pad_seconds) * sample_rate))
    end = min(wave.shape[0], int((end_s + pad_seconds) * sample_rate))
    if end <= start:
        return wave[:0]
    return wave[start:end]


def crop_last_seconds(wave: np.ndarray, sample_rate: int, max_seconds: float) -> np.ndarray:
    max_samples = int(max_seconds * sample_rate)
    if wave.shape[0] <= max_samples:
        return wave
    return wave[-max_samples:]


def tile_pad(wave: np.ndarray, target_samples: int) -> np.ndarray:
    """Repeat the clip to fill a static bucket; never zero-pads (the encoder normalizes over the clip)."""
    if wave.shape[0] == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if wave.shape[0] >= target_samples:
        return wave[:target_samples]
    repeats = int(np.ceil(target_samples / wave.shape[0]))
    return np.tile(wave, repeats)[:target_samples].astype(np.float32, copy=False)


def bucket_samples(n_samples: int, sample_rate: int, buckets_seconds: list[float]) -> int:
    """Smallest static bucket that holds the clip; the largest bucket when the clip exceeds them all."""
    sizes = sorted(int(round(b * sample_rate)) for b in buckets_seconds)
    for size in sizes:
        if n_samples <= size:
            return size
    return sizes[-1]


def rms_dbfs(wave: np.ndarray) -> float:
    if wave.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    return float(20.0 * np.log10(max(rms, 1e-9)))
