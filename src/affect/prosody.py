"""Cheap prosody features in numpy: autocorrelation F0 over voiced frames, pause ratio, speech-rate proxy."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

FRAME_MS = 40.0
HOP_MS = 20.0
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
VOICING_THRESHOLD = 0.55
ENERGY_FLOOR_RATIO = 0.08
ENVELOPE_PEAK_RATIO = 0.5
MIN_PEAK_GAP_MS = 80.0


class ProsodyFeatures(BaseModel):
    f0_median_hz: float
    f0_iqr_hz: float
    voiced_ratio: float
    pause_ratio: float
    speech_rate_proxy: float
    rms_dbfs: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.f0_median_hz, self.f0_iqr_hz, self.voiced_ratio, self.pause_ratio, self.speech_rate_proxy, self.rms_dbfs],
            dtype=np.float32,
        )


def _frame(wave: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if wave.shape[0] < frame_len:
        return wave[None, :frame_len] if wave.shape[0] > 0 else np.zeros((0, frame_len), dtype=np.float32)
    n_frames = 1 + (wave.shape[0] - frame_len) // hop
    indices = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return wave[indices]


def _autocorrelation(frames: np.ndarray) -> np.ndarray:
    n = frames.shape[1]
    spectrum = np.fft.rfft(frames * np.hanning(n)[None, :], n=2 * n, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :n]
    return autocorr


def estimate_f0_track(wave: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame F0 in Hz (nan when unvoiced) and per-frame RMS."""
    frame_len = int(sample_rate * FRAME_MS / 1000.0)
    hop = int(sample_rate * HOP_MS / 1000.0)
    frames = _frame(wave.astype(np.float32, copy=False), frame_len, hop)
    if frames.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    autocorr = _autocorrelation(frames)
    lag_min = int(sample_rate / F0_MAX_HZ)
    lag_max = min(int(sample_rate / F0_MIN_HZ), frame_len - 1)
    energy = autocorr[:, 0:1] + 1e-9
    normalized = autocorr[:, lag_min : lag_max + 1] / energy
    best = np.argmax(normalized, axis=1)
    peak = normalized[np.arange(normalized.shape[0]), best]
    f0 = sample_rate / (best + lag_min).astype(np.float64)
    energy_floor = ENERGY_FLOOR_RATIO * float(np.max(rms)) if rms.size else 0.0
    voiced = (peak >= VOICING_THRESHOLD) & (rms >= energy_floor)
    f0 = np.where(voiced, f0, np.nan)
    return f0, rms


def _speech_rate_proxy(rms: np.ndarray, sample_rate: int) -> float:
    if rms.size < 3:
        return 0.0
    kernel = np.ones(3) / 3.0
    envelope = np.convolve(rms, kernel, mode="same")
    threshold = ENVELOPE_PEAK_RATIO * float(np.max(envelope))
    min_gap = max(1, int(MIN_PEAK_GAP_MS / HOP_MS))
    peaks = 0
    last_peak = -min_gap
    for i in range(1, envelope.shape[0] - 1):
        if envelope[i] >= threshold and envelope[i] >= envelope[i - 1] and envelope[i] >= envelope[i + 1]:
            if i - last_peak >= min_gap:
                peaks += 1
                last_peak = i
    seconds = envelope.shape[0] * HOP_MS / 1000.0
    return float(peaks / seconds) if seconds > 0 else 0.0


def prosody_features(
    wave: np.ndarray,
    sample_rate: int,
    speech_probabilities: np.ndarray | None = None,
    speech_threshold: float = 0.5,
) -> ProsodyFeatures:
    f0, rms = estimate_f0_track(wave, sample_rate)
    voiced = f0[~np.isnan(f0)]
    if voiced.size:
        f0_median = float(np.median(voiced))
        f0_iqr = float(np.percentile(voiced, 75) - np.percentile(voiced, 25))
    else:
        f0_median = 0.0
        f0_iqr = 0.0
    voiced_ratio = float(voiced.size / f0.size) if f0.size else 0.0
    if speech_probabilities is not None and speech_probabilities.size:
        pause_ratio = float(np.mean(speech_probabilities < speech_threshold))
    elif rms.size:
        pause_ratio = float(np.mean(rms < ENERGY_FLOOR_RATIO * float(np.max(rms))))
    else:
        pause_ratio = 0.0
    overall_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) if wave.size else 0.0
    return ProsodyFeatures(
        f0_median_hz=f0_median,
        f0_iqr_hz=f0_iqr,
        voiced_ratio=voiced_ratio,
        pause_ratio=pause_ratio,
        speech_rate_proxy=_speech_rate_proxy(rms, sample_rate),
        rms_dbfs=float(20.0 * np.log10(max(overall_rms, 1e-9))),
    )
