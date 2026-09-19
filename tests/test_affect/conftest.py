"""Shared fixtures for affect runtime tests: synthetic audio and a fake ONNX session."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from affect.config import AffectConfig  # noqa: E402
from affect.manifest import ModelManifest  # noqa: E402

SAMPLE_RATE = 16000
EMBEDDING_DIM = 8
FLOAT_TOLERANCE = 1e-6


class _FakeIO:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession: deterministic outputs from the input length."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, categorical: int = 0) -> None:
        self.embedding_dim = embedding_dim
        self.categorical = categorical
        self.calls: list[tuple[int, ...]] = []
        self._outputs = ["embedding", "avd"] + (["cat_logits"] if categorical else [])

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO("waveform", [1, "time"])]

    def get_outputs(self) -> list[_FakeIO]:
        return [_FakeIO(name, [1, None]) for name in self._outputs]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names, feeds):
        wave = feeds["waveform"]
        self.calls.append(tuple(wave.shape))
        n_samples = wave.shape[1]
        rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)) + 1e-9)
        embedding = np.full((1, self.embedding_dim), rms, dtype=np.float32)
        arousal = min(1.0, rms * 4.0)
        avd = np.array([[arousal, 0.5, 0.5 + 0.1 * np.sign(n_samples - SAMPLE_RATE * 3)]], dtype=np.float32)
        outputs = [embedding, avd]
        if self.categorical:
            logits = np.zeros((1, self.categorical), dtype=np.float32)
            logits[0, 0] = 1.0
            outputs.append(logits)
        return [outputs[self._outputs.index(name)] for name in output_names]


def make_tone(seconds: float, freq_hz: float = 140.0, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_noise(seconds: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(seconds * sample_rate))).astype(np.float32)


@pytest.fixture
def affect_config() -> AffectConfig:
    return AffectConfig()


@pytest.fixture
def fake_session() -> _FakeSession:
    return _FakeSession()


@pytest.fixture
def fake_manifest() -> ModelManifest:
    return ModelManifest(
        version="test-0",
        family="fake",
        embedding_dim=EMBEDDING_DIM,
        population_avd_mean=[0.5, 0.5, 0.5],
        population_avd_std=[0.15, 0.15, 0.15],
    )


@pytest.fixture
def tone_2s() -> np.ndarray:
    return make_tone(2.0)


@pytest.fixture
def tone_5s() -> np.ndarray:
    return make_tone(5.0)
