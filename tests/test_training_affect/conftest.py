"""Fixtures for the training package: a tiny random WavLM saved locally so nothing is downloaded."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

SAMPLE_RATE = 16000
NUM_CLASSES = 8


@pytest.fixture(scope="session")
def tiny_wavlm_dir(tmp_path_factory) -> Path:
    from transformers import WavLMConfig, WavLMModel

    config = WavLMConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        vocab_size=32,
        num_buckets=32,
        max_bucket_distance=80,
    )
    torch.manual_seed(0)
    model = WavLMModel(config)
    path = tmp_path_factory.mktemp("tiny_wavlm")
    model.save_pretrained(str(path))
    return path


def write_wav(path: Path, seconds: float, freq_hz: float, sample_rate: int = SAMPLE_RATE) -> None:
    import wave

    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = (0.3 * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(signal.tobytes())


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> Path:
    from training.affect.data.manifest import Segment, write_manifest

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    segments = []
    rng = np.random.default_rng(0)
    for index in range(12):
        wav = audio_dir / f"seg{index}.wav"
        write_wav(wav, 1.5 + 0.25 * (index % 4), 120.0 + 20 * index)
        probs = rng.dirichlet(np.ones(NUM_CLASSES))
        segments.append(Segment(
            id=f"syn{index}", path=str(wav), source="synthetic", split="train" if index < 8 else "dev",
            speaker=f"spk{index % 3}", seconds=1.5 + 0.25 * (index % 4),
            class_probs=[float(p) for p in probs], label=["angry", "sad", "happy", "surprise", "fear", "disgust", "contempt", "neutral"][int(probs.argmax())],
            avd=[float(v) for v in rng.uniform(0.2, 0.8, 3)], exertion=float(index % 2),
        ))
    path = tmp_path / "synthetic.jsonl"
    write_manifest(segments, path)
    return path
