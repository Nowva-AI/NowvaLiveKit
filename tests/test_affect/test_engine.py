"""Tests for AffectEngine with a fake ONNX session, plus a real-model test that skips when no model is exported."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from affect.config import EngineConfig, load_affect_config
from affect.engine import AffectEngine, AffectResult
from affect.manifest import ModelManifest

from .conftest import EMBEDDING_DIM, SAMPLE_RATE, _FakeSession, make_tone

TOLERANCE = 1e-6


def _engine(session: _FakeSession, manifest: ModelManifest, **config_overrides) -> AffectEngine:
    config = EngineConfig(warmup=False, **config_overrides)
    engine = AffectEngine(Path("/nonexistent"), config, manifest=manifest, session=session)
    engine.initialize()
    return engine


class TestFakeSession:
    def test_shapes_and_fields(self, fake_session, fake_manifest, tone_2s) -> None:
        engine = _engine(fake_session, fake_manifest)
        result = engine.infer(tone_2s)
        assert isinstance(result, AffectResult)
        assert result.embedding.shape == (EMBEDDING_DIM,)
        assert 0.0 <= result.arousal <= 1.0
        assert result.n_samples == tone_2s.shape[0]
        assert result.provider == "CPUExecutionProvider"
        assert result.padded_to is None
        assert fake_session.calls[-1] == (1, tone_2s.shape[0])

    def test_avd_order_mapping(self, fake_session, fake_manifest, tone_5s) -> None:
        manifest = fake_manifest.model_copy(update={"avd_order": ["valence", "dominance", "arousal"]})
        engine = _engine(fake_session, manifest)
        result = engine.infer(tone_5s)
        raw = fake_session.run(["avd"], {"waveform": tone_5s[None, :]})[0][0]
        assert result.valence == pytest.approx(float(raw[0]), abs=TOLERANCE)
        assert result.arousal == pytest.approx(float(raw[2]), abs=TOLERANCE)

    def test_categorical_probs(self, fake_manifest, tone_2s) -> None:
        session = _FakeSession(categorical=3)
        manifest = fake_manifest.model_copy(update={"output_cat_logits": "cat_logits", "categorical_labels": ["a", "b", "c"]})
        engine = _engine(session, manifest)
        result = engine.infer(tone_2s)
        assert result.cat_probs is not None
        assert sum(result.cat_probs.values()) == pytest.approx(1.0, abs=1e-6)
        assert max(result.cat_probs, key=result.cat_probs.get) == "a"

    def test_io_validation_rejects_wrong_names(self, fake_session, fake_manifest) -> None:
        manifest = fake_manifest.model_copy(update={"input_name": "audio"})
        with pytest.raises(ValueError):
            _engine(fake_session, manifest)

    def test_static_buckets_tile_pad(self, fake_manifest, tone_2s) -> None:
        class _CoreMLSession(_FakeSession):
            def get_providers(self):
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        session = _CoreMLSession()
        engine = _engine(session, fake_manifest, static_shape_buckets_seconds=[2.0, 4.0, 8.0])
        assert engine.static_buckets
        wave = make_tone(3.0)
        result = engine.infer(wave)
        assert result.padded_to == 4 * SAMPLE_RATE
        assert session.calls[-1] == (1, 4 * SAMPLE_RATE)
        assert result.n_samples == wave.shape[0]

    def test_warmup_runs_each_bucket(self, fake_manifest) -> None:
        class _CoreMLSession(_FakeSession):
            def get_providers(self):
                return ["CoreMLExecutionProvider"]

        session = _CoreMLSession()
        config = EngineConfig(warmup=True, static_shape_buckets_seconds=[1.0, 2.0])
        engine = AffectEngine(Path("/nonexistent"), config, manifest=fake_manifest, session=session)
        engine.initialize()
        assert session.calls == [(1, SAMPLE_RATE), (1, 2 * SAMPLE_RATE)]

    def test_infer_async_uses_executor(self, fake_session, fake_manifest, tone_2s) -> None:
        engine = _engine(fake_session, fake_manifest)

        async def _run() -> AffectResult:
            return await engine.infer_async(tone_2s)

        result = asyncio.run(_run())
        assert result.n_samples == tone_2s.shape[0]

    def test_missing_model_raises(self, tmp_path: Path, fake_manifest) -> None:
        fake_manifest.save(tmp_path)
        engine = AffectEngine(tmp_path, EngineConfig(warmup=False, providers=["cpu"]))
        with pytest.raises(FileNotFoundError):
            engine.initialize()


class TestRealModel:
    def test_exported_model_runs(self) -> None:
        config = load_affect_config(env={})
        model_dir = config.resolve_model_dir()
        if not (model_dir / "model.onnx").exists():
            pytest.skip("no exported affect model at models/affect/current")
        engine = AffectEngine(model_dir, EngineConfig(warmup=False, providers=["cpu"]))
        engine.initialize()
        result = engine.infer(make_tone(2.0, freq_hz=180.0))
        assert result.embedding.shape[0] == engine.manifest.embedding_dim
        assert 0.0 <= result.arousal <= 1.0 and 0.0 <= result.valence <= 1.0
        assert result.infer_ms > 0.0


class TestProviderFallback:
    def test_failed_provider_falls_back_to_cpu(self, tmp_path: Path, fake_manifest, monkeypatch) -> None:
        import onnxruntime as ort

        fake_manifest.save(tmp_path)
        (tmp_path / "model.onnx").write_bytes(b"not a real model")
        attempts: list[list] = []

        class _Session(_FakeSession):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                providers = kwargs.get("providers") or []
                attempts.append(list(providers))
                if any((p if isinstance(p, str) else p[0]) == "CoreMLExecutionProvider" for p in providers):
                    raise RuntimeError("Failed to create MLModel: output size is too small")

        monkeypatch.setattr(ort, "InferenceSession", _Session)
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        engine = AffectEngine(tmp_path, EngineConfig(warmup=False, providers=["coreml", "cpu"]), manifest=fake_manifest)
        engine.initialize()
        assert len(attempts) == 2
        assert attempts[1] == ["CPUExecutionProvider"]
        assert engine.provider == "CPUExecutionProvider"
        assert engine.static_buckets is False

    def test_single_provider_failure_raises(self, tmp_path: Path, fake_manifest, monkeypatch) -> None:
        import onnxruntime as ort

        fake_manifest.save(tmp_path)
        (tmp_path / "model.onnx").write_bytes(b"x")

        def _boom(*args, **kwargs):
            raise RuntimeError("bad graph")

        monkeypatch.setattr(ort, "InferenceSession", _boom)
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        engine = AffectEngine(tmp_path, EngineConfig(warmup=False, providers=["cpu"]), manifest=fake_manifest)
        with pytest.raises(RuntimeError):
            engine.initialize()
