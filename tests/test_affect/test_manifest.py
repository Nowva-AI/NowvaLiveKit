"""Tests for the affect model manifest sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from affect.manifest import MANIFEST_FILENAME, ModelManifest


def _manifest(**overrides) -> ModelManifest:
    base = dict(version="v0", family="fake", embedding_dim=16)
    base.update(overrides)
    return ModelManifest(**base)


class TestRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        manifest = _manifest(categorical_labels=["neutral", "happy"], output_cat_logits="cat_logits")
        manifest.save(tmp_path)
        assert (tmp_path / MANIFEST_FILENAME).exists()
        loaded = ModelManifest.load(tmp_path)
        assert loaded == manifest

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ModelManifest.load(tmp_path)


class TestIOValidation:
    def test_valid_io_passes(self) -> None:
        _manifest().validate_session_io(["waveform"], ["embedding", "avd"])

    def test_missing_input_fails(self) -> None:
        with pytest.raises(ValueError):
            _manifest().validate_session_io(["audio"], ["embedding", "avd"])

    def test_missing_categorical_output_fails(self) -> None:
        manifest = _manifest(output_cat_logits="cat_logits")
        with pytest.raises(ValueError):
            manifest.validate_session_io(["waveform"], ["embedding", "avd"])

    def test_bad_avd_order_fails(self) -> None:
        manifest = _manifest(avd_order=["arousal", "arousal", "valence"])
        with pytest.raises(ValueError):
            manifest.validate_session_io(["waveform"], ["embedding", "avd"])

    def test_avd_index_follows_order(self) -> None:
        manifest = _manifest(avd_order=["valence", "arousal", "dominance"])
        assert manifest.avd_index("arousal") == 1
        assert manifest.avd_index("valence") == 0
