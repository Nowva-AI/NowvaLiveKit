"""Tests for affect config loading and env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from affect.config import (
    DEFAULT_CONFIG_PATH,
    AffectConfig,
    load_affect_config,
)


class TestDefaults:
    def test_defaults_construct(self) -> None:
        config = AffectConfig()
        assert config.enabled is True
        assert config.audio.model_sample_rate == 16000
        assert config.state.enter_z > config.state.exit_z
        assert config.style.adapter == "cartesia_inline"

    def test_repo_yaml_matches_schema(self) -> None:
        assert DEFAULT_CONFIG_PATH.exists()
        config = load_affect_config(DEFAULT_CONFIG_PATH, env={})
        assert config.trigger.llm_wait_ms == 40
        assert config.engine.providers[-1] == "cpu"

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        config = load_affect_config(tmp_path / "nope.yaml", env={})
        assert config == AffectConfig()

    def test_non_mapping_yaml_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a list\n")
        with pytest.raises(ValueError):
            load_affect_config(path, env={})


class TestEnvOverrides:
    def test_enabled_flag(self) -> None:
        config = load_affect_config(DEFAULT_CONFIG_PATH, env={"AFFECT_ENABLED": "0"})
        assert config.enabled is False

    def test_model_dir_and_wait(self) -> None:
        config = load_affect_config(
            DEFAULT_CONFIG_PATH,
            env={"AFFECT_MODEL_DIR": "/tmp/m", "AFFECT_LLM_WAIT_MS": "15"},
        )
        assert config.model_dir == "/tmp/m"
        assert config.resolve_model_dir() == Path("/tmp/m")
        assert config.trigger.llm_wait_ms == 15

    def test_record_inject_adapter_providers(self) -> None:
        config = load_affect_config(
            DEFAULT_CONFIG_PATH,
            env={
                "AFFECT_RECORD": "1",
                "AFFECT_INJECT_AS": "user_prefix",
                "AFFECT_STYLE_ADAPTER": "none",
                "AFFECT_PROVIDERS": "cpu",
            },
        )
        assert config.recorder.enabled is True
        assert config.trigger.inject_as == "user_prefix"
        assert config.style.adapter == "none"
        assert config.engine.providers == ["cpu"]

    def test_invalid_adapter_rejected(self) -> None:
        with pytest.raises(ValueError):
            load_affect_config(DEFAULT_CONFIG_PATH, env={"AFFECT_STYLE_ADAPTER": "bogus"})

    def test_relative_paths_resolve_under_project_root(self) -> None:
        config = AffectConfig()
        assert config.resolve_model_dir().is_absolute()
        assert config.resolve_path("data/affect/profiles").name == "profiles"
