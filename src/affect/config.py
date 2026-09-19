"""Affect perception configuration: pydantic models loaded from config/affect.yaml with env overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "affect.yaml"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "affect" / "current"

ENV_ENABLED = "AFFECT_ENABLED"
ENV_MODEL_DIR = "AFFECT_MODEL_DIR"
ENV_RECORD = "AFFECT_RECORD"
ENV_LLM_WAIT_MS = "AFFECT_LLM_WAIT_MS"
ENV_STYLE_ADAPTER = "AFFECT_STYLE_ADAPTER"
ENV_INJECT_AS = "AFFECT_INJECT_AS"
ENV_PROVIDERS = "AFFECT_PROVIDERS"

InjectMode = Literal["system", "user_prefix"]
AdapterName = Literal["cartesia_inline", "cartesia_extra", "qwen3", "none"]
ProviderName = Literal["tensorrt", "cuda", "coreml", "cpu"]


class AudioConfig(BaseModel):
    model_sample_rate: int = 16000
    min_voiced_seconds: float = 1.0
    max_seconds: float = 8.0
    trim_pad_seconds: float = 0.1
    speech_prob_threshold: float = 0.5


class TriggerConfig(BaseModel):
    early_silence_seconds: float = 0.25
    early_min_voiced_seconds: float = 1.0
    llm_wait_ms: int = 40
    coaching_wait_ms: int = 0
    inject_as: InjectMode = "system"


class EngineConfig(BaseModel):
    providers: list[ProviderName] = Field(default_factory=lambda: ["tensorrt", "cuda", "coreml", "cpu"])
    intra_op_threads: int = 4
    static_shape_buckets_seconds: list[float] = Field(default_factory=lambda: [2.0, 4.0, 8.0])
    coreml_static_buckets: bool = True
    warmup: bool = True
    trt_engine_cache_dir: str = "models/affect/trt_cache"
    trt_fp16: bool = True


class BaselineConfig(BaseModel):
    enrollment_target_seconds: float = 45.0
    enrollment_floor_seconds: float = 30.0
    min_samples_before_gate: int = 5
    neutral_z_max: float = 2.5
    shrinkage_prior_n: float = 10.0
    cross_session_ema_alpha: float = 0.2
    min_std_ratio: float = 0.5
    max_samples: int = 400
    require_ready_for_confidence: bool = True
    profile_dir: str = "data/affect/profiles"


class StateConfig(BaseModel):
    window_utterances: int = 3
    window_seconds: float = 120.0
    enter_z: float = 1.5
    enter_z_single: float = 2.5
    exit_z: float = 1.0
    min_dwell_utterances: int = 2
    effort_working_ratio: float = 1.15
    effort_near_limit_ratio: float = 1.35


class StyleConfig(BaseModel):
    adapter: AdapterName = "cartesia_inline"
    speed_min: float = 0.9
    speed_max: float = 1.1
    volume_min: float = 0.9
    volume_max: float = 1.15
    pace_step: float = 0.1
    energy_volume_step: float = 0.075


class RecorderConfig(BaseModel):
    enabled: bool = False
    output_dir: str = "data/affect/recordings"


class AffectConfig(BaseModel):
    enabled: bool = True
    model_dir: str = str(DEFAULT_MODEL_DIR)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)

    def resolve_model_dir(self) -> Path:
        path = Path(self.model_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _env_flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _apply_env_overrides(config: AffectConfig, env: Mapping[str, str]) -> AffectConfig:
    updates: dict[str, Any] = {}
    if ENV_ENABLED in env:
        updates["enabled"] = _env_flag(env[ENV_ENABLED])
    if ENV_MODEL_DIR in env and env[ENV_MODEL_DIR]:
        updates["model_dir"] = env[ENV_MODEL_DIR]
    if ENV_RECORD in env:
        updates["recorder"] = config.recorder.model_copy(update={"enabled": _env_flag(env[ENV_RECORD])})
    if ENV_LLM_WAIT_MS in env:
        updates["trigger"] = config.trigger.model_copy(update={"llm_wait_ms": int(env[ENV_LLM_WAIT_MS])})
    if ENV_INJECT_AS in env and env[ENV_INJECT_AS]:
        trigger = updates.get("trigger", config.trigger)
        updates["trigger"] = trigger.model_copy(update={"inject_as": env[ENV_INJECT_AS]})
    if ENV_STYLE_ADAPTER in env and env[ENV_STYLE_ADAPTER]:
        updates["style"] = config.style.model_copy(update={"adapter": env[ENV_STYLE_ADAPTER]})
    if ENV_PROVIDERS in env and env[ENV_PROVIDERS]:
        providers = [p.strip() for p in env[ENV_PROVIDERS].split(",") if p.strip()]
        updates["engine"] = config.engine.model_copy(update={"providers": providers})
    if not updates:
        return config
    return AffectConfig.model_validate({**config.model_dump(), **{k: (v.model_dump() if isinstance(v, BaseModel) else v) for k, v in updates.items()}})


def load_affect_config(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> AffectConfig:
    """Load config/affect.yaml (or defaults when absent) and apply AFFECT_* env overrides."""
    config_path = path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Affect config at {config_path} must be a mapping")
        raw = loaded
    config = AffectConfig.model_validate(raw)
    return _apply_env_overrides(config, os.environ if env is None else env)
