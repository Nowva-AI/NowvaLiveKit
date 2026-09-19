"""Experiment configuration schema for affect training: one YAML per experiment, every knob explicit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

EncoderName = Literal["wavlm", "hubert", "wav2vec2", "whisper"]
PoolingName = Literal["mean_std", "attentive_stats"]
LayerSelect = Literal["weighted", "last", "mean_last4"]
ClassWeighting = Literal["none", "inverse_freq"]

MSP_PODCAST_CLASSES = ["angry", "sad", "happy", "surprise", "fear", "disgust", "contempt", "neutral"]


class EncoderConfig(BaseModel):
    name: EncoderName = "wavlm"
    pretrained: str = "microsoft/wavlm-base-plus"
    freeze_feature_extractor: bool = True
    freeze_bottom_layers: int = 0
    freeze_all: bool = False
    gradient_checkpointing: bool = False
    whisper_max_seconds: float = 30.0


class PoolingConfig(BaseModel):
    layer_select: LayerSelect = "weighted"
    pooling: PoolingName = "mean_std"
    attention_hidden: int = 128


class AffectModelConfig(BaseModel):
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    pooling: PoolingConfig = Field(default_factory=PoolingConfig)
    class_labels: list[str] = Field(default_factory=lambda: list(MSP_PODCAST_CLASSES))
    categorical: bool = True
    avd: bool = True
    exertion: bool = False
    head_hidden: int = 256
    dropout: float = 0.2
    normalize_input_in_graph: bool = True


class DataMixEntry(BaseModel):
    manifest: str
    weight: float = 1.0
    use_categorical: bool = True
    use_avd: bool = True
    use_exertion: bool = False


class AugmentConfig(BaseModel):
    enabled: bool = True
    apm_prob: float = 0.5
    apm_warmup_seconds: float = 1.5
    rir_dir: str | None = None
    rir_prob: float = 0.3
    noise_dir: str | None = None
    noise_prob: float = 0.3
    noise_snr_db: tuple[float, float] = (5.0, 25.0)
    gain_db: tuple[float, float] = (-6.0, 6.0)
    speed_perturb: list[float] = Field(default_factory=lambda: [0.9, 1.0, 1.1])
    speed_prob: float = 0.3


class DataConfig(BaseModel):
    train: list[DataMixEntry] = Field(default_factory=list)
    dev: list[DataMixEntry] = Field(default_factory=list)
    test: list[DataMixEntry] = Field(default_factory=list)
    sample_rate: int = 16000
    crop_seconds: float = 10.0
    min_seconds: float = 1.0
    keep_other_class: bool = True
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    num_workers: int = 4


class LossConfig(BaseModel):
    w_cat: float = 1.5
    w_avd: float = 0.4
    w_exertion: float = 1.0
    class_weighting: ClassWeighting = "inverse_freq"
    focal_gamma: float = 0.0
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.3
    mixup_layers: tuple[int, int] = (0, 0)
    label_smoothing: float = 0.0


class OptimConfig(BaseModel):
    encoder_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    epochs_stage1: int = 15
    epochs_stage2: int = 5
    batch_size: int = 16
    grad_accum: int = 2
    bf16: bool = True
    max_grad_norm: float = 1.0
    early_stop_patience: int = 4
    select_metric: str = "dev/avg_ccc"


class DistillConfig(BaseModel):
    teacher_cache: str
    w_ccc: float = 1.0
    w_quadrant: float = 0.5
    w_cat_kl: float = 1.0
    w_embedding_cos: float = 0.0


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    model: AffectModelConfig = Field(default_factory=AffectModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    distill: DistillConfig | None = None
    output_dir: str = "experiments"
    log_every: int = 25
    eval_every_epochs: int = 1
    max_train_steps: int | None = None
    notes: str = ""

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name
