"""Model manifest: the sidecar model.json that describes an exported affect ONNX graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MANIFEST_FILENAME = "model.json"
MODEL_FILENAME = "model.onnx"
SPEAKER_HEADS_FILENAME = "heads_spknorm.npz"

AVD_DIMENSIONS = ("arousal", "dominance", "valence")


class ModelManifest(BaseModel):
    version: str
    family: str
    sample_rate: int = 16000
    min_seconds: float = 1.0
    max_seconds: float = 8.0
    input_name: str = "waveform"
    output_embedding: str = "embedding"
    output_avd: str = "avd"
    output_cat_logits: str | None = None
    avd_order: list[str] = Field(default_factory=lambda: list(AVD_DIMENSIONS))
    avd_range: tuple[float, float] = (0.0, 1.0)
    categorical_labels: list[str] = Field(default_factory=list)
    embedding_dim: int
    input_normalized_in_graph: bool = True
    population_avd_mean: list[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    population_avd_std: list[float] = Field(default_factory=lambda: [0.15, 0.15, 0.15])
    eval_metrics: dict[str, float] = Field(default_factory=dict)
    source: str = ""
    notes: str = ""

    @classmethod
    def load(cls, model_dir: Path) -> ModelManifest:
        path = model_dir / MANIFEST_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Affect model manifest not found: {path}")
        return cls.model_validate(json.loads(path.read_text()))

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / MANIFEST_FILENAME
        path.write_text(json.dumps(self.model_dump(), indent=2) + "\n")
        return path

    def model_path(self, model_dir: Path) -> Path:
        return model_dir / MODEL_FILENAME

    def expected_outputs(self) -> list[str]:
        outputs = [self.output_embedding, self.output_avd]
        if self.output_cat_logits:
            outputs.append(self.output_cat_logits)
        return outputs

    def validate_session_io(self, input_names: list[str], output_names: list[str]) -> None:
        if self.input_name not in input_names:
            raise ValueError(
                f"ONNX input {self.input_name!r} missing; graph inputs are {input_names}"
            )
        missing = [name for name in self.expected_outputs() if name not in output_names]
        if missing:
            raise ValueError(f"ONNX outputs {missing} missing; graph outputs are {output_names}")
        if len(self.avd_order) != 3 or set(self.avd_order) != set(AVD_DIMENSIONS):
            raise ValueError(f"avd_order must be a permutation of {AVD_DIMENSIONS}, got {self.avd_order}")

    def avd_index(self, dimension: str) -> int:
        return self.avd_order.index(dimension)

    def to_summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "family": self.family,
            "embedding_dim": self.embedding_dim,
            "categorical": bool(self.output_cat_logits),
            "eval_metrics": dict(self.eval_metrics),
        }
