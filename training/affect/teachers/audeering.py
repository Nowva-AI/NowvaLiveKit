"""audEERING wav2vec2-large-robust-12 dimensional emotion model: loading, inference, and the V1a ONNX export."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from training.affect.models.affect_model import normalize_waveform

logger = logging.getLogger(__name__)

AUDEERING_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
# A local copy of config.json + model.safetensors (e.g. models/affect/_hf/audeering) is used when
# present: the hub client's transfer path has been seen to stall on some networks; curl does not.
AUDEERING_LOCAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "affect" / "_hf" / "audeering"
AUDEERING_AVD_ORDER = ["arousal", "dominance", "valence"]
AUDEERING_SAMPLE_RATE = 16000
# Reported on MSP-Podcast test-1 by the model card (approximate; verify in RESULTS.md).
AUDEERING_REPORTED_CCC = {"arousal": 0.745, "dominance": 0.655, "valence": 0.638}


class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, final_dropout: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(final_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class AudeeringEmotionModel(nn.Module):
    """wav2vec2 encoder + mean pooling + regression head, matching the published w2v2-how-to definition."""

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        if model_id is None:
            model_id = str(AUDEERING_LOCAL_DIR) if (AUDEERING_LOCAL_DIR / "model.safetensors").exists() else AUDEERING_MODEL_ID
        config = Wav2Vec2Config.from_pretrained(model_id)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config.hidden_size, config.num_labels, config.final_dropout)
        self.config = config
        self._load_weights(model_id)

    def _load_weights(self, model_id: str) -> None:
        from safetensors.torch import load_file

        local = Path(model_id) / "model.safetensors"
        if local.exists():
            path = str(local)
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_id, "model.safetensors")
        state = load_file(path)
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("wav2vec2.masked_spec_embed")]
        missing = [k for k in missing if "masked_spec_embed" not in k]
        if unexpected:
            logger.warning("audEERING load: unexpected keys %s", unexpected[:5])
        if missing:
            raise RuntimeError(f"audEERING load: missing keys {missing[:10]}")

    def forward(self, normalized_wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.wav2vec2(normalized_wave).last_hidden_state
        pooled = hidden.mean(dim=1)
        return pooled, self.classifier(pooled)


class AudeeringExportWrapper(nn.Module):
    """waveform [1, T] (raw float in [-1, 1]) → embedding [1, 1024], avd [1, 3] with normalization in-graph."""

    def __init__(self, model: AudeeringEmotionModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor):
        pooled, logits = self.model(normalize_waveform(waveform))
        return pooled, logits


@torch.no_grad()
def predict_avd(model: AudeeringEmotionModel, wave: torch.Tensor) -> torch.Tensor:
    """wave: [B, T] raw float; returns [B, 3] arousal, dominance, valence in about [0, 1]."""
    model.eval()
    _, logits = model(normalize_waveform(wave))
    return logits


def export_audeering(out_dir: Path, promote: bool = False) -> Path:
    from affect.manifest import ModelManifest
    from training.affect.export import (
        check_parity,
        export_onnx,
        finalize_model_dir,
        padding_sensitivity,
        promote_to_current,
    )

    model = AudeeringEmotionModel()
    wrapper = AudeeringExportWrapper(model).eval()
    output_names = ["embedding", "avd"]
    onnx_path = export_onnx(wrapper, out_dir, output_names, AUDEERING_SAMPLE_RATE)
    parity = check_parity(wrapper, onnx_path, output_names, AUDEERING_SAMPLE_RATE)
    parity["padding_avd_delta_2s"] = padding_sensitivity(wrapper, AUDEERING_SAMPLE_RATE)
    manifest = ModelManifest(
        version=f"bootstrap-audeering-{out_dir.name}",
        family="wav2vec2-large-robust-12-avd",
        sample_rate=AUDEERING_SAMPLE_RATE,
        min_seconds=1.0,
        max_seconds=8.0,
        avd_order=list(AUDEERING_AVD_ORDER),
        avd_range=(0.0, 1.0),
        embedding_dim=int(model.config.hidden_size),
        input_normalized_in_graph=True,
        population_avd_mean=[0.5, 0.5, 0.5],
        population_avd_std=[0.15, 0.15, 0.15],
        source=AUDEERING_MODEL_ID,
        notes=(
            "V1a bootstrap. Dimensional only (no categorical head). Trained by audEERING on "
            "MSP-Podcast v1.7 train, so MSP-Podcast numbers are contaminated. Population norms "
            "are placeholders until measured on Nowva recordings."
        ),
    )
    finalize_model_dir(out_dir, manifest, parity)
    logger.info("audEERING export complete: %s parity=%s", out_dir, parity)
    if promote:
        promote_to_current(out_dir, out_dir.parent)
    return out_dir
