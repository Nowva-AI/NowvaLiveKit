"""AffectModel: encoder → layer selection → pooling → heads, with Manifold MixUp and an export-ready forward."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from training.affect.config import AffectModelConfig
from training.affect.models.encoder import SpeechEncoder
from training.affect.models.heads import AVDHead, CategoricalHead, ExertionHead
from training.affect.models.pooling import LayerSelector, TemporalPooling

INPUT_NORM_EPSILON = 1e-7


@dataclass
class MixupSpec:
    lam: float
    permutation: torch.Tensor
    layer_index: int


def normalize_waveform(wave: torch.Tensor) -> torch.Tensor:
    mean = wave.mean(dim=-1, keepdim=True)
    var = wave.var(dim=-1, keepdim=True, unbiased=False)
    return (wave - mean) / torch.sqrt(var + INPUT_NORM_EPSILON)


class AffectModel(nn.Module):
    def __init__(self, config: AffectModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SpeechEncoder(config.encoder)
        self.layer_selector = LayerSelector(config.pooling, self.encoder.num_hidden_states)
        self.pooling = TemporalPooling(config.pooling, self.encoder.hidden_size)
        self.embedding_dim = self.pooling.output_dim
        hidden = config.head_hidden
        self.cat_head = (
            CategoricalHead(self.embedding_dim, hidden, len(config.class_labels), config.dropout)
            if config.categorical
            else None
        )
        self.avd_head = AVDHead(self.embedding_dim, hidden, config.dropout) if config.avd else None
        self.exertion_head = ExertionHead(self.embedding_dim, hidden, config.dropout) if config.exertion else None
        self._mixup: MixupSpec | None = None
        self._mixup_handle = None

    # -- Manifold MixUp -----------------------------------------------------------------

    def _encoder_layers(self) -> list[nn.Module]:
        if self.encoder.is_whisper:
            return list(self.encoder.encoder.layers)
        return list(self.encoder.encoder.encoder.layers)

    def arm_mixup(self, lam: float, permutation: torch.Tensor, layer_range: tuple[int, int]) -> MixupSpec:
        layers = self._encoder_layers()
        low, high = layer_range
        high = min(high, len(layers) - 1)
        layer_index = random.randint(low, max(low, high))
        spec = MixupSpec(lam=lam, permutation=permutation, layer_index=layer_index)

        def _hook(module: nn.Module, inputs: tuple, output):
            hidden = output[0] if isinstance(output, tuple) else output
            mixed = spec.lam * hidden + (1.0 - spec.lam) * hidden[spec.permutation]
            if isinstance(output, tuple):
                return (mixed,) + tuple(output[1:])
            return mixed

        self.disarm_mixup()
        self._mixup_handle = layers[layer_index].register_forward_hook(_hook)
        self._mixup = spec
        return spec

    def disarm_mixup(self) -> None:
        if self._mixup_handle is not None:
            self._mixup_handle.remove()
            self._mixup_handle = None
        self._mixup = None

    # -- Forward -------------------------------------------------------------------------

    def embed(self, wave: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.config.normalize_input_in_graph:
            wave = normalize_waveform(wave)
        hidden_states, frame_mask = self.encoder(wave, lengths)
        selected = self.layer_selector(hidden_states)
        return self.pooling(selected, frame_mask)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embedding = self.embed(wave, lengths)
        outputs: dict[str, torch.Tensor] = {"embedding": embedding}
        if self.avd_head is not None:
            outputs["avd"] = self.avd_head(embedding)
        if self.cat_head is not None:
            outputs["cat_logits"] = self.cat_head(embedding)
        if self.exertion_head is not None:
            outputs["exertion_logit"] = self.exertion_head(embedding).squeeze(-1)
        return outputs

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        encoder_params = self.encoder.trainable_parameters()
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids]
        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups


class ExportWrapper(nn.Module):
    """Single-input graph for ONNX: waveform [1, T] → embedding, avd (+ cat_logits)."""

    def __init__(self, model: AffectModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor):
        outputs = self.model(waveform)
        result = [outputs["embedding"], outputs["avd"]]
        if "cat_logits" in outputs:
            result.append(outputs["cat_logits"])
        return tuple(result)
