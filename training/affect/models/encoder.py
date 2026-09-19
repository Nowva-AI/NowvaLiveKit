"""Pretrained speech encoders exposing all hidden layers, with partial freezing and Whisper position slicing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.affect.config import EncoderConfig

WHISPER_HOP = 160
WHISPER_N_FFT = 400
WHISPER_N_MELS = 80
WHISPER_CONV_STRIDE = 2


def _whisper_mel_filters(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    from transformers.audio_utils import mel_filter_bank

    filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    return torch.tensor(filters, dtype=torch.float32)


class WhisperLogMel(nn.Module):
    """Torch re-implementation of WhisperFeatureExtractor so the front-end exports to ONNX."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = WHISPER_N_MELS) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(WHISPER_N_FFT), persistent=False)
        self.register_buffer("filters", _whisper_mel_filters(sample_rate, WHISPER_N_FFT, n_mels), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wave, WHISPER_N_FFT, WHISPER_HOP, window=self.window, return_complex=True)
        magnitudes = spec[..., :-1].abs() ** 2
        mel = self.filters.T @ magnitudes
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0


class SpeechEncoder(nn.Module):
    """Wraps a HF WavLM / HuBERT / wav2vec2 model or a Whisper encoder; returns every layer's hidden states."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.is_whisper = config.name == "whisper"
        if self.is_whisper:
            from transformers import WhisperModel

            whisper = WhisperModel.from_pretrained(config.pretrained)
            self.encoder = whisper.encoder
            self.frontend = WhisperLogMel()
            self.hidden_size = int(self.encoder.config.d_model)
            self.num_layers = len(self.encoder.layers)
            self.feat_extract_norm = "layer"
        else:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(config.pretrained)
            self.hidden_size = int(self.encoder.config.hidden_size)
            self.num_layers = int(self.encoder.config.num_hidden_layers)
            self.feat_extract_norm = str(getattr(self.encoder.config, "feat_extract_norm", "group"))
        if config.gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        self._apply_freezing()

    @property
    def num_hidden_states(self) -> int:
        return self.num_layers + 1

    def _apply_freezing(self) -> None:
        cfg = self.config
        if cfg.freeze_all:
            for param in self.encoder.parameters():
                param.requires_grad = False
            return
        if self.is_whisper:
            if cfg.freeze_feature_extractor:
                for module in (self.encoder.conv1, self.encoder.conv2):
                    for param in module.parameters():
                        param.requires_grad = False
            for index, layer in enumerate(self.encoder.layers):
                if index < cfg.freeze_bottom_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            return
        if cfg.freeze_feature_extractor:
            for name, param in self.encoder.named_parameters():
                if name.startswith("feature_extractor") or name.startswith("feature_projection"):
                    param.requires_grad = False
        layers = self.encoder.encoder.layers
        for index, layer in enumerate(layers):
            if index < cfg.freeze_bottom_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        if cfg.freeze_bottom_layers > 0 and hasattr(self.encoder.encoder, "pos_conv_embed"):
            for param in self.encoder.encoder.pos_conv_embed.parameters():
                param.requires_grad = False

    def frame_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        if self.is_whisper:
            mel_frames = torch.div(sample_lengths, WHISPER_HOP, rounding_mode="floor")
            return torch.div(mel_frames + 1, WHISPER_CONV_STRIDE, rounding_mode="floor")
        return self.encoder._get_feat_extract_output_lengths(sample_lengths)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch, n_samples = wave.shape
        if lengths is None:
            lengths = torch.full((batch,), n_samples, dtype=torch.long, device=wave.device)
        if self.is_whisper:
            hidden_states = self._forward_whisper(wave)
        else:
            attention_mask = None
            if self.feat_extract_norm == "layer":
                attention_mask = (torch.arange(n_samples, device=wave.device)[None, :] < lengths[:, None]).long()
            outputs = self.encoder(wave, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)
        n_frames = hidden_states[-1].shape[1]
        frame_lengths = torch.clamp(self.frame_lengths(lengths), min=1, max=n_frames)
        frame_mask = torch.arange(n_frames, device=wave.device)[None, :] < frame_lengths[:, None]
        return hidden_states, frame_mask

    def _forward_whisper(self, wave: torch.Tensor) -> list[torch.Tensor]:
        # Mirrors WhisperEncoder.forward but slices the positional table to the real
        # frame count, so short utterances are not padded to 30 s.
        encoder = self.encoder
        features = self.frontend(wave)
        x = F.gelu(encoder.conv1(features))
        x = F.gelu(encoder.conv2(x))
        x = x.permute(0, 2, 1)
        n_frames = x.shape[1]
        max_frames = encoder.embed_positions.weight.shape[0]
        if n_frames > max_frames:
            raise ValueError(f"Whisper encoder supports at most {max_frames} frames, got {n_frames}")
        x = x + encoder.embed_positions.weight[:n_frames]
        x = F.dropout(x, p=float(encoder.dropout), training=self.training)
        hidden_states = [x]
        for layer in encoder.layers:
            layer_out = layer(x, attention_mask=None, layer_head_mask=None)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            hidden_states.append(x)
        hidden_states[-1] = encoder.layer_norm(x)
        return hidden_states

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def receptive_seconds(self, seconds: float) -> int:
        return int(math.ceil(seconds))
