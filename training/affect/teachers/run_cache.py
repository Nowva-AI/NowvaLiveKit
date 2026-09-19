"""Score a manifest with the off-the-shelf teachers and write soft targets to a TargetCache."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from training.affect.data.manifest import read_manifest
from training.affect.teachers.cache import TargetCache

logger = logging.getLogger(__name__)


def _read_wave(path: str, start_s: float | None, end_s: float | None, sample_rate: int = 16000) -> np.ndarray:
    from training.affect.data.dataset import _read_audio

    return _read_audio(path, sample_rate, start_s, end_s)


def cache_targets(
    manifest: Path,
    out: Path,
    use_audeering: bool = True,
    use_emotion2vec: bool = True,
    device: str = "cuda",
    limit: int | None = None,
) -> int:
    cache = TargetCache(out)
    audeering = None
    e2v = None
    if use_audeering:
        from training.affect.teachers.audeering import AudeeringEmotionModel, predict_avd

        audeering = AudeeringEmotionModel().to(device).eval()
    if use_emotion2vec:
        from training.affect.teachers.emotion2vec import Emotion2VecTeacher, map_to_msp

        e2v = Emotion2VecTeacher(device=device)
    written = 0
    for index, segment in enumerate(read_manifest(manifest)):
        if limit is not None and index >= limit:
            break
        if segment.id in cache:
            continue
        row: dict[str, np.ndarray] = {}
        if audeering is not None:
            wave = torch.from_numpy(_read_wave(segment.path, segment.start_s, segment.end_s))[None, :].to(device)
            row["avd"] = predict_avd(audeering, wave)[0].float().cpu().numpy().clip(0.0, 1.0)
        if e2v is not None:
            scores = e2v.predict(Path(segment.path))
            probs, mask = map_to_msp(scores)
            row["class_probs"] = probs
            row["class_mask"] = mask.astype(np.float32)
        cache.put(segment.id, **row)
        written += 1
        if written % 500 == 0:
            logger.info("cached %d segments", written)
    cache.flush()
    return written
