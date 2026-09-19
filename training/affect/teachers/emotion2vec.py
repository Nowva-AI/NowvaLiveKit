"""emotion2vec+ categorical teacher via FunASR, with its 9 classes mapped onto the MSP-Podcast 8 (contempt masked)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from training.affect.config import MSP_PODCAST_CLASSES

logger = logging.getLogger(__name__)

EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_large"
# emotion2vec+ output order (FunASR): angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
EMOTION2VEC_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
E2V_TO_MSP = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprise",
}
MASKED_CLASSES = ("contempt",)


def map_to_msp(e2v_scores: np.ndarray, labels: list[str] = MSP_PODCAST_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs over MSP classes, class mask). 'other' and 'unknown' mass is dropped and the rest renormalized."""
    probs = np.zeros(len(labels), dtype=np.float32)
    for source, score in zip(EMOTION2VEC_LABELS, e2v_scores):
        target = E2V_TO_MSP.get(source)
        if target is not None:
            probs[labels.index(target)] += float(score)
    mask = np.array([label not in MASKED_CLASSES for label in labels])
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return probs, mask


class Emotion2VecTeacher:
    def __init__(self, model_id: str = EMOTION2VEC_MODEL_ID, device: str = "cuda") -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ImportError("pip install funasr (training environment) to use the emotion2vec+ teacher") from exc
        self.model = AutoModel(model=model_id, hub="hf", device=device, disable_update=True)

    def predict(self, wav_path: Path) -> np.ndarray:
        result = self.model.generate(str(wav_path), granularity="utterance", extract_embedding=False)
        scores = np.asarray(result[0]["scores"], dtype=np.float32)
        if scores.shape[0] != len(EMOTION2VEC_LABELS):
            raise RuntimeError(f"emotion2vec+ returned {scores.shape[0]} scores, expected {len(EMOTION2VEC_LABELS)}")
        return scores
