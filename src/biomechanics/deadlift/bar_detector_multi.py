"""Barbell keypoints on every synced camera view at once, for the deadlift's 3D bar
tracker (PLAN.md §3.6, §6). Same YOLO11n-pose weights as BarbellDetector (2 keypoints
= the plate hubs), but one batched predict over all views, and every candidate per
view is kept: the floor bar is picked across views, never by score alone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from biomechanics.config import BarbellTrackingConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/barbell_keypoints.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IMGSZ = 640
AUTO_DEVICE = "auto"
# The floor bar, a racked bar, a spare bar: a few real candidates per view at most. It
# also bounds the cross-view association (hypotheses grow as (1 + 2 * candidates) ** views).
DEFAULT_MAX_CANDIDATES_PER_VIEW = 4
NUM_BAR_KEYPOINTS = 2


class BarCandidate2D(BaseModel):
    """One bar detection in one view, in raw (distorted) pixels.

    end_a_px / end_b_px are the model's two keypoints in its output order. Which one is
    the subject's left is decided in 3D, so the order may differ between views.
    """
    end_a_px: tuple[float, float]
    end_b_px: tuple[float, float]
    end_confidences: tuple[float, float]
    box_xyxy: tuple[float, float, float, float]
    score: float


def _to_numpy(value: Any) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def _candidates_from_result(result: Any, max_candidates: int) -> list[BarCandidate2D]:
    boxes, keypoints = result.boxes, result.keypoints
    if boxes is None or keypoints is None:
        return []
    scores = _to_numpy(boxes.conf).reshape(-1)
    if scores.size == 0:
        return []
    boxes_xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    keypoints_xy = _to_numpy(keypoints.xy)
    keypoint_conf = getattr(keypoints, "conf", None)
    keypoint_conf = (
        _to_numpy(keypoint_conf)
        if keypoint_conf is not None
        else np.repeat(scores[:, None], NUM_BAR_KEYPOINTS, axis=1)
    )
    candidates = []
    for i in np.argsort(-scores)[:max_candidates]:
        if keypoints_xy[i].shape[0] < NUM_BAR_KEYPOINTS:
            continue
        candidates.append(BarCandidate2D(
            end_a_px=(float(keypoints_xy[i, 0, 0]), float(keypoints_xy[i, 0, 1])),
            end_b_px=(float(keypoints_xy[i, 1, 0]), float(keypoints_xy[i, 1, 1])),
            end_confidences=(float(keypoint_conf[i, 0]), float(keypoint_conf[i, 1])),
            box_xyxy=tuple(float(value) for value in boxes_xyxy[i]),
            score=float(scores[i]),
        ))
    return candidates


class MultiViewBarDetector:
    """Loads its weights on first use; check `available` (and `error`) before detecting.

    model injects an already-loaded Ultralytics-like model (anything with the YOLO
    predict signature), e.g. a TensorRT engine loaded elsewhere, or a stub in tests.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = AUTO_DEVICE,
        max_candidates_per_view: int = DEFAULT_MAX_CANDIDATES_PER_VIEW,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.device = device
        self.max_candidates_per_view = int(max_candidates_per_view)
        self._model = model
        self._error: str | None = None

    @classmethod
    def from_config(cls, config: BarbellTrackingConfig) -> MultiViewBarDetector:
        """The same weights and thresholds as the 2D BarbellDetector."""
        return cls(config.model_path, config.conf_threshold, config.imgsz, config.device)

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    @property
    def error(self) -> str | None:
        """Why the detector is unavailable (missing weights, no ultralytics); None when it is."""
        self._load()
        return self._error

    def detect(self, views: dict[str, np.ndarray]) -> dict[str, list[BarCandidate2D]]:
        """Every candidate per camera (best score first) from one batched predict over all
        BGR views. Raises RuntimeError when the detector is unavailable."""
        if not views:
            return {}
        if not self.available:
            raise RuntimeError(f"bar detector unavailable: {self._error}")
        camera_ids = list(views)
        device_argument = {} if self.device == AUTO_DEVICE else {"device": self.device}
        results = self._model.predict(
            [views[camera_id] for camera_id in camera_ids],
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
            **device_argument,
        )
        return {
            camera_id: _candidates_from_result(result, self.max_candidates_per_view)
            for camera_id, result in zip(camera_ids, results)
        }

    def _load(self) -> None:
        if self._model is not None or self._error is not None:
            return
        if not self.model_path.exists():
            self._error = (
                f"bar model not found at {self.model_path}: the YOLO11n-pose weights are not in "
                f"the repo (docs/deadlift/PLAN.md §6, Q7)"
            )
            logger.warning("Multi-view bar detector unavailable: %s", self._error)
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._error = f"ultralytics is not installed ({exc})"
            logger.warning("Multi-view bar detector unavailable: %s", self._error)
            return
        logger.info("Loading bar YOLO model from %s", self.model_path)
        self._model = YOLO(str(self.model_path))
