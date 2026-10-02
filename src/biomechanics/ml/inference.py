"""
BiLSTM Inference Orchestrator

Wires feature extraction → sequence buffering → model forward pass → rep counting
into a single ``process_skeleton()`` call for the pipeline. Live skeletons are
Y-down with missing keypoints zeroed at the hip centre; the model sees the Y-up
training convention and each missing keypoint at its last measured position.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from biomechanics.utils.types import Skeleton3D, RepData
from biomechanics.ml.feature_extractor import LandmarkFeatureExtractor
from biomechanics.ml.sequence_buffer import SequenceBuffer
from biomechanics.ml.bilstm_model import BiLSTMRepModel
from biomechanics.ml.bilstm_counter import BiLSTMRepCounter, BiLSTMCounterConfig

# Keypoints below this confidence are missing (the IK solver's floor).
MIN_KEYPOINT_CONFIDENCE = 0.1


class BiLSTMInference:
    """
    End-to-end BiLSTM rep counting inference.

    Processes one Skeleton3D per call, internally managing the feature
    buffer and model state. Returns RepData when a rep is completed.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        config: Optional[BiLSTMCounterConfig] = None,
    ):
        self._model_path = model_path
        self._device = device

        self._extractor = LandmarkFeatureExtractor(input_y_down=True)
        self._buffer = SequenceBuffer(window_size=30)
        self._last_measured_points: Optional[np.ndarray] = None
        self._ever_measured: Optional[np.ndarray] = None
        self._counter = BiLSTMRepCounter(config)
        self._model: Optional[BiLSTMRepModel] = None  # lazy loaded

    def _load_model(self) -> None:
        self._model = BiLSTMRepModel.from_checkpoint(self._model_path, self._device)

    def process_skeleton(
        self, skeleton: Skeleton3D
    ) -> Tuple[Optional[RepData], Optional[int]]:
        """
        Process one frame through the full BiLSTM pipeline.

        Returns (rep_data, shallow_depth_class) — see BiLSTMRepCounter.update.
        Returns (None, None) during the cold-start period (first ~30 frames).
        """
        if self._model is None:
            self._load_model()

        features = self._extractor.extract_points(self._hold_last_measured(skeleton))
        sequence = self._buffer.push(features)

        if sequence is None:
            return None, None  # cold-start

        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self._device)
            logits = self._model(x)                    # (1, 30, num_classes)
            probs = F.softmax(logits, dim=-1)          # (1, 30, num_classes)
            current_probs = probs[0, -1].cpu().numpy() # (num_classes,)

        return self._counter.update(
            current_probs,
            timestamp=skeleton.timestamp,
            frame_index=skeleton.frame_index,
        )

    def _hold_last_measured(self, skeleton: Skeleton3D) -> np.ndarray:
        points = skeleton.to_numpy()
        measured = np.array([kp.confidence for kp in skeleton.keypoints]) >= MIN_KEYPOINT_CONFIDENCE
        if self._last_measured_points is None:
            self._last_measured_points = points.copy()
            self._ever_measured = measured.copy()
        held = np.where((~measured & self._ever_measured)[:, None], self._last_measured_points, points)
        self._last_measured_points = np.where(measured[:, None], points, self._last_measured_points)
        self._ever_measured |= measured
        return held

    @property
    def rep_count(self) -> int:
        return self._counter.rep_count

    @property
    def in_rep(self) -> bool:
        return self._counter.in_rep

    @property
    def current_probability(self) -> float:
        """Backward compat: probability of being in a counted rep."""
        return self._counter.smoothed_probability

    @property
    def current_depth_class(self) -> int:
        return self._counter.predicted_depth_class

    @property
    def current_class_probabilities(self) -> np.ndarray:
        return self._counter.smoothed_probabilities

    def set_min_depth_class(self, depth_class: int) -> None:
        self._counter.set_min_depth_class(depth_class)

    def reject_last_rep(self) -> None:
        self._counter.reject_last_rep()

    def set_assessment_mode(self, enabled: bool) -> None:
        """Delegate to the inner counter so any completed rep counts."""
        self._counter.set_assessment_mode(enabled)

    def reset(self) -> None:
        """Reset buffer and counter state (e.g. between sets)."""
        self._buffer.reset()
        self._counter.reset()
        self._last_measured_points = None
        self._ever_measured = None
