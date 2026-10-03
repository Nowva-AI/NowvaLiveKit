"""
Tests for the BiLSTM-based Rep Counting Module (5-class depth classification)

Covers: feature extraction, sequence buffering, BiLSTM model,
5-class probability counter, and end-to-end inference.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.utils.types import CocoKeypoints as CK, Skeleton3D, Point3D, RepData, DEPTH_CLASS_NAMES, NUM_DEPTH_CLASSES
from biomechanics.ml.feature_extractor import LandmarkFeatureExtractor
from biomechanics.ml.sequence_buffer import SequenceBuffer
from biomechanics.ml.bilstm_model import BiLSTMRepModel
from biomechanics.ml.bilstm_counter import BiLSTMRepCounter, BiLSTMCounterConfig, BiLSTMCounterState


# ============================================================================
# Helpers
# ============================================================================

def _make_skeleton(scale: float = 1.0, y_offset: float = 0.0) -> Skeleton3D:
    """
    Create a synthetic standing-pose Skeleton3D (COCO 17 keypoints).

    scale: multiplies all coordinates (simulates distance from camera)
    y_offset: shifts hips/knees/ankles down (simulates squat depth)
    """
    raw = np.array([
        [0.00,  0.60,  0.00],  # 0  nose
        [-0.03, 0.62,  0.00],  # 1  left_eye
        [0.03,  0.62,  0.00],  # 2  right_eye
        [-0.06, 0.60,  0.00],  # 3  left_ear
        [0.06,  0.60,  0.00],  # 4  right_ear
        [-0.18, 0.45,  0.00],  # 5  left_shoulder
        [0.18,  0.45,  0.00],  # 6  right_shoulder
        [-0.30, 0.25,  0.00],  # 7  left_elbow
        [0.30,  0.25,  0.00],  # 8  right_elbow
        [-0.35, 0.10,  0.00],  # 9  left_wrist
        [0.35,  0.10,  0.00],  # 10 right_wrist
        [-0.10, 0.00,  0.00],  # 11 left_hip
        [0.10,  0.00,  0.00],  # 12 right_hip
        [-0.10, -0.40 + y_offset, 0.00],  # 13 left_knee
        [0.10,  -0.40 + y_offset, 0.00],  # 14 right_knee
        [-0.10, -0.80 + y_offset, 0.00],  # 15 left_ankle
        [0.10,  -0.80 + y_offset, 0.00],  # 16 right_ankle
    ]) * scale

    return Skeleton3D.from_numpy(raw, timestamp=0.0, frame_index=0)


FEATURE_TOL = 1e-9
KNEE_Y_DIFF_INDEX = 11


def _y_down(skeleton: Skeleton3D) -> Skeleton3D:
    points = skeleton.to_numpy()
    points[:, 1] *= -1.0
    confidences = np.array([kp.confidence for kp in skeleton.keypoints])
    return Skeleton3D.from_numpy(
        points, confidences=confidences, timestamp=skeleton.timestamp, frame_index=skeleton.frame_index,
    )


def _without(skeleton: Skeleton3D, index: int) -> Skeleton3D:
    """MediaPipe's missing keypoint: zeroed at the hip centre with confidence 0."""
    points = skeleton.to_numpy()
    points[index] = 0.0
    confidences = np.array([kp.confidence for kp in skeleton.keypoints])
    confidences[index] = 0.0
    return Skeleton3D.from_numpy(
        points, confidences=confidences, timestamp=skeleton.timestamp, frame_index=skeleton.frame_index,
    )


def _one_hot(cls: int, num_classes: int = 5) -> np.ndarray:
    """Create a one-hot probability vector for the given class."""
    v = np.zeros(num_classes)
    v[cls] = 1.0
    return v


def _peaked(cls: int, peak: float = 0.8, num_classes: int = 5) -> np.ndarray:
    """Create a probability vector peaked at the given class."""
    rest = (1.0 - peak) / (num_classes - 1)
    v = np.full(num_classes, rest)
    v[cls] = peak
    return v


# ============================================================================
# TestLandmarkFeatureExtractor
# ============================================================================

class TestLandmarkFeatureExtractor:
    def setup_method(self):
        self.extractor = LandmarkFeatureExtractor()

    def test_feature_dim(self):
        skeleton = _make_skeleton()
        features = self.extractor.extract(skeleton)
        assert features.shape == (LandmarkFeatureExtractor.FEATURE_DIM,)

    def test_output_is_finite(self):
        skeleton = _make_skeleton()
        features = self.extractor.extract(skeleton)
        assert np.all(np.isfinite(features))

    def test_normalization_invariance(self):
        """Same proportions at different scales should give similar features."""
        features_1x = self.extractor.extract(_make_skeleton(scale=1.0))
        features_2x = self.extractor.extract(_make_skeleton(scale=2.0))

        np.testing.assert_allclose(features_1x[:4], features_2x[:4], atol=0.1)
        np.testing.assert_allclose(features_1x[4:10], features_2x[4:10], atol=0.01)
        np.testing.assert_allclose(features_1x[10:], features_2x[10:], atol=0.01)

    def test_different_poses_differ(self):
        """Standing vs squat should produce different feature vectors."""
        standing = self.extractor.extract(_make_skeleton(y_offset=0.0))
        squatting = self.extractor.extract(_make_skeleton(y_offset=0.3))
        assert not np.allclose(standing, squatting, atol=0.01)

    def test_angles_are_degrees(self):
        """Joint angles should be in degrees (0-180 range)."""
        features = self.extractor.extract(_make_skeleton())
        angles = features[:4]
        assert np.all(angles >= 0) and np.all(angles <= 180)

    def test_with_sample_skeleton(self, sample_skeleton_3d):
        """Works with the shared fixture skeleton."""
        features = self.extractor.extract(sample_skeleton_3d)
        assert features.shape == (LandmarkFeatureExtractor.FEATURE_DIM,)
        assert np.all(np.isfinite(features))

    def test_y_down_input_gives_the_y_up_training_features(self):
        """The model was trained on Y-up skeletons; live skeletons are Y-down."""
        y_up = _make_skeleton(y_offset=0.2)
        y_down = _y_down(y_up)

        training_features = LandmarkFeatureExtractor().extract(y_up)
        live_features = LandmarkFeatureExtractor(input_y_down=True).extract(y_down)

        assert live_features == pytest.approx(training_features, abs=FEATURE_TOL)
        # Training convention: a knee below the hip has a negative y difference.
        assert live_features[KNEE_Y_DIFF_INDEX] < 0.0


# ============================================================================
# TestSequenceBuffer
# ============================================================================

class TestSequenceBuffer:
    def test_cold_start_returns_none(self):
        buf = SequenceBuffer(window_size=30)
        feat = np.zeros(14)
        for _ in range(29):
            assert buf.push(feat) is None
        assert not buf.is_ready

    def test_ready_at_window_size(self):
        buf = SequenceBuffer(window_size=30)
        feat = np.ones(14)
        result = None
        for i in range(30):
            result = buf.push(feat * i)
        assert result is not None
        assert result.shape == (30, 14)
        assert buf.is_ready

    def test_sliding_window(self):
        """After 31 pushes, oldest frame is dropped."""
        buf = SequenceBuffer(window_size=3)
        buf.push(np.array([1.0]))
        buf.push(np.array([2.0]))
        buf.push(np.array([3.0]))
        result = buf.push(np.array([4.0]))
        assert result is not None
        np.testing.assert_array_equal(result[:, 0], [2.0, 3.0, 4.0])

    def test_reset(self):
        buf = SequenceBuffer(window_size=5)
        for _ in range(5):
            buf.push(np.zeros(3))
        assert buf.is_ready
        buf.reset()
        assert not buf.is_ready
        assert buf.push(np.zeros(3)) is None


# ============================================================================
# TestBiLSTMRepModel
# ============================================================================

class TestBiLSTMRepModel:
    def test_forward_shape_5class(self):
        """Default model outputs 5 classes."""
        model = BiLSTMRepModel(input_dim=14, hidden_dim=64, num_layers=2)
        x = torch.randn(2, 30, 14)
        logits = model(x)
        assert logits.shape == (2, 30, 5)

    def test_forward_shape_explicit_2class(self):
        """Backward compat: explicit num_classes=2 still works."""
        model = BiLSTMRepModel(input_dim=14, hidden_dim=64, num_classes=2)
        x = torch.randn(1, 30, 14)
        logits = model(x)
        assert logits.shape == (1, 30, 2)

    def test_output_range_after_softmax(self):
        model = BiLSTMRepModel(input_dim=14)
        x = torch.randn(1, 30, 14)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        assert (probs >= 0).all() and (probs <= 1).all()
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, 30), atol=1e-5, rtol=0)

    def test_deterministic(self):
        model = BiLSTMRepModel(input_dim=14, hidden_dim=32)
        model.eval()
        x = torch.randn(1, 10, 14)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ============================================================================
# TestBiLSTMRepCounter (5-class)
# ============================================================================

class TestBiLSTMRepCounter:
    def test_initial_state(self):
        counter = BiLSTMRepCounter()
        assert counter.state == BiLSTMCounterState.UP
        assert counter.rep_count == 0
        assert not counter.in_rep
        assert counter.predicted_depth_class == 0

    def test_enter_down_at_min_depth(self):
        """Feeding class >= min_depth_class should enter DOWN."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_depth_class=3)
        counter = BiLSTMRepCounter(config)
        counter.update(_one_hot(3))  # parallel
        assert counter.state == BiLSTMCounterState.DOWN
        assert counter.in_rep

    def test_no_enter_below_min_depth(self):
        """Classes below min_depth_class should NOT enter DOWN."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_depth_class=3)
        counter = BiLSTMRepCounter(config)
        counter.update(_one_hot(2))  # half squat - below parallel threshold
        assert counter.state == BiLSTMCounterState.UP

    def test_quarter_squat_not_counted_at_default(self):
        """Quarter squats should NOT count as reps with default min_depth_class=3."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2)
        counter = BiLSTMRepCounter(config)
        # Go to quarter depth
        counter.update(_one_hot(1))
        assert counter.state == BiLSTMCounterState.UP  # not deep enough
        # Even staying there many frames
        for _ in range(20):
            counter.update(_one_hot(1))
        assert counter.state == BiLSTMCounterState.UP
        assert counter.rep_count == 0

    def test_rep_completion_parallel(self):
        """Parallel depth rep should be counted."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=3, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        # Enter DOWN with parallel
        counter.update(_one_hot(3))
        assert counter.state == BiLSTMCounterState.DOWN

        # Stay in DOWN
        for _ in range(3):
            counter.update(_one_hot(3))

        # Return to standing
        result, _ = counter.update(_one_hot(0))
        assert result is not None
        assert isinstance(result, RepData)
        assert counter.rep_count == 1
        assert counter.state == BiLSTMCounterState.UP
        assert result.depth_class == 3
        assert result.depth_class_name == "Parallel"

    def test_rep_completion_deep(self):
        """Deep squat rep with max depth tracking."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        # Descend: parallel → deep
        counter.update(_one_hot(3))
        counter.update(_one_hot(4))
        counter.update(_one_hot(4))

        # Return to standing
        result, _ = counter.update(_one_hot(0))
        assert result is not None
        assert result.max_depth_class == 4
        assert result.depth_class_name == "Deep"

    def test_min_frames_enforced(self):
        """Quick DOWN→UP should NOT count as rep."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=12)
        counter = BiLSTMRepCounter(config)

        counter.update(_one_hot(3))  # enter DOWN
        counter.update(_one_hot(3))  # 1 frame in DOWN
        result, _ = counter.update(_one_hot(0))  # try to exit

        # min_frames not met — should stay in DOWN (0 doesn't cause transition)
        assert counter.rep_count == 0

    def test_multiple_reps(self):
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        for _ in range(3):
            counter.update(_one_hot(3))  # enter DOWN
            counter.update(_one_hot(4))  # stay (deep)
            counter.update(_one_hot(4))  # meet min_frames
            counter.update(_one_hot(0))  # exit → rep counted
            counter.update(_one_hot(0))  # stay UP

        assert counter.rep_count == 3

    def test_configurable_min_depth(self):
        """min_depth_class=1 should count quarter squats."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2, min_depth_class=1)
        counter = BiLSTMRepCounter(config)

        counter.update(_one_hot(1))  # quarter — now counts
        assert counter.state == BiLSTMCounterState.DOWN

        counter.update(_one_hot(1))
        counter.update(_one_hot(1))
        result, _ = counter.update(_one_hot(0))
        assert result is not None
        assert counter.rep_count == 1
        assert result.depth_class == 1
        assert result.depth_class_name == "Quarter"

    def test_configurable_min_depth_deep_only(self):
        """min_depth_class=4 should only count deep squats."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2, min_depth_class=4)
        counter = BiLSTMRepCounter(config)

        # Parallel squat should NOT count
        counter.update(_one_hot(3))
        assert counter.state == BiLSTMCounterState.UP

        # Deep should count
        counter.update(_one_hot(4))
        assert counter.state == BiLSTMCounterState.DOWN
        counter.update(_one_hot(4))
        counter.update(_one_hot(4))
        result, _ = counter.update(_one_hot(0))
        assert result is not None
        assert counter.rep_count == 1

    def test_ema_smoothing_delays_transition(self):
        """With default EMA (alpha=0.2), a single high-class frame shouldn't trigger DOWN
        when preceded by standing frames."""
        counter = BiLSTMRepCounter(BiLSTMCounterConfig(min_depth_class=3))

        # Build up standing EMA
        counter.update(_one_hot(0))
        counter.update(_one_hot(0))
        # Single deep frame — EMA should still be mostly standing
        counter.update(_one_hot(4))
        assert counter.state == BiLSTMCounterState.UP

    def test_max_depth_tracking(self):
        """During a rep, max_depth_seen tracks the deepest class."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        counter.update(_one_hot(3))  # parallel
        assert counter._max_depth_seen == 3
        counter.update(_one_hot(4))  # deep
        assert counter._max_depth_seen == 4
        counter.update(_one_hot(3))  # back to parallel, max stays 4
        assert counter._max_depth_seen == 4
        result, _ = counter.update(_one_hot(0))
        assert result is not None
        assert result.max_depth_class == 4

    def test_smoothed_probabilities_shape(self):
        config = BiLSTMCounterConfig(num_classes=5)
        counter = BiLSTMRepCounter(config)
        counter.update(_peaked(0))
        probs = counter.smoothed_probabilities
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 0.01

    def test_in_rep_probability(self):
        """in_rep_probability sums classes >= min_depth_class."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_depth_class=3, num_classes=5)
        counter = BiLSTMRepCounter(config)
        # Feed probabilities spread across classes
        probs = np.array([0.1, 0.1, 0.1, 0.4, 0.3])
        counter.update(probs)
        assert abs(counter.in_rep_probability - 0.7) < 0.01  # 0.4 + 0.3

    def test_reset(self):
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=1, min_depth_class=3)
        counter = BiLSTMRepCounter(config)
        counter.update(_one_hot(4))
        counter.update(_one_hot(4))
        counter.update(_one_hot(0))
        assert counter.rep_count == 1
        counter.reset()
        assert counter.rep_count == 0
        assert counter.state == BiLSTMCounterState.UP
        assert counter.predicted_depth_class == 0

    def test_rep_data_fields(self):
        """Completed rep should have all depth-related fields populated."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        counter.update(_one_hot(3), timestamp=1.0, frame_index=10)
        counter.update(_one_hot(4), timestamp=1.1, frame_index=11)
        counter.update(_one_hot(4), timestamp=1.2, frame_index=12)
        result, _ = counter.update(_one_hot(0), timestamp=1.3, frame_index=13)

        assert result is not None
        assert result.rep_number == 1
        assert result.start_time == 1.0
        assert result.end_time == 1.3
        assert result.start_frame == 10
        assert result.end_frame == 13
        assert result.depth_class == 4
        assert result.depth_class_name == "Deep"
        assert result.max_depth_class == 4


# ============================================================================
# TestShallowRepDetection
# ============================================================================

class TestShallowRepDetection:
    def test_quarter_squat_reports_shallow(self):
        """A sustained quarter squat returning to standing is a shallow rep."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=3, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        for _ in range(4):
            rep, shallow = counter.update(_one_hot(1))
            assert rep is None
            assert shallow is None

        rep, shallow = counter.update(_one_hot(0))
        assert rep is None
        assert shallow == 1
        assert counter.rep_count == 0

    def test_half_squat_reports_max_class(self):
        """Shallow rep reports the deepest class reached, not the last one."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=3, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        counter.update(_one_hot(1))
        counter.update(_one_hot(2))
        counter.update(_one_hot(2))
        counter.update(_one_hot(1))
        rep, shallow = counter.update(_one_hot(0))

        assert rep is None
        assert shallow == 2

    def test_brief_dip_is_not_shallow_rep(self):
        """A descent shorter than min_rep_frames is noise, not a rep attempt."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=12, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        counter.update(_one_hot(1))
        counter.update(_one_hot(1))
        rep, shallow = counter.update(_one_hot(0))

        assert rep is None
        assert shallow is None

    def test_counted_rep_never_reports_shallow(self):
        """Passing through shallow classes on the way down must not fire."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=2, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        shallow_events = []
        for probs in [_one_hot(1)] * 5 + [_one_hot(2)] * 5 + [_one_hot(4)] * 5:
            _, shallow = counter.update(probs)
            shallow_events.append(shallow)

        rep, shallow = counter.update(_one_hot(0))
        shallow_events.append(shallow)

        assert rep is not None
        assert counter.rep_count == 1
        assert all(s is None for s in shallow_events)

    def test_shallow_then_deep_rep(self):
        """A shallow attempt must not poison the rep that follows it."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=3, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        for _ in range(4):
            counter.update(_one_hot(1))
        _, shallow = counter.update(_one_hot(0))
        assert shallow == 1

        for _ in range(4):
            counter.update(_one_hot(4))
        rep, shallow = counter.update(_one_hot(0))

        assert rep is not None
        assert rep.max_depth_class == 4
        assert shallow is None
        assert counter.rep_count == 1

    def test_consecutive_shallow_reps_all_report(self):
        """Every shallow rep fires — the lifter needs a cue on each one."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=3, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        shallow_count = 0
        for _ in range(4):
            for _ in range(4):
                counter.update(_one_hot(2))
            _, shallow = counter.update(_one_hot(0))
            if shallow is not None:
                shallow_count += 1
            counter.update(_one_hot(0))

        assert shallow_count == 4
        assert counter.rep_count == 0

    def test_assessment_mode_has_no_shallow_reps(self):
        """Assessment counts any descent, so nothing is rejected for depth."""
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=3, min_depth_class=3)
        counter = BiLSTMRepCounter(config)
        counter.set_assessment_mode(True)

        for _ in range(4):
            counter.update(_one_hot(1))
        rep, shallow = counter.update(_one_hot(0))

        assert shallow is None
        assert rep is not None
        assert counter.rep_count == 1

    def test_reset_clears_shallow_tracking(self):
        config = BiLSTMCounterConfig(ema_alpha=1.0, min_rep_frames=3, min_depth_class=3)
        counter = BiLSTMRepCounter(config)

        for _ in range(4):
            counter.update(_one_hot(1))
        counter.reset()

        rep, shallow = counter.update(_one_hot(0))
        assert shallow is None


# ============================================================================
# TestBiLSTMInference (5-class)
# ============================================================================

class TestBiLSTMInference:
    def test_cold_start_returns_none(self):
        """First 29 frames should return None (buffer not full)."""
        from biomechanics.ml.inference import BiLSTMInference

        with patch.object(BiLSTMInference, '_load_model'):
            inf = BiLSTMInference(model_path="dummy.pt")
            mock_model = BiLSTMRepModel(input_dim=14, hidden_dim=32, num_classes=5)
            mock_model.eval()
            inf._model = mock_model

            for i in range(29):
                skeleton = _make_skeleton()
                skeleton.frame_index = i
                result, _ = inf.process_skeleton(skeleton)
                assert result is None

    def test_produces_output_after_warmup(self):
        """After 30 frames, should produce depth class output."""
        from biomechanics.ml.inference import BiLSTMInference

        inf = BiLSTMInference(model_path="dummy.pt")
        mock_model = BiLSTMRepModel(input_dim=14, hidden_dim=32, num_classes=5)
        mock_model.eval()
        inf._model = mock_model

        for i in range(30):
            skeleton = _make_skeleton()
            skeleton.frame_index = i
            inf.process_skeleton(skeleton)

        assert inf.current_probability >= 0.0
        assert inf.current_depth_class in range(5)
        probs = inf.current_class_probabilities
        assert probs.shape == (5,)

    def test_rep_count_starts_at_zero(self):
        from biomechanics.ml.inference import BiLSTMInference

        inf = BiLSTMInference(model_path="dummy.pt")
        assert inf.rep_count == 0
        assert not inf.in_rep

    def test_reset(self):
        from biomechanics.ml.inference import BiLSTMInference

        inf = BiLSTMInference(model_path="dummy.pt")
        inf._model = BiLSTMRepModel(input_dim=14, hidden_dim=32, num_classes=5)
        inf._model.eval()

        for i in range(30):
            s = _make_skeleton()
            s.frame_index = i
            inf.process_skeleton(s)

        inf.reset()
        assert inf.rep_count == 0
        assert not inf._buffer.is_ready

    def test_current_depth_class_property(self):
        """current_depth_class returns an integer in [0, 4]."""
        from biomechanics.ml.inference import BiLSTMInference

        inf = BiLSTMInference(model_path="dummy.pt")
        inf._model = BiLSTMRepModel(input_dim=14, hidden_dim=32, num_classes=5)
        inf._model.eval()

        for i in range(30):
            s = _make_skeleton()
            s.frame_index = i
            inf.process_skeleton(s)

        assert isinstance(inf.current_depth_class, (int, np.integer))
        assert 0 <= inf.current_depth_class <= 4

    def test_live_skeletons_are_read_as_y_down(self):
        """Production skeletons are Y-down; the model sees the Y-up training convention."""
        from biomechanics.ml.inference import BiLSTMInference

        inf = BiLSTMInference(model_path="dummy.pt")
        inf._model = BiLSTMRepModel(input_dim=14, hidden_dim=32, num_classes=5)
        inf._model.eval()
        live = _y_down(_make_skeleton(y_offset=0.2))

        inf.process_skeleton(live)

        expected = LandmarkFeatureExtractor().extract(_make_skeleton(y_offset=0.2))
        assert inf._buffer._buffer[-1] == pytest.approx(expected, abs=FEATURE_TOL)

    def test_missing_keypoint_holds_its_last_measured_position(self):
        """A dropped knee sits at the hip centre (0, 0, 0); the model must not see it fold."""
        from biomechanics.ml.inference import BiLSTMInference

        inf = BiLSTMInference(model_path="dummy.pt")
        inf._model = BiLSTMRepModel(input_dim=14, hidden_dim=32, num_classes=5)
        inf._model.eval()
        live = _y_down(_make_skeleton(y_offset=0.2))

        inf.process_skeleton(live)
        measured = np.array(inf._buffer._buffer[-1])
        inf.process_skeleton(_without(live, CK.LEFT_KNEE))

        assert inf._buffer._buffer[-1] == pytest.approx(measured, abs=FEATURE_TOL)

    def test_reset_forgets_held_positions(self):
        from biomechanics.ml.inference import BiLSTMInference

        inf = BiLSTMInference(model_path="dummy.pt")
        inf._model = BiLSTMRepModel(input_dim=14, hidden_dim=32, num_classes=5)
        inf._model.eval()
        live = _y_down(_make_skeleton(y_offset=0.2))
        inf.process_skeleton(live)

        inf.reset()
        inf.process_skeleton(_without(live, CK.LEFT_KNEE))

        unheld = LandmarkFeatureExtractor(input_y_down=True).extract(_without(live, CK.LEFT_KNEE))
        assert inf._buffer._buffer[-1] == pytest.approx(unheld, abs=FEATURE_TOL)
