"""
Tests for BiomechanicsPipeline.

Single-camera tests mock cv2.VideoCapture with a synthetic video so no real
camera is needed. Multi-camera tests drive the real pipeline with a fake
triangulated provider (world-frame 21-keypoint squats, real capture
sequence numbers) and a fake clock through readiness and reps.
"""

import math
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultType
from biomechanics.utils.types import CocoKeypoints as CK, PipelineFrame, Skeleton3D
from conftest import SYNTHETIC_FPS, squat_depth_profile, world_squat_points

FRAME_DT_S = 1.0 / SYNTHETIC_FPS
CLOCK_START_S = 1.7e9
PROVIDER_CONFIDENCE = 0.9
VALGUS_SHIFT_M = 0.12
KALMAN_LAG_FRAMES = 2
MAX_PREDICTED_FRAMES = 5
DROPOUT_FRAMES = 3
READINESS_WARM_UP_FRAMES = 10
RESET_REPEATS = 3
# A perfectly collinear synthetic leg reads exactly 0.0 deg, which is
# indistinguishable from the IK-zeroing regression; real lifters never lock
# out to the millimetre, so standing keeps a small residual bend.
MIN_DEPTH_RATIO = 0.02
SPIKE_DEG = 15.0
# Knee flexion of the synthetic bottom hold is 120 deg; the spike lands once we are there.
SPIKE_TRIGGER_KNEE_DEG = 119.0
DEPTH_MATCH_TOL_DEG = 1.0
ATHLETE_PARAMS = {
    "shoulder_width_m": 0.38,
    "femur_avg_m": 0.45,
    "torso_avg_m": 0.52,
    "hip_width_m": 0.24,
    "tibia_avg_m": 0.43,
    "foot_avg_m": 0.18,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _standing_points() -> np.ndarray:
    """Standing skeleton that passes both standing and readiness gates."""
    points = np.zeros((17, 3))
    points[CK.NOSE] = [0.0, 1.70, 0.0]
    points[CK.LEFT_EYE] = [0.03, 1.72, -0.02]
    points[CK.RIGHT_EYE] = [-0.03, 1.72, -0.02]
    points[CK.LEFT_EAR] = [0.07, 1.70, 0.0]
    points[CK.RIGHT_EAR] = [-0.07, 1.70, 0.0]
    points[CK.LEFT_SHOULDER] = [0.20, 1.50, 0.0]
    points[CK.RIGHT_SHOULDER] = [-0.20, 1.50, 0.0]
    points[CK.LEFT_ELBOW] = [0.25, 1.25, 0.0]
    points[CK.RIGHT_ELBOW] = [-0.25, 1.25, 0.0]
    points[CK.LEFT_WRIST] = [0.25, 1.00, 0.0]
    points[CK.RIGHT_WRIST] = [-0.25, 1.00, 0.0]
    points[CK.LEFT_HIP] = [0.10, 1.00, 0.0]
    points[CK.RIGHT_HIP] = [-0.10, 1.00, 0.0]
    points[CK.LEFT_KNEE] = [0.10, 0.55, 0.0]
    points[CK.RIGHT_KNEE] = [-0.10, 0.55, 0.0]
    points[CK.LEFT_ANKLE] = [0.10, 0.10, 0.0]
    points[CK.RIGHT_ANKLE] = [-0.10, 0.10, 0.0]
    return points


def _make_fake_capture(num_frames: int = 20):
    """Create a mock cv2.VideoCapture that yields synthetic frames."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.set.return_value = True

    call_count = 0

    def read_side_effect():
        nonlocal call_count
        if call_count >= num_frames:
            return False, None
        call_count += 1
        # 720p synthetic frame with random noise
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        return True, frame

    mock_cap.read.side_effect = read_side_effect
    mock_cap.release.return_value = None
    return mock_cap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBiomechanicsPipeline:

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_pipeline_processes_frames(self, mock_video_capture_cls):
        """Pipeline should process synthetic frames and return PipelineFrames."""
        mock_video_capture_cls.return_value = _make_fake_capture(20)

        from biomechanics.pipeline import BiomechanicsPipeline

        config = BiomechanicsConfig()
        pipeline = BiomechanicsPipeline(config)

        frames_processed = 0
        for _ in range(20):
            result = pipeline.process_frame()

            assert isinstance(result, PipelineFrame)
            assert isinstance(result.frame_index, int)
            assert isinstance(result.timestamp, float)
            assert isinstance(result.latency_ms, dict)
            assert "capture" in result.latency_ms

            frames_processed += 1

        pipeline.release()
        assert frames_processed == 20

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_pipeline_frame_types(self, mock_video_capture_cls):
        """PipelineFrame fields should have correct types when populated."""
        mock_video_capture_cls.return_value = _make_fake_capture(5)

        from biomechanics.pipeline import BiomechanicsPipeline

        config = BiomechanicsConfig()
        pipeline = BiomechanicsPipeline(config)

        result = pipeline.process_frame()

        # Always present
        assert result.frame_index >= 1
        assert result.timestamp > 0
        assert all(isinstance(v, float) for v in result.latency_ms.values())

        # Faults is always a list (possibly empty)
        assert isinstance(result.faults, list)

        pipeline.release()

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_pipeline_handles_no_capture(self, mock_video_capture_cls):
        """Pipeline should return a valid PipelineFrame even when capture fails."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.set.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap.release.return_value = None
        mock_video_capture_cls.return_value = mock_cap

        from biomechanics.pipeline import BiomechanicsPipeline

        config = BiomechanicsConfig()
        pipeline = BiomechanicsPipeline(config)

        result = pipeline.process_frame()

        assert isinstance(result, PipelineFrame)
        assert result.skeleton_2d is None
        assert result.skeleton_3d is None
        assert result.joint_angles is None
        assert "capture" in result.latency_ms

        pipeline.release()

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_pipeline_latency_keys(self, mock_video_capture_cls):
        """When pose succeeds, latency_ms should include all layer keys."""
        mock_video_capture_cls.return_value = _make_fake_capture(5)

        from biomechanics.pipeline import BiomechanicsPipeline

        config = BiomechanicsConfig()
        pipeline = BiomechanicsPipeline(config)

        # Process a few frames — at least one should get pose if mediapipe works,
        # but on CI without a real person, pose may return None. That's fine;
        # we just check that capture key is always present.
        for _ in range(5):
            result = pipeline.process_frame()
            assert "capture" in result.latency_ms
            # If pose succeeded, ik and faults should also be present
            if result.joint_angles is not None:
                assert "pose" in result.latency_ms
                assert "ik" in result.latency_ms
                assert "faults" in result.latency_ms

        pipeline.release()

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_pipeline_release(self, mock_video_capture_cls):
        """release() should not raise."""
        mock_video_capture_cls.return_value = _make_fake_capture(1)

        from biomechanics.pipeline import BiomechanicsPipeline

        config = BiomechanicsConfig()
        pipeline = BiomechanicsPipeline(config)
        pipeline.process_frame()
        pipeline.release()  # Should not raise


class TestPresenceOnlyMode:
    """Rest periods must not advance gates or collect analysis data."""

    def _pipeline_with_standing_pose(self, mock_video_capture_cls):
        """Pipeline with mocked capture and a pose estimator that always
        returns a valid standing skeleton."""
        mock_video_capture_cls.return_value = _make_fake_capture(60)

        from biomechanics.pipeline import BiomechanicsPipeline

        config = BiomechanicsConfig()
        pipeline = BiomechanicsPipeline(config)

        points = _standing_points()

        def fake_estimate_both(frame):
            skeleton_3d = Skeleton3D.from_numpy(
                points, confidences=np.ones(17),
                timestamp=time.time(), frame_index=0,
            )
            return None, skeleton_3d

        pipeline._pose_estimator = MagicMock()
        pipeline._pose_estimator.estimate_both.side_effect = fake_estimate_both

        # Seed a frame directly so the first process_frame() call does not
        # race the background capture thread.
        with pipeline._frame_lock:
            pipeline._latest_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        return pipeline

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_presence_only_skips_gate_and_analysis(self, mock_video_capture_cls):
        """Standing frames during rest must not latch the readiness gate,
        produce joint angles, count reps, or emit faults."""
        pipeline = self._pipeline_with_standing_pose(mock_video_capture_cls)
        pipeline.presence_only = True

        for _ in range(10):
            result = pipeline.process_frame()

        # Presence is still tracked (raw view only — nothing was analysed)
        assert result.skeleton_3d_raw is not None
        assert result.skeleton_3d is None

        # But nothing downstream runs
        assert result.joint_angles is None
        assert result.rep_data is None
        assert result.faults == []
        assert not pipeline.is_ready
        assert pipeline._readiness_gate.progress[0] == 0

        pipeline.release()

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_gate_advances_when_presence_only_cleared(self, mock_video_capture_cls):
        """Identical standing frames DO advance the readiness gate in
        normal mode — the contrast case for presence-only."""
        pipeline = self._pipeline_with_standing_pose(mock_video_capture_cls)

        for _ in range(4):
            result = pipeline.process_frame()

        assert pipeline._readiness_gate.progress[0] == 4
        assert result.joint_angles is None  # gate not yet latched

        pipeline.release()


class TestLoggedRepCount:
    """The displayed count must be the count that actually gets logged."""

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_rep_count_follows_hip_counter_without_bilstm(self, mock_video_capture_cls):
        mock_video_capture_cls.return_value = _make_fake_capture(1)

        from biomechanics.pipeline import BiomechanicsPipeline

        pipeline = BiomechanicsPipeline(BiomechanicsConfig())
        assert pipeline._bilstm is None

        pipeline._rep_counter.rep_count = 7
        assert pipeline.rep_count == 7

        pipeline.release()

    @patch("biomechanics.pipeline.cv2.VideoCapture")
    def test_rep_count_follows_bilstm_when_enabled(self, mock_video_capture_cls):
        """The hip counter accepts shallow reps the BiLSTM rejects — the
        HUD has to follow the counter that feeds the session tracker."""
        mock_video_capture_cls.return_value = _make_fake_capture(1)

        from biomechanics.pipeline import BiomechanicsPipeline

        pipeline = BiomechanicsPipeline(BiomechanicsConfig())
        pipeline._bilstm = MagicMock()
        pipeline._bilstm.rep_count = 6

        pipeline._rep_counter.rep_count = 11
        assert pipeline.rep_count == 6

        pipeline.release()


# ---------------------------------------------------------------------------
# Multi-camera: fake triangulated provider + fake clock
# ---------------------------------------------------------------------------

class _FakeClock:
    """One wall-aligned clock for capture timestamps, time.time() and perf_counter()."""

    def __init__(self) -> None:
        self.now_s = CLOCK_START_S

    def advance(self, seconds: float) -> None:
        self.now_s += seconds

    def time(self) -> float:
        return self.now_s

    def perf_counter(self) -> float:
        return self.now_s


class _FakeProvider:
    """Stands in for MultiCameraPoseProvider: world-frame skeletons stamped
    with a strictly increasing capture sequence, or None for a dropout."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self._queue: list[np.ndarray | None] = []
        self.sequence = 0
        self.swap_count = 0
        self.temporal_resets = 0
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)

    def push(self, points: np.ndarray | None, confidences: np.ndarray | None = None) -> None:
        self._queue.append((points, confidences))

    def get_pose(self):
        points, confidences = self._queue.pop(0)
        self.sequence += 1
        if points is None:
            return self.frame, None, None
        if confidences is None:
            confidences = np.full(len(points), PROVIDER_CONFIDENCE)
        skeleton = Skeleton3D.from_numpy(
            points, confidences=confidences,
            timestamp=self._clock.time(), frame_index=self.sequence,
        )
        return self.frame, None, skeleton

    def reset_temporal_state(self) -> None:
        self.temporal_resets += 1

    def release(self) -> None:
        pass


def _build_multi_camera_pipeline(monkeypatch) -> tuple:
    monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
    from biomechanics import pipeline as pipeline_module

    clock = _FakeClock()
    monkeypatch.setattr(
        pipeline_module, "time",
        types.SimpleNamespace(time=clock.time, perf_counter=clock.perf_counter),
    )
    pipe = pipeline_module.BiomechanicsPipeline(BiomechanicsConfig(), defer_capture=True)
    provider = _FakeProvider(clock)
    pipe._multi_camera_provider = provider
    # The synthetic squat bottoms ~4 cm above parallel, just past the default
    # target; count every descent here. TestDepthTargetGate covers the gate.
    pipe.set_depth_target(None)
    return pipe, provider, clock


def _step(
    pipe,
    provider: _FakeProvider,
    clock: _FakeClock,
    points: np.ndarray | None,
    confidences: np.ndarray | None = None,
) -> PipelineFrame:
    provider.push(points, confidences)
    result = pipe.process_frame()
    clock.advance(FRAME_DT_S)
    return result


def _confidences_without(keypoint_idx: int, num_keypoints: int) -> np.ndarray:
    confidences = np.full(num_keypoints, PROVIDER_CONFIDENCE)
    confidences[keypoint_idx] = 0.0
    return confidences


def _run_frames(pipe, provider, clock, depths: list[float], valgus_m: float = 0.0) -> list[PipelineFrame]:
    return [
        _step(pipe, provider, clock, world_squat_points(max(depth, MIN_DEPTH_RATIO), valgus_m=valgus_m))
        for depth in depths
    ]


def _reps(count: int) -> list[float]:
    depths: list[float] = [0.0] * READINESS_WARM_UP_FRAMES
    for _ in range(count):
        depths += squat_depth_profile()
    return depths


def _squat_thresholds(pipe) -> dict:
    """Every threshold attribute of every squat rule, keyed by rule and attribute."""
    snapshot = {}
    for rule in pipe._rule_engine.rules:
        for name, value in vars(rule).items():
            if isinstance(value, float) and not name.startswith("_last"):
                snapshot[(rule.fault_type.value, name)] = value
            elif isinstance(value, dict):
                for key, tier_value in value.items():
                    snapshot[(rule.fault_type.value, name, key)] = tier_value
    return snapshot


class TestMultiCameraPipeline:

    def test_frame_index_follows_provider_sequence(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        results = _run_frames(pipe, provider, clock, _reps(1))

        indices = [result.frame_index for result in results]
        assert indices == list(range(1, len(results) + 1))
        assert all(later > earlier for earlier, later in zip(indices, indices[1:]))

    def test_gated_frames_carry_raw_skeleton_only(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        first = _step(pipe, provider, clock, world_squat_points(0.0))

        assert not pipe.is_ready
        assert first.skeleton_3d is None
        assert first.joint_angles is None
        assert first.skeleton_3d_raw is not None
        # The raw view is hip-centred, like the analysis view will be.
        raw = first.skeleton_3d_raw.to_numpy()
        assert np.linalg.norm((raw[CK.LEFT_HIP] + raw[CK.RIGHT_HIP]) / 2.0) == pytest.approx(0.0)

    def test_ready_frames_expose_analysis_display_and_foot_state(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        results = _run_frames(pipe, provider, clock, _reps(1))
        last = results[-1]

        assert pipe.is_ready
        assert last.skeleton_3d is not None
        assert last.skeleton_3d_display is not None
        assert last.skeleton_3d_raw is not None
        assert last.foot_state is not None and last.foot_state.valid
        # Analysis lags, display does not.
        assert last.skeleton_3d.frame_index == last.frame_index - KALMAN_LAG_FRAMES
        assert last.skeleton_3d_display.frame_index == last.frame_index
        assert "pre_ik" in last.latency_ms

    def test_reps_are_counted(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        results = _run_frames(pipe, provider, clock, _reps(2))

        rep_numbers = [result.rep_data.rep_number for result in results if result.rep_data is not None]
        assert rep_numbers == [1, 2]

    def test_knee_flexion_is_never_exactly_zero(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        results = _run_frames(pipe, provider, clock, _reps(1))

        analysed = [result.joint_angles for result in results if result.joint_angles is not None]
        assert analysed
        for angles in analysed:
            for knee_flexion in (angles.knee_flexion_l, angles.knee_flexion_r):
                assert math.isfinite(knee_flexion)
                assert knee_flexion != 0.0

    def test_valgus_fault_fires_in_second_rep(self, monkeypatch):
        """W1 regression: triangulated frames once shared frame_index 0, so
        every rule fired at most once per session."""
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        results = _run_frames(pipe, provider, clock, _reps(2), valgus_m=VALGUS_SHIFT_M)

        valgus_reps = {
            fault.rep_number
            for result in results
            for fault in result.faults
            if fault.fault_type == FaultType.KNEE_VALGUS
        }
        assert 2 in valgus_reps

    def test_observed_reps_never_move_a_threshold(self, monkeypatch):
        """A2 regression: calibration once turned each athlete's faults into their
        normal (a caved first rep raised the valgus bar). Thresholds are absolute."""
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        before = _squat_thresholds(pipe)
        pipe.apply_athlete_params(ATHLETE_PARAMS)
        _run_frames(pipe, provider, clock, _reps(1), valgus_m=VALGUS_SHIFT_M)

        for _ in range(RESET_REPEATS):
            pipe.reset_readiness_gate()
            _run_frames(pipe, provider, clock, _reps(1), valgus_m=VALGUS_SHIFT_M)

        assert _squat_thresholds(pipe) == pytest.approx(before)
        assert provider.temporal_resets == RESET_REPEATS

    def test_body_calibration_completes_and_survives_reset(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        assert pipe.body_calibration.to_athlete_params() is None

        _run_frames(pipe, provider, clock, _reps(3))
        params = pipe.body_calibration.to_athlete_params()

        assert params is not None
        assert params["femur_avg_m"] == pytest.approx(ATHLETE_PARAMS["femur_avg_m"], abs=0.01)
        assert params["tibia_avg_m"] == pytest.approx(ATHLETE_PARAMS["tibia_avg_m"], abs=0.01)

        pipe.reset_readiness_gate()
        assert pipe.body_calibration.is_complete
        assert pipe.body_calibration.to_athlete_params() == params

    def test_apply_athlete_params_round_trip(self, monkeypatch):
        pipe, _, _ = _build_multi_camera_pipeline(monkeypatch)

        pipe.apply_athlete_params(ATHLETE_PARAMS)

        assert pipe.body_calibration.is_complete
        assert pipe.body_calibration.to_athlete_params() == pytest.approx(ATHLETE_PARAMS)
        # The measured femur now normalizes depth.
        assert pipe._femur_length_m() == pytest.approx(ATHLETE_PARAMS["femur_avg_m"])

    def test_dropout_is_predicted_then_dropped(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        _run_frames(pipe, provider, clock, _reps(1))
        assert pipe.is_ready

        predicted = [_step(pipe, provider, clock, None) for _ in range(DROPOUT_FRAMES)]
        assert all(result.skeleton_3d is not None for result in predicted)
        assert all(result.skeleton_3d_raw is None for result in predicted)

        remaining = MAX_PREDICTED_FRAMES + KALMAN_LAG_FRAMES + 2 - DROPOUT_FRAMES
        tail = [_step(pipe, provider, clock, None) for _ in range(remaining)]
        assert tail[-1].skeleton_3d is None
        assert pipe.is_ready

        # A returning pose is visible at once in the raw view; the lagged
        # analysis stream re-seeds once the lag window refills.
        back = [_step(pipe, provider, clock, world_squat_points(MIN_DEPTH_RATIO)) for _ in range(KALMAN_LAG_FRAMES + 1)]
        assert back[0].skeleton_3d_raw is not None
        assert back[-1].skeleton_3d is not None

    def test_single_frame_spike_does_not_change_rep_depth(self, monkeypatch):
        """Rep depth is the max of a running median of knee flexion, so one
        transient IK frame cannot over-read it."""
        clean_pipe, clean_provider, clean_clock = _build_multi_camera_pipeline(monkeypatch)
        profile = _reps(1)
        clean_results = _run_frames(clean_pipe, clean_provider, clean_clock, profile)

        spiked_pipe, spiked_provider, spiked_clock = _build_multi_camera_pipeline(monkeypatch)
        real_solve = spiked_pipe._ik_solver.solve
        spiked_frames: list[int] = []

        def spiked_solve(skeleton):
            angles = real_solve(skeleton)
            if not spiked_frames and angles.avg_knee_flexion > SPIKE_TRIGGER_KNEE_DEG:
                spiked_frames.append(skeleton.frame_index)
                angles.knee_flexion_l += SPIKE_DEG
                angles.knee_flexion_r += SPIKE_DEG
            return angles

        monkeypatch.setattr(spiked_pipe._ik_solver, "solve", spiked_solve)
        spiked_results = _run_frames(spiked_pipe, spiked_provider, spiked_clock, profile)
        assert len(spiked_frames) == 1

        clean_depth = next(r.rep_data.max_depth_angle for r in clean_results if r.rep_data is not None)
        spiked_depth = next(r.rep_data.max_depth_angle for r in spiked_results if r.rep_data is not None)
        assert spiked_depth == pytest.approx(clean_depth, abs=DEPTH_MATCH_TOL_DEG)

        # The bottom frame comes from the same statistic: it is not the spiked frame.
        _, spiked_bottom_angles = spiked_pipe.consume_bottom_frame()
        spiked_bottom_knee = (spiked_bottom_angles["knee_flexion_l"] + spiked_bottom_angles["knee_flexion_r"]) / 2.0
        assert spiked_bottom_knee == pytest.approx(clean_depth, abs=DEPTH_MATCH_TOL_DEG)


LOST_ANKLE_FRAMES = 25


class TestMissingKeypointsDoNotFakeMotion:
    def test_lost_ankle_never_starts_a_rep(self, monkeypatch):
        """A lost ankle sits at the hip origin after re-centring; the rep
        signal must read it as missing, not as the hips dropping to the floor."""
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        standing = world_squat_points(MIN_DEPTH_RATIO)
        _run_frames(pipe, provider, clock, [0.0] * READINESS_WARM_UP_FRAMES)
        assert pipe.is_ready

        no_left_ankle = _confidences_without(CK.LEFT_ANKLE, len(standing))
        results = [
            _step(pipe, provider, clock, standing, no_left_ankle)
            for _ in range(LOST_ANKLE_FRAMES)
        ]
        results += [_step(pipe, provider, clock, standing) for _ in range(LOST_ANKLE_FRAMES)]

        assert pipe.rep_count == 0
        assert all(result.rep_data is None for result in results)
        assert not pipe._rep_counter.in_rep

    def test_rep_signal_is_nan_when_an_ankle_is_missing(self):
        from biomechanics.profiles import get_profile

        profile = get_profile("Barbell Back Squat")
        points = world_squat_points(MIN_DEPTH_RATIO)
        confidences = _confidences_without(CK.RIGHT_ANKLE, len(points))
        skeleton = Skeleton3D.from_numpy(points, confidences=confidences, timestamp=0.0, frame_index=0)

        assert math.isnan(profile.get_rep_signal(skeleton))
        assert not math.isnan(
            profile.get_rep_signal(
                Skeleton3D.from_numpy(points, confidences=np.full(len(points), PROVIDER_CONFIDENCE),
                                      timestamp=0.0, frame_index=0)
            )
        )

    def test_rules_do_not_fire_on_predicted_frames(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        _run_frames(pipe, provider, clock, _reps(1))
        assert pipe.is_ready

        evaluated_on: list[bool] = []
        real_evaluate = pipe._rule_engine.evaluate

        def _counting_evaluate(*args, **kwargs):
            evaluated_on.append(True)
            return real_evaluate(*args, **kwargs)

        monkeypatch.setattr(pipe._rule_engine, "evaluate", _counting_evaluate)

        _step(pipe, provider, clock, world_squat_points(MIN_DEPTH_RATIO))
        assert len(evaluated_on) == 1

        predicted = [_step(pipe, provider, clock, None) for _ in range(DROPOUT_FRAMES)]
        assert all(result.skeleton_3d is not None for result in predicted)
        assert len(evaluated_on) == 1


class TestPerRepVerdicts:
    def test_rep_fault_lands_on_the_completed_rep(self, monkeypatch):
        """Whole-rep verdicts belong to the rep that produced them, in both the
        rep's fault list and the frame that completes it."""
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        rep_results = _run_frames(pipe, provider, clock, _reps(1), valgus_m=VALGUS_SHIFT_M)
        completed = [result for result in rep_results if result.rep_data is not None]
        assert len(completed) == 1
        rep_data = completed[0].rep_data
        knee_faults = [fault for fault in rep_data.faults if fault.fault_type == FaultType.KNEE_VALGUS]
        assert knee_faults
        assert knee_faults[0].rep_number == rep_data.rep_number
        assert FaultType.KNEE_VALGUS in {fault.fault_type for fault in completed[0].faults}

    def test_completed_rep_carries_its_features(self, monkeypatch):
        """The same numbers the rules judged travel on the rep to the diagnosis."""
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        rep_results = _run_frames(pipe, provider, clock, _reps(1))
        rep_data = next(result.rep_data for result in rep_results if result.rep_data is not None)
        features = rep_data.features
        assert features["rep_number"] == rep_data.rep_number
        assert math.isfinite(features["depth_ratio"])
        assert features["depth_ratio"] < SHALLOW_SYNTHETIC_DEPTH_RATIO_CEILING
        assert math.isfinite(features["concentric_velocity_mps"])

    def test_set_reset_clears_per_rep_rule_state(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        _run_frames(pipe, provider, clock, [0.0] * READINESS_WARM_UP_FRAMES)
        # Half a rep, then the set ends mid-descent.
        for depth in squat_depth_profile()[: len(squat_depth_profile()) // 2]:
            _step(pipe, provider, clock, world_squat_points(max(depth, MIN_DEPTH_RATIO)))
        pipe.reset_readiness_gate()
        bar_rule = pipe._rule_engine.get_rule(FaultType.BILATERAL_ASYMMETRY)
        assert bar_rule._samples == 0
        assert not bar_rule._in_rep_prev
        assert len(pipe._rep_trajectory) == 0
        assert pipe._rep_setup is None


# The synthetic squat's thigh stops 5° above horizontal: a depth ratio of
# sin(5°) ≈ 0.087, just past the default target's 0.08 tolerance.
SHALLOW_SYNTHETIC_DEPTH_RATIO_CEILING = 0.15
LENIENT_DEPTH_TARGET = 0.10


class TestDepthTargetGate:
    def test_rep_short_of_the_target_is_not_counted_and_says_why(self, monkeypatch):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        pipe.set_depth_target(0.0)
        results = _run_frames(pipe, provider, clock, _reps(1))

        assert all(result.rep_data is None for result in results)
        shallow = [result for result in results if result.shallow_rep_class is not None]
        assert len(shallow) == 1
        depth_faults = [fault for fault in shallow[0].faults if fault.fault_type == FaultType.DEPTH]
        assert depth_faults and depth_faults[0].details["shallow_rep"] is True
        assert pipe.rep_count == 0

    def test_rep_within_the_athletes_target_counts(self, monkeypatch):
        """A target above parallel (limited range) accepts the same rep."""
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        pipe.set_depth_target(LENIENT_DEPTH_TARGET)
        results = _run_frames(pipe, provider, clock, _reps(2))

        reps = [result.rep_data for result in results if result.rep_data is not None]
        assert [rep.rep_number for rep in reps] == [1, 2]
        assert all(rep.depth_target_met for rep in reps)

    def test_uncalibrated_squat_starts_with_a_lenient_target(self, monkeypatch):
        """Refusing reps of an athlete not yet measured is worse than counting a high one."""
        from biomechanics import pipeline as pipeline_module

        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
        pipe = pipeline_module.BiomechanicsPipeline(BiomechanicsConfig(), defer_capture=True)
        depth = BiomechanicsConfig().faults.depth
        assert pipe._rule_engine.depth_target_ratio == depth.uncalibrated_target_ratio
        assert depth.uncalibrated_target_ratio > depth.default_target_ratio


CALIBRATED_DEPTH_TARGET = 0.12


def _pipeline_for(monkeypatch, exercise_name: str, bilstm_enabled: bool = False):
    from biomechanics import pipeline as pipeline_module

    monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
    config = BiomechanicsConfig()
    config.bilstm.enabled = bilstm_enabled
    return pipeline_module.BiomechanicsPipeline(config, exercise_name=exercise_name, defer_capture=True)


class TestExerciseProfiles:
    def test_bilstm_counts_squats_only(self, monkeypatch):
        """The squat-trained BiLSTM would never see an overhead press rep and
        would override the press's own counter."""
        assert _pipeline_for(monkeypatch, "Barbell Back Squat", bilstm_enabled=True)._bilstm is not None
        assert _pipeline_for(monkeypatch, "Barbell Overhead Press", bilstm_enabled=True)._bilstm is None

    def test_set_exercise_rebuilds_rules_and_counting(self, monkeypatch):
        pipe = _pipeline_for(monkeypatch, "Barbell Back Squat", bilstm_enabled=True)
        pipe.set_exercise("Barbell Overhead Press")

        assert pipe.profile.name == "overhead_press"
        assert pipe._bilstm is None
        assert not pipe._depth_gated
        assert {rule.fault_type for rule in pipe._rule_engine.rules} == {
            FaultType.LOCKOUT, FaultType.ELBOW_FLARE, FaultType.BAR_PATH, FaultType.BILATERAL_ASYMMETRY,
        }

    def test_depth_target_survives_a_switch_away_and_back(self, monkeypatch):
        pipe = _pipeline_for(monkeypatch, "Barbell Back Squat")
        pipe.set_depth_target(CALIBRATED_DEPTH_TARGET)

        pipe.set_exercise("Barbell Overhead Press")
        pipe.set_exercise("Barbell Back Squat")

        assert pipe._depth_gated
        assert pipe._rule_engine.depth_target_ratio == pytest.approx(CALIBRATED_DEPTH_TARGET)

    def test_squat_bilstm_only_segments_after_a_switch_back(self, monkeypatch):
        """Rebuilt for the squat, the BiLSTM must open a rep on any descent and
        leave counting to the depth target; at its default parallel class an
        athlete whose target sits above parallel would silently never count."""
        from biomechanics.ml.bilstm_counter import ASSESSMENT_MIN_DEPTH_CLASS

        pipe = _pipeline_for(monkeypatch, "Barbell Back Squat", bilstm_enabled=True)
        pipe.set_depth_target(CALIBRATED_DEPTH_TARGET)

        pipe.set_exercise("Reverse Lunge")
        pipe.set_exercise("Barbell Back Squat")

        assert pipe._depth_gated
        assert pipe._bilstm._counter.config.min_depth_class == ASSESSMENT_MIN_DEPTH_CLASS
        assert ASSESSMENT_MIN_DEPTH_CLASS < BiomechanicsConfig().bilstm.min_depth_class
        assert pipe._rule_engine.depth_target_ratio == pytest.approx(CALIBRATED_DEPTH_TARGET)

    def test_calibration_loaded_during_another_exercise_reaches_the_squat(self, monkeypatch):
        """A workout that opens with a press still hands its squat the stored depth target."""
        from biomechanics.calibration import apply_calibration_to_rule_engine

        pipe = _pipeline_for(monkeypatch, "Barbell Overhead Press")
        apply_calibration_to_rule_engine(pipe._rule_engine, {"depth_target_ratio": CALIBRATED_DEPTH_TARGET})

        pipe.set_exercise("Barbell Back Squat")

        assert pipe._rule_engine.depth_target_ratio == pytest.approx(CALIBRATED_DEPTH_TARGET)

    def test_uncalibrated_squat_after_another_exercise_keeps_the_lenient_target(self, monkeypatch):
        pipe = _pipeline_for(monkeypatch, "Barbell Overhead Press")
        pipe.set_exercise("Barbell Back Squat")

        lenient = BiomechanicsConfig().faults.depth.uncalibrated_target_ratio
        assert pipe._rule_engine.depth_target_ratio == pytest.approx(lenient)


class TestTwoCameraRigs:
    def test_two_camera_rig_lowers_the_body_measurement_gate(self, monkeypatch):
        from biomechanics.pipeline import TWO_CAMERA_MEASUREMENT_CONFIDENCE
        from biomechanics.triangulation.triangulator import TWO_VIEW_CONFIDENCE_CAP

        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
        from biomechanics import pipeline as pipeline_module

        config = BiomechanicsConfig()
        config.triangulation.device_ids = [0, 1]
        pipe = pipeline_module.BiomechanicsPipeline(config, defer_capture=True)
        assert pipe.body_calibration._min_endpoint_confidence == pytest.approx(TWO_CAMERA_MEASUREMENT_CONFIDENCE)
        assert TWO_CAMERA_MEASUREMENT_CONFIDENCE < TWO_VIEW_CONFIDENCE_CAP


# ---------------------------------------------------------------------------
# Camera calibration: world-state reset on install, provider wiring
# ---------------------------------------------------------------------------

CALIBRATION_BUFFER_FRAMES = 123
BAR_DETECTION_STRIDE = 4
CAMERA_KEYS = {1: "usb_left"}


class _FakeBarbellDetector:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _build_recording_provider_pipeline(monkeypatch, config: BiomechanicsConfig) -> tuple:
    """Real pipeline whose MultiCameraPoseProvider only records its constructor arguments."""
    monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
    import biomechanics.barbell_tracking as barbell_tracking_module
    import biomechanics.pose.multi_camera as multi_camera_module
    from biomechanics import pipeline as pipeline_module

    received: dict[str, object] = {}

    class _RecordingProvider:
        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

    monkeypatch.setattr(multi_camera_module, "MultiCameraPoseProvider", _RecordingProvider)
    monkeypatch.setattr(barbell_tracking_module, "BarbellDetector", _FakeBarbellDetector)
    pipe = pipeline_module.BiomechanicsPipeline(config, defer_capture=True)
    return pipe, received


class TestCameraCalibrationChange:
    @pytest.mark.parametrize("world_frame_changed", [False, True])
    def test_every_install_resets_temporal_and_foot_state_but_keeps_body_measurements(
        self, monkeypatch, world_frame_changed
    ):
        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        _run_frames(pipe, provider, clock, _reps(3))
        params = pipe.body_calibration.to_athlete_params()
        resets_before = provider.temporal_resets

        pipe.on_calibration_changed(world_frame_changed=world_frame_changed)
        result = _step(pipe, provider, clock, world_squat_points(MIN_DEPTH_RATIO))

        assert provider.temporal_resets == resets_before + 1
        # Kalman history is gone: the lagged analysis frame is the current one.
        assert result.skeleton_3d.frame_index == result.frame_index
        # Even a kept world frame shifts ~1 cm: anchors and floor restart rather than carry a bias.
        assert not result.foot_state.valid
        assert params is not None
        assert pipe.body_calibration.to_athlete_params() == params


class TestCameraCalibrationWiring:
    def test_provider_receives_camera_calibration_config(self, monkeypatch):
        config = BiomechanicsConfig()
        config.camera_calibration.camera_keys = CAMERA_KEYS
        config.camera_calibration.calibration_buffer_frames = CALIBRATION_BUFFER_FRAMES
        config.camera_calibration.bar_detection_stride = BAR_DETECTION_STRIDE

        _, received = _build_recording_provider_pipeline(monkeypatch, config)

        assert received["camera_keys"] == CAMERA_KEYS
        assert received["calibration_buffer_frames"] == CALIBRATION_BUFFER_FRAMES
        assert received["bar_detection_stride"] == BAR_DETECTION_STRIDE
        assert received["bar_detector"] is None

    def test_bar_scale_gives_the_provider_a_detector_without_per_frame_tracking(self, monkeypatch, tmp_path):
        model_path = tmp_path / "barbell_keypoints.pt"
        model_path.write_bytes(b"")
        config = BiomechanicsConfig()
        config.camera_calibration.use_bar_scale = True
        config.barbell_tracking.model_path = str(model_path)

        pipe, received = _build_recording_provider_pipeline(monkeypatch, config)

        assert isinstance(received["bar_detector"], _FakeBarbellDetector)
        assert received["bar_detector"].kwargs["model_path"] == str(model_path)
        # Tracking is off, so sets never pay for bar detection.
        assert pipe._barbell_detector is None

    def test_bar_scale_without_the_model_falls_back_to_height(self, monkeypatch, tmp_path, caplog):
        config = BiomechanicsConfig()
        config.camera_calibration.use_bar_scale = True
        config.barbell_tracking.model_path = str(tmp_path / "missing.pt")

        with caplog.at_level("WARNING", logger="biomechanics.pipeline"):
            pipe, received = _build_recording_provider_pipeline(monkeypatch, config)

        assert received["bar_detector"] is None
        assert pipe._barbell_detector is None
        assert any("use_bar_scale" in record.getMessage() for record in caplog.records)

    def test_configured_calibration_file_that_does_not_exist_yet_is_left_to_the_session(self, monkeypatch, tmp_path):
        config = BiomechanicsConfig()
        config.triangulation.calibration_file = str(tmp_path / "not_written_yet.json")

        _, received = _build_recording_provider_pipeline(monkeypatch, config)

        assert received["device_ids"] == config.triangulation.device_ids

    def test_barbell_tracking_shares_its_detector_with_the_provider(self, monkeypatch):
        config = BiomechanicsConfig()
        config.barbell_tracking.enabled = True

        pipe, received = _build_recording_provider_pipeline(monkeypatch, config)

        assert isinstance(pipe._barbell_detector, _FakeBarbellDetector)
        assert received["bar_detector"] is pipe._barbell_detector


class _RecordingIPCClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


class TestPipelineToDiagnosis:
    def test_live_reps_reach_the_set_diagnosis(self, monkeypatch):
        """Pipeline output must survive the session tracker into a diagnosis.

        The tracker once crashed on every rep (inside a try/except) because
        the pipeline's trajectory samples changed type — no set was ever
        diagnosed, and no unit test drove real pipeline output through it.
        """
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.coaching.session_tracker import SessionTracker
        from biomechanics.config import IPCConfig

        pipe, provider, clock = _build_multi_camera_pipeline(monkeypatch)
        client = _RecordingIPCClient()
        tracker = SessionTracker(IPCBridge(client, ipc_config=IPCConfig()))
        tracker.set_athlete_params(ATHLETE_PARAMS, {})

        # Consume on the frame that completes each rep, as pipeline_process does.
        for depth in _reps(3):
            result = _step(
                pipe, provider, clock,
                world_squat_points(max(depth, MIN_DEPTH_RATIO), valgus_m=VALGUS_SHIFT_M),
            )
            if result.rep_data is None:
                continue
            bottom_kpts, bottom_angles = pipe.consume_bottom_frame()
            tracker.on_rep_complete(
                result.rep_data,
                bottom_kpts=bottom_kpts,
                bottom_angles=bottom_angles,
                standing_kpts=pipe.consume_standing_frame(),
                trajectory_samples=pipe.consume_rep_trajectory(),
            )

        assert len(tracker._rep_kinematic_buffer) == 3
        assert all(trajectory is not None for trajectory in tracker._rep_trajectory_buffer)
        tracker._end_current_set()

        diagnoses = [message for message in client.messages if message["type"] == "diagnosis_complete"]
        assert len(diagnoses) == 1
        symptoms = {symptom["symptom_id"] for symptom in diagnoses[0]["diagnosis"]["detected_symptoms"]}
        assert "knee_not_tracking_toes" in symptoms
        rep_messages = [message for message in client.messages if message["type"] == "rep_complete"]
        assert all(message["features"]["rep_number"] == message["rep_number"] for message in rep_messages)
