"""Behavioural tests for the fixed-lag keypoint Kalman smoother (pre-IK contract C4) on synthetic squats."""
from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.utils.keypoint_kalman import (
    INITIAL_VELOCITY_VARIANCE,
    MAX_REJECTION_S,
    MIN_SNAP_CONFIDENCE,
    REACQUIRE_AGREEING_FRAMES,
    TIMEOUT_AGREEING_FRAMES,
    FixedLagKeypointSmoother,
    KeypointKalmanOutput,
)
from biomechanics.utils.types import CocoKeypoints as CK

FPS = 30.0
FRAME_DT_S = 1.0 / FPS
START_TIMESTAMP_S = 1000.0
KEYPOINT_COUNT = 21
LAG_FRAMES = 2
PROCESS_NOISE = 10.0
UNCERTAINTY_SCALE_M = 0.02
MIN_OUTPUT_CONFIDENCE = 0.15
MAX_PREDICTED_FRAMES = 5
NOISE_STD_M = 0.01
THREE_PX_NOISE_STD_M = 0.0087  # hip-centred per-axis std at 3 px (velocity-clamp audit)
SEEDS = (0, 1, 2, 3, 4)
FINE_RATE_HZ = 3000.0

EXACT_TOLERANCE_M = 1e-9
PEAK_ERROR_TOLERANCE_M = 0.005
NOISE_FREE_PEAK_TOLERANCE_M = 0.001
MAX_KNEE_CAVE_PEAK_LOSS_RATIO = 0.25
MIN_JITTER_REDUCTION_RATIO = 0.70
MISSING_KEYPOINT_TOLERANCE_M = 0.01
OUTLIER_TOLERANCE_M = 0.03
BURST_DISPLAY_TOLERANCE_M = 0.05  # undelayed stream extrapolates a noisy velocity for the burst's length
VELOCITY_TOLERANCE_M_PER_S = 0.001
PREDICTION_TOLERANCE_M = 0.001
PROFILE_TOLERANCE = 0.1

OUTLIER_OFFSET_M = 0.30
BURST_OFFSET_M = 0.60
BURST_CONFIDENCES = (0.20, 0.55)
GOOD_CONFIDENCE = 0.55  # u = 1.8 cm, consistent with 1 cm per-axis noise
JUMP_OFFSET_M = 1.0
PERSISTENT_JUMP_FRAMES = 10
TIMEOUT_OFFSET_M = 2.0
KNEE_CAVE_M = 0.08
KNEE_CAVE_DURATION_S = 0.30
EXPLOSIVE_PEAK_SPEED_M_PER_S = 3.0
EXPLOSIVE_PEAK_ACCELERATION_M_PER_S2 = 80.0

STANDING_POSITIONS_M: dict[int, tuple[float, float, float]] = {
    CK.NOSE: (0.0, -0.62, -0.08), CK.LEFT_EYE: (0.03, -0.66, -0.06), CK.RIGHT_EYE: (-0.03, -0.66, -0.06),
    CK.LEFT_EAR: (0.07, -0.64, 0.0), CK.RIGHT_EAR: (-0.07, -0.64, 0.0),
    CK.LEFT_SHOULDER: (0.19, -0.50, 0.0), CK.RIGHT_SHOULDER: (-0.19, -0.50, 0.0),
    CK.LEFT_ELBOW: (0.22, -0.22, 0.0), CK.RIGHT_ELBOW: (-0.22, -0.22, 0.0),
    CK.LEFT_WRIST: (0.23, 0.05, -0.02), CK.RIGHT_WRIST: (-0.23, 0.05, -0.02),
    CK.LEFT_HIP: (0.13, 0.0, 0.0), CK.RIGHT_HIP: (-0.13, 0.0, 0.0),
    CK.LEFT_KNEE: (0.17, 0.45, -0.02), CK.RIGHT_KNEE: (-0.17, 0.45, -0.02),
    CK.LEFT_ANKLE: (0.20, 0.90, 0.0), CK.RIGHT_ANKLE: (-0.20, 0.90, 0.0),
    CK.LEFT_FOOT_INDEX: (0.25, 0.95, -0.18), CK.RIGHT_FOOT_INDEX: (-0.25, 0.95, -0.18),
    CK.LEFT_HEEL: (0.20, 0.95, 0.05), CK.RIGHT_HEEL: (-0.20, 0.95, 0.05),
}
STANDING_POSE = np.array([STANDING_POSITIONS_M[index] for index in range(KEYPOINT_COUNT)])

# Displacement at full depth (Y down, +Z back): upper body down and back, knees down and forward, feet static.
HIP_KEYPOINTS = (CK.LEFT_HIP, CK.RIGHT_HIP)
KNEE_KEYPOINTS = (CK.LEFT_KNEE, CK.RIGHT_KNEE)
FOOT_KEYPOINTS = (CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX, CK.LEFT_HEEL, CK.RIGHT_HEEL)
UPPER_BODY_DISPLACEMENT_M = (0.0, 0.42, 0.20)
HIP_DISPLACEMENT_M = (0.0, 0.45, 0.25)
KNEE_DISPLACEMENT_M = (0.0, 0.20, -0.15)
SQUAT_DISPLACEMENT_M = np.array([
    HIP_DISPLACEMENT_M if index in HIP_KEYPOINTS
    else KNEE_DISPLACEMENT_M if index in KNEE_KEYPOINTS
    else (0.0, 0.0, 0.0) if index in FOOT_KEYPOINTS
    else UPPER_BODY_DISPLACEMENT_M
    for index in range(KEYPOINT_COUNT)
])
EXPLOSIVE_DIRECTION = np.array([0.0, 0.8, -0.6])


def _confidence_for_noise(axis_std_m: float) -> float:
    # C3: u is the 3D position std; isotropic per-axis noise sigma gives u = sqrt(3) * sigma.
    uncertainty_m = math.sqrt(3.0) * axis_std_m
    return 1.0 / (1.0 + (uncertainty_m / UNCERTAINTY_SCALE_M) ** 2)


def _timestamp(frame: int) -> float:
    return START_TIMESTAMP_S + frame * FRAME_DT_S


def _make_smoother(**overrides: float) -> FixedLagKeypointSmoother:
    settings = dict(lag_frames=LAG_FRAMES, process_noise=PROCESS_NOISE, uncertainty_scale_m=UNCERTAINTY_SCALE_M,
                    max_predicted_frames=MAX_PREDICTED_FRAMES, min_output_confidence=MIN_OUTPUT_CONFIDENCE)
    settings.update(overrides)
    return FixedLagKeypointSmoother(**settings)


def _depth_profile(descent_s: float = 1.0, ascent_s: float = 0.8, stand_s: float = 1.0,
                   reps: int = 6) -> tuple[np.ndarray, list[tuple[int, int]]]:
    depth = [0.0] * int(stand_s * FPS)
    windows = []
    for _ in range(reps):
        start = len(depth)
        descent_frames, ascent_frames = int(round(descent_s * FPS)), int(round(ascent_s * FPS))
        depth += [0.5 * (1.0 - math.cos(math.pi * (i + 1) / descent_frames)) for i in range(descent_frames)]
        depth += [0.5 * (1.0 + math.cos(math.pi * (i + 1) / ascent_frames)) for i in range(ascent_frames)]
        windows.append((start, len(depth) - 1))
        depth += [0.0] * int(stand_s * FPS)
    return np.array(depth), windows


def _squat_session(reps: int = 6) -> tuple[np.ndarray, list[tuple[int, int]]]:
    depth, windows = _depth_profile(reps=reps)
    return STANDING_POSE[None] + depth[:, None, None] * SQUAT_DISPLACEMENT_M[None], windows


def _standing_session(frame_count: int) -> np.ndarray:
    return np.repeat(STANDING_POSE[None], frame_count, axis=0)


def _add_noise(points: np.ndarray, axis_std_m: float, seed: int) -> np.ndarray:
    return points + np.random.default_rng(seed).normal(0.0, axis_std_m, points.shape)


def _explosive_fine_displacement(stand_s: float = 0.5, cruise_s: float = 0.10, reps: int = 3) -> np.ndarray:
    # Raised-cosine acceleration pulses: 0 -> +v, +v -> -v (bounce), -v -> 0; peaks exactly v and a.
    pulse_s = 2.0 * EXPLOSIVE_PEAK_SPEED_M_PER_S / EXPLOSIVE_PEAK_ACCELERATION_M_PER_S2
    speed = EXPLOSIVE_PEAK_SPEED_M_PER_S
    segments = [(stand_s, 0.0, 0.0), (pulse_s, 0.0, speed), (cruise_s, speed, speed),
                (2.0 * pulse_s, speed, -speed), (cruise_s, -speed, -speed), (pulse_s, -speed, 0.0)]
    velocity = []
    for _ in range(reps):
        for duration_s, start_speed, end_speed in segments:
            phase = np.arange(int(round(duration_s * FINE_RATE_HZ))) / (duration_s * FINE_RATE_HZ)
            velocity.append(start_speed + (end_speed - start_speed)
                            * (phase - np.sin(2.0 * math.pi * phase) / (2.0 * math.pi)))
    velocity.append(np.zeros(int(stand_s * FINE_RATE_HZ)))
    return np.cumsum(np.concatenate(velocity)) / FINE_RATE_HZ


def _explosive_displacement() -> np.ndarray:
    return _explosive_fine_displacement()[::int(FINE_RATE_HZ / FPS)]


def _run(points: np.ndarray, confidences: np.ndarray,
         smoother: FixedLagKeypointSmoother | None = None) -> dict[str, np.ndarray]:
    smoother = smoother or _make_smoother()
    frame_count = len(points)
    lagged = np.empty_like(points)
    current = np.empty_like(points)
    lagged_confidences = np.empty(points.shape[:2])
    current_confidences = np.empty(points.shape[:2])
    rejected = np.zeros(points.shape[:2], dtype=bool)
    for frame in range(frame_count):
        output = smoother.update(points[frame], confidences[frame], _timestamp(frame))
        lagged[frame] = output.lagged_points
        current[frame] = output.current_points
        lagged_confidences[frame] = output.lagged_confidences
        current_confidences[frame] = output.current_confidences
        rejected[frame] = output.gate_rejected
    aligned = np.concatenate([lagged[LAG_FRAMES:], np.repeat(lagged[-1:], LAG_FRAMES, axis=0)])
    return dict(aligned=aligned, current=current, lagged_confidences=lagged_confidences,
                current_confidences=current_confidences, rejected=rejected)


def _uniform_confidences(points: np.ndarray, confidence: float) -> np.ndarray:
    return np.full(points.shape[:2], confidence)


def _reference_fixed_lag(measurements: np.ndarray, variances: np.ndarray, lag: int) -> np.ndarray:
    # Textbook matrix Kalman filter + RTS over each lag window, one scalar sequence, 30 Hz.
    transition = np.array([[1.0, FRAME_DT_S], [0.0, 1.0]])
    process = PROCESS_NOISE * np.array([[FRAME_DT_S ** 3 / 3.0, FRAME_DT_S ** 2 / 2.0],
                                        [FRAME_DT_S ** 2 / 2.0, FRAME_DT_S]])
    state = np.array([measurements[0], 0.0])
    covariance = np.diag([variances[0], INITIAL_VELOCITY_VARIANCE])
    filtered, filtered_cov, predicted, predicted_cov = [state], [covariance], [state], [covariance]
    lagged = np.empty(len(measurements))
    lagged[0] = measurements[0]
    for frame in range(1, len(measurements)):
        state_prior = transition @ state
        cov_prior = transition @ covariance @ transition.T + process
        gain = cov_prior[:, 0] / (cov_prior[0, 0] + variances[frame])
        state = state_prior + gain * (measurements[frame] - state_prior[0])
        covariance = cov_prior - np.outer(gain, cov_prior[0])
        filtered.append(state), filtered_cov.append(covariance)
        predicted.append(state_prior), predicted_cov.append(cov_prior)
        start = max(0, frame - lag)
        smoothed = state
        for k in range(frame - 1, start - 1, -1):
            smoother_gain = filtered_cov[k] @ transition.T @ np.linalg.inv(predicted_cov[k + 1])
            smoothed = filtered[k] + smoother_gain @ (smoothed - predicted[k + 1])
        lagged[frame] = smoothed[0]
    return lagged


class TestConstruction:
    def test_negative_lag_raises(self) -> None:
        with pytest.raises(ValueError):
            FixedLagKeypointSmoother(lag_frames=-1)

    def test_non_positive_process_noise_raises(self) -> None:
        with pytest.raises(ValueError):
            FixedLagKeypointSmoother(process_noise=0.0)

    def test_floor_above_ceiling_raises(self) -> None:
        with pytest.raises(ValueError):
            FixedLagKeypointSmoother(measurement_std_floor_m=0.1, measurement_std_ceiling_m=0.05)


class TestInitialisation:
    def test_returns_none_until_first_valid_measurement(self) -> None:
        smoother = _make_smoother()
        assert smoother.predict_missing(_timestamp(0)) is None
        assert smoother.update(STANDING_POSE, np.zeros(KEYPOINT_COUNT), _timestamp(1)) is None
        output = smoother.update(STANDING_POSE, np.full(KEYPOINT_COUNT, 0.5), _timestamp(2))
        assert isinstance(output, KeypointKalmanOutput)
        np.testing.assert_allclose(output.current_points, STANDING_POSE, atol=EXACT_TOLERANCE_M)
        np.testing.assert_allclose(output.lagged_points, STANDING_POSE, atol=EXACT_TOLERANCE_M)
        assert output.lagged_timestamp == pytest.approx(_timestamp(2), abs=EXACT_TOLERANCE_M)

    def test_never_seeds_from_zero_confidence_keypoint(self) -> None:
        smoother = _make_smoother()
        garbage = STANDING_POSE.copy()
        garbage[CK.LEFT_ANKLE] = (5.0, 5.0, 5.0)
        confidences = np.full(KEYPOINT_COUNT, 0.5)
        confidences[CK.LEFT_ANKLE] = 0.0
        output = smoother.update(garbage, confidences, _timestamp(0))
        assert output.current_confidences[CK.LEFT_ANKLE] == 0.0
        assert np.linalg.norm(output.current_points[CK.LEFT_ANKLE]) < MISSING_KEYPOINT_TOLERANCE_M
        output = smoother.update(STANDING_POSE, np.full(KEYPOINT_COUNT, 0.5), _timestamp(1))
        np.testing.assert_allclose(output.current_points[CK.LEFT_ANKLE], STANDING_POSE[CK.LEFT_ANKLE],
                                   atol=EXACT_TOLERANCE_M)
        assert output.current_confidences[CK.LEFT_ANKLE] > 0.0
        assert not output.gate_rejected[CK.LEFT_ANKLE]

    def test_reset_returns_to_uninitialised(self) -> None:
        smoother = _make_smoother()
        smoother.update(STANDING_POSE, np.full(KEYPOINT_COUNT, 0.5), _timestamp(0))
        smoother.reset()
        assert smoother.predict_missing(_timestamp(1)) is None
        moved = STANDING_POSE + 1.0
        output = smoother.update(moved, np.full(KEYPOINT_COUNT, 0.5), _timestamp(2))
        np.testing.assert_allclose(output.current_points, moved, atol=EXACT_TOLERANCE_M)


class TestClosedFormSmoothing:
    def test_lagged_output_matches_matrix_fixed_lag_smoother(self) -> None:
        rng = np.random.default_rng(7)
        frame_count = 60
        points = np.cumsum(rng.normal(0.0, 0.02, (frame_count, 2, 3)), axis=0)
        confidences = np.stack([np.full(frame_count, 0.3), np.full(frame_count, 0.8)], axis=1)
        smoother = _make_smoother(gate_sigma=1e9, gate_min_radius_m=1e9)
        lagged = np.stack([smoother.update(points[f], confidences[f], _timestamp(f)).lagged_points
                           for f in range(frame_count)])
        for keypoint in range(2):
            variance = UNCERTAINTY_SCALE_M ** 2 * (1.0 / confidences[:, keypoint] - 1.0)
            for axis in range(3):
                expected = _reference_fixed_lag(points[:, keypoint, axis], variance, LAG_FRAMES)
                np.testing.assert_allclose(lagged[:, keypoint, axis], expected, atol=EXACT_TOLERANCE_M)

    def test_lagged_timestamp_uses_available_history_then_full_lag(self) -> None:
        smoother = _make_smoother()
        lagged_timestamps = [smoother.update(STANDING_POSE, np.full(KEYPOINT_COUNT, 0.5), _timestamp(f)).lagged_timestamp
                             for f in range(6)]
        expected = [_timestamp(0)] * (LAG_FRAMES + 1) + [_timestamp(f - LAG_FRAMES) for f in range(LAG_FRAMES + 1, 6)]
        assert lagged_timestamps == pytest.approx(expected, abs=EXACT_TOLERANCE_M)

    def test_velocities_match_constant_velocity_motion(self) -> None:
        velocity = np.array([0.3, -1.0, 0.5])
        points = STANDING_POSE[None] + np.arange(60)[:, None, None] * FRAME_DT_S * velocity
        smoother = _make_smoother()
        for frame in range(60):
            output = smoother.update(points[frame], np.full(KEYPOINT_COUNT, 0.5), _timestamp(frame))
        np.testing.assert_allclose(output.velocities, np.tile(velocity, (KEYPOINT_COUNT, 1)),
                                   atol=VELOCITY_TOLERANCE_M_PER_S)
        np.testing.assert_allclose(output.lagged_points, points[-1 - LAG_FRAMES], atol=PREDICTION_TOLERANCE_M)


class TestSquatAccuracy:
    def test_knee_height_peak_error_within_5mm_at_1cm_noise(self) -> None:
        truth, windows = _squat_session()
        confidences = _uniform_confidences(truth, _confidence_for_noise(NOISE_STD_M))
        peak_errors = []
        for seed in SEEDS:
            aligned = _run(_add_noise(truth, NOISE_STD_M, seed), confidences)["aligned"]
            for start, end in windows:
                peak_errors.append(aligned[start:end + 1, CK.LEFT_KNEE, 1].max()
                                   - truth[start:end + 1, CK.LEFT_KNEE, 1].max())
        assert float(np.mean(np.abs(peak_errors))) < PEAK_ERROR_TOLERANCE_M

    def test_noise_free_peak_is_preserved(self) -> None:
        truth, windows = _squat_session(reps=2)
        aligned = _run(truth, _uniform_confidences(truth, _confidence_for_noise(NOISE_STD_M)))["aligned"]
        for keypoint in (CK.LEFT_KNEE, CK.LEFT_HIP):
            for start, end in windows:
                assert aligned[start:end + 1, keypoint, 1].max() == pytest.approx(
                    truth[start:end + 1, keypoint, 1].max(), abs=NOISE_FREE_PEAK_TOLERANCE_M)

    def test_knee_cave_excursion_peak_loss_below_25_percent(self) -> None:
        truth, windows = _squat_session()
        half_width = int(round(KNEE_CAVE_DURATION_S * FPS / 2.0))
        descent_frames = int(round(1.0 * FPS))
        for start, _ in windows:
            centre = start + descent_frames + half_width + 2
            for frame in range(centre - half_width, centre + half_width + 1):
                phase = (frame - centre) / (2.0 * half_width)
                truth[frame, CK.LEFT_KNEE, 0] -= KNEE_CAVE_M * 0.5 * (1.0 + math.cos(2.0 * math.pi * phase))
        confidences = _uniform_confidences(truth, _confidence_for_noise(NOISE_STD_M))
        standing_x = STANDING_POSE[CK.LEFT_KNEE, 0]
        losses = []
        for points in [truth] + [_add_noise(truth, NOISE_STD_M, seed) for seed in SEEDS]:
            aligned = _run(points, confidences)["aligned"]
            for start, end in windows:
                true_peak = standing_x - truth[start:end + 1, CK.LEFT_KNEE, 0].min()
                estimated_peak = standing_x - aligned[start:end + 1, CK.LEFT_KNEE, 0].min()
                losses.append(1.0 - estimated_peak / true_peak)
        noise_free_losses = losses[:len(windows)]
        assert max(noise_free_losses) < MAX_KNEE_CAVE_PEAK_LOSS_RATIO
        assert float(np.mean(losses)) < MAX_KNEE_CAVE_PEAK_LOSS_RATIO

    def test_standing_jitter_reduced_by_at_least_70_percent(self) -> None:
        frame_count = 300
        noisy = _add_noise(_standing_session(frame_count), NOISE_STD_M, seed=1)
        aligned = _run(noisy, _uniform_confidences(noisy, _confidence_for_noise(NOISE_STD_M)))["aligned"]
        settled = slice(30, frame_count - LAG_FRAMES)
        raw_jitter = np.sqrt(np.mean(np.sum(np.diff(noisy[settled], axis=0) ** 2, axis=-1)))
        smoothed_jitter = np.sqrt(np.mean(np.sum(np.diff(aligned[settled], axis=0) ** 2, axis=-1)))
        assert 1.0 - smoothed_jitter / raw_jitter >= MIN_JITTER_REDUCTION_RATIO

    def test_explosive_tempo_causes_no_gate_rejections(self) -> None:
        displacement = _explosive_displacement()
        truth = STANDING_POSE[None] + displacement[:, None, None] * EXPLOSIVE_DIRECTION
        confidences = _uniform_confidences(truth, _confidence_for_noise(THREE_PX_NOISE_STD_M))
        for seed in SEEDS:
            result = _run(_add_noise(truth, THREE_PX_NOISE_STD_M, seed), confidences)
            assert int(result["rejected"].sum()) == 0

    def test_explosive_profile_reaches_speed_and_acceleration_targets(self) -> None:
        velocity = np.diff(_explosive_fine_displacement()) * FINE_RATE_HZ
        acceleration = np.diff(velocity) * FINE_RATE_HZ
        assert float(np.abs(velocity).max()) == pytest.approx(EXPLOSIVE_PEAK_SPEED_M_PER_S, abs=PROFILE_TOLERANCE)
        assert float(np.abs(acceleration).max()) == pytest.approx(EXPLOSIVE_PEAK_ACCELERATION_M_PER_S2,
                                                                  abs=PROFILE_TOLERANCE)


class TestTiming:
    def test_duplicate_timestamp_leaves_output_unchanged(self) -> None:
        noisy = _add_noise(_standing_session(22), NOISE_STD_M, seed=3)
        confidences = np.full(KEYPOINT_COUNT, 0.5)
        with_duplicate, without_duplicate = _make_smoother(), _make_smoother()
        for frame in range(21):
            previous = with_duplicate.update(noisy[frame], confidences, _timestamp(frame))
            without_duplicate.update(noisy[frame], confidences, _timestamp(frame))
        duplicate = with_duplicate.update(noisy[frame] + 0.5, confidences, _timestamp(frame))
        backwards = with_duplicate.update(noisy[frame] - 0.5, confidences, _timestamp(frame) - FRAME_DT_S)
        assert duplicate is previous
        assert backwards is previous
        after_duplicate = with_duplicate.update(noisy[21], confidences, _timestamp(21))
        reference = without_duplicate.update(noisy[21], confidences, _timestamp(21))
        np.testing.assert_array_equal(after_duplicate.lagged_points, reference.lagged_points)
        np.testing.assert_array_equal(after_duplicate.current_points, reference.current_points)

    def test_gap_over_half_second_reinitialises_without_crawl(self) -> None:
        smoother = _make_smoother()
        confidences = np.full(KEYPOINT_COUNT, 0.5)
        for frame in range(30):
            smoother.update(STANDING_POSE, confidences, _timestamp(frame))
        moved = STANDING_POSE + np.array([0.0, 0.4, 0.0])
        resumed_timestamp = _timestamp(29) + 0.6
        output = smoother.update(moved, confidences, resumed_timestamp)
        np.testing.assert_allclose(output.current_points, moved, atol=EXACT_TOLERANCE_M)
        np.testing.assert_allclose(output.lagged_points, moved, atol=EXACT_TOLERANCE_M)
        assert output.lagged_timestamp == pytest.approx(resumed_timestamp, abs=EXACT_TOLERANCE_M)
        assert not output.gate_rejected.any()


class TestMissingKeypoints:
    def test_zero_confidence_keypoint_moves_estimate_less_than_1cm(self) -> None:
        # Triangulator emits (0, 0, 0) at confidence 0 (C3); mid-descent the knee is ~50 cm from the origin.
        truth, windows = _squat_session(reps=2)
        confidences = _uniform_confidences(truth, _confidence_for_noise(NOISE_STD_M))
        dropout_start = windows[1][0] + 15
        dropout = slice(dropout_start, dropout_start + 2)
        checked = slice(dropout_start - LAG_FRAMES, dropout_start + 8)
        dropped_confidences = confidences.copy()
        dropped_confidences[dropout, CK.LEFT_KNEE] = 0.0
        # Noise-free: neither output may move. Noisy: the analysis stream stays within the gated-outlier bound
        # (the undelayed display estimate additionally carries the filtered velocity's noise over 2 frames).
        cases = ((truth, ("aligned", "current"), MISSING_KEYPOINT_TOLERANCE_M),
                 (_add_noise(truth, NOISE_STD_M, seed=2), ("aligned",), OUTLIER_TOLERANCE_M))
        for points, keys, tolerance_m in cases:
            dropped = points.copy()
            dropped[dropout, CK.LEFT_KNEE] = 0.0
            result = _run(dropped, dropped_confidences)
            for key in keys:
                error = np.linalg.norm(result[key][checked, CK.LEFT_KNEE] - truth[checked, CK.LEFT_KNEE], axis=-1)
                assert float(error.max()) < tolerance_m
            assert result["current_confidences"][dropout, CK.LEFT_KNEE].min() >= MIN_OUTPUT_CONFIDENCE

    def test_non_finite_measurement_is_never_emitted(self) -> None:
        smoother = _make_smoother()
        confidences = np.full(KEYPOINT_COUNT, 0.9)
        for frame in range(10):
            smoother.update(STANDING_POSE, confidences, _timestamp(frame))
        corrupted = STANDING_POSE.copy()
        corrupted[CK.RIGHT_KNEE] = (np.nan, np.inf, 0.0)
        for frame in range(10, 14):
            output = smoother.update(corrupted, confidences, _timestamp(frame))
            assert np.isfinite(output.current_points).all()
            assert np.isfinite(output.lagged_points).all()
            assert np.isfinite(output.velocities).all()
        np.testing.assert_allclose(output.current_points[CK.RIGHT_KNEE], STANDING_POSE[CK.RIGHT_KNEE],
                                   atol=PREDICTION_TOLERANCE_M)

    def test_predict_missing_extrapolates_all_keypoints(self) -> None:
        velocity = np.array([0.0, 1.0, 0.0])
        smoother = _make_smoother()
        for frame in range(40):
            accepted = smoother.update(STANDING_POSE + frame * FRAME_DT_S * velocity,
                                       np.full(KEYPOINT_COUNT, 0.8), _timestamp(frame))
        for step in range(1, 4):
            output = smoother.predict_missing(_timestamp(39 + step))
            expected = STANDING_POSE + (39 + step) * FRAME_DT_S * velocity
            np.testing.assert_allclose(output.current_points, expected, atol=PREDICTION_TOLERANCE_M)
            assert output.current_timestamp == pytest.approx(_timestamp(39 + step), abs=EXACT_TOLERANCE_M)
            assert output.lagged_timestamp == pytest.approx(_timestamp(39 + step - LAG_FRAMES), abs=EXACT_TOLERANCE_M)
            assert (output.current_confidences >= MIN_OUTPUT_CONFIDENCE).all()
            assert (output.current_confidences < accepted.current_confidences).all()

    def test_max_predicted_frames_then_zero_confidence_then_clean_reinit(self) -> None:
        smoother = _make_smoother()
        confidences = np.full(KEYPOINT_COUNT, 0.8)
        for frame in range(30):
            smoother.update(STANDING_POSE, confidences, _timestamp(frame))
        missing = confidences.copy()
        missing[CK.LEFT_ANKLE] = 0.0
        for miss in range(1, MAX_PREDICTED_FRAMES + 1):
            output = smoother.update(STANDING_POSE, missing, _timestamp(29 + miss))
            assert output.current_confidences[CK.LEFT_ANKLE] >= MIN_OUTPUT_CONFIDENCE
        output = smoother.update(STANDING_POSE, missing, _timestamp(30 + MAX_PREDICTED_FRAMES))
        assert output.current_confidences[CK.LEFT_ANKLE] == 0.0
        assert np.isfinite(output.current_points).all()
        returned = STANDING_POSE.copy()
        returned[CK.LEFT_ANKLE] += (0.0, 0.0, 0.5)
        reinit_frame = 31 + MAX_PREDICTED_FRAMES
        output = smoother.update(returned, confidences, _timestamp(reinit_frame))
        np.testing.assert_allclose(output.current_points[CK.LEFT_ANKLE], returned[CK.LEFT_ANKLE],
                                   atol=EXACT_TOLERANCE_M)
        assert output.current_confidences[CK.LEFT_ANKLE] > 0.0
        assert not output.gate_rejected[CK.LEFT_ANKLE]
        for frame in range(reinit_frame + 1, reinit_frame + 1 + LAG_FRAMES):
            output = smoother.update(returned, confidences, _timestamp(frame))
        np.testing.assert_allclose(output.lagged_points[CK.LEFT_ANKLE], returned[CK.LEFT_ANKLE],
                                   atol=PREDICTION_TOLERANCE_M)
        np.testing.assert_allclose(output.current_points[CK.LEFT_ANKLE], returned[CK.LEFT_ANKLE],
                                   atol=PREDICTION_TOLERANCE_M)


class TestInnovationGate:
    def test_single_frame_30cm_outlier_is_rejected(self) -> None:
        frame_count, outlier_frame = 90, 60
        truth = _standing_session(frame_count)
        noisy = _add_noise(truth, NOISE_STD_M, seed=4)
        noisy[outlier_frame, CK.LEFT_KNEE, 0] += OUTLIER_OFFSET_M
        result = _run(noisy, _uniform_confidences(noisy, _confidence_for_noise(NOISE_STD_M)))
        assert result["rejected"][outlier_frame, CK.LEFT_KNEE]
        assert int(result["rejected"].sum()) == 1
        for key in ("aligned", "current"):
            error = np.linalg.norm(result[key][outlier_frame - 2:outlier_frame + 5, CK.LEFT_KNEE]
                                   - truth[outlier_frame - 2:outlier_frame + 5, CK.LEFT_KNEE], axis=-1)
            assert float(error.max()) < OUTLIER_TOLERANCE_M
        assert result["current_confidences"][outlier_frame, CK.LEFT_KNEE] >= MIN_OUTPUT_CONFIDENCE

    def test_short_consistent_outlier_burst_is_not_reacquired(self) -> None:
        # Audit B1: RTMPose outlier runs average 1.6 frames (p90 3) with a constant offset; swaps last 2-6 frames.
        frame_count, burst_start = 90, 60
        truth = _standing_session(frame_count)
        checked = slice(burst_start - LAG_FRAMES, burst_start + 12)
        for burst_frames in (1, 2, 3):
            for burst_confidence in BURST_CONFIDENCES:
                noisy = _add_noise(truth, NOISE_STD_M, seed=6)
                confidences = _uniform_confidences(noisy, GOOD_CONFIDENCE)
                burst = slice(burst_start, burst_start + burst_frames)
                noisy[burst, CK.LEFT_KNEE, 0] += BURST_OFFSET_M
                confidences[burst, CK.LEFT_KNEE] = burst_confidence
                result = _run(noisy, confidences)
                assert result["rejected"][burst, CK.LEFT_KNEE].all()
                for key, tolerance_m in (("aligned", OUTLIER_TOLERANCE_M), ("current", BURST_DISPLAY_TOLERANCE_M)):
                    error = np.linalg.norm(result[key][checked, CK.LEFT_KNEE] - truth[checked, CK.LEFT_KNEE], axis=-1)
                    assert float(error.max()) < tolerance_m

    def test_persistent_jump_reacquires_within_four_frames(self) -> None:
        frame_count, jump_start = 100, 60
        truth = _standing_session(frame_count)
        truth[jump_start:jump_start + PERSISTENT_JUMP_FRAMES, CK.LEFT_KNEE, 2] += JUMP_OFFSET_M
        noisy = _add_noise(truth, NOISE_STD_M, seed=5)
        result = _run(noisy, _uniform_confidences(noisy, GOOD_CONFIDENCE))
        error = np.linalg.norm(result["current"][:, CK.LEFT_KNEE] - truth[:, CK.LEFT_KNEE], axis=-1)
        for transition in (jump_start, jump_start + PERSISTENT_JUMP_FRAMES):
            reacquired = transition + REACQUIRE_AGREEING_FRAMES
            assert result["rejected"][transition:reacquired, CK.LEFT_KNEE].all()
            assert float(error[reacquired:reacquired + 5].max()) < OUTLIER_TOLERANCE_M
        assert result["current_confidences"][:, CK.LEFT_KNEE].min() > 0.0

    def test_low_confidence_jump_never_snaps_and_recovers_after_loss(self) -> None:
        frame_count, jump_start = 100, 60
        truth = _standing_session(frame_count)
        truth[jump_start:, CK.LEFT_KNEE, 2] += BURST_OFFSET_M
        noisy = _add_noise(truth, NOISE_STD_M, seed=8)
        confidences = _uniform_confidences(noisy, GOOD_CONFIDENCE)
        confidences[jump_start:, CK.LEFT_KNEE] = MIN_SNAP_CONFIDENCE
        result = _run(noisy, confidences)
        error = np.linalg.norm(result["current"][:, CK.LEFT_KNEE] - truth[:, CK.LEFT_KNEE], axis=-1)
        lost_frame = jump_start + MAX_PREDICTED_FRAMES
        blind = slice(jump_start, lost_frame + 1)
        assert result["rejected"][blind, CK.LEFT_KNEE].all()
        assert float(error[blind].min()) > BURST_OFFSET_M - OUTLIER_TOLERANCE_M
        assert result["current_confidences"][lost_frame, CK.LEFT_KNEE] == 0.0
        reseeded = slice(lost_frame + 1, lost_frame + 6)
        assert float(error[reseeded].max()) < OUTLIER_TOLERANCE_M
        assert (result["current_confidences"][reseeded, CK.LEFT_KNEE] > 0.0).all()

    def test_timeout_snaps_only_when_last_rejections_agree(self) -> None:
        # The gate widens as predict-only covariance grows, so the 0.2 s backstop only matters for gross errors.
        timeout_frames = int(round(MAX_REJECTION_S * FPS))
        alternating = [TIMEOUT_OFFSET_M * (1.0 if index % 2 == 0 else -1.0) for index in range(timeout_frames)]
        agreeing_tail = alternating[:timeout_frames - TIMEOUT_AGREEING_FRAMES] + [TIMEOUT_OFFSET_M] * TIMEOUT_AGREEING_FRAMES
        for offsets, expect_snap in ((alternating, False), (agreeing_tail, True)):
            smoother = _make_smoother(max_predicted_frames=20)
            confidences = np.full(KEYPOINT_COUNT, 0.8)
            for frame in range(30):
                smoother.update(STANDING_POSE, confidences, _timestamp(frame))
            for index, offset in enumerate(offsets):
                displaced = STANDING_POSE.copy()
                displaced[CK.LEFT_WRIST, 0] += offset
                output = smoother.update(displaced, confidences, _timestamp(30 + index))
                assert output.gate_rejected[CK.LEFT_WRIST]
                if index < timeout_frames - 1:
                    np.testing.assert_allclose(output.current_points[CK.LEFT_WRIST], STANDING_POSE[CK.LEFT_WRIST],
                                               atol=PREDICTION_TOLERANCE_M)
            expected = displaced[CK.LEFT_WRIST] if expect_snap else STANDING_POSE[CK.LEFT_WRIST]
            np.testing.assert_allclose(output.current_points[CK.LEFT_WRIST], expected, atol=PREDICTION_TOLERANCE_M)


class TestConfidenceOutput:
    def test_smoothing_raises_confidence_of_accepted_measurements(self) -> None:
        low, high = _make_smoother(), _make_smoother()
        for frame in range(40):
            low_output = low.update(STANDING_POSE, np.full(KEYPOINT_COUNT, 0.2), _timestamp(frame))
            high_output = high.update(STANDING_POSE, np.full(KEYPOINT_COUNT, 0.5), _timestamp(frame))
        assert (high_output.current_confidences > 0.5).all()
        assert (high_output.current_confidences <= 1.0).all()
        assert (low_output.current_confidences > 0.2).all()
        assert (low_output.current_confidences < high_output.current_confidences).all()
        assert (high_output.lagged_confidences > 0.5).all()

    def test_predicted_confidence_decays_to_floor(self) -> None:
        smoother = _make_smoother()
        confidences = np.full(KEYPOINT_COUNT, 0.5)
        for frame in range(30):
            output = smoother.update(STANDING_POSE, confidences, _timestamp(frame))
        previous = output.current_confidences[CK.NOSE]
        missing = confidences.copy()
        missing[CK.NOSE] = 0.0
        for miss in range(1, MAX_PREDICTED_FRAMES + 1):
            output = smoother.update(STANDING_POSE, missing, _timestamp(29 + miss))
            decayed = output.current_confidences[CK.NOSE]
            assert decayed <= previous
            assert decayed >= MIN_OUTPUT_CONFIDENCE
            previous = decayed
