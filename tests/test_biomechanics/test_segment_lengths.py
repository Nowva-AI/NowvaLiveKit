"""
Tests for the session-scoped body segment length estimator: accuracy on noisy
synthetic squats, outlier robustness, completion rules, and athlete params.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from biomechanics.utils.segment_lengths import (
    DEFAULT_FOOT_LENGTH_M,
    FALLBACK_MIN_ACCEPTED_FRAMES,
    FORWARD_LEAN_SCALE_MAX,
    FORWARD_LEAN_SCALE_MIN,
    MIN_ENDPOINT_CONFIDENCE,
    MIN_SAMPLES_PER_SEGMENT,
    REFERENCE_FEMUR_TO_TORSO_RATIO,
    BodyProportions,
    SegmentLengthEstimator,
)
from biomechanics.utils.types import CocoKeypoints as CK

# C3: conf = 1 / (1 + (u / UNCERTAINTY_SCALE_M)^2), u = 3D position std
UNCERTAINTY_SCALE_M = 0.02
# Noisiest keypoint the confidence gate still accepts (conf 0.4 -> u = 2.4 cm)
GATE_WORST_CASE_SIGMA_M = UNCERTAINTY_SCALE_M * math.sqrt(1.0 / MIN_ENDPOINT_CONFIDENCE - 1.0)
# Per-keypoint uncertainty range whose upper third the gate rejects
C3_MIX_UNCERTAINTY_RANGE_M = (0.010, 0.035)

FPS = 30.0
USER_HEIGHT_M = 1.885
HIP_HALF_WIDTH_M = 0.0725 * USER_HEIGHT_M
SHOULDER_HALF_WIDTH_M = 0.105 * USER_HEIGHT_M
FEMUR_M = 0.245 * USER_HEIGHT_M
TIBIA_M = 0.235 * USER_HEIGHT_M
TORSO_M = 0.29 * USER_HEIGHT_M
FOOT_M = 0.21
TOE_OUT_DEG = 15.0
STAND_S = 2.0
REP_S = 2.6
KEYPOINT_COUNT = 21
TRACKED_CONFIDENCE = 0.9

NOISE_SIGMA_M = 0.02
OUTLIER_PROBABILITY = 0.01
OUTLIER_RANGE_M = (0.10, 0.30)
SEQUENCE_FRAMES = 900

ACCURACY_TOLERANCE_M = 0.005
PARAM_TOLERANCE_M = 1e-9
SCALE_TOLERANCE = 1e-9
ACCURACY_SEEDS = (0, 1, 2, 3, 4)
WORST_CASE_SEED_COUNT = 20

LEGACY_ATHLETE_PARAM_KEYS = {
    "shoulder_width_m", "femur_avg_m", "torso_avg_m", "hip_width_m", "tibia_avg_m", "foot_avg_m",
}


def _squat_frame(
    depth: float,
    femur_l_m: float = FEMUR_M,
    femur_r_m: float = FEMUR_M,
    tibia_m: float = TIBIA_M,
) -> np.ndarray:
    # Hip-centred, Y-down, forward = -Z. Exact segment lengths at any depth.
    points = np.zeros((KEYPOINT_COUNT, 3))
    thigh_rad = math.radians(100.0 * depth)
    shank_rad = math.radians(35.0 * depth)
    lean_rad = math.radians(8.0 + 34.0 * depth)
    toe_out_rad = math.radians(TOE_OUT_DEG)
    sides = (
        (1.0, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL, femur_l_m),
        (-1.0, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL, femur_r_m),
    )
    for sign, hip, knee, ankle, toe, heel, femur_m in sides:
        points[hip] = [sign * HIP_HALF_WIDTH_M, 0.0, 0.0]
        points[knee] = points[hip] + femur_m * np.array([0.0, math.cos(thigh_rad), -math.sin(thigh_rad)])
        points[ankle] = points[knee] + tibia_m * np.array([0.0, math.cos(shank_rad), math.sin(shank_rad)])
        foot_direction = np.array([sign * math.sin(toe_out_rad), 0.28, -math.cos(toe_out_rad)])
        points[toe] = points[ankle] + FOOT_M * foot_direction / np.linalg.norm(foot_direction)
        points[heel] = points[ankle] + np.array([0.0, 0.05, 0.05])
    trunk_direction = np.array([0.0, -math.cos(lean_rad), -math.sin(lean_rad)])
    shoulder_mid = TORSO_M * trunk_direction
    points[CK.LEFT_SHOULDER] = shoulder_mid + [SHOULDER_HALF_WIDTH_M, 0.0, 0.0]
    points[CK.RIGHT_SHOULDER] = shoulder_mid - [SHOULDER_HALF_WIDTH_M, 0.0, 0.0]
    points[CK.NOSE] = shoulder_mid + 0.2 * trunk_direction
    return points


def _squat_sequence(frame_count: int = SEQUENCE_FRAMES, **frame_overrides: float) -> tuple[np.ndarray, np.ndarray]:
    times_s = np.arange(frame_count) / FPS
    rep_phase = np.clip((times_s - STAND_S) / REP_S, 0.0, None)
    depths = np.where(times_s < STAND_S, 0.0, 0.5 - 0.5 * np.cos(2.0 * np.pi * rep_phase))
    rep_counts = np.floor(rep_phase).astype(int)
    frames = np.stack([_squat_frame(depth, **frame_overrides) for depth in depths])
    return frames, rep_counts


def _add_noise(
    frames: np.ndarray,
    rng: np.random.Generator,
    sigma_m: float = NOISE_SIGMA_M,
    outlier_probability: float = OUTLIER_PROBABILITY,
) -> np.ndarray:
    noisy = frames + rng.normal(0.0, sigma_m, frames.shape)
    outlier_mask = rng.random(frames.shape[:2]) < outlier_probability
    directions = rng.normal(size=frames.shape)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    magnitudes_m = rng.uniform(*OUTLIER_RANGE_M, frames.shape[:2])
    return noisy + np.where(outlier_mask[..., None], directions * magnitudes_m[..., None], 0.0)


def _run_until_complete(
    estimator: SegmentLengthEstimator,
    frames: np.ndarray,
    rep_counts: np.ndarray,
    confidences: np.ndarray | None = None,
) -> int:
    confidences = np.full(frames.shape[1], TRACKED_CONFIDENCE) if confidences is None else confidences
    for i, (points, rep_count) in enumerate(zip(frames, rep_counts)):
        estimator.record(points, confidences, int(rep_count))
        if estimator.is_complete:
            return i
    return len(frames)


def _athlete_params(femur_avg_m: float = FEMUR_M, torso_avg_m: float = TORSO_M) -> dict[str, float]:
    return {
        "shoulder_width_m": 2.0 * SHOULDER_HALF_WIDTH_M,
        "femur_avg_m": femur_avg_m,
        "torso_avg_m": torso_avg_m,
        "hip_width_m": 2.0 * HIP_HALF_WIDTH_M,
        "tibia_avg_m": TIBIA_M,
        "foot_avg_m": FOOT_M,
    }


class TestAccuracy:
    @pytest.mark.parametrize("seed", ACCURACY_SEEDS)
    def test_noisy_squats_measure_femur_and_tibia_within_5mm(self, seed: int) -> None:
        frames, rep_counts = _squat_sequence()
        noisy = _add_noise(frames, np.random.default_rng(seed))
        estimator = SegmentLengthEstimator()

        _run_until_complete(estimator, noisy, rep_counts)

        params = estimator.to_athlete_params()
        assert params is not None
        assert params["femur_avg_m"] == pytest.approx(FEMUR_M, abs=ACCURACY_TOLERANCE_M)
        assert params["tibia_avg_m"] == pytest.approx(TIBIA_M, abs=ACCURACY_TOLERANCE_M)

    def test_gate_worst_case_noise_p95_error_within_5mm(self) -> None:
        # Every accepted keypoint at the noisiest level the gate lets through;
        # the acceptance criterion is p95 over sessions, single seeds can exceed it.
        frames, rep_counts = _squat_sequence()
        femur_errors_m = []
        tibia_errors_m = []
        for seed in range(WORST_CASE_SEED_COUNT):
            noisy = _add_noise(frames, np.random.default_rng(seed), sigma_m=GATE_WORST_CASE_SIGMA_M)
            estimator = SegmentLengthEstimator()
            _run_until_complete(estimator, noisy, rep_counts)
            params = estimator.to_athlete_params()
            assert params is not None
            femur_errors_m.append(abs(params["femur_avg_m"] - FEMUR_M))
            tibia_errors_m.append(abs(params["tibia_avg_m"] - TIBIA_M))

        assert np.percentile(femur_errors_m, 95) < ACCURACY_TOLERANCE_M
        assert np.percentile(tibia_errors_m, 95) < ACCURACY_TOLERANCE_M

    @pytest.mark.parametrize("seed", ACCURACY_SEEDS)
    def test_c3_confidence_mix_measures_within_5mm(self, seed: int) -> None:
        # Per-keypoint noise std drawn per frame; confidence follows C3, so the
        # gate itself decides which frames count and rejects the noisiest third.
        rng = np.random.default_rng(seed)
        frames, rep_counts = _squat_sequence()
        uncertainties_m = rng.uniform(*C3_MIX_UNCERTAINTY_RANGE_M, frames.shape[:2])
        noisy = frames + rng.normal(size=frames.shape) * uncertainties_m[..., None]
        confidences = 1.0 / (1.0 + (uncertainties_m / UNCERTAINTY_SCALE_M) ** 2)
        estimator = SegmentLengthEstimator()

        for points, frame_confidences, rep_count in zip(noisy, confidences, rep_counts):
            estimator.record(points, frame_confidences, int(rep_count))
            if estimator.is_complete:
                break

        params = estimator.to_athlete_params()
        assert params is not None
        assert params["femur_avg_m"] == pytest.approx(FEMUR_M, abs=ACCURACY_TOLERANCE_M)
        assert params["tibia_avg_m"] == pytest.approx(TIBIA_M, abs=ACCURACY_TOLERANCE_M)

    def test_torso_length_reported_from_side_segments(self) -> None:
        frames, rep_counts = _squat_sequence()
        estimator = SegmentLengthEstimator()
        _run_until_complete(estimator, _add_noise(frames, np.random.default_rng(7)), rep_counts)

        expected_torso_side_m = float(np.linalg.norm(frames[0][CK.LEFT_SHOULDER] - frames[0][CK.LEFT_HIP]))
        assert estimator.to_athlete_params()["torso_avg_m"] == pytest.approx(
            expected_torso_side_m, abs=2.0 * ACCURACY_TOLERANCE_M,
        )

    def test_outlier_burst_moves_estimate_less_than_5mm(self) -> None:
        # A 30 cm lateral knee burst over 30 frames shifts a plain mean of femur_l by ~1 cm
        burst_frames = 30
        burst_offset_m = np.array([0.30, 0.0, 0.0])
        frames, rep_counts = _squat_sequence()
        noisy = _add_noise(frames, np.random.default_rng(11))
        burst = noisy.copy()
        burst[:burst_frames, CK.LEFT_KNEE] += burst_offset_m

        clean_estimator = SegmentLengthEstimator()
        burst_estimator = SegmentLengthEstimator()
        _run_until_complete(clean_estimator, noisy, rep_counts)
        _run_until_complete(burst_estimator, burst, rep_counts)

        clean_params = clean_estimator.to_athlete_params()
        burst_params = burst_estimator.to_athlete_params()
        assert burst_params["femur_avg_m"] == pytest.approx(clean_params["femur_avg_m"], abs=ACCURACY_TOLERANCE_M)
        assert burst_params["tibia_avg_m"] == pytest.approx(clean_params["tibia_avg_m"], abs=ACCURACY_TOLERANCE_M)


class TestCompletion:
    def test_incomplete_before_minimum_samples(self) -> None:
        frames, _ = _squat_sequence(100)
        estimator = SegmentLengthEstimator()

        for i, points in enumerate(frames):
            estimator.record(points, np.full(KEYPOINT_COUNT, TRACKED_CONFIDENCE), i % 2)

        assert not estimator.is_complete
        assert estimator.progress == (100, MIN_SAMPLES_PER_SEGMENT)
        assert estimator.to_athlete_params() is None
        assert estimator.body_proportions is None

    def test_single_rep_count_does_not_complete(self) -> None:
        frames, _ = _squat_sequence(400)
        estimator = SegmentLengthEstimator()

        _run_until_complete(estimator, frames, np.zeros(len(frames), dtype=int))

        assert not estimator.is_complete

    def test_completes_once_samples_span_two_rep_counts(self) -> None:
        frames, _ = _squat_sequence(400)
        rep_counts = np.zeros(len(frames), dtype=int)
        rep_counts[MIN_SAMPLES_PER_SEGMENT:] = 1
        estimator = SegmentLengthEstimator()

        completed_at = _run_until_complete(estimator, frames, rep_counts)

        assert completed_at == MIN_SAMPLES_PER_SEGMENT
        assert estimator.is_complete
        assert estimator.progress == (MIN_SAMPLES_PER_SEGMENT, MIN_SAMPLES_PER_SEGMENT)

    def test_endpoint_below_confidence_gate_is_rejected(self) -> None:
        frames, rep_counts = _squat_sequence(400)
        confidences = np.full(KEYPOINT_COUNT, TRACKED_CONFIDENCE)
        confidences[CK.LEFT_KNEE] = MIN_ENDPOINT_CONFIDENCE - 0.01
        estimator = SegmentLengthEstimator()

        _run_until_complete(estimator, frames, rep_counts, confidences)

        assert not estimator.is_complete
        assert estimator.progress == (0, MIN_SAMPLES_PER_SEGMENT)

    def test_endpoint_at_confidence_gate_is_accepted(self) -> None:
        frames, rep_counts = _squat_sequence(400)
        confidences = np.full(KEYPOINT_COUNT, MIN_ENDPOINT_CONFIDENCE)
        estimator = SegmentLengthEstimator()

        _run_until_complete(estimator, frames, rep_counts, confidences)

        assert estimator.is_complete

    def test_implausible_lengths_are_rejected(self) -> None:
        frames, rep_counts = _squat_sequence(400)
        frames[:, CK.RIGHT_ANKLE] += np.array([0.0, 0.5, 0.0])
        estimator = SegmentLengthEstimator()

        _run_until_complete(estimator, frames, rep_counts)

        assert not estimator.is_complete
        assert estimator.progress == (0, MIN_SAMPLES_PER_SEGMENT)

    def test_wide_confidence_interval_delays_completion(self) -> None:
        frames, rep_counts = _squat_sequence(400)
        noisy = _add_noise(frames, np.random.default_rng(3), sigma_m=0.06, outlier_probability=0.0)
        estimator = SegmentLengthEstimator()

        _run_until_complete(estimator, noisy, rep_counts)

        assert not estimator.is_complete

    def test_bilateral_mismatch_blocks_strict_completion_until_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        frames, rep_counts = _squat_sequence(SEQUENCE_FRAMES, femur_r_m=FEMUR_M + 0.03)
        noisy = _add_noise(frames, np.random.default_rng(5), sigma_m=0.005, outlier_probability=0.0)
        estimator = SegmentLengthEstimator()

        with caplog.at_level(logging.WARNING, logger="biomechanics.utils.segment_lengths"):
            completed_at = _run_until_complete(estimator, noisy, rep_counts)

        assert completed_at == FALLBACK_MIN_ACCEPTED_FRAMES - 1
        assert "fallback" in caplog.text
        assert estimator.to_athlete_params()["femur_avg_m"] == pytest.approx(FEMUR_M + 0.015, abs=ACCURACY_TOLERANCE_M)

    def test_seventeen_keypoint_layout_completes_with_default_foot(self) -> None:
        frames, rep_counts = _squat_sequence(400)
        estimator = SegmentLengthEstimator()

        _run_until_complete(estimator, frames[:, :17], rep_counts, np.full(17, TRACKED_CONFIDENCE))

        assert estimator.is_complete
        assert estimator.to_athlete_params()["foot_avg_m"] == pytest.approx(DEFAULT_FOOT_LENGTH_M, abs=PARAM_TOLERANCE_M)

    def test_frames_after_completion_do_not_change_measurement(self) -> None:
        frames, rep_counts = _squat_sequence(400)
        estimator = SegmentLengthEstimator()
        completed_at = _run_until_complete(estimator, frames, rep_counts)
        params_at_completion = estimator.to_athlete_params()

        stretched = frames * 1.1
        for points, rep_count in zip(stretched[completed_at:], rep_counts[completed_at:]):
            estimator.record(points, np.full(KEYPOINT_COUNT, TRACKED_CONFIDENCE), int(rep_count))

        assert estimator.to_athlete_params() == params_at_completion


class TestAthleteParams:
    def test_to_athlete_params_uses_legacy_keys(self) -> None:
        frames, rep_counts = _squat_sequence(400)
        estimator = SegmentLengthEstimator()
        _run_until_complete(estimator, frames, rep_counts)

        params = estimator.to_athlete_params()

        assert set(params) == LEGACY_ATHLETE_PARAM_KEYS
        assert params["hip_width_m"] == pytest.approx(2.0 * HIP_HALF_WIDTH_M, abs=ACCURACY_TOLERANCE_M)
        assert params["shoulder_width_m"] == pytest.approx(2.0 * SHOULDER_HALF_WIDTH_M, abs=ACCURACY_TOLERANCE_M)
        assert params["foot_avg_m"] == pytest.approx(FOOT_M, abs=ACCURACY_TOLERANCE_M)

    def test_round_trip_through_athlete_params(self) -> None:
        stored = _athlete_params()

        estimator = SegmentLengthEstimator.from_athlete_params(stored)

        assert estimator.is_complete
        restored = estimator.to_athlete_params()
        assert set(restored) == LEGACY_ATHLETE_PARAM_KEYS
        for key, value in stored.items():
            assert restored[key] == pytest.approx(value, abs=PARAM_TOLERANCE_M)
        assert estimator.body_proportions.femur_length_avg == pytest.approx(FEMUR_M, abs=PARAM_TOLERANCE_M)

    def test_restored_estimator_ignores_new_frames(self) -> None:
        stored = _athlete_params()
        estimator = SegmentLengthEstimator.from_athlete_params(stored)
        frames, rep_counts = _squat_sequence(400, femur_l_m=0.40, femur_r_m=0.40)

        _run_until_complete(estimator, frames, rep_counts)

        assert estimator.to_athlete_params()["femur_avg_m"] == pytest.approx(FEMUR_M, abs=PARAM_TOLERANCE_M)

    def test_to_athlete_params_returns_a_copy(self) -> None:
        estimator = SegmentLengthEstimator.from_athlete_params(_athlete_params())

        estimator.to_athlete_params()["femur_avg_m"] = 0.0

        assert estimator.to_athlete_params()["femur_avg_m"] == pytest.approx(FEMUR_M, abs=PARAM_TOLERANCE_M)


class TestBodyProportions:
    def test_forward_lean_scale_above_one_for_long_femur(self) -> None:
        torso_m = 0.50
        femur_m = torso_m * REFERENCE_FEMUR_TO_TORSO_RATIO * 1.1

        proportions = SegmentLengthEstimator.from_athlete_params(_athlete_params(femur_m, torso_m)).body_proportions

        assert proportions.forward_lean_scale == pytest.approx(1.1, abs=SCALE_TOLERANCE)

    def test_forward_lean_scale_below_one_for_short_femur(self) -> None:
        torso_m = 0.50
        femur_m = torso_m * REFERENCE_FEMUR_TO_TORSO_RATIO * 0.9

        proportions = SegmentLengthEstimator.from_athlete_params(_athlete_params(femur_m, torso_m)).body_proportions

        assert proportions.forward_lean_scale == pytest.approx(0.9, abs=SCALE_TOLERANCE)

    def test_reference_proportions_give_unit_scale(self) -> None:
        proportions = SegmentLengthEstimator.from_athlete_params(_athlete_params()).body_proportions

        assert proportions.forward_lean_scale == pytest.approx(
            (FEMUR_M / TORSO_M) / REFERENCE_FEMUR_TO_TORSO_RATIO, abs=SCALE_TOLERANCE,
        )
        assert proportions.forward_lean_scale == pytest.approx(1.0, abs=0.01)

    def test_forward_lean_scale_is_clamped(self) -> None:
        long_femur = SegmentLengthEstimator.from_athlete_params(_athlete_params(0.70, 0.35)).body_proportions
        short_femur = SegmentLengthEstimator.from_athlete_params(_athlete_params(0.30, 0.70)).body_proportions

        assert long_femur.forward_lean_scale == pytest.approx(FORWARD_LEAN_SCALE_MAX, abs=SCALE_TOLERANCE)
        assert short_femur.forward_lean_scale == pytest.approx(FORWARD_LEAN_SCALE_MIN, abs=SCALE_TOLERANCE)

    def test_removed_scale_fields_are_absent(self) -> None:
        assert "valgus_scale" not in BodyProportions.model_fields
        assert "pelvis_tilt_coupling" not in BodyProportions.model_fields
