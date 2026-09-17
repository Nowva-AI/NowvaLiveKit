"""Tests for the world-frame foot contact model on synthetic noisy squat sequences."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.utils.foot_contact import FootContactModel, FootState
from biomechanics.utils.keypoint_kalman import FixedLagKeypointSmoother
from biomechanics.utils.types import CocoKeypoints as CK

FPS = 30.0
NUM_KEYPOINTS = 21
WHITE_SIGMA_M = 0.015
DRIFT_SIGMA_M = 0.006
DRIFT_TAU_S = 0.5
STANCE_WIDTH_M = 0.38
TOE_OUT_DEG = 15.0
FLOOR_Y_M = 0.95
ANKLE_HEIGHT_M = 0.075
TOE_FORWARD_M = 0.19
TOE_HEIGHT_M = 0.025
HEEL_BACK_M = 0.06
HEEL_HEIGHT_M = 0.03
MTP_FORWARD_M = 0.14
HEEL_RISE_DEG = 15.0  # rear-foot pitch about the ball of the foot: 3.37 cm ankle rise at the bottom
HIP_HALF_M = 0.137
HIP_DROP_M = 0.52
HIP_BACK_M = 0.22
KNEE_FORWARD_M = 0.12
STAND_S = 2.0
DESCENT_S = 1.2
BOTTOM_S = 0.3
ASCENT_S = 1.0
BETWEEN_S = 1.2
STEP_START_S = 2.0
STEP_DURATION_S = 0.6
STEP_LIFT_M = 0.05
SNAP_FRACTION = 0.85
EVAL_START_FRAME = 45
STEP_STAND_FRAMES = 90
STEP_LIFT_M = 0.07
STEP_NOISE_M = 0.01
STEP_FOOT_INDICES = [CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL]
REPLANT_DEADLINE_S = 0.5
POST_REPLANT_ANKLE_ERROR_CM = 2.0
SMOOTHED_CONFIDENCE = 0.64  # triangulator confidence for ~1.5 cm position noise (C3)
FOOT_INDICES = [CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX, CK.LEFT_HEEL, CK.RIGHT_HEEL]
ANKLE_INDICES = [CK.LEFT_ANKLE, CK.RIGHT_ANKLE]
TOE_INDICES = [CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]
HEEL_INDICES = [CK.LEFT_HEEL, CK.RIGHT_HEEL]
M_TO_CM = 100.0


def _smooth_step(fraction: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(math.pi * np.clip(fraction, 0.0, 1.0))


def _depth_profile(n_reps: int, stand_s: float, bottom_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rep_len_s = DESCENT_S + bottom_s + ASCENT_S + BETWEEN_S
    n_frames = int(round((stand_s + n_reps * rep_len_s) * FPS))
    t = np.arange(n_frames) / FPS
    fraction = np.zeros(n_frames)
    rep_index = np.full(n_frames, -1)
    for rep in range(n_reps):
        t0 = stand_s + rep * rep_len_s
        t1 = t0 + DESCENT_S
        t2 = t1 + bottom_s
        t3 = t2 + ASCENT_S
        descent = (t >= t0) & (t < t1)
        fraction[descent] = _smooth_step((t[descent] - t0) / DESCENT_S)
        fraction[(t >= t1) & (t < t2)] = 1.0
        ascent = (t >= t2) & (t < t3)
        fraction[ascent] = 1.0 - _smooth_step((t[ascent] - t2) / ASCENT_S)
        rep_index[(t >= t0) & (t < t3)] = rep
    return t, fraction, rep_index


def _pivot_about_mtp(
    mtp: np.ndarray, forward: np.ndarray, along_m: float, height_m: float, pitch_rad: float,
) -> np.ndarray:
    # A rear-foot point (along the foot axis, at a height above the floor) pitched heel-up about the ball.
    along = along_m * math.cos(pitch_rad) + height_m * math.sin(pitch_rad)
    height = -along_m * math.sin(pitch_rad) + height_m * math.cos(pitch_rad)
    return mtp + along * forward + [0.0, -height, 0.0]


def _roll_matrix(roll_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    return np.array([[math.cos(roll), -math.sin(roll), 0.0],
                     [math.sin(roll), math.cos(roll), 0.0],
                     [0.0, 0.0, 1.0]])


def _squat_sequence(
    n_reps: int = 3,
    stand_s: float = STAND_S,
    bottom_s: float = BOTTOM_S,
    heel_rise_deg: tuple[float, float] = (0.0, 0.0),
    hip_shift_m: float = 0.0,
    initial_width_m: float | None = None,
    roll_deg: float = 0.0,
    foot_snap: bool = False,
    seed: int = 0,
) -> dict:
    """Noisy world-frame squat (Y-down, forward = -Z): feet planted on a floor, optional faults."""
    t, fraction, rep_index = _depth_profile(n_reps, stand_s, bottom_s)
    n_frames = len(t)
    world = np.zeros((n_frames, NUM_KEYPOINTS, 3))
    heel_rise_true = np.zeros((n_frames, 2))
    width_true = np.full(n_frames, STANCE_WIDTH_M)
    toe_out = math.radians(TOE_OUT_DEG)
    for i in range(n_frames):
        s = fraction[i]
        width = STANCE_WIDTH_M
        lift = 0.0
        if initial_width_m is not None:
            step_fraction = (t[i] - STEP_START_S) / STEP_DURATION_S
            width = initial_width_m + (STANCE_WIDTH_M - initial_width_m) * float(_smooth_step(np.array(step_fraction)))
            if 0.0 < step_fraction < 1.0:
                lift = STEP_LIFT_M * math.sin(math.pi * step_fraction)
        width_true[i] = width
        hip_mid = np.array([hip_shift_m * s, HIP_DROP_M * s, HIP_BACK_M * s])
        hips = (hip_mid + [HIP_HALF_M, 0.0, 0.0], hip_mid - [HIP_HALF_M, 0.0, 0.0])
        for side, sign in ((0, 1.0), (1, -1.0)):
            forward = np.array([sign * math.sin(toe_out), 0.0, -math.cos(toe_out)])
            ankle_rest = np.array([sign * width / 2.0, FLOOR_Y_M - ANKLE_HEIGHT_M - lift, 0.0])
            toe = ankle_rest + TOE_FORWARD_M * forward + [0.0, ANKLE_HEIGHT_M - TOE_HEIGHT_M, 0.0]
            # Heel rise pivots the rear foot about the ball of the foot: ankle and heel swing up and forward.
            mtp = ankle_rest + MTP_FORWARD_M * forward + [0.0, ANKLE_HEIGHT_M, 0.0]
            pitch_rad = math.radians(heel_rise_deg[side] * s)
            ankle = _pivot_about_mtp(mtp, forward, -MTP_FORWARD_M, ANKLE_HEIGHT_M, pitch_rad)
            heel = _pivot_about_mtp(mtp, forward, -(MTP_FORWARD_M + HEEL_BACK_M), HEEL_HEIGHT_M, pitch_rad)
            knee = (hips[side] + ankle) / 2.0 + [0.0, 0.0, -(0.03 + KNEE_FORWARD_M * s)]
            heel_rise_true[i, side] = (ankle_rest[1] - ankle[1]) + lift
            hip_idx, knee_idx, ankle_idx, toe_idx, heel_idx = (
                (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL) if side == 0
                else (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL))
            world[i, hip_idx] = hips[side]
            world[i, knee_idx] = knee
            world[i, ankle_idx] = ankle
            world[i, toe_idx] = toe
            world[i, heel_idx] = heel
        shoulder_mid = hip_mid + [0.0, -0.5, 0.15 * s]
        world[i, CK.LEFT_SHOULDER] = shoulder_mid + [0.2, 0.0, 0.0]
        world[i, CK.RIGHT_SHOULDER] = shoulder_mid - [0.2, 0.0, 0.0]
        world[i, CK.LEFT_ELBOW] = world[i, CK.LEFT_SHOULDER] + [0.05, 0.15, 0.1]
        world[i, CK.RIGHT_ELBOW] = world[i, CK.RIGHT_SHOULDER] + [-0.05, 0.15, 0.1]
        world[i, CK.LEFT_WRIST] = world[i, CK.LEFT_SHOULDER] + [0.12, -0.02, 0.05]
        world[i, CK.RIGHT_WRIST] = world[i, CK.RIGHT_SHOULDER] + [-0.12, -0.02, 0.05]
        head = shoulder_mid + [0.0, -0.2, 0.0]
        world[i, CK.NOSE] = head + [0.0, 0.0, -0.09]
        world[i, CK.LEFT_EYE] = head + [0.03, -0.03, -0.08]
        world[i, CK.RIGHT_EYE] = head + [-0.03, -0.03, -0.08]
        world[i, CK.LEFT_EAR] = head + [0.07, 0.0, 0.0]
        world[i, CK.RIGHT_EAR] = head + [-0.07, 0.0, 0.0]

    ideal = world @ _roll_matrix(roll_deg).T
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, WHITE_SIGMA_M, size=ideal.shape)
    rho = math.exp(-1.0 / (DRIFT_TAU_S * FPS))
    drift = np.zeros_like(ideal)
    drift[0] = rng.normal(0.0, DRIFT_SIGMA_M, size=ideal.shape[1:])
    innovation = rng.normal(0.0, DRIFT_SIGMA_M * math.sqrt(1.0 - rho ** 2), size=ideal.shape)
    for i in range(1, n_frames):
        drift[i] = rho * drift[i - 1] + innovation[i]
    points = ideal + white + drift
    if foot_snap:
        # At each rep bottom one foot's keypoints are drawn at the knee (real RTMPose failure), alternating feet.
        for i in np.flatnonzero(fraction > SNAP_FRACTION):
            knee_idx, ankle_idx, toe_idx, heel_idx = (
                (CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL) if rep_index[i] % 2 == 0
                else (CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL))
            knee = points[i, knee_idx]
            points[i, ankle_idx] = knee + [0.0, 0.04, -0.02] + white[i, ankle_idx]
            points[i, toe_idx] = knee + [0.0, 0.07, -0.08] + white[i, toe_idx]
            points[i, heel_idx] = knee + [0.0, 0.07, 0.02] + white[i, heel_idx]
    return {
        "t": t, "fraction": fraction, "points": points, "ideal": ideal,
        "heel_rise_true": heel_rise_true, "width_true": width_true,
        "hip_mid_ideal": (ideal[:, CK.LEFT_HIP] + ideal[:, CK.RIGHT_HIP]) / 2.0,
    }


def _min_jerk(fraction: float) -> float:
    fraction = min(max(fraction, 0.0), 1.0)
    return fraction ** 3 * (10.0 - 15.0 * fraction + 6.0 * fraction ** 2)


def _step_sequence(step_m: float, step_s: float, seed: int = 0) -> dict:
    """Standing, then the left foot steps forward by step_m in step_s (min-jerk, 7 cm lift), then standing."""
    base = _squat_sequence(n_reps=0, stand_s=1.0)["ideal"][0]
    step_frames = int(round(step_s * FPS))
    n_frames = STEP_STAND_FRAMES + step_frames + STEP_STAND_FRAMES
    ideal = np.repeat(base[None], n_frames, axis=0)
    for i in range(STEP_STAND_FRAMES, n_frames):
        fraction = (i - STEP_STAND_FRAMES) / step_frames
        ideal[i, STEP_FOOT_INDICES, 2] -= step_m * _min_jerk(fraction)
        ideal[i, STEP_FOOT_INDICES, 1] -= STEP_LIFT_M * math.sin(math.pi * min(fraction, 1.0))
    rng = np.random.default_rng(seed)
    points = ideal + rng.normal(0.0, STEP_NOISE_M, size=ideal.shape)
    return {"points": points, "ideal": ideal, "step_start": STEP_STAND_FRAMES,
            "step_end": STEP_STAND_FRAMES + step_frames}


def _run_model(
    points: np.ndarray, confidences: np.ndarray | None = None, model: FootContactModel | None = None,
) -> tuple[np.ndarray, list[FootState]]:
    model = model or FootContactModel()
    confidences = np.ones(NUM_KEYPOINTS) if confidences is None else confidences
    outputs = np.zeros_like(points)
    states: list[FootState] = []
    for i in range(len(points)):
        outputs[i], state = model.update(points[i], confidences, i / FPS)
        states.append(state)
    return outputs, states


def _rmse_cm(estimate: np.ndarray, truth: np.ndarray, frames: np.ndarray, indices: list[int]) -> float:
    error = np.linalg.norm(estimate[frames][:, indices] - truth[frames][:, indices], axis=2)
    return float(np.sqrt(np.mean(error ** 2))) * M_TO_CM


def _rise_at_bottom_cm(states: list[FootState], sequence: dict) -> tuple[float, float]:
    bottom = sequence["fraction"] == 1.0
    rise = np.array([[state.heel_rise_l_cm, state.heel_rise_r_cm] for state in states])
    return float(np.nanmean(rise[bottom, 0])), float(np.nanmean(rise[bottom, 1]))


class TestFootStabilization:

    def test_non_foot_keypoints_pass_through_unchanged(self) -> None:
        sequence = _squat_sequence(foot_snap=True, heel_rise_deg=(0.0, HEEL_RISE_DEG))
        outputs, _ = _run_model(sequence["points"])
        assert np.array_equal(outputs[:, :CK.LEFT_ANKLE], sequence["points"][:, :CK.LEFT_ANKLE])

    def test_clean_squat_foot_error_below_raw(self) -> None:
        sequence = _squat_sequence()
        outputs, _ = _run_model(sequence["points"])
        frames = np.arange(len(outputs)) >= EVAL_START_FRAME
        raw_ankle = _rmse_cm(sequence["points"], sequence["ideal"], frames, ANKLE_INDICES)
        assert _rmse_cm(outputs, sequence["ideal"], frames, ANKLE_INDICES) < 1.0 < raw_ankle
        assert _rmse_cm(outputs, sequence["ideal"], frames, TOE_INDICES) < 1.0

    def test_foot_snap_to_knee_rejected(self) -> None:
        sequence = _squat_sequence(foot_snap=True)
        outputs, states = _run_model(sequence["points"])
        frames = np.arange(len(outputs)) >= EVAL_START_FRAME
        assert _rmse_cm(outputs, sequence["ideal"], frames, ANKLE_INDICES) < 1.5
        assert _rmse_cm(outputs, sequence["ideal"], frames, TOE_INDICES) < 1.5
        rise_l, rise_r = _rise_at_bottom_cm(states, sequence)
        assert max(rise_l, rise_r) < 0.5

    def test_stance_change_after_warmup_is_tracked(self) -> None:
        stand_s = STEP_START_S + STEP_DURATION_S + 1.0
        sequence = _squat_sequence(initial_width_m=0.30, stand_s=stand_s)
        outputs, states = _run_model(sequence["points"])
        settled = sequence["t"] >= stand_s
        width = np.array([state.stance_width_cm for state in states])
        width_error = width[settled] - sequence["width_true"][settled] * M_TO_CM
        assert abs(float(np.mean(width_error))) < 1.0
        assert _rmse_cm(outputs, sequence["ideal"], settled, ANKLE_INDICES) < 1.0

    def test_duplicate_timestamp_repeats_last_output(self) -> None:
        sequence = _squat_sequence(n_reps=1)
        model = FootContactModel()
        confidences = np.ones(NUM_KEYPOINTS)
        for i in range(90):
            outputs, state = model.update(sequence["points"][i], confidences, i / FPS)
        repeat, repeat_state = model.update(sequence["points"][90], confidences, 89 / FPS)
        assert np.array_equal(repeat, outputs)
        assert repeat_state == state

    def test_reset_clears_session_state(self) -> None:
        sequence = _squat_sequence(n_reps=1)
        model = FootContactModel()
        _, states = _run_model(sequence["points"], model=model)
        assert states[-1].valid
        model.reset()
        _, state = model.update(sequence["points"][0], np.ones(NUM_KEYPOINTS), 0.0)
        assert not state.valid
        assert not state.planted_l and not state.planted_r


class TestFootStep:

    @pytest.mark.parametrize("step_s, drop_odd_ticks", [(0.5, False), (0.3, False), (0.5, True), (0.3, True)])
    def test_step_is_released_and_replanted_promptly(self, step_s: float, drop_odd_ticks: bool) -> None:
        sequence = _step_sequence(step_m=0.25, step_s=step_s)
        model = FootContactModel()
        confidences = np.ones(NUM_KEYPOINTS)
        release_frame = None
        replant_frame = None
        post_replant_errors_cm = []
        for i in range(len(sequence["points"])):
            if drop_odd_ticks and i % 2 == 1:
                continue
            outputs, state = model.update(sequence["points"][i], confidences, i / FPS)
            if i < sequence["step_start"]:
                continue
            if release_frame is None and not state.planted_l:
                release_frame = i
            if release_frame is not None and replant_frame is None and state.planted_l:
                replant_frame = i
            if replant_frame is not None:
                post_replant_errors_cm.append(
                    np.linalg.norm(outputs[CK.LEFT_ANKLE] - sequence["ideal"][i, CK.LEFT_ANKLE]) * M_TO_CM)
        assert release_frame is not None and release_frame < sequence["step_end"]
        assert replant_frame is not None
        assert (replant_frame - sequence["step_end"]) / FPS <= REPLANT_DEADLINE_S
        assert float(np.sqrt(np.mean(np.square(post_replant_errors_cm)))) < POST_REPLANT_ANKLE_ERROR_CM
        assert state.planted_r

    def test_foot_snap_is_not_a_step(self) -> None:
        sequence = _squat_sequence(foot_snap=True)
        _, states = _run_model(sequence["points"])
        snapped = sequence["fraction"] > SNAP_FRACTION
        planted = np.array([[state.planted_l, state.planted_r] for state in states])
        assert planted[snapped].all()

    def test_smoothed_foot_snap_is_not_a_step(self) -> None:
        # Production runs the Kalman smoother first; its fixed-lag output smears a snap over a few frames.
        sequence = _squat_sequence(foot_snap=True)
        smoother = FixedLagKeypointSmoother()
        model = FootContactModel()
        confidences = np.full(NUM_KEYPOINTS, SMOOTHED_CONFIDENCE)
        ankle_errors_cm = []
        planted = []
        for i in range(len(sequence["points"])):
            smoothed = smoother.update(sequence["points"][i], confidences, i / FPS)
            if smoothed is None:
                continue
            lagged_frame = int(round(smoothed.lagged_timestamp * FPS))
            outputs, state = model.update(
                smoothed.lagged_points, smoothed.lagged_confidences, smoothed.lagged_timestamp)
            if lagged_frame >= EVAL_START_FRAME:
                ankle_errors_cm.append(np.linalg.norm(
                    outputs[ANKLE_INDICES] - sequence["ideal"][lagged_frame, ANKLE_INDICES], axis=1) * M_TO_CM)
            if sequence["fraction"][lagged_frame] > SNAP_FRACTION:
                planted.append([state.planted_l, state.planted_r])
        assert float(np.sqrt(np.mean(np.square(ankle_errors_cm)))) < 1.5
        assert np.array(planted).all()


class TestHeelRise:

    def test_one_sided_heel_rise_measured_on_that_foot_only(self) -> None:
        sequence = _squat_sequence(heel_rise_deg=(0.0, HEEL_RISE_DEG))
        bottom = sequence["fraction"] == 1.0
        assert float(np.mean(sequence["heel_rise_true"][bottom, 1])) * M_TO_CM == pytest.approx(3.37, abs=0.05)
        outputs, states = _run_model(sequence["points"])
        rise_l, rise_r = _rise_at_bottom_cm(states, sequence)
        assert rise_r >= 2.8
        assert abs(rise_l) < 0.5
        frames = np.arange(len(outputs)) >= EVAL_START_FRAME
        assert _rmse_cm(outputs, sequence["ideal"], frames, [CK.LEFT_ANKLE]) < 1.0
        assert _rmse_cm(outputs, sequence["ideal"], frames, TOE_INDICES) < 1.0

    def test_bilateral_heel_rise(self) -> None:
        sequence = _squat_sequence(heel_rise_deg=(HEEL_RISE_DEG, HEEL_RISE_DEG))
        _, states = _run_model(sequence["points"])
        rise_l, rise_r = _rise_at_bottom_cm(states, sequence)
        assert rise_l >= 2.5
        assert rise_r >= 2.5

    def test_heel_rise_survives_foot_snap(self) -> None:
        sequence = _squat_sequence(heel_rise_deg=(0.0, HEEL_RISE_DEG), foot_snap=True)
        _, states = _run_model(sequence["points"])
        rise_l, rise_r = _rise_at_bottom_cm(states, sequence)
        assert rise_r >= 1.5
        assert abs(rise_l) < 0.5

    def test_raised_hold_never_becomes_floor(self) -> None:
        sequence = _squat_sequence(n_reps=1, bottom_s=3.0, heel_rise_deg=(0.0, HEEL_RISE_DEG))
        _, states = _run_model(sequence["points"])
        bottom_frames = np.flatnonzero(sequence["fraction"] == 1.0)
        late_hold = bottom_frames[-15:]
        rise_r = np.array([states[i].heel_rise_r_cm for i in late_hold])
        assert float(np.nanmean(rise_r)) >= 2.8

    def test_clean_squat_reports_no_rise(self) -> None:
        sequence = _squat_sequence()
        _, states = _run_model(sequence["points"])
        rise_l, rise_r = _rise_at_bottom_cm(states, sequence)
        assert abs(rise_l) < 0.5
        assert abs(rise_r) < 0.5

    def test_missing_heels_use_ankle_and_toe(self) -> None:
        sequence = _squat_sequence(heel_rise_deg=(0.0, HEEL_RISE_DEG))
        points = sequence["points"].copy()
        points[:, HEEL_INDICES] = 0.0
        confidences = np.ones(NUM_KEYPOINTS)
        confidences[HEEL_INDICES] = 0.0
        outputs, states = _run_model(points, confidences)
        assert states[-1].valid
        rise_l, rise_r = _rise_at_bottom_cm(states, sequence)
        assert rise_r >= 2.8
        assert abs(rise_l) < 0.5
        assert np.array_equal(outputs[:, HEEL_INDICES], points[:, HEEL_INDICES])
        frames = np.arange(len(outputs)) >= EVAL_START_FRAME
        assert _rmse_cm(outputs, sequence["ideal"], frames, TOE_INDICES) < 1.0


class TestFloorState:

    def test_valid_only_after_both_feet_planted(self) -> None:
        sequence = _squat_sequence(n_reps=1)
        _, states = _run_model(sequence["points"])
        assert not states[0].valid
        assert math.isnan(states[0].stance_width_cm)
        assert states[-1].valid
        standing = sequence["fraction"] == 0.0
        planted = np.array([[state.planted_l, state.planted_r] for state in states])
        assert planted[standing][EVAL_START_FRAME:].mean() > 0.95

    def test_roll_tilt_estimated_from_anchors(self) -> None:
        sequence = _squat_sequence(n_reps=1, roll_deg=5.0)
        _, states = _run_model(sequence["points"])
        assert states[-1].floor_roll_deg == pytest.approx(5.0, abs=1.0)
        assert states[-1].stance_width_cm == pytest.approx(STANCE_WIDTH_M * M_TO_CM, abs=1.5)

    def test_toe_out_from_anchors(self) -> None:
        sequence = _squat_sequence(n_reps=1)
        _, states = _run_model(sequence["points"])
        assert states[-1].toe_out_l_deg == pytest.approx(TOE_OUT_DEG, abs=3.0)
        assert states[-1].toe_out_r_deg == pytest.approx(TOE_OUT_DEG, abs=3.0)

    def test_hip_height_tracks_squat_depth(self) -> None:
        sequence = _squat_sequence(n_reps=2)
        _, states = _run_model(sequence["points"])
        hip_height = np.array([state.hip_height_above_floor_cm for state in states])
        standing = (sequence["fraction"] == 0.0) & (np.arange(len(states)) >= EVAL_START_FRAME)
        bottom = sequence["fraction"] == 1.0
        drop_cm = float(np.mean(hip_height[standing]) - np.mean(hip_height[bottom]))
        assert drop_cm == pytest.approx(HIP_DROP_M * M_TO_CM, abs=2.0)
        assert float(np.std(hip_height[standing])) < 1.5

    def test_lateral_hip_offset_follows_hip_shift(self) -> None:
        sequence = _squat_sequence(n_reps=2, hip_shift_m=0.06)
        _, states = _run_model(sequence["points"])
        offset = np.array([state.lateral_hip_offset_cm for state in states])
        bottom = sequence["fraction"] == 1.0
        # The model reads the input (noisy) hips against its foot anchors, so compare with the same noisy
        # hip midpoint measured against the true ankle midpoint: the residual is the anchor placement error.
        hip_mid_x = (sequence["points"][:, CK.LEFT_HIP, 0] + sequence["points"][:, CK.RIGHT_HIP, 0]) / 2.0
        ankle_mid_x = (sequence["ideal"][:, CK.LEFT_ANKLE, 0] + sequence["ideal"][:, CK.RIGHT_ANKLE, 0]) / 2.0
        expected_cm = float(np.mean((hip_mid_x - ankle_mid_x)[bottom])) * M_TO_CM
        assert expected_cm == pytest.approx(6.0, abs=1.0)
        assert float(np.mean(offset[bottom])) == pytest.approx(expected_cm, abs=1.0)
