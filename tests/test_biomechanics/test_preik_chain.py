"""Integration tests for the pre-IK chain (C10).

Drives the real chain over a synthetic world-frame squat with capture-clock
timestamps and asserts on the chain's end-to-end contract: lagged hip-centred
analysis, undelayed display, foot contact in multi-camera mode, bounded
prediction through dropouts, and per-set reset scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.utils.preik_chain import PreIKChain, PreIKResult, build_preik_chain
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Skeleton3D
from conftest import SYNTHETIC_FPS, SYNTHETIC_NUM_KEYPOINTS, squat_depth_profile, world_squat_points

CLOCK_START_S = 1.7e9
DEFAULT_CONFIDENCE = 0.9
LAG_FRAMES = 2
MAX_PREDICTED_FRAMES = 5
WARM_UP_FRAMES = 60
POSITION_TOL_M = 0.01
HIP_CENTRE_TOL_M = 1e-9


def _timestamp(frame_index: int) -> float:
    return CLOCK_START_S + frame_index / SYNTHETIC_FPS


def _make_skeleton(
    points: np.ndarray, frame_index: int, confidences: np.ndarray | None = None,
) -> Skeleton3D:
    if confidences is None:
        confidences = np.full(len(points), DEFAULT_CONFIDENCE)
    return Skeleton3D.from_numpy(
        points, confidences=confidences,
        timestamp=_timestamp(frame_index), frame_index=frame_index,
    )


def _centred(points: np.ndarray) -> np.ndarray:
    return points - (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0


def _hip_midpoint(skeleton: Skeleton3D) -> np.ndarray:
    points = skeleton.to_numpy()
    return (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0


def _multi_camera_chain() -> PreIKChain:
    return build_preik_chain(BiomechanicsConfig(), multi_camera=True)


def _single_camera_chain() -> PreIKChain:
    return build_preik_chain(BiomechanicsConfig(), multi_camera=False)


def _run_squat(chain: PreIKChain, hip_centred_input: bool = False) -> tuple[list[np.ndarray], list[PreIKResult | None]]:
    """Feed standing warm-up then one rep; return the input points and chain outputs per frame."""
    inputs: list[np.ndarray] = []
    outputs: list[PreIKResult | None] = []
    depths = [0.0] * WARM_UP_FRAMES + squat_depth_profile()
    for frame_index, depth in enumerate(depths):
        points = world_squat_points(depth)
        if hip_centred_input:
            points = _centred(points)
        inputs.append(points)
        outputs.append(chain.run(_make_skeleton(points, frame_index)))
    return inputs, outputs


class TestMultiCameraChain:

    def test_stage_names(self):
        assert _multi_camera_chain().stage_names == ("raw", "kalman", "foot_contact", "recentre")

    def test_analysis_is_hip_centred(self):
        _, outputs = _run_squat(_multi_camera_chain())
        for result in outputs[LAG_FRAMES:]:
            assert result is not None
            assert np.linalg.norm(_hip_midpoint(result.analysis)) < HIP_CENTRE_TOL_M

    def test_analysis_lags_two_frames(self):
        inputs, outputs = _run_squat(_multi_camera_chain())
        for frame_index in range(WARM_UP_FRAMES, len(inputs)):
            result = outputs[frame_index]
            assert result.analysis.timestamp == pytest.approx(_timestamp(frame_index - LAG_FRAMES))
            assert result.analysis.frame_index == frame_index - LAG_FRAMES
            expected = _centred(inputs[frame_index - LAG_FRAMES])
            deviation = np.linalg.norm(result.analysis.to_numpy() - expected, axis=1).max()
            assert deviation < POSITION_TOL_M

    def test_display_is_undelayed(self):
        inputs, outputs = _run_squat(_multi_camera_chain())
        for frame_index in range(WARM_UP_FRAMES, len(inputs)):
            result = outputs[frame_index]
            assert result.display is not None
            assert result.display.timestamp == pytest.approx(_timestamp(frame_index))
            assert result.display.frame_index == frame_index
            deviation = np.linalg.norm(result.display.to_numpy() - _centred(inputs[frame_index]), axis=1).max()
            assert deviation < POSITION_TOL_M

    def test_analysis_world_keeps_the_floor(self):
        _, outputs = _run_squat(_multi_camera_chain())
        world = outputs[-1].analysis_world
        assert world is not None
        assert world.to_numpy()[CK.LEFT_ANKLE][1] == pytest.approx(0.0, abs=POSITION_TOL_M)

    def test_foot_state_valid_after_warm_up(self):
        _, outputs = _run_squat(_multi_camera_chain())
        foot_state = outputs[WARM_UP_FRAMES - 1].foot_state
        assert foot_state is not None
        assert foot_state.valid
        assert foot_state.planted_l and foot_state.planted_r
        assert foot_state.heel_rise_l_cm == pytest.approx(0.0)

    def test_velocities_match_frame_count(self):
        _, outputs = _run_squat(_multi_camera_chain())
        assert outputs[-1].velocities.shape == (SYNTHETIC_NUM_KEYPOINTS, 3)

    def test_predict_missing_continues_then_stops(self):
        chain = _multi_camera_chain()
        _, outputs = _run_squat(chain)
        next_index = len(outputs)

        predicted: list[PreIKResult | None] = []
        for offset in range(MAX_PREDICTED_FRAMES + LAG_FRAMES + 2):
            predicted.append(chain.predict_missing(_timestamp(next_index + offset)))

        for result in predicted[:MAX_PREDICTED_FRAMES]:
            assert result is not None
            assert np.linalg.norm(_hip_midpoint(result.analysis)) < HIP_CENTRE_TOL_M
        assert predicted[-1] is None

    def test_predict_missing_before_any_measurement_is_none(self):
        assert _multi_camera_chain().predict_missing(_timestamp(0)) is None

    def test_hips_missing_on_first_frame_is_none(self):
        chain = _multi_camera_chain()
        confidences = np.full(SYNTHETIC_NUM_KEYPOINTS, DEFAULT_CONFIDENCE)
        confidences[CK.LEFT_HIP] = 0.0
        assert chain.run(_make_skeleton(world_squat_points(0.0), 0, confidences)) is None

    def test_reset_clears_kalman_but_keeps_foot_anchors(self):
        chain = _multi_camera_chain()
        _, outputs = _run_squat(chain)
        assert outputs[-1].foot_state.valid

        chain.reset()
        next_index = len(outputs)
        result = chain.run(_make_skeleton(world_squat_points(0.0), next_index))

        assert result is not None
        # No history after the reset: the lagged output IS the current frame.
        assert result.analysis.timestamp == pytest.approx(_timestamp(next_index))
        assert result.analysis.frame_index == next_index
        # Foot contact is session-scoped: a fresh chain would still be warming up.
        assert result.foot_state.valid

    def test_tap_receives_every_stage(self):
        chain = _multi_camera_chain()
        seen: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        chain.set_tap(lambda stage, points, confidences: seen.append((stage, points.shape, confidences.shape)))

        chain.run(_make_skeleton(world_squat_points(0.0), 0))

        assert [stage for stage, _, _ in seen] == list(chain.stage_names)
        assert all(shape == (SYNTHETIC_NUM_KEYPOINTS, 3) for _, shape, _ in seen)
        assert all(shape == (SYNTHETIC_NUM_KEYPOINTS,) for _, _, shape in seen)

    def test_tap_can_be_removed(self):
        chain = _multi_camera_chain()
        seen: list[str] = []
        chain.set_tap(lambda stage, points, confidences: seen.append(stage))
        chain.set_tap(None)

        chain.run(_make_skeleton(world_squat_points(0.0), 0))

        assert seen == []


class TestSingleCameraChain:

    def test_stage_names(self):
        assert _single_camera_chain().stage_names == ("raw", "kalman")

    def test_no_foot_state_and_no_world_view(self):
        _, outputs = _run_squat(_single_camera_chain(), hip_centred_input=True)
        assert outputs[-1].foot_state is None
        assert outputs[-1].analysis_world is None

    def test_hip_centred_input_stays_hip_centred_and_lagged(self):
        inputs, outputs = _run_squat(_single_camera_chain(), hip_centred_input=True)
        for frame_index in range(WARM_UP_FRAMES, len(inputs)):
            result = outputs[frame_index]
            assert np.linalg.norm(_hip_midpoint(result.analysis)) < POSITION_TOL_M
            assert result.analysis.timestamp == pytest.approx(_timestamp(frame_index - LAG_FRAMES))
            deviation = np.linalg.norm(result.analysis.to_numpy() - inputs[frame_index - LAG_FRAMES], axis=1).max()
            assert deviation < POSITION_TOL_M

    def test_hips_missing_on_first_frame_is_none(self):
        chain = _single_camera_chain()
        confidences = np.full(SYNTHETIC_NUM_KEYPOINTS, DEFAULT_CONFIDENCE)
        confidences[CK.RIGHT_HIP] = 0.0
        assert chain.run(_make_skeleton(_centred(world_squat_points(0.0)), 0, confidences)) is None
