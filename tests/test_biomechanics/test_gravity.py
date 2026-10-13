"""
Tests for measured gravity on the rig and the shared sagittal sign conventions
(SHARED-PLUMBING.md §3): per-camera gravity files map into the world through
each camera's rotation, up = -gravity, and every metric uses up as is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from biomechanics.triangulation.calibration import CalibrationResult, CameraCalibration
from biomechanics.utils import gravity as gravity_module
from biomechanics.utils.frame_context import GRAVITY_BODY, GRAVITY_MEASURED
from biomechanics.utils.geometry import WORLD_UP, forward_offset, lean_back_deg, sagittal_forward
from biomechanics.utils.gravity import gravity_path, load_camera_gravity, measured_up

ANGLE_TOLERANCE_DEG = 0.5
VECTOR_TOLERANCE = 1e-9
WORLD_TILT_DEG = 3.0
MOVED_CAMERA_DEG = 2.0
CAMERA_YAWS_DEG = (0.0, 35.0, -40.0)
TRUNK_LENGTH_M = 0.5
BAR_HEIGHT_M = 2.15
BAR_AHEAD_M = 0.08
LEAN_BACK_DEG = 10.0
LATERAL_RIGHT = np.array([-1.0, 0.0, 0.0])


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ])


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])


def _calibration(rotations: list[np.ndarray]) -> CalibrationResult:
    cameras = {
        str(index): CameraCalibration(
            camera_id=str(index),
            projection_matrix=np.zeros((3, 4)),
            intrinsic_matrix=np.eye(3),
            rotation_matrix=rotation,
            translation_vector=np.zeros(3),
            reprojection_error=0.0,
            resolution=(1280, 720),
        )
        for index, rotation in enumerate(rotations)
    }
    return CalibrationResult(cameras=cameras)


def _write_gravity(directory: Path, camera_key: str, gravity_camera: np.ndarray) -> None:
    gravity_path(camera_key, directory).write_text(json.dumps({"gravity": list(gravity_camera)}))


def _true_gravity_world() -> np.ndarray:
    # The rig's world Y sits WORLD_TILT_DEG off true gravity.
    return _rotation_x(WORLD_TILT_DEG) @ np.array([0.0, 1.0, 0.0])


class TestMeasuredUp:
    def test_gravity_files_map_through_each_cameras_rotation(self, tmp_path: Path):
        rotations = [_rotation_y(yaw) for yaw in CAMERA_YAWS_DEG]
        for index, rotation in enumerate(rotations):
            _write_gravity(tmp_path, str(index), rotation @ _true_gravity_world())

        up = measured_up(_calibration(rotations), {}, tmp_path)

        assert up == pytest.approx(-_true_gravity_world(), abs=VECTOR_TOLERANCE)
        assert up[1] < 0.0

    def test_camera_keys_name_the_files(self, tmp_path: Path):
        _write_gravity(tmp_path, "usb_left", _true_gravity_world())
        up = measured_up(_calibration([np.eye(3)]), {"0": "usb_left"}, tmp_path)
        assert up == pytest.approx(-_true_gravity_world(), abs=VECTOR_TOLERANCE)

    def test_a_camera_that_moved_is_dropped(self, tmp_path: Path, caplog):
        rotations = [np.eye(3), np.eye(3), np.eye(3)]
        _write_gravity(tmp_path, "0", _true_gravity_world())
        _write_gravity(tmp_path, "1", _true_gravity_world())
        _write_gravity(tmp_path, "2", _rotation_x(4 * MOVED_CAMERA_DEG) @ _true_gravity_world())

        up = measured_up(_calibration(rotations), {}, tmp_path)

        assert up == pytest.approx(-_true_gravity_world(), abs=VECTOR_TOLERANCE)
        assert "disagree" in caplog.text

    def test_no_measurement_means_none(self, tmp_path: Path):
        assert measured_up(_calibration([np.eye(3)]), {}, tmp_path) is None

    def test_unreadable_files_are_ignored(self, tmp_path: Path):
        gravity_path("0", tmp_path).write_text("not json")
        gravity_path("1", tmp_path).write_text(json.dumps({"gravity": [0.0, 0.0, 0.0]}))
        assert load_camera_gravity(gravity_path("0", tmp_path)) is None
        assert load_camera_gravity(gravity_path("1", tmp_path)) is None
        assert measured_up(_calibration([np.eye(3), np.eye(3)]), {}, tmp_path) is None


def _sagittal_scene(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(forward axis, lateral) of a lifter standing on a floor whose true up is `up`."""
    return sagittal_forward(up, LATERAL_RIGHT), LATERAL_RIGHT


class TestSharedSignConvention:
    """An upright trunk reads 0° of lean and a bar ahead of the feet reads a
    positive forward offset, whether up is the body axis or a tilted measurement."""

    @pytest.mark.parametrize("source", [GRAVITY_BODY, GRAVITY_MEASURED])
    def test_upright_trunk_reads_no_lean_and_a_bar_ahead_reads_positive(self, source: str):
        up = WORLD_UP if source == GRAVITY_BODY else -_true_gravity_world()
        forward, _ = _sagittal_scene(up)
        trunk = TRUNK_LENGTH_M * up
        bar_from_midfoot = BAR_HEIGHT_M * up + BAR_AHEAD_M * forward

        assert lean_back_deg(trunk, up, forward) == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)
        assert forward_offset(bar_from_midfoot, up, forward) == pytest.approx(BAR_AHEAD_M, abs=1e-6)

    def test_forward_points_toward_the_toes_in_the_production_frame(self):
        forward = sagittal_forward(WORLD_UP, LATERAL_RIGHT)
        assert forward == pytest.approx(np.array([0.0, 0.0, -1.0]))

    def test_leaning_back_reads_positive(self):
        up = -_true_gravity_world()
        forward, _ = _sagittal_scene(up)
        angle = math.radians(LEAN_BACK_DEG)
        trunk = TRUNK_LENGTH_M * (math.cos(angle) * up - math.sin(angle) * forward)
        assert lean_back_deg(trunk, up, forward) == pytest.approx(LEAN_BACK_DEG, abs=ANGLE_TOLERANCE_DEG)

    def test_negated_up_fails_the_upright_trunk(self):
        up = WORLD_UP
        forward = sagittal_forward(up, LATERAL_RIGHT)
        trunk = TRUNK_LENGTH_M * up
        assert abs(lean_back_deg(trunk, -up, forward)) > 90.0

    def test_the_body_axis_on_a_tilted_rig_biases_the_lean(self):
        true_up = -_true_gravity_world()
        forward = sagittal_forward(WORLD_UP, LATERAL_RIGHT)
        reading = lean_back_deg(TRUNK_LENGTH_M * true_up, WORLD_UP, forward)
        assert abs(reading) == pytest.approx(WORLD_TILT_DEG, abs=ANGLE_TOLERANCE_DEG)


class TestPipelineGravity:
    def test_the_rig_uses_measured_gravity_when_the_calibration_has_it(self, monkeypatch, tmp_path: Path):
        from conftest import SyntheticSession

        monkeypatch.setattr(gravity_module, "GRAVITY_DIR", tmp_path)
        for camera_id in ("0", "1", "2"):
            _write_gravity(tmp_path, camera_id, _true_gravity_world())
        session = SyntheticSession(monkeypatch)
        assert session.pipeline._gravity()[1] == GRAVITY_BODY

        session.provider.calibration = _calibration([np.eye(3)] * 3)
        up, source = session.pipeline._gravity()

        assert source == GRAVITY_MEASURED
        assert up == pytest.approx(-_true_gravity_world(), abs=VECTOR_TOLERANCE)

    def test_single_camera_keeps_world_up(self, monkeypatch):
        from conftest import SyntheticSession

        session = SyntheticSession(monkeypatch, multi_camera=False)
        up, source = session.pipeline._gravity()
        assert source == GRAVITY_BODY
        assert up == pytest.approx(WORLD_UP)
