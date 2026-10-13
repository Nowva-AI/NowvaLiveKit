"""
Tests for the single camera's mount (SHARED-PLUMBING.md §3): configured and
env-entered pitch, certification, the delta tripwire, the session's
certification across exercise switches, and levelling a pitched camera's
skeleton (which brings the depth axis in at sin(pitch)).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig, CaptureConfig
from biomechanics.pipeline import _standing_pitch_deg
from biomechanics.profiles.overhead_press import OverheadPressProfile
from biomechanics.profiles.romanian_deadlift import RomanianDeadliftProfile
from biomechanics.utils.camera_mount import (
    PITCH_ENV,
    CameraMount,
    PitchCertification,
    PitchDeltaTripwire,
    camera_mount_from_config,
)
from biomechanics.utils.frame_context import MOUNT_INCLINOMETER, MOUNT_NONE, MOUNT_RACK
from biomechanics.utils.geometry import level_camera_frame
from biomechanics.utils.types import Skeleton3D
from conftest import SyntheticSession, world_squat_points
from test_press_poses import camera_pitched, counted_reps, hands_down, hold, press_rep, run_session
from test_rdl_poses import STANDING, RdlPose, camera_view, rdl_rep

RACK_PITCH_DEG = 1.5
ENTERED_PITCH_DEG = 2.5
LENS_HEIGHT_M = 0.95
TOLERANCE_DEG = 3.0
CAMERA_PITCH_DEG = 15.0
ANGLE_TOLERANCE_DEG = 1e-6
POSITION_TOLERANCE_M = 1e-9
SIGMA_XY_M = 0.01
SIGMA_Z_M = 0.03
NOISE_SAMPLES = 20000
NOISE_RATIO_TOLERANCE = 0.05
RNG_SEED = 7
RDL_TOLERANCE_DEG = 2.5
PRESS_NAME = "Barbell Overhead Press"
RDL_NAME = "Barbell Romanian Deadlift"
SQUAT_NAME = "Barbell Back Squat"
SQUAT_STAND_S = 2.0
SQUAT_STANDING_DEPTH = 0.02
SWITCH_FPS = 30.0
SWITCH_MOUNT_PITCH_DEG = 2.0
SWITCH_LENS_HEIGHT_M = 0.95
# Beyond both exercises' delta tolerances (press 3°, RDL 2.5°).
KNOCK_DEG = -4.0
TRIP_DELTA_TOLERANCE_DEG = 0.5
RDL_STAND_S = 2.0
RDL_BOTTOM = RdlPose(70.0, 45.0, 0.0)


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ])


def _pitched_camera_view(world_points: np.ndarray, pitch_deg: float) -> np.ndarray:
    """World points as a camera pitched by pitch_deg (lens up = positive) sees them."""
    return world_points @ _rotation_x(-pitch_deg).T


class TestCameraMountConfig:
    def test_no_mount_by_default(self, monkeypatch):
        monkeypatch.delenv(PITCH_ENV, raising=False)
        mount = camera_mount_from_config(BiomechanicsConfig().capture)
        assert mount.source == MOUNT_NONE
        assert math.isnan(mount.pitch_deg) and math.isnan(mount.height_m)
        assert not mount.is_measured

    def test_a_rack_mount_with_its_pitch_is_measured(self, monkeypatch):
        monkeypatch.delenv(PITCH_ENV, raising=False)
        capture = CaptureConfig(mount_source=MOUNT_RACK, mount_pitch_deg=RACK_PITCH_DEG, mount_height_m=LENS_HEIGHT_M)
        mount = camera_mount_from_config(capture)
        assert mount == CameraMount(source=MOUNT_RACK, pitch_deg=RACK_PITCH_DEG, height_m=LENS_HEIGHT_M)
        assert mount.is_measured

    def test_a_mount_without_a_pitch_is_not_measured(self, monkeypatch):
        monkeypatch.delenv(PITCH_ENV, raising=False)
        assert not camera_mount_from_config(CaptureConfig(mount_source=MOUNT_INCLINOMETER)).is_measured

    def test_env_pitch_counts_as_an_inclinometer_reading(self, monkeypatch):
        monkeypatch.setenv(PITCH_ENV, str(ENTERED_PITCH_DEG))
        mount = camera_mount_from_config(CaptureConfig())
        assert mount.source == MOUNT_INCLINOMETER
        assert mount.pitch_deg == pytest.approx(ENTERED_PITCH_DEG)
        assert mount.is_measured

    def test_env_pitch_overrides_a_rack_pitch_and_keeps_the_rack(self, monkeypatch):
        monkeypatch.setenv(PITCH_ENV, str(ENTERED_PITCH_DEG))
        mount = camera_mount_from_config(CaptureConfig(mount_source=MOUNT_RACK, mount_pitch_deg=RACK_PITCH_DEG))
        assert mount.source == MOUNT_RACK
        assert mount.pitch_deg == pytest.approx(ENTERED_PITCH_DEG)

    def test_a_malformed_env_pitch_is_ignored(self, monkeypatch, caplog):
        monkeypatch.setenv(PITCH_ENV, "level")
        assert not camera_mount_from_config(CaptureConfig()).is_measured
        assert PITCH_ENV in caplog.text


class TestPipelineCertification:
    def test_a_measured_single_camera_is_certified(self, monkeypatch):
        from conftest import SyntheticSession

        monkeypatch.setenv(PITCH_ENV, str(ENTERED_PITCH_DEG))
        session = SyntheticSession(monkeypatch, multi_camera=False)
        assert session.pipeline.pitch_certified
        session.pipeline.revoke_pitch_certification(TOLERANCE_DEG + 1.0)
        assert not session.pipeline.pitch_certified

    def test_the_rig_is_never_pitch_certified(self, monkeypatch):
        from conftest import SyntheticSession

        monkeypatch.setenv(PITCH_ENV, str(ENTERED_PITCH_DEG))
        assert not SyntheticSession(monkeypatch, multi_camera=True).pipeline.pitch_certified

    def test_an_unmeasured_camera_is_not_certified(self, monkeypatch):
        from conftest import SyntheticSession

        monkeypatch.delenv(PITCH_ENV, raising=False)
        assert not SyntheticSession(monkeypatch, multi_camera=False).pipeline.pitch_certified


class TestPitchDeltaTripwire:
    def test_without_a_reference_nothing_trips(self):
        assert PitchDeltaTripwire(TOLERANCE_DEG).check(30.0) is None

    def test_a_change_within_tolerance_holds(self):
        tripwire = PitchDeltaTripwire(TOLERANCE_DEG)
        tripwire.pool_reference(-4.0, 1)
        assert tripwire.check(-4.0 + TOLERANCE_DEG) is None
        assert not tripwire.tripped

    def test_a_change_beyond_tolerance_trips_once_and_stays_tripped(self):
        tripwire = PitchDeltaTripwire(TOLERANCE_DEG)
        tripwire.pool_reference(-4.0, 1)
        assert tripwire.check(1.0) == pytest.approx(5.0)
        assert tripwire.tripped
        assert tripwire.check(-4.0) is None
        assert tripwire.tripped_delta_deg == pytest.approx(5.0)

    def test_nan_readings_are_ignored(self):
        tripwire = PitchDeltaTripwire(TOLERANCE_DEG)
        tripwire.pool_reference(0.0, 1)
        assert tripwire.check(math.nan) is None


class TestPitchCertification:
    def test_each_exercise_keeps_its_own_tripwire_for_the_session(self):
        certification = PitchCertification()
        press = certification.tripwire("overhead_press", TOLERANCE_DEG)
        rdl = certification.tripwire("romanian_deadlift", RDL_TOLERANCE_DEG)
        assert certification.tripwire("overhead_press", TOLERANCE_DEG) is press
        assert rdl is not press
        assert press.tolerance_deg == pytest.approx(TOLERANCE_DEG)
        assert rdl.tolerance_deg == pytest.approx(RDL_TOLERANCE_DEG)

    def test_revocation_is_recorded_once(self):
        certification = PitchCertification()
        assert not certification.revoked
        assert certification.revoke(4.0)
        assert not certification.revoke(5.0)
        assert certification.revoked
        assert certification.revoked_delta_deg == pytest.approx(4.0)

    def test_pooled_readings_set_the_reference_once_enough_are_in(self):
        tripwire = PitchDeltaTripwire(TOLERANCE_DEG)
        tripwire.pool_reference(1.0, 3)
        tripwire.pool_reference(2.0, 3)
        assert math.isnan(tripwire.reference_deg)
        tripwire.pool_reference(6.0, 3)
        assert tripwire.reference_deg == pytest.approx(3.0)


def _switching_session(monkeypatch) -> SyntheticSession:
    monkeypatch.delenv(PITCH_ENV, raising=False)
    monkeypatch.setattr(OverheadPressProfile, "coaching_ready", True)
    monkeypatch.setattr(RomanianDeadliftProfile, "coaching_ready", True)
    config = BiomechanicsConfig()
    config.capture.mount_source = MOUNT_INCLINOMETER
    config.capture.mount_pitch_deg = SWITCH_MOUNT_PITCH_DEG
    config.capture.mount_height_m = SWITCH_LENS_HEIGHT_M
    return SyntheticSession(monkeypatch, multi_camera=False, config=config, exercise_name=PRESS_NAME)


def _end_set(session: SyntheticSession) -> None:
    session.pipeline.reset_readiness_gate()
    session.pipeline.reset_rep_counter()


def _press_set(session: SyntheticSession, pitch_deg: float) -> list:
    poses = hands_down(1.0, SWITCH_FPS) + hold(1.5, SWITCH_FPS) + press_rep(SWITCH_FPS) + hold(0.5, SWITCH_FPS)
    results = run_session(session, poses, SWITCH_FPS, transform=lambda points: camera_pitched(points, pitch_deg))
    _end_set(session)
    return results


def _press_reference_sets(session: SyntheticSession, pitch_deg: float) -> list:
    """The press sets whose rack-hold pitch pools the press tripwire's reference."""
    results = []
    for _ in range(BiomechanicsConfig().faults.press.camera.tripwire_reference_sets):
        results += _press_set(session, pitch_deg)
    return results


def _rdl_set(session: SyntheticSession, pitch_deg: float) -> list:
    poses = [STANDING] * int(RDL_STAND_S * SWITCH_FPS) + rdl_rep(SWITCH_FPS, bottom=RDL_BOTTOM, stand_s=0.0)
    poses += [STANDING] * int(SWITCH_FPS)
    results = [session.step(camera_view(pose.points(), pitch_deg), fps=SWITCH_FPS) for pose in poses]
    _end_set(session)
    return results


def _squat_standing(session: SyntheticSession, pitch_deg: float) -> list:
    points = camera_pitched(world_squat_points(SQUAT_STANDING_DEPTH), pitch_deg)
    results = [session.step(points, fps=SWITCH_FPS) for _ in range(int(SQUAT_STAND_S * SWITCH_FPS))]
    _end_set(session)
    return results


def _revoked(results: list) -> list[float]:
    return [result.pitch_certification_revoked_deg for result in results if result.pitch_certification_revoked_deg]


class TestCertificationAcrossExerciseSwitches:
    def test_each_new_analyser_gets_its_profiles_session_tripwire(self, monkeypatch):
        session = _switching_session(monkeypatch)
        first = session.pipeline._rep_analyzer._tripwire
        session.pipeline.set_exercise(SQUAT_NAME)
        session.pipeline.set_exercise(PRESS_NAME)
        assert session.pipeline._rep_analyzer._tripwire is first
        assert session.pipeline._pitch_certification.tripwire(OverheadPressProfile.name, TOLERANCE_DEG) is first

    def test_a_camera_knocked_during_squats_trips_the_press_on_its_return(self, monkeypatch):
        session = _switching_session(monkeypatch)
        assert not _revoked(_press_reference_sets(session, SWITCH_MOUNT_PITCH_DEG))

        session.pipeline.set_exercise(SQUAT_NAME)
        assert not _revoked(_squat_standing(session, SWITCH_MOUNT_PITCH_DEG + KNOCK_DEG))
        session.pipeline.set_exercise(PRESS_NAME)
        knocked = _press_set(session, SWITCH_MOUNT_PITCH_DEG + KNOCK_DEG)

        assert _revoked(knocked) == [pytest.approx(abs(KNOCK_DEG), abs=TRIP_DELTA_TOLERANCE_DEG)]
        assert not session.pipeline.pitch_certified

    def test_a_camera_knocked_during_another_exercise_trips_the_press_on_its_return(self, monkeypatch):
        session = _switching_session(monkeypatch)
        certified = _press_reference_sets(session, SWITCH_MOUNT_PITCH_DEG)
        assert len(counted_reps(certified)) == BiomechanicsConfig().faults.press.camera.tripwire_reference_sets
        assert not _revoked(certified)

        session.pipeline.set_exercise(RDL_NAME)
        knocked_rdl = _rdl_set(session, SWITCH_MOUNT_PITCH_DEG + KNOCK_DEG)
        # The RDL has no certification reading of its own yet: it cannot see the knock.
        assert len(counted_reps(knocked_rdl)) == 1 and not _revoked(knocked_rdl)
        assert session.pipeline.pitch_certified

        session.pipeline.set_exercise(PRESS_NAME)
        knocked_press = _press_set(session, SWITCH_MOUNT_PITCH_DEG + KNOCK_DEG)

        assert _revoked(knocked_press) == [pytest.approx(abs(KNOCK_DEG), abs=TRIP_DELTA_TOLERANCE_DEG)]
        assert not session.pipeline.pitch_certified

    def test_the_rdl_reference_survives_a_press_in_between(self, monkeypatch):
        session = _switching_session(monkeypatch)
        _press_set(session, SWITCH_MOUNT_PITCH_DEG)
        session.pipeline.set_exercise(RDL_NAME)
        for _ in range(BiomechanicsConfig().faults.rdl.camera.reference_sets):
            assert not _revoked(_rdl_set(session, SWITCH_MOUNT_PITCH_DEG))

        session.pipeline.set_exercise(PRESS_NAME)
        assert not _revoked(_press_set(session, SWITCH_MOUNT_PITCH_DEG))
        session.pipeline.set_exercise(RDL_NAME)
        knocked = _rdl_set(session, SWITCH_MOUNT_PITCH_DEG + KNOCK_DEG)

        assert _revoked(knocked) == [pytest.approx(abs(KNOCK_DEG), abs=TRIP_DELTA_TOLERANCE_DEG)]
        assert not session.pipeline.pitch_certified
        # Revoked for the session: the press that follows is judged uncertified too.
        session.pipeline.set_exercise(PRESS_NAME)
        assert not session.pipeline.pitch_certified


class TestLevelCameraFrame:
    def test_levelling_recovers_a_pitched_cameras_heights(self):
        world = world_squat_points(0.02)
        camera_view = _pitched_camera_view(world, CAMERA_PITCH_DEG)

        levelled = level_camera_frame(camera_view, CAMERA_PITCH_DEG)

        assert levelled == pytest.approx(world, abs=POSITION_TOLERANCE_M)

    def test_pitch_sign_matches_the_standing_pitch_estimate(self):
        world = world_squat_points(0.02)
        camera_view = _pitched_camera_view(world, CAMERA_PITCH_DEG)
        posture = _standing_pitch_deg(Skeleton3D.from_numpy(world))
        standing = _standing_pitch_deg(Skeleton3D.from_numpy(camera_view))
        levelled = _standing_pitch_deg(Skeleton3D.from_numpy(level_camera_frame(camera_view, CAMERA_PITCH_DEG)))

        assert standing - posture == pytest.approx(CAMERA_PITCH_DEG, abs=ANGLE_TOLERANCE_DEG)
        assert levelled == pytest.approx(posture, abs=ANGLE_TOLERANCE_DEG)

    def test_levelling_mixes_in_depth_noise_at_sin_pitch(self):
        """Anisotropic camera noise (σz = 3σxy): a levelled height's noise is
        sqrt(σy² cos²θ + σz² sin²θ), not σy."""
        rng = np.random.default_rng(RNG_SEED)
        noise = rng.normal(0.0, 1.0, size=(NOISE_SAMPLES, 3)) * np.array([SIGMA_XY_M, SIGMA_XY_M, SIGMA_Z_M])

        levelled_heights = level_camera_frame(noise, CAMERA_PITCH_DEG)[:, 1]

        pitch = math.radians(CAMERA_PITCH_DEG)
        expected = math.sqrt((SIGMA_XY_M * math.cos(pitch)) ** 2 + (SIGMA_Z_M * math.sin(pitch)) ** 2)
        assert float(np.std(levelled_heights)) == pytest.approx(expected, rel=NOISE_RATIO_TOLERANCE)
        assert float(np.std(levelled_heights)) > SIGMA_XY_M
