"""
Press start tiers and recorded-only verdicts (docs/exercises/ohp.md §4.1, §5.0;
SHARED-PLUMBING.md §4): cue_start_tier by capture mode and single-camera pitch
band, the pipeline's recorded-only stamping, approximate caps outside a validated
band, and a certified mount revoked by the delta tripwire on the rep that trips it.
"""

from __future__ import annotations

import math

import pytest

from biomechanics.analysis.press_features import PITCH_BAND_LEVEL, PITCH_BAND_UNKNOWN
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.observability import APPROXIMATE, OBSERVABLE, SINGLE_CAMERA, TRIANGULATED
from biomechanics.profiles.overhead_press import (
    RIG_START_TIERS,
    SINGLE_CAMERA_START_TIERS,
    OverheadPressProfile,
)
from biomechanics.utils.frame_context import FrameContext
from biomechanics.utils.types import FaultSeverity, Skeleton3D
from test_press_poses import (
    FPS_MONO,
    FPS_RIG,
    camera_pitched,
    counted_reps,
    faults_of,
    hands_down,
    hold,
    press_rep,
    press_session,
    run_session,
    world_press_points,
)

MILD_TILT_DEG = 5.0
MODERATE_TILT_DEG = 7.0
SEVERE_TILT_DEG = 10.0
MODERATE_DIP_M = 0.07
TILTED_MOUNT_DEG = 6.0
TRIPWIRE_CAMERA_MOVE_DEG = 4.0


def _context(capture_mode: str, certified: bool = False, pitch_deg: float = math.nan) -> FrameContext:
    return FrameContext(
        skeleton_3d=Skeleton3D.from_numpy(world_press_points()), angles=None, timestamp=0.0,
        phase="idle", in_rep=False, capture_mode=capture_mode, pitch_certified=certified,
        mount_pitch_deg=pitch_deg,
    )


def _press_config(**press) -> BiomechanicsConfig:
    config = BiomechanicsConfig()
    for name, value in press.items():
        setattr(config.faults.press, name, value)
    return config


def _mounted(pitch_deg: float | None, **press) -> BiomechanicsConfig:
    config = BiomechanicsConfig()
    if pitch_deg is not None:
        config.capture.mount_source = "inclinometer"
        config.capture.mount_pitch_deg = pitch_deg
    for name, value in press.items():
        setattr(config.faults.press, name, value)
    return config


def _uneven_set(fps: float, tilts: list[float], **rep) -> list[dict]:
    poses = hands_down(1.0, fps) + hold(1.5, fps)
    for tilt in tilts:
        poses += press_rep(fps, tilt_top_deg=tilt, **rep)
    return poses + hold(1.0, fps)


def _end_set(session) -> None:
    session.pipeline.reset_readiness_gate()
    session.pipeline.reset_rep_counter()


def _uneven(results: list) -> dict[int, object]:
    return {fault.rep_number: fault for fault in faults_of(results, "press_uneven")}


class TestCueStartTier:
    @pytest.mark.parametrize(("certified", "pitch_deg", "leg_drive", "uneven"), [
        (True, 0.0, FaultSeverity.MODERATE, FaultSeverity.SEVERE),
        (True, -2.5, FaultSeverity.MODERATE, FaultSeverity.SEVERE),
        (True, 6.0, FaultSeverity.SEVERE, FaultSeverity.SEVERE),
        (True, 15.0, FaultSeverity.SEVERE, FaultSeverity.SEVERE),
        (False, 0.0, FaultSeverity.SEVERE, FaultSeverity.SEVERE),
    ])
    def test_single_camera_pitch_bands(self, certified: bool, pitch_deg: float, leg_drive, uneven):
        profile, config = OverheadPressProfile(), _press_config()
        ctx = _context(SINGLE_CAMERA, certified, pitch_deg)
        assert profile.cue_start_tier("press_leg_drive", ctx, config) == leg_drive
        assert profile.cue_start_tier("press_uneven", ctx, config) == uneven

    def test_uneven_promotion_moves_only_the_level_band(self):
        profile, config = OverheadPressProfile(), _press_config(uneven_promoted=True)
        assert profile.cue_start_tier("press_uneven", _context(SINGLE_CAMERA, True, 1.0), config) == FaultSeverity.MODERATE
        assert profile.cue_start_tier("press_uneven", _context(SINGLE_CAMERA, True, 6.0), config) == FaultSeverity.SEVERE
        assert profile.cue_start_tier("press_uneven", _context(SINGLE_CAMERA), config) == FaultSeverity.SEVERE

    def test_the_tier_follows_the_config_each_call_receives(self):
        profile, promoted = OverheadPressProfile(), _press_config(uneven_promoted=True)
        level = _context(SINGLE_CAMERA, True, 1.0)
        # Building layers from another config leaves nothing behind in the profile.
        profile.create_fault_rules(_press_config())
        profile.create_rep_analyzer(_press_config())
        assert profile.cue_start_tier("press_uneven", level, promoted) == FaultSeverity.MODERATE
        assert profile.cue_start_tier("press_uneven", level, _press_config()) == FaultSeverity.SEVERE

    def test_rig_tiers_ignore_the_mount(self):
        profile, config = OverheadPressProfile(), _press_config()
        ctx = _context(TRIANGULATED)
        assert profile.cue_start_tier("press_uneven", ctx, config) == FaultSeverity.MODERATE
        assert profile.cue_start_tier("press_layback", ctx, config) == FaultSeverity.MILD
        assert profile.cue_start_tier("press_bar_forward", ctx, config) == FaultSeverity.MODERATE

    def test_grip_is_a_moderate_note_in_both_modes(self):
        profile, config = OverheadPressProfile(), _press_config()
        assert profile.cue_start_tier("press_grip", _context(SINGLE_CAMERA), config) == FaultSeverity.MODERATE
        assert profile.cue_start_tier("press_grip", _context(TRIANGULATED), config) == FaultSeverity.MODERATE

    def test_static_tiers_follow_the_capture_mode(self, monkeypatch):
        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "false")
        assert OverheadPressProfile().min_cue_tiers is SINGLE_CAMERA_START_TIERS
        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
        assert OverheadPressProfile().min_cue_tiers is RIG_START_TIERS
        with pytest.raises(TypeError):
            SINGLE_CAMERA_START_TIERS["press_uneven"] = FaultSeverity.MILD


class TestRecordedOnlyInThePipeline:
    def test_level_mount_default_uneven_starts_at_severe(self, monkeypatch):
        session = press_session(monkeypatch, config=_mounted(0.0))
        results = run_session(session, _uneven_set(FPS_MONO, [MILD_TILT_DEG, MODERATE_TILT_DEG, SEVERE_TILT_DEG]), FPS_MONO)
        reps = counted_reps(results)
        uneven = _uneven(results)
        assert uneven[1].severity == FaultSeverity.MILD and uneven[1].details["recorded_only"]
        assert uneven[2].severity == FaultSeverity.MODERATE and uneven[2].details["recorded_only"]
        assert uneven[3].severity == FaultSeverity.SEVERE and not uneven[3].details.get("recorded_only")
        assert uneven[3].details["observability"] == OBSERVABLE
        assert reps[0].is_clean and reps[1].is_clean and not reps[2].is_clean

    def test_promoted_level_mount_cues_a_moderate_tilt(self, monkeypatch):
        session = press_session(monkeypatch, config=_mounted(0.0, uneven_promoted=True))
        uneven = _uneven(run_session(session, _uneven_set(FPS_MONO, [MILD_TILT_DEG, MODERATE_TILT_DEG]), FPS_MONO))
        assert uneven[1].details["recorded_only"]
        assert not uneven[2].details.get("recorded_only")
        assert uneven[2].details["pitch_band"] == PITCH_BAND_LEVEL

    def test_uncertified_camera_is_recap_only(self, monkeypatch):
        session = press_session(monkeypatch, config=_mounted(None))
        uneven = _uneven(run_session(session, _uneven_set(FPS_MONO, [MODERATE_TILT_DEG, SEVERE_TILT_DEG]), FPS_MONO))
        assert uneven[1].details["recorded_only"] and uneven[1].details["observability"] == APPROXIMATE
        assert not uneven[2].details.get("recorded_only") and uneven[2].details["observability"] == APPROXIMATE
        assert uneven[2].details["pitch_band"] == PITCH_BAND_UNKNOWN

    @pytest.mark.parametrize(("validated", "observability"), [
        pytest.param(set(), APPROXIMATE, id="none"),
        pytest.param({"press_leg_drive"}, APPROXIMATE, id="leg-drive-only"),
        pytest.param({"press_uneven"}, OBSERVABLE, id="uneven"),
    ])
    def test_tilted_band_is_severe_once_that_fault_is_validated(
        self, monkeypatch, validated: set[str], observability: str,
    ):
        session = press_session(monkeypatch, config=_mounted(TILTED_MOUNT_DEG, tilted_band_validated_faults=validated))
        poses = _uneven_set(FPS_MONO, [SEVERE_TILT_DEG])
        results = run_session(session, poses, FPS_MONO, transform=lambda points: camera_pitched(points, TILTED_MOUNT_DEG))
        [fault] = faults_of(results, "press_uneven")
        assert fault.severity == FaultSeverity.SEVERE
        assert fault.details["observability"] == observability
        assert not fault.details.get("recorded_only")

    def test_rig_cues_uneven_from_moderate(self, monkeypatch):
        session = press_session(monkeypatch, multi_camera=True)
        uneven = _uneven(run_session(session, _uneven_set(FPS_RIG, [MILD_TILT_DEG, MODERATE_TILT_DEG]), FPS_RIG))
        assert uneven[1].details["recorded_only"]
        assert not uneven[2].details.get("recorded_only")
        assert uneven[2].details["observability"] == OBSERVABLE


class TestTripwireBetweenSets:
    def test_the_rep_that_trips_the_wire_is_judged_uncertified(self, monkeypatch):
        """At 30 fps the first knocked rep's rack hold completes the set's reading
        (at 12 fps it takes about two reps): its own verdicts are recorded-only."""
        config = _mounted(0.0, uneven_promoted=True)
        session = press_session(monkeypatch, config=config)
        profile = session.pipeline.profile
        static_tiers = dict(profile.min_cue_tiers)
        for _ in range(config.faults.press.camera.tripwire_reference_sets):
            level = hands_down(1.0, FPS_RIG) + hold(1.5, FPS_RIG) + press_rep(FPS_RIG, tilt_top_deg=MODERATE_TILT_DEG)
            [verdict] = faults_of(run_session(session, level + hold(0.5, FPS_RIG), FPS_RIG), "press_uneven")
            assert not verdict.details.get("recorded_only") and verdict.details["observability"] == OBSERVABLE
            _end_set(session)
        assert session.pipeline.pitch_certified

        moved = hands_down(1.0, FPS_RIG) + hold(1.5, FPS_RIG)
        for _ in range(2):
            moved += press_rep(FPS_RIG, tilt_top_deg=MODERATE_TILT_DEG, dip_m=MODERATE_DIP_M)
        later = run_session(
            session, moved + hold(1.0, FPS_RIG), FPS_RIG,
            transform=lambda points: camera_pitched(points, TRIPWIRE_CAMERA_MOVE_DEG),
        )

        first_rep_frame = next(index for index, result in enumerate(later) if result.rep_data is not None)
        revoked = [index for index, result in enumerate(later) if result.pitch_certification_revoked_deg]
        assert revoked == [first_rep_frame]
        assert not session.pipeline.pitch_certified
        reps = counted_reps(later)
        assert len(reps) == 2
        for rep in reps:
            verdicts = {fault.fault_type: fault for fault in rep.faults}
            for fault_type in ("press_uneven", "press_leg_drive"):
                assert verdicts[fault_type].severity == FaultSeverity.MODERATE
                assert verdicts[fault_type].details["recorded_only"]
                assert verdicts[fault_type].details["observability"] == APPROXIMATE
            assert rep.is_clean
        assert dict(profile.min_cue_tiers) == static_tiers
