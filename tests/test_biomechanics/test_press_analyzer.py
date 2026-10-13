"""
PressRepAnalyzer (docs/exercises/ohp.md §3, SHARED-PLUMBING.md §2): every
rejection reason, the first rep after the latch, rejected reps that leave no
verdict and no reference update, velocity loss reaching RepData, the rack
predicate ANDed into the readiness gate, counter baselines, and touch-and-go order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.analysis.press_features import REJECT_NO_LOCKOUT, REJECT_NO_RACK_START
from biomechanics.config import BiomechanicsConfig
from biomechanics.profiles.overhead_press import OverheadPressProfile
from biomechanics.utils.standing_gate import PROFILE_CHECK_FAILURE
from biomechanics.utils.types import JointAngles
from test_press_poses import (
    FPS_MONO,
    FPS_RIG,
    LATCH_FRAMES,
    RACK_BAR_HEIGHT_M,
    AnalyzerHarness,
    bare_hand_reach,
    camera_pitched,
    counted_reps,
    faults_of,
    hands_down,
    hold,
    lockout_bar_height,
    press_rep,
    press_session,
    run_session,
    wrists_clipped_above,
)

PRESS = BiomechanicsConfig().faults.press
MISSED_TOP_M = 0.30
SHORT_TOP_M = 0.18
CLIP_HIGH_M = 0.45
CLIP_LOW_M = 0.30
DEEP_KNEE_BEND_M = 0.5
SLOWING_PRESS_S = (0.6, 0.8, 1.3)
# Tripwire sessions: anisotropic keypoint noise (sigma_z = 3 sigma_xy), rack-hold
# lean drawn per rep, and a per-set posture offset (bar-in-hand repeatability).
TRIPWIRE_SETS = 5
TRIPWIRE_REPS_PER_SET = 5
TRIPWIRE_SIGMA_M = np.array([0.01, 0.01, 0.03])
NATURAL_LEAN_MAX_DEG = 3.0
POSTURE_SIGMA_DEG = 0.5
CLEAN_SESSION_SEEDS = range(24)
KNOCK_SESSION_SEEDS = range(30)
KNOCK_DEG = 4.0
TRIP_DELTA_TOLERANCE_DEG = 1.5
# A 4° knock sits just past the 3° tolerance: measured 93% over 60 sessions (the
# residual band runs to about 4.3°, like the RDL's 3.1-3.9°).
KNOCK_DETECTION_MIN_RATE = 0.9


def _tripwire_session(seed: int, knock_from_set: int | None = None, sets: int = TRIPWIRE_SETS) -> list[float | None]:
    """A certified level-mount session at 12 fps; the per-set tripwire result
    (the tripping delta, or None). The camera turns KNOCK_DEG from knock_from_set on."""
    rng = np.random.default_rng(seed)
    harness = AnalyzerHarness(pitch_certified=True, mount_pitch_deg=0.0)
    trips: list[float | None] = []
    for set_index in range(sets):
        poses = hands_down(1.0, FPS_MONO) + hold(1.5, FPS_MONO)
        for _ in range(TRIPWIRE_REPS_PER_SET):
            poses += press_rep(FPS_MONO, layback_deg=float(rng.uniform(0.0, NATURAL_LEAN_MAX_DEG)))
        posture_deg = float(rng.normal(0.0, POSTURE_SIGMA_DEG))
        camera_deg = KNOCK_DEG if knock_from_set is not None and set_index >= knock_from_set else 0.0
        harness.run(
            poses + hold(1.0, FPS_MONO),
            transform=lambda points: camera_pitched(points, camera_deg + posture_deg)
            + rng.normal(0.0, 1.0, points.shape) * TRIPWIRE_SIGMA_M,
        )
        trips.append(harness.analyzer.pitch_tripwire_delta())
        harness.reset_set()
    return trips


def _unrack(fps: float, hold_s: float = 1.5, **pose) -> list[dict]:
    return hands_down(1.0, fps) + hold(hold_s, fps, **pose)


class _CallLog:
    """Wraps an analyser's protocol methods and records their order."""

    def __init__(self, analyzer) -> None:
        self.calls: list[tuple] = []
        for name in ("on_rep_start", "finish_rep", "rejection_reason", "discard_rep"):
            original = getattr(analyzer, name)
            setattr(analyzer, name, self._wrap(name, original))

    def _wrap(self, name: str, original):
        def logged(*args, **kwargs):
            result = original(*args, **kwargs)
            detail = kwargs.get("piston", args[0] if args and name == "on_rep_start" else None)
            if name == "rejection_reason":
                detail = result
            elif name == "finish_rep":
                detail = args[0]
            self.calls.append((name, detail))
            return result
        return logged


class TestRejectionReasons:
    def test_missed_press_is_no_lockout_with_a_missed_event(self):
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO) + press_rep(FPS_MONO, top_m=MISSED_TOP_M) + hold(1.0, FPS_MONO)
        harness = AnalyzerHarness().run(poses)
        assert len(harness.counted) == 1
        [(reason, features)] = harness.rejections
        assert reason == REJECT_NO_LOCKOUT
        assert features.rom_cm >= PRESS.acceptance.missed_min_travel_cm
        [missed] = harness.rejected_faults
        assert missed.fault_type == "press_missed"
        assert missed.details["observability"] == "observable"
        assert missed.rep_number == 2

    def test_the_missed_event_floor_follows_the_config_each_call_receives(self):
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO) + press_rep(FPS_MONO, top_m=MISSED_TOP_M) + hold(1.0, FPS_MONO)
        [(reason, features)] = AnalyzerHarness().run(poses).rejections
        strict = BiomechanicsConfig()
        strict.faults.press.acceptance.missed_min_travel_cm = features.rom_cm + 1.0
        profile = OverheadPressProfile()
        assert profile.on_rep_rejected(reason, features, JointAngles(), strict) == []
        assert len(profile.on_rep_rejected(reason, features, JointAngles(), BiomechanicsConfig())) == 1

    def test_short_attempt_under_the_counter_minimum_is_silent(self):
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO, top_m=SHORT_TOP_M) + hold(1.0, FPS_MONO)
        harness = AnalyzerHarness().run(poses)
        assert not harness.features and not harness.rejected_faults

    def test_bare_hand_reach_is_no_rack_start_and_emits_nothing(self):
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO) + hands_down(1.0, FPS_MONO) + bare_hand_reach(FPS_MONO)
        poses += hands_down(1.5, FPS_MONO)
        harness = AnalyzerHarness().run(poses)
        assert len(harness.counted) == 1
        assert [reason for reason, _ in harness.rejections] == [REJECT_NO_RACK_START]
        assert harness.rejected_faults == []

    def test_clipped_wrists_high_up_count_with_lockout_unmeasured(self):
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO) + hold(1.0, FPS_MONO)
        harness = AnalyzerHarness().run(poses, confidence=wrists_clipped_above(CLIP_HIGH_M))
        [features] = harness.counted
        assert features.wrists_clipped
        assert features.clip_height_ratio >= PRESS.acceptance.clipped_wrists_min_ratio
        assert features.top_height_ratio < PRESS.acceptance.top_height_min_ratio
        assert math.isnan(features.lockout_elbow_l_deg) and math.isnan(features.lockout_elbow_r_deg)
        assert math.isnan(features.concentric_velocity_mps)

    def test_clipped_wrists_low_down_are_not_waived(self):
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO) + hold(1.0, FPS_MONO)
        harness = AnalyzerHarness().run(poses, confidence=wrists_clipped_above(CLIP_LOW_M))
        assert not harness.counted
        assert [reason for reason, _ in harness.rejections] == [REJECT_NO_LOCKOUT]


class TestFirstRep:
    def test_first_rep_right_after_the_latch_counts_with_no_idle_frames(self):
        poses = hands_down(1.0, FPS_MONO) + hold(LATCH_FRAMES / FPS_MONO, FPS_MONO)
        poses += press_rep(FPS_MONO, hold_s=0.0) + hold(1.0, FPS_MONO)
        harness = AnalyzerHarness().run(poses)
        [features] = harness.counted
        assert features.first_rep_after_latch
        assert features.rack_start

    def test_counter_baseline_is_the_rack_from_the_latch_to_the_set_end(self):
        harness = AnalyzerHarness()
        assert harness.analyzer.counter_baseline() is None
        harness.run(_unrack(FPS_MONO))
        assert harness.analyzer.counter_baseline() == pytest.approx(100.0 * RACK_BAR_HEIGHT_M)
        harness.analyzer.reset_set()
        assert harness.analyzer.counter_baseline() is None


class TestRejectedRepsInThePipeline:
    def test_rejected_rep_gets_no_verdict_and_no_reference_update(self, monkeypatch):
        session = press_session(monkeypatch)
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO)
        poses += press_rep(FPS_MONO, top_m=MISSED_TOP_M, tilt_top_deg=12.0, dip_m=0.1) + press_rep(FPS_MONO)
        results = run_session(session, poses + hold(1.0, FPS_MONO), FPS_MONO)
        assert [rep.rep_number for rep in counted_reps(results)] == [1, 2]
        faults = faults_of(results)
        assert [fault.fault_type for fault in faults] == ["press_missed"]
        assert "recorded_only" not in faults[0].details
        reference = session.pipeline._rule_engine.reference
        assert len(reference._set_velocities) == 2
        assert session.pipeline.rep_count == 2

    def test_press_missed_has_no_cue(self, monkeypatch):
        assert "press_missed" not in OverheadPressProfile().get_fault_to_cue_map()
        assert "press_grip" not in OverheadPressProfile().get_fault_to_cue_map()

    def test_velocity_loss_reaches_rep_data(self, monkeypatch):
        session = press_session(monkeypatch, multi_camera=True)
        poses = _unrack(FPS_RIG)
        for press_s in SLOWING_PRESS_S:
            poses += press_rep(FPS_RIG, press_s=press_s)
        reps = counted_reps(run_session(session, poses + hold(1.0, FPS_RIG), FPS_RIG))
        assert len(reps) == 3
        assert math.isnan(reps[0].features["velocity_loss_pct"])
        assert reps[2].features["velocity_loss_pct"] > PRESS.velocity_loss_pct.moderate
        assert reps[2].ascent_time == pytest.approx(reps[2].features["concentric_time_s"])
        assert reps[2].descent_time == pytest.approx(reps[2].features["eccentric_time_s"])
        assert reps[2].features["highlight_score"] == pytest.approx(reps[2].features["concentric_velocity_mps"])
        [loss] = [fault for fault in reps[2].faults if fault.fault_type == "press_velocity_loss"]
        assert loss.details["observability"] == "approximate"

    def test_reset_rep_counter_reseeds_the_rack_baseline(self, monkeypatch):
        session = press_session(monkeypatch)
        run_session(session, _unrack(FPS_MONO) + press_rep(FPS_MONO), FPS_MONO)
        session.pipeline.reset_rep_counter()
        assert session.pipeline.rep_counter._standing_baseline == pytest.approx(100.0 * RACK_BAR_HEIGHT_M, abs=0.5)


class TestReadinessGate:
    def test_rack_pose_latches_and_the_readiness_context_has_no_angles(self, monkeypatch):
        session = press_session(monkeypatch)
        analyzer = session.pipeline._rep_analyzer
        contexts = []
        original = analyzer.is_ready
        analyzer.is_ready = lambda ctx: contexts.append(ctx) or original(ctx)
        run_session(session, _unrack(FPS_MONO), FPS_MONO)
        assert session.pipeline.is_ready
        assert contexts and all(ctx.angles is None for ctx in contexts)

    def test_hands_at_the_sides_never_latch(self, monkeypatch):
        session = press_session(monkeypatch)
        run_session(session, hands_down(2.0, FPS_MONO), FPS_MONO)
        assert not session.pipeline.is_ready
        assert session.pipeline._readiness_gate.last_failure == PROFILE_CHECK_FAILURE

    def test_rack_pose_with_bent_knees_fails_the_gates_own_checks(self, monkeypatch):
        session = press_session(monkeypatch)
        run_session(session, hands_down(1.0, FPS_MONO) + hold(2.0, FPS_MONO, hip_drop_m=DEEP_KNEE_BEND_M), FPS_MONO)
        assert not session.pipeline.is_ready
        assert session.pipeline._readiness_gate.last_failure not in (None, PROFILE_CHECK_FAILURE)


def _v_turnarounds(fps: float, reps: int, low_progress: float, half_s: float, **pose) -> list[dict]:
    """Lockout -> straight-line lower to low_progress -> straight-line press, reps times:
    a sharp turnaround the counter itself splits as a touch-and-go."""
    top = lockout_bar_height()
    frames = int(round(half_s * fps))
    heights: list[float] = []
    for _ in range(reps):
        heights += [top - (top - RACK_BAR_HEIGHT_M) * (1.0 - low_progress) * (i + 1) / frames for i in range(frames)]
        heights += [heights[-1] + (top - heights[-1]) * (i + 1) / frames for i in range(frames)]
    return [dict(pose, bar_height_m=height) for height in heights]


class TestTouchAndGo:
    def test_ended_rep_finishes_before_the_piston_rep_starts(self, monkeypatch):
        session = press_session(monkeypatch, multi_camera=True)
        log = _CallLog(session.pipeline._rep_analyzer)
        grip = {"grip_tilt_deg": 16.0}
        poses = _unrack(FPS_RIG, **grip) + press_rep(FPS_RIG, lower_s=0.0, **grip)
        poses += _v_turnarounds(FPS_RIG, 2, 0.2, 0.5, **grip)
        poses += press_rep(FPS_RIG, hold_s=0.0, press_s=0.0, start_progress=1.0, **grip) + hold(1.0, FPS_RIG, **grip)
        results = run_session(session, poses, FPS_RIG)
        reps = counted_reps(results)
        assert [rep.rep_number for rep in reps] == [1, 2, 3]
        assert [rep.features["piston"] for rep in reps] == [False, True, True]
        piston_starts = [i for i, call in enumerate(log.calls) if call == ("on_rep_start", True)]
        assert len(piston_starts) == 2
        for start in piston_starts:
            assert log.calls[start - 2][0] == "finish_rep"
            assert log.calls[start - 1] == ("rejection_reason", None)
        for rep in reps[1:]:
            assert rep.features["grip_tilt_deg"] is None or math.isnan(rep.features["grip_tilt_deg"])
        assert [fault.rep_number for fault in faults_of(results, "press_grip")] == [1]

    def test_turnaround_the_counter_settles_on_is_still_a_continuation(self, monkeypatch):
        """At 12 fps the counter ends a chin touch-and-go where the bar stops and opens
        the next rep a frame later; the press still counts it as a continuation."""
        session = press_session(monkeypatch)
        poses = _unrack(FPS_MONO) + press_rep(FPS_MONO, end_progress=0.2)
        poses += press_rep(FPS_MONO, hold_s=0.0, start_progress=0.2, end_progress=0.2)
        poses += press_rep(FPS_MONO, hold_s=0.0, start_progress=0.2) + hold(1.0, FPS_MONO)
        results = run_session(session, poses, FPS_MONO)
        reps = counted_reps(results)
        assert [rep.rep_number for rep in reps] == [1, 2, 3]
        assert [rep.features["piston"] for rep in reps] == [False, True, True]
        assert all(rep.features["rack_start"] for rep in reps)
        assert not faults_of(results, "press_missed")


class TestPitchTripwire:
    def test_noisy_clean_sessions_never_trip(self):
        for seed in CLEAN_SESSION_SEEDS:
            assert _tripwire_session(seed) == [None] * TRIPWIRE_SETS, seed

    def test_a_4_degree_knock_between_sets_trips_the_next_set(self):
        knock_from = PRESS.camera.tripwire_reference_sets
        tripped = 0
        for seed in KNOCK_SESSION_SEEDS:
            trips = _tripwire_session(seed, knock_from_set=knock_from, sets=knock_from + 1)
            assert trips[:knock_from] == [None] * knock_from, seed
            if trips[knock_from] is not None:
                assert trips[knock_from] == pytest.approx(KNOCK_DEG, abs=TRIP_DELTA_TOLERANCE_DEG), seed
                tripped += 1
        assert tripped >= KNOCK_DETECTION_MIN_RATE * len(KNOCK_SESSION_SEEDS)

    def test_the_reference_sets_are_never_checked(self):
        reference_sets = PRESS.camera.tripwire_reference_sets
        trips = _tripwire_session(0, knock_from_set=1, sets=reference_sets)
        assert trips == [None] * reference_sets
