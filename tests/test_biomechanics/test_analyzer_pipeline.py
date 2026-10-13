"""
Tests for the pipeline's rep-analyser path (docs/exercises/SHARED-PLUMBING.md §1-§4):
a profile with a RepAnalyzer owns the rep signal and per-rep features, the
squat blocks are skipped, rejections run in the spec's order with a
pipeline-owned streak, readiness and counter baselines come from the analyser,
and verdicts below a fault's start tier are stamped recorded-only (a rejected
rep's too, before its shallow_rep cue is chosen, and at the uncertified tier
from the frame a pitch tripwire fires on).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import BaseModel

from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.faults.fault_types import FaultRule, FaultType, RepFaultRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.squat import SquatProfile
from biomechanics.utils.frame_context import GRAVITY_BODY, FrameContext
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.standing_gate import PROFILE_CHECK_FAILURE
from biomechanics.utils.types import LEG_KEYPOINTS, CocoKeypoints as CK, FaultEvent, FaultSeverity
from conftest import SYNTHETIC_FPS, SyntheticSession, squat_depth_profile, world_squat_points

READINESS_WARM_UP_FRAMES = 10
LATCH_FRAMES = 5
ANALYSER_DEPTH_DEG = 42.0
ECCENTRIC_S = 1.25
CONCENTRIC_S = 0.75
ANNOTATED_VELOCITY_LOSS_PCT = 12.5
SEEDED_BASELINE_CM = -88.0
PISTON_TOP_DEPTH = 0.3
PISTON_TURN_S = 0.5
PISTON_REPS = 3
SHALLOW_DIP_DEPTH = 0.3
TRIPWIRE_DELTA_DEG = 4.0
TRIPWIRE_AFTER_OBSERVATIONS = 3
# Far enough after the first analysed frame that the rule engine's 0.5 s
# de-duplication lets a verdict through on both.
TRIPWIRE_WITH_VERDICT_OBSERVATIONS = 20
MOUNT_PITCH_DEG = "2.0"
CUE_AFTER_STREAK = 2
MAX_CUES_PER_SET = 2
SHALLOW_CUE = "rdl_lower"


class _Features(BaseModel):
    exercise: str = "fake_lift"
    rep_number: int
    max_depth_angle: float = math.nan
    concentric_velocity_mps: float = math.nan
    velocity_loss_pct: float = math.nan
    annotated: bool = False
    eccentric_time_s: float = math.nan
    concentric_time_s: float = math.nan


class _FakeAnalyzer:
    """Squat-shaped signal (hip above ankle, cm); logs every call into a shared list."""

    def __init__(self, calls: list, reasons: list[str | None] | None = None) -> None:
        self.rep_signal = math.nan
        self.calls = calls
        self.reasons = list(reasons or [])
        self.observed: list[FrameContext] = []

    def observe(self, ctx: FrameContext) -> None:
        self.observed.append(ctx)
        points = ctx.skeleton_3d.to_numpy()
        hip_y = (points[CK.LEFT_HIP][1] + points[CK.RIGHT_HIP][1]) / 2.0
        ankle_y = (points[CK.LEFT_ANKLE][1] + points[CK.RIGHT_ANKLE][1]) / 2.0
        self.rep_signal = (hip_y - ankle_y) * 100.0

    def on_rep_start(self, piston: bool) -> None:
        self.calls.append(("on_rep_start", piston))

    def finish_rep(self, rep_number: int) -> _Features:
        self.calls.append(("finish_rep", rep_number))
        return _Features(
            rep_number=rep_number, max_depth_angle=ANALYSER_DEPTH_DEG,
            eccentric_time_s=ECCENTRIC_S, concentric_time_s=CONCENTRIC_S,
        )

    def rejection_reason(self, features: _Features) -> str | None:
        reason = self.reasons.pop(0) if self.reasons else None
        self.calls.append(("rejection_reason", reason))
        return reason

    def discard_rep(self) -> None:
        self.calls.append(("discard_rep",))

    def reset_set(self) -> None:
        self.calls.append(("reset_set",))


class _ReadyAnalyzer(_FakeAnalyzer):
    def __init__(self, calls: list, ready: bool) -> None:
        super().__init__(calls)
        self.ready = ready
        self.readiness_contexts: list[FrameContext] = []

    def is_ready(self, ctx: FrameContext) -> bool:
        self.readiness_contexts.append(ctx)
        return self.ready


class _SeededAnalyzer(_FakeAnalyzer):
    def counter_baseline(self) -> float | None:
        return SEEDED_BASELINE_CM


class _TrippingAnalyzer(_FakeAnalyzer):
    def __init__(self, calls: list, trip_after: int = TRIPWIRE_AFTER_OBSERVATIONS) -> None:
        super().__init__(calls)
        self.trip_after = trip_after

    def pitch_tripwire_delta(self) -> float | None:
        return TRIPWIRE_DELTA_DEG if len(self.observed) >= self.trip_after else None


class _RecordingReference(SessionReference):
    def __init__(self) -> None:
        super().__init__()
        self.updated: list[int] = []

    def annotate(self, features) -> None:
        features.annotated = True
        features.velocity_loss_pct = ANNOTATED_VELOCITY_LOSS_PCT

    def update(self, features) -> None:
        self.updated.append(features.rep_number)


class _VerdictRule(RepFaultRule):
    """One verdict per rep at a fixed severity; logs discards."""

    def __init__(self, fault_type: FaultType, severity: FaultSeverity | None, calls: list | None = None) -> None:
        self._type = fault_type
        self._severity = severity
        self._calls = calls if calls is not None else []

    @property
    def fault_type(self) -> FaultType:
        return self._type

    def judge_rep(self, features, reference, angles) -> FaultEvent | None:
        if self._severity is None:
            return None
        return self._create_fault_event(
            self._severity, 1.0, "verdict", angles, features.rep_number, details={"side": None},
        )

    def discard_rep(self) -> None:
        self._calls.append(("rule_discard",))


class _ObservationRule(FaultRule):
    """A moderate frame verdict on chosen analyser observations (1 = the first analysed frame)."""

    def __init__(self, fault_type: FaultType, analyzer: _FakeAnalyzer, observations: frozenset[int]) -> None:
        self._type = fault_type
        self._analyzer = analyzer
        self._observations = observations

    @property
    def fault_type(self) -> FaultType:
        return self._type

    def evaluate(self, angles, history, in_rep: bool = False, rep_number: int = 0) -> FaultEvent | None:
        if len(self._analyzer.observed) not in self._observations:
            return None
        return self._create_fault_event(FaultSeverity.MODERATE, 1.5, "verdict", angles, rep_number)


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


def _fault(fault_type: str, rep_number: int, shallow: bool = False) -> FaultEvent:
    return FaultEvent(
        fault_type=fault_type, severity=FaultSeverity.MODERATE, severity_score=1.5,
        message="rejected", rep_number=rep_number, details={"shallow_rep": shallow} if shallow else {},
    )


class _AnalyserProfile(ExerciseProfile):
    name = "fake_lift"

    def __init__(
        self,
        analyzer: _FakeAnalyzer,
        calls: list,
        rules: list | None = None,
        rejected_types: tuple[str, ...] = ("rdl_partial",),
    ) -> None:
        self.analyzer = analyzer
        self.calls = calls
        self.rules = rules or []
        self.rejected_types = rejected_types
        self.reference = _RecordingReference()
        self.cue_args: list[tuple[int, int]] = []
        self.session: SyntheticSession | None = None

    def create_fault_rules(self, config):
        return list(self.rules)

    def create_rep_analyzer(self, config):
        return self.analyzer

    def create_session_reference(self, config):
        return self.reference

    def on_rep_rejected(self, reason, features, angles, config):
        rep_count = self.session.pipeline.rep_count if self.session is not None else None
        self.calls.append(("on_rep_rejected", reason, rep_count))
        return [
            _fault(fault_type, features.rep_number, shallow=(index == len(self.rejected_types) - 1))
            for index, fault_type in enumerate(self.rejected_types)
        ]

    def shallow_rep_cue(self, fault, streak, cued_this_set, config):
        self.cue_args.append((streak, cued_this_set))
        if streak >= CUE_AFTER_STREAK and cued_this_set < MAX_CUES_PER_SET:
            return SHALLOW_CUE
        return None


class _AlwaysCueProfile(_AnalyserProfile):
    def shallow_rep_cue(self, fault, streak, cued_this_set, config):
        self.cue_args.append((streak, cued_this_set))
        return SHALLOW_CUE


class _CertifiedTierProfile(_AnalyserProfile):
    """Cues from mild only while the mount is certified, severe otherwise."""

    def cue_start_tier(self, fault_type, ctx, config):
        return FaultSeverity.MILD if ctx.pitch_certified else FaultSeverity.SEVERE


def _session(monkeypatch, profile: ExerciseProfile, multi_camera: bool = True) -> SyntheticSession:
    session = SyntheticSession(monkeypatch, multi_camera=multi_camera)
    session.pipeline._profile = profile
    session.pipeline._build_exercise_layers()
    if isinstance(profile, _AnalyserProfile):
        profile.session = session
    return session


def _reps(count: int) -> list[float]:
    depths = [0.0] * READINESS_WARM_UP_FRAMES
    for _ in range(count):
        depths += squat_depth_profile()
    return depths


def _cosine_move(start: float, end: float, seconds: float) -> list[float]:
    frames = int(round(seconds * SYNTHETIC_FPS))
    return [start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * (i + 1) / frames)) for i in range(frames)]


def _linear_move(start: float, end: float, seconds: float) -> list[float]:
    frames = int(round(seconds * SYNTHETIC_FPS))
    return [start + (end - start) * (i + 1) / frames for i in range(frames)]


def _counted(results) -> list:
    return [result.rep_data for result in results if result.rep_data is not None]


class TestAnalyserOwnsTheRep:
    def test_reps_come_from_the_analyser_signal_and_carry_its_features(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))

        reps = _counted(session.run(_reps(2)))

        assert [rep.rep_number for rep in reps] == [1, 2]
        for rep in reps:
            assert rep.features["exercise"] == "fake_lift"
            assert rep.max_depth_angle == pytest.approx(ANALYSER_DEPTH_DEG)
            assert rep.depth_target_met
            assert rep.descent_time == pytest.approx(ECCENTRIC_S)
            assert rep.ascent_time == pytest.approx(CONCENTRIC_S)

    def test_squat_blocks_are_skipped(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))

        session.run(_reps(1))

        assert session.pipeline.consume_rep_trajectory() == []
        assert session.pipeline.consume_standing_frame() is None
        assert session.pipeline.consume_bottom_frame() == (None, None)

    def test_features_are_dumped_after_the_engine_annotates_them(self, monkeypatch):
        calls: list = []
        profile = _AnalyserProfile(_FakeAnalyzer(calls), calls)
        session = _session(monkeypatch, profile)

        reps = _counted(session.run(_reps(2)))

        assert all(rep.features["annotated"] for rep in reps)
        assert all(rep.features["velocity_loss_pct"] == pytest.approx(ANNOTATED_VELOCITY_LOSS_PCT) for rep in reps)
        assert profile.reference.updated == [1, 2]

    def test_observe_sees_only_measured_frames_after_the_latch(self, monkeypatch):
        calls: list = []
        analyzer = _FakeAnalyzer(calls)
        session = _session(monkeypatch, _AnalyserProfile(analyzer, calls))
        depths = _reps(1)
        dropout = frozenset({READINESS_WARM_UP_FRAMES + 40, READINESS_WARM_UP_FRAMES + 41})

        results = session.run(depths, dropped=dropout)

        analysed = [result for result in results if result.skeleton_3d is not None]
        measured = {result.skeleton_3d.timestamp for result in analysed if not result.extrapolated_keypoints}
        assert 0 < len(analyzer.observed) == len(measured) < len(analysed)
        first_ready = next(index for index, result in enumerate(results) if result.skeleton_3d is not None)
        assert first_ready >= LATCH_FRAMES - 1
        assert [ctx.timestamp for ctx in analyzer.observed] == sorted(measured)

    def test_a_frame_the_kalman_warm_up_repeats_is_observed_once(self, monkeypatch):
        calls: list = []
        analyzer = _FakeAnalyzer(calls)
        session = _session(monkeypatch, _AnalyserProfile(analyzer, calls))

        results = session.run(_reps(1))

        analysed = [result.skeleton_3d.timestamp for result in results if result.skeleton_3d is not None]
        assert len(analysed) > len(set(analysed))
        observed = [ctx.timestamp for ctx in analyzer.observed]
        assert all(later > earlier for earlier, later in zip(observed, observed[1:]))
        assert observed[0] == analysed[0]
        assert [rep.rep_number for rep in _counted(results)] == [1]

    def test_frame_context_carries_the_rig_frame(self, monkeypatch):
        calls: list = []
        analyzer = _FakeAnalyzer(calls)
        session = _session(monkeypatch, _AnalyserProfile(analyzer, calls))

        session.run([0.0] * (READINESS_WARM_UP_FRAMES * 2))

        ctx = analyzer.observed[-1]
        assert ctx.angles is not None
        assert ctx.capture_mode == "triangulated"
        assert ctx.gravity_source == GRAVITY_BODY
        assert np.allclose(ctx.up, WORLD_UP)
        assert ctx.skeleton_world is not None
        assert ctx.raw_confidences is not None and len(ctx.raw_confidences) == len(ctx.skeleton_3d.keypoints)
        assert not ctx.pitch_certified
        assert math.isnan(ctx.camera_pitch_deg)

    def test_get_rep_signal_is_never_called_on_the_analyser_path(self, monkeypatch):
        calls: list = []
        profile = _AnalyserProfile(_FakeAnalyzer(calls), calls)
        session = _session(monkeypatch, profile)
        with pytest.raises(NotImplementedError):
            profile.get_rep_signal(None)

        assert len(_counted(session.run(_reps(1)))) == 1


class TestRepBoundaries:
    def test_each_rep_starts_then_finishes(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))

        session.run(_reps(2))

        assert calls == [
            ("on_rep_start", False), ("finish_rep", 1), ("rejection_reason", None),
            ("on_rep_start", False), ("finish_rep", 2), ("rejection_reason", None),
        ]

    def test_touch_and_go_finishes_the_ended_rep_before_starting_the_next(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))
        depths = [0.0] * (READINESS_WARM_UP_FRAMES + int(SYNTHETIC_FPS)) + _cosine_move(0.0, 1.0, 1.0)
        # Sharp turnarounds (no easing): the rep cannot settle at the top first.
        for _ in range(PISTON_REPS - 1):
            depths += _linear_move(1.0, PISTON_TOP_DEPTH, PISTON_TURN_S) + _linear_move(PISTON_TOP_DEPTH, 1.0, 1.0)
        depths += _cosine_move(1.0, 0.0, 1.0) + [0.0] * int(SYNTHETIC_FPS)

        reps = _counted(session.run(depths))

        assert [rep.rep_number for rep in reps] == list(range(1, PISTON_REPS + 1))
        piston_starts = [index for index, call in enumerate(calls) if call == ("on_rep_start", True)]
        assert len(piston_starts) == PISTON_REPS - 1
        for index in piston_starts:
            assert calls[index - 2][0] == "finish_rep"
            assert calls[index - 1][0] == "rejection_reason"

    def test_a_dip_too_shallow_to_count_is_discarded(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))
        depths = [0.0] * READINESS_WARM_UP_FRAMES + [depth * SHALLOW_DIP_DEPTH for depth in squat_depth_profile()]

        session.run(depths + [0.0] * int(SYNTHETIC_FPS))

        assert ("on_rep_start", False) in calls
        assert ("discard_rep",) in calls
        assert not any(call[0] == "finish_rep" for call in calls)

    def test_reset_readiness_gate_resets_the_analysers_set(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))
        session.pipeline.reset_readiness_gate()
        assert calls == [("reset_set",)]

    def test_set_exercise_builds_a_fresh_analyser_and_session_reference(self, monkeypatch):
        from biomechanics import pipeline as pipeline_module

        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))
        profiles: list[_AnalyserProfile] = []

        def _fresh_profile(exercise_name: str) -> _AnalyserProfile:
            profiles.append(_AnalyserProfile(_FakeAnalyzer(calls), calls))
            return profiles[-1]

        monkeypatch.setattr(pipeline_module, "get_profile", _fresh_profile)
        session.pipeline.set_exercise("Fake Lift")
        first_analyzer, first_reference = session.pipeline._rep_analyzer, session.pipeline._rule_engine.reference
        session.pipeline.set_exercise("Fake Lift")

        assert first_analyzer is profiles[0].analyzer and first_reference is profiles[0].reference
        assert session.pipeline._rep_analyzer is profiles[1].analyzer is not first_analyzer
        assert session.pipeline._rule_engine.reference is profiles[1].reference


class TestRejection:
    def test_rejected_rep_is_uncounted_before_any_verdict_or_reference_update(self, monkeypatch):
        calls: list = []
        rule = _VerdictRule(FaultType.PRESS_UNEVEN, FaultSeverity.MODERATE, calls)
        profile = _AnalyserProfile(_FakeAnalyzer(calls, reasons=["partial"]), calls, rules=[rule])
        session = _session(monkeypatch, profile)

        results = session.run(_reps(2))

        reps = _counted(results)
        assert [rep.rep_number for rep in reps] == [1]
        assert profile.reference.updated == [1]
        rejection = calls.index(("rejection_reason", "partial"))
        assert calls[rejection + 1:rejection + 4] == [
            ("rule_discard",), ("discard_rep",), ("on_rep_rejected", "partial", 0),
        ]
        verdicts = [fault for result in results for fault in result.faults if fault.fault_type == "press_uneven"]
        assert len(verdicts) == 1

    def test_rejected_faults_pass_the_observability_gate(self, monkeypatch):
        calls: list = []
        profile = _AnalyserProfile(
            _FakeAnalyzer(calls, reasons=["no_lockout"]), calls, rejected_types=("press_layback", "rdl_partial"),
        )
        session = _session(monkeypatch, profile)
        session.pipeline._rule_engine.capture_mode = "single_camera"

        results = session.run(_reps(1))

        rejected = [result for result in results if result.analyzer_shallow_rep]
        assert len(rejected) == 1
        assert [fault.fault_type for fault in rejected[0].faults] == ["rdl_partial"]
        assert rejected[0].faults[0].details["observability"] == "observable"
        assert rejected[0].shallow_rep_class is None

    def test_streak_counts_rejections_per_reason_and_a_counted_rep_resets_it(self, monkeypatch):
        calls: list = []
        reasons = ["partial", "partial", "ignore", "partial", None, "partial"]
        profile = _AnalyserProfile(_FakeAnalyzer(calls, reasons=reasons), calls)
        session = _session(monkeypatch, profile)

        results = session.run(_reps(len(reasons)))

        assert profile.cue_args == [(1, 0), (2, 0), (3, 1), (1, 2)]
        shallow = [result for result in results if result.analyzer_shallow_rep]
        assert [result.shallow_rep_cue for result in shallow] == [None, SHALLOW_CUE, SHALLOW_CUE, None]
        assert len(_counted(results)) == 1

    def test_ignore_emits_nothing(self, monkeypatch):
        calls: list = []
        profile = _AnalyserProfile(_FakeAnalyzer(calls, reasons=["ignore"]), calls)
        session = _session(monkeypatch, profile)

        results = session.run(_reps(1))

        assert ("on_rep_rejected", "ignore", 0) in calls
        assert not any(result.faults or result.analyzer_shallow_rep for result in results)
        assert profile.cue_args == []

    def test_another_reason_leaves_the_streak_unchanged(self, monkeypatch):
        calls: list = []
        profile = _AnalyserProfile(_FakeAnalyzer(calls, reasons=["partial", "no_lockout", "partial"]), calls)
        session = _session(monkeypatch, profile)

        session.run(_reps(3))

        assert [streak for streak, _ in profile.cue_args] == [1, 1, 2]

    def test_set_reset_clears_the_streak_and_the_cue_count(self, monkeypatch):
        calls: list = []
        profile = _AnalyserProfile(_FakeAnalyzer(calls, reasons=["partial", "partial", "partial"]), calls)
        session = _session(monkeypatch, profile)

        session.run(_reps(2))
        session.pipeline.reset_readiness_gate()
        session.run(_reps(1))

        assert profile.cue_args == [(1, 0), (2, 0), (1, 0)]


class TestReadiness:
    def test_analyser_readiness_is_anded_with_the_standing_gate(self, monkeypatch):
        calls: list = []
        analyzer = _ReadyAnalyzer(calls, ready=False)
        session = _session(monkeypatch, _AnalyserProfile(analyzer, calls))

        session.run([0.0] * READINESS_WARM_UP_FRAMES * 3)

        assert not session.pipeline.is_ready
        assert session.pipeline._readiness_gate.last_failure == PROFILE_CHECK_FAILURE
        ctx = analyzer.readiness_contexts[-1]
        assert ctx.angles is None
        assert ctx.raw_confidences is not None
        hip_mid = (ctx.skeleton_3d.to_numpy()[CK.LEFT_HIP] + ctx.skeleton_3d.to_numpy()[CK.RIGHT_HIP]) / 2.0
        assert np.allclose(hip_mid, 0.0)

        analyzer.ready = True
        results = session.run([0.0] * LATCH_FRAMES)
        assert session.pipeline.is_ready
        assert not results[LATCH_FRAMES - 2].skeleton_3d

    def test_profile_readiness_check_gates_a_profile_without_an_analyser(self, monkeypatch):
        contexts: list[FrameContext] = []

        class _RackedSquat(SquatProfile):
            def get_readiness_check(self):
                def _check(ctx: FrameContext) -> bool:
                    contexts.append(ctx)
                    return False
                return _check

        session = _session(monkeypatch, _RackedSquat())
        session.run([0.0] * READINESS_WARM_UP_FRAMES * 2)

        assert not session.pipeline.is_ready
        assert contexts and all(ctx.angles is None for ctx in contexts)

    def test_counter_baseline_seeds_the_counter_at_the_latch_and_after_every_reset(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_SeededAnalyzer(calls), calls))
        seeds: list[float] = []
        counter = session.pipeline.rep_counter
        original_seed = counter.seed_baseline
        monkeypatch.setattr(counter, "seed_baseline", lambda value: (seeds.append(value), original_seed(value)))

        session.run([0.0] * READINESS_WARM_UP_FRAMES)
        assert seeds == [SEEDED_BASELINE_CM]
        session.pipeline.reset_rep_counter()
        assert seeds == [SEEDED_BASELINE_CM, SEEDED_BASELINE_CM]
        assert counter._standing_baseline == pytest.approx(SEEDED_BASELINE_CM)

    def test_without_counter_baseline_a_reset_counter_finds_its_own(self, monkeypatch):
        calls: list = []
        session = _session(monkeypatch, _AnalyserProfile(_FakeAnalyzer(calls), calls))
        session.run([0.0] * READINESS_WARM_UP_FRAMES)
        session.pipeline.reset_rep_counter()
        assert session.pipeline.rep_counter._standing_baseline is None


class TestRecordedOnly:
    def test_verdicts_below_the_start_tier_are_recorded_only_and_keep_the_rep_clean(self, monkeypatch):
        calls: list = []
        rules = [_VerdictRule(FaultType.PRESS_UNEVEN, FaultSeverity.MILD)]
        profile = _AnalyserProfile(_FakeAnalyzer(calls), calls, rules=rules)
        profile.min_cue_tiers = {"press_uneven": FaultSeverity.MODERATE}
        session = _session(monkeypatch, profile)

        rep = _counted(session.run(_reps(1)))[0]

        assert [fault.details.get("recorded_only") for fault in rep.faults] == [True]
        assert rep.is_clean

    def test_verdicts_at_the_start_tier_are_cueable(self, monkeypatch):
        calls: list = []
        rules = [_VerdictRule(FaultType.PRESS_GRIP, FaultSeverity.MODERATE)]
        profile = _AnalyserProfile(_FakeAnalyzer(calls), calls, rules=rules)
        profile.min_cue_tiers = {"press_grip": FaultSeverity.MODERATE}
        session = _session(monkeypatch, profile)

        rep = _counted(session.run(_reps(1)))[0]

        assert "recorded_only" not in rep.faults[0].details
        assert not rep.is_clean

    def test_the_session_aware_start_tier_reads_the_frame_context(self, monkeypatch):
        calls: list = []
        seen: list[FrameContext] = []

        class _SessionAwareProfile(_AnalyserProfile):
            def cue_start_tier(self, fault_type, ctx, config):
                seen.append(ctx)
                return FaultSeverity.SEVERE

        profile = _SessionAwareProfile(
            _FakeAnalyzer(calls, reasons=["partial"]), calls,
            rules=[_VerdictRule(FaultType.PRESS_UNEVEN, FaultSeverity.MODERATE)],
        )
        session = _session(monkeypatch, profile)

        results = session.run(_reps(2))

        faults = [fault for result in results for fault in result.faults]
        assert {fault.fault_type for fault in faults} == {"rdl_partial", "press_uneven"}
        assert all(fault.details.get("recorded_only") for fault in faults)
        assert seen and all(isinstance(ctx, FrameContext) and ctx.angles is not None for ctx in seen)

    def test_a_recorded_only_shallow_verdict_is_sent_cue_less_and_flagged(self, monkeypatch):
        from biomechanics.pipeline_process import _send_shallow_rep

        calls: list = []
        profile = _AlwaysCueProfile(_FakeAnalyzer(calls, reasons=["partial"]), calls)
        profile.min_cue_tiers = {"rdl_partial": FaultSeverity.SEVERE}
        session = _session(monkeypatch, profile)

        rejected = [result for result in session.run(_reps(1)) if result.analyzer_shallow_rep]

        assert len(rejected) == 1
        assert rejected[0].faults[0].severity == FaultSeverity.MODERATE
        assert rejected[0].faults[0].details["recorded_only"] is True
        assert rejected[0].shallow_rep_cue is None
        assert profile.cue_args == []
        assert session.pipeline._shallow_cues_this_set == 0
        client = _RecordingClient()
        _send_shallow_rep(IPCBridge(client), rejected[0], set_number=1)
        message = next(message for message in client.messages if message["type"] == "shallow_rep")
        assert message["cue"] is None
        assert message["recorded_only"] is True

    def test_a_cueable_shallow_verdict_carries_no_recorded_only_flag(self, monkeypatch):
        from biomechanics.pipeline_process import _send_shallow_rep

        calls: list = []
        profile = _AlwaysCueProfile(_FakeAnalyzer(calls, reasons=["partial"]), calls)
        profile.min_cue_tiers = {"rdl_partial": FaultSeverity.MODERATE}
        session = _session(monkeypatch, profile)

        rejected = [result for result in session.run(_reps(1)) if result.analyzer_shallow_rep]

        assert rejected[0].shallow_rep_cue == SHALLOW_CUE
        assert session.pipeline._shallow_cues_this_set == 1
        client = _RecordingClient()
        _send_shallow_rep(IPCBridge(client), rejected[0], set_number=1)
        message = next(message for message in client.messages if message["type"] == "shallow_rep")
        assert message["cue"] == SHALLOW_CUE
        assert "recorded_only" not in message


class TestProfileFlagsInThePipeline:
    def test_a_profile_that_does_not_feed_body_calibration_never_measures_the_body(self, monkeypatch):
        class _NoMeasureSquat(SquatProfile):
            feeds_body_calibration = False

        measured = _session(monkeypatch, SquatProfile())
        measured.run([0.0] * int(3 * SYNTHETIC_FPS))
        assert measured.pipeline.body_calibration.progress[0] > 0

        unmeasured = _session(monkeypatch, _NoMeasureSquat())
        unmeasured.run([0.0] * int(3 * SYNTHETIC_FPS))
        assert unmeasured.pipeline.body_calibration.progress[0] == 0

    def test_tracking_keypoints_decide_what_counts_as_lost(self, monkeypatch):
        class _HandsTracked(SquatProfile):
            tracking_keypoints = LEG_KEYPOINTS + (CK.LEFT_WRIST, CK.RIGHT_WRIST)

        confidences = np.full(21, 0.9)
        confidences[[CK.LEFT_WRIST, CK.RIGHT_WRIST]] = 0.0
        squat = _session(monkeypatch, SquatProfile())
        assert squat.step(world_squat_points(0.02), confidences).missing_keypoints == []

        hands = _session(monkeypatch, _HandsTracked())
        assert hands.step(world_squat_points(0.02), confidences).missing_keypoints == ["left_wrist", "right_wrist"]


class TestPitchTripwire:
    def test_a_tripped_delta_revokes_certification_for_the_session(self, monkeypatch):
        monkeypatch.setenv("NOWVA_CAMERA_PITCH_DEG", MOUNT_PITCH_DEG)
        calls: list = []
        analyzer = _TrippingAnalyzer(calls)
        session = _session(monkeypatch, _AnalyserProfile(analyzer, calls), multi_camera=False)
        assert session.pipeline.pitch_certified

        results = session.run([0.0] * READINESS_WARM_UP_FRAMES * 2)

        revoked = [result.pitch_certification_revoked_deg for result in results if result.pitch_certification_revoked_deg]
        assert revoked == [TRIPWIRE_DELTA_DEG]
        assert analyzer.observed[0].pitch_certified
        assert not analyzer.observed[-1].pitch_certified
        assert not session.pipeline.pitch_certified
        session.pipeline.set_exercise("Barbell Back Squat")
        assert not session.pipeline.pitch_certified

    def test_a_verdict_on_the_tripping_frame_is_judged_uncertified(self, monkeypatch):
        monkeypatch.setenv("NOWVA_CAMERA_PITCH_DEG", MOUNT_PITCH_DEG)
        calls: list = []
        analyzer = _TrippingAnalyzer(calls, trip_after=TRIPWIRE_WITH_VERDICT_OBSERVATIONS)
        rule = _ObservationRule(
            FaultType.RDL_HIP_SINK, analyzer, frozenset({1, TRIPWIRE_WITH_VERDICT_OBSERVATIONS}),
        )
        session = _session(monkeypatch, _CertifiedTierProfile(analyzer, calls, rules=[rule]), multi_camera=False)

        results = session.run([0.0] * READINESS_WARM_UP_FRAMES * 4)

        tripping = [index for index, result in enumerate(results) if result.pitch_certification_revoked_deg]
        assert len(tripping) == 1
        before = [fault for result in results[:tripping[0]] for fault in result.faults]
        assert [fault.details.get("recorded_only") for fault in before] == [None]
        assert [fault.details.get("recorded_only") for fault in results[tripping[0]].faults] == [True]
