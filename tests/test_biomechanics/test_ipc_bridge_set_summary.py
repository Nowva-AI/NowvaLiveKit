"""
Producer-side delivery contract for non-squat exercises (SHARED-PLUMBING.md §4):
recorded-only verdicts are flagged, never cued and never counted; a profile that
does not report depth sends null depth fields (every profile but the squat,
legacy rep-counting ones included); cache_cues carries the fields a non-squat
profile needs and leaves the squat payload unchanged.
"""

from __future__ import annotations

import math

import pytest

from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.coaching.session_tracker import SessionTracker
from biomechanics.config import BiomechanicsConfig, CoachingConfig, IPCConfig
from biomechanics.profiles import get_profile
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.utils.types import FaultEvent, FaultSeverity, RepData

FAULT_COOLDOWN_S = 3.0
REP_DURATION_S = 2.0
DEPTH_DEG = 95.0
CUSTOM_IDLE_TIMEOUT_S = 20.0
RANGE_LABEL = "mid-shin"
SHALLOW_CUE = "rdl_lower"
SQUAT_CACHE_CUES_KEYS = {"type", "exercise_name", "profile", "cues"}
# Rep-counting profiles from before the shared plumbing: none of them is a depth exercise.
LEGACY_REP_COUNTING_EXERCISES = (
    "Barbell Row", "Barbell Curl", "Reverse Lunge", "Bulgarian Split Squat",
    "Skull Crusher", "Overhead Tricep Extension", "Barbell Conventional Deadlift",
)


class _NonDepthProfile(ExerciseProfile):
    name = "fake_lift"
    reports_depth = False
    set_idle_timeout_s = CUSTOM_IDLE_TIMEOUT_S

    def get_fault_to_cue_map(self):
        return {"press_uneven": "press_uneven"}

    def create_set_diagnosis(self):
        return object()

    def get_hedged_recap_faults(self, config):
        return frozenset({"press_uneven"})


def _fault(
    fault_type: str = "press_uneven",
    severity: FaultSeverity = FaultSeverity.MILD,
    recorded_only: bool = False,
    timestamp: float = 10.0,
    rep_number: int = 1,
    setup_note: bool = False,
) -> FaultEvent:
    details = {"side": "left", "observability": "observable"}
    if recorded_only:
        details["recorded_only"] = True
    if setup_note:
        details["setup_note"] = True
    return FaultEvent(
        fault_type=fault_type, severity=severity, severity_score=1.0, message="verdict",
        timestamp=timestamp, rep_number=rep_number, details=details,
    )


def _rep(rep_number: int, faults: list[FaultEvent] | None = None, **features) -> RepData:
    start = rep_number * 10.0
    return RepData(
        rep_number=rep_number, start_time=start, end_time=start + REP_DURATION_S,
        start_frame=0, end_frame=1, max_depth_angle=DEPTH_DEG, faults=faults or [], features=features,
    )


@pytest.fixture
def bridge(mock_ipc_client) -> IPCBridge:
    return IPCBridge(mock_ipc_client, CoachingConfig(), IPCConfig(fault_cooldown_seconds=FAULT_COOLDOWN_S))


def _messages(bridge: IPCBridge, message_type: str) -> list[dict]:
    return [message for message in bridge.ipc_client.messages if message["type"] == message_type]


class TestRecordedOnlyOnTheWire:
    def test_a_recorded_only_fault_is_flagged_and_never_cued(self, bridge):
        bridge.prepare_exercise("Barbell Back Squat")
        bridge.send_fault(_fault("knee_valgus", recorded_only=True))

        message = _messages(bridge, "fault")[0]
        assert message["recorded_only"] is True
        assert message["cue"] is None

    def test_a_cueable_fault_carries_no_flag(self, bridge):
        bridge.prepare_exercise("Barbell Back Squat")
        bridge.send_fault(_fault("knee_valgus"))

        message = _messages(bridge, "fault")[0]
        assert "recorded_only" not in message
        assert message["cue"] == "knees_out_left"

    def test_a_recorded_only_fault_never_holds_back_a_cueable_one(self, bridge):
        bridge.prepare_exercise("Barbell Back Squat")
        bridge.send_fault(_fault("knee_valgus", recorded_only=True, timestamp=10.0))
        bridge.send_fault(_fault("knee_valgus", timestamp=11.0))

        assert [message.get("recorded_only", False) for message in _messages(bridge, "fault")] == [True, False]

    def test_rep_complete_keeps_recorded_only_out_of_faults_in_rep(self, bridge):
        recorded = _fault("press_uneven", recorded_only=True)
        cueable = _fault("press_grip", FaultSeverity.MODERATE)
        bridge.send_rep_complete(_rep(1, [recorded, cueable]))

        message = _messages(bridge, "rep_complete")[0]
        assert message["faults_in_rep"] == ["press_grip"]
        assert [entry["details"].get("recorded_only", False) for entry in message["faults_detailed"]] == [True, False]

    def test_a_rep_whose_only_verdict_is_recorded_only_is_clean(self, bridge):
        rep = _rep(1, [_fault(recorded_only=True)])
        assert rep.is_clean
        bridge.send_rep_complete(rep)
        assert _messages(bridge, "rep_complete")[0]["is_clean"] is True

    def test_a_rep_whose_only_significant_verdict_is_a_setup_note_is_clean(self, bridge):
        grip_note = _fault("press_grip", FaultSeverity.MODERATE, setup_note=True)
        noted = _rep(1, [grip_note])
        faulted = _rep(2, [_fault("press_grip", FaultSeverity.MODERATE, setup_note=True), _fault()])
        assert noted.is_clean and not faulted.is_clean

        bridge.send_rep_complete(noted)
        bridge.send_set_complete(1, [noted, faulted])

        message = _messages(bridge, "rep_complete")[0]
        assert message["is_clean"] is True
        assert message["faults_in_rep"] == ["press_grip"]
        summary = _messages(bridge, "set_complete")[0]
        assert summary["clean_reps"] == 1
        assert set(summary["fault_summary"]) == {"press_uneven"}

    def test_set_summary_never_counts_recorded_only_verdicts(self, bridge):
        reps = [_rep(1, [_fault(recorded_only=True)]), _rep(2, [_fault(recorded_only=True), _fault("press_grip")])]
        bridge.send_set_complete(1, reps)

        summary = _messages(bridge, "set_complete")[0]
        assert set(summary["fault_summary"]) == {"press_grip"}
        assert summary["clean_reps"] == 1


class TestDepthFields:
    def test_squat_reports_depth(self, bridge):
        bridge.prepare_exercise("Barbell Back Squat")
        bridge.send_rep_complete(_rep(1))
        bridge.send_set_complete(1, [_rep(1), _rep(2)])

        assert _messages(bridge, "rep_complete")[0]["max_depth_angle"] == pytest.approx(DEPTH_DEG)
        summary = _messages(bridge, "set_complete")[0]
        assert summary["avg_depth"] == pytest.approx(DEPTH_DEG)
        assert summary["depth_consistency"] == pytest.approx(0.0)

    def test_a_profile_without_depth_sends_null_depth(self, bridge):
        bridge.prepare_exercise("Barbell Bench Press", _NonDepthProfile())
        bridge.send_rep_complete(_rep(1))
        bridge.send_set_complete(1, [_rep(1), _rep(2)])

        rep_message = _messages(bridge, "rep_complete")[0]
        assert rep_message["max_depth_angle"] is None
        assert rep_message["depth_category"] is None
        summary = _messages(bridge, "set_complete")[0]
        assert summary["avg_depth"] is None
        assert summary["depth_consistency"] is None


class TestLegacyProfilesWithoutDepth:
    @pytest.mark.parametrize("exercise_name", LEGACY_REP_COUNTING_EXERCISES)
    def test_a_switch_to_a_legacy_profile_sends_null_depth_end_to_end(
        self, monkeypatch, bridge, exercise_name: str,
    ):
        from biomechanics.pipeline_process import _activate_profile
        from conftest import SyntheticSession

        session = SyntheticSession(monkeypatch, multi_camera=False)
        tracker = SessionTracker(bridge, CoachingConfig())
        _activate_profile(session.pipeline, tracker, bridge, "Barbell Back Squat", startup=True)
        assert tracker.reports_depth and bridge.reports_depth

        _activate_profile(session.pipeline, tracker, bridge, exercise_name)
        tracker.on_rep_complete(_rep(1))
        tracker.on_rep_complete(_rep(2))
        tracker.force_end_set()

        assert session.pipeline.profile.name != "untracked"
        assert _messages(bridge, "cache_cues")[-1]["reports_depth"] is False
        assert not tracker.reports_depth and not bridge.reports_depth
        for message in _messages(bridge, "rep_complete"):
            assert message["max_depth_angle"] is None
            assert message["depth_category"] is None
            assert "depth_target_met" not in message["highlights"]
        summary = _messages(bridge, "set_complete")[0]
        assert summary["avg_depth"] is None and summary["depth_consistency"] is None
        assert tracker.last_set_summary["avg_depth"] is None


class TestCacheCues:
    def test_the_squat_payload_is_unchanged(self, bridge):
        bridge.prepare_exercise("Barbell Back Squat", get_profile("Barbell Back Squat"), BiomechanicsConfig())
        assert set(_messages(bridge, "cache_cues")[0]) == SQUAT_CACHE_CUES_KEYS

    def test_a_non_squat_profile_sends_what_differs_from_the_squat(self, bridge):
        bridge.prepare_exercise("Barbell Bench Press", _NonDepthProfile(), BiomechanicsConfig())

        message = _messages(bridge, "cache_cues")[0]
        assert message["fault_to_cue"] == {"press_uneven": "press_uneven"}
        assert message["reports_depth"] is False
        assert message["set_idle_timeout_s"] == pytest.approx(CUSTOM_IDLE_TIMEOUT_S)
        assert message["waits_for_diagnosis"] is True
        assert message["hedged_recap_faults"] == ["press_uneven"]

    def test_an_untracked_exercise_sends_its_empty_cue_map(self, bridge):
        bridge.prepare_exercise("Barbell Bench Press")
        message = _messages(bridge, "cache_cues")[0]
        assert message["fault_to_cue"] == {}
        assert message["reports_depth"] is False


class TestShallowRep:
    def test_the_squat_message_is_unchanged(self, bridge):
        bridge.send_shallow_rep(2)
        assert _messages(bridge, "shallow_rep")[0] == {
            "type": "shallow_rep", "depth_class": 2, "depth_class_name": "Half", "cue": "deeper",
        }

    def test_an_analyser_rejection_carries_its_cue_and_range_label(self, bridge):
        fault = _fault("rdl_partial", FaultSeverity.MODERATE)
        bridge.send_shallow_rep(None, fault=fault, set_number=2, cue_key=SHALLOW_CUE, label=RANGE_LABEL)

        message = _messages(bridge, "shallow_rep")[0]
        assert message["depth_class"] is None
        assert message["depth_class_name"] == RANGE_LABEL
        assert message["label"] == RANGE_LABEL
        assert message["cue"] == SHALLOW_CUE
        assert message["fault_type"] == "rdl_partial"

    def test_a_recorded_only_rejection_is_flagged_at_the_top_level(self, bridge):
        fault = _fault("rdl_partial", FaultSeverity.MODERATE, recorded_only=True)
        bridge.send_shallow_rep(None, fault=fault, set_number=2, cue_key=None, label=RANGE_LABEL)

        message = _messages(bridge, "shallow_rep")[0]
        assert message["recorded_only"] is True
        assert message["cue"] is None

    def test_a_cueable_rejection_carries_no_flag(self, bridge):
        fault = _fault("rdl_partial", FaultSeverity.MODERATE)
        bridge.send_shallow_rep(None, fault=fault, set_number=2, cue_key=SHALLOW_CUE, label=RANGE_LABEL)
        assert "recorded_only" not in _messages(bridge, "shallow_rep")[0]

    def test_a_cue_less_rejection_still_reaches_the_agent(self, bridge):
        bridge.send_shallow_rep(None, cue_key=None)
        message = _messages(bridge, "shallow_rep")[0]
        assert message["cue"] is None
        assert message["depth_class_name"] == ""


class TestSessionTrackerSummaries:
    def test_a_profile_without_depth_summarises_without_depth(self, bridge):
        tracker = SessionTracker(bridge, CoachingConfig())
        tracker.on_exercise_changed(diagnosis_enabled=False, reports_depth=False)
        tracker.on_rep_complete(_rep(1))
        tracker.force_end_set()

        assert tracker.last_set_summary["avg_depth"] is None
        assert tracker.last_set_summary["depth_consistency"] is None
        assert tracker.session_stats["avg_depth"] is None

    def test_tracker_fault_summary_skips_recorded_only(self, bridge):
        tracker = SessionTracker(bridge, CoachingConfig())
        tracker.on_rep_complete(_rep(1, [_fault(recorded_only=True), _fault("press_grip")]))
        tracker.force_end_set()

        assert set(tracker.last_set_summary["fault_summary"]) == {"press_grip"}

    def test_a_setup_note_is_neither_a_rep_fault_nor_unclean_for_the_tracker(self, bridge):
        tracker = SessionTracker(bridge, CoachingConfig())
        tracker.on_exercise_changed(diagnosis_enabled=False, reports_depth=False)
        grip_note = _fault("press_grip", FaultSeverity.MODERATE, setup_note=True)
        tracker.on_rep_complete(_rep(1, highlight_score=1.0))
        tracker.on_rep_complete(_rep(2, [grip_note], highlight_score=5.0))
        tracker.force_end_set()

        assert tracker.last_set_summary["fault_summary"] == {}
        assert tracker.last_set_summary["clean_reps"] == 2
        # The noted rep can still be the best one.
        best = ["best_rep_so_far" in message["highlights"] for message in _messages(bridge, "rep_complete")]
        assert best == [False, True]

    def test_depth_target_highlight_only_when_the_exercise_reports_depth(self, bridge):
        tracker = SessionTracker(bridge, CoachingConfig())
        tracker.on_exercise_changed(diagnosis_enabled=False, reports_depth=False)
        tracker.on_rep_complete(_rep(1))

        assert "depth_target_met" not in _messages(bridge, "rep_complete")[0]["highlights"]

    def test_highlight_score_names_the_best_clean_rep_when_the_engine_does_not_score(self, bridge):
        tracker = SessionTracker(bridge, CoachingConfig())
        tracker.on_exercise_changed(diagnosis_enabled=False, reports_depth=False)
        reps = [
            _rep(1, highlight_score=1.0),
            _rep(2, highlight_score=2.0),
            _rep(3, highlight_score=1.5),
            _rep(4, [_fault("press_grip", FaultSeverity.MODERATE)], highlight_score=5.0),
            _rep(5, highlight_score=math.nan),
            _rep(6),
        ]
        for rep in reps:
            tracker.on_rep_complete(rep)

        best = ["best_rep_so_far" in message["highlights"] for message in _messages(bridge, "rep_complete")]
        assert best == [False, True, False, False, False, False]
