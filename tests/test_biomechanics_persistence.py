"""
Tests for the biomechanics persistence layer: cue outcome evaluation,
rep row building, and the recorder's ordered-operation state machine.
The ORM apply layer is exercised through op payload shapes only — no DB.
"""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db.biomechanics_persistence import (
    BiomechanicsRecorder,
    _aggregate_effectiveness,
    _build_rep_row,
    _extract_fault_series,
    build_baseline_summary,
    evaluate_cue_outcome,
    mean_fault_severity,
)

SCORE_TOLERANCE = 1e-9

USER_ID = uuid.uuid4()


def _fault_msg(fault_type="knee_valgus", severity_score=0.6, rep_number=1):
    return {
        "type": "fault",
        "fault_type": fault_type,
        "severity": "moderate",
        "severity_score": severity_score,
        "message": "Knees caving in",
        "cue": "knees_out",
        "rep_number": rep_number,
    }


def _rep_msg(rep_number=1, faults_detailed=None, set_number=1):
    return {
        "type": "rep_complete",
        "rep_number": rep_number,
        "set_number": set_number,
        "max_depth_angle": 95.0,
        "is_clean": not faults_detailed,
        "depth_class_int": 2,
        "rep_duration_ms": 2100,
        "descent_time_s": 1.2,
        "ascent_time_s": 0.9,
        "faults_detailed": faults_detailed or [],
        "rep_kinematic_summary": {"rep_number": rep_number, "knee_valgus_l": 8.0},
        "bottom_kpts": [[0.1, 0.2, 0.3]] * 17,
        "standing_kpts": [[0.1, 0.9, 0.3]] * 17,
    }


def _set_msg(set_number=1, mean_score=0.8, per_rep_scores=None, session_causes=None):
    return {
        "type": "diagnosis_complete",
        "set_number": set_number,
        "diagnosis": {
            "confidence": 0.9,
            "session_causes": session_causes or [],
        },
        "scoring": {
            "mean_score": mean_score,
            "per_dimension": {
                "depth": 0.9,
                "trunk_control": 0.8,
                "knee_tracking": 0.7,
                "symmetry": 0.85,
            },
            "best_rep": 1,
            "worst_rep": 2,
            "trend_slope": -0.01,
            "per_rep_scores": per_rep_scores or [],
        },
    }


def _fault_detail(fault_type="knee_valgus", severity_score=0.4):
    return {
        "fault_type": fault_type,
        "severity": "mild",
        "severity_score": severity_score,
        "message": "",
        "details": {},
    }


def _make_recorder():
    return BiomechanicsRecorder(user_id=USER_ID)


def _ops_by_kind(ops, kind):
    return [payload for op_kind, payload in ops if op_kind == kind]


class TestEvaluateCueOutcome:
    def test_fault_absent_next_rep_is_effective(self):
        outcome = evaluate_cue_outcome("knee_valgus", 0.6, [])
        assert outcome["present_next_rep"] is False
        assert outcome["severity_next_rep"] is None
        assert outcome["effective"] is True

    def test_severity_decreased_is_effective(self):
        outcome = evaluate_cue_outcome(
            "knee_valgus", 0.6, [_fault_detail(severity_score=0.3)]
        )
        assert outcome["present_next_rep"] is True
        assert outcome["severity_next_rep"] == pytest.approx(0.3, abs=SCORE_TOLERANCE)
        assert outcome["effective"] is True

    def test_severity_increased_is_not_effective(self):
        outcome = evaluate_cue_outcome(
            "knee_valgus", 0.6, [_fault_detail(severity_score=0.8)]
        )
        assert outcome["effective"] is False

    def test_severity_equal_is_not_effective(self):
        outcome = evaluate_cue_outcome(
            "knee_valgus", 0.6, [_fault_detail(severity_score=0.6)]
        )
        assert outcome["effective"] is False

    def test_other_fault_types_ignored(self):
        outcome = evaluate_cue_outcome(
            "knee_valgus", 0.6, [_fault_detail(fault_type="butt_wink")]
        )
        assert outcome["present_next_rep"] is False
        assert outcome["effective"] is True

    def test_worst_occurrence_judged_not_first(self):
        # Same fault twice in the next rep: first milder, second worse.
        outcome = evaluate_cue_outcome(
            "knee_valgus",
            0.6,
            [_fault_detail(severity_score=0.3), _fault_detail(severity_score=0.8)],
        )
        assert outcome["severity_next_rep"] == pytest.approx(0.8, abs=SCORE_TOLERANCE)
        assert outcome["effective"] is False


class TestMeanFaultSeverity:
    def test_absent_fault_returns_zero(self):
        assert mean_fault_severity("knee_valgus", [[], []]) == pytest.approx(
            0.0, abs=SCORE_TOLERANCE
        )

    def test_mean_across_reps(self):
        reps_faults = [
            [_fault_detail(severity_score=0.4)],
            [],
            [_fault_detail(severity_score=0.8)],
        ]
        assert mean_fault_severity("knee_valgus", reps_faults) == pytest.approx(
            0.6, abs=SCORE_TOLERANCE
        )


class TestBuildRepRow:
    def test_extracts_columns_from_message(self):
        row = _build_rep_row(_rep_msg(rep_number=3, set_number=2))
        assert row["rep_number"] == 3
        assert row["set_number"] == 2
        assert row["is_clean"] is True
        assert row["depth_class"] == 2
        assert row["max_depth_angle"] == pytest.approx(95.0, abs=SCORE_TOLERANCE)
        assert row["timing"]["rep_duration_ms"] == 2100
        assert row["kinematics"]["knee_valgus_l"] == pytest.approx(
            8.0, abs=SCORE_TOLERANCE
        )
        assert len(row["bottom_kpts"]) == 17

    def test_missing_optional_fields_default(self):
        row = _build_rep_row({"rep_number": 1})
        assert row["faults"] == []
        assert row["kinematics"] is None
        assert row["bottom_kpts"] is None


class TestRecorderOps:
    def test_session_start_op(self):
        recorder = _make_recorder()
        ops = recorder._ops_for("session_start", {})
        inserts = _ops_by_kind(ops, "insert_session")
        assert len(inserts) == 1
        assert inserts[0]["id"] == recorder.session_id
        assert inserts[0]["user_id"] == USER_ID
        assert inserts[0]["exercise"] == "squat"

    def test_string_user_id_coerced_to_uuid(self):
        recorder = BiomechanicsRecorder(user_id=str(USER_ID))
        assert recorder._user_id == USER_ID

    def test_invalid_user_id_raises(self):
        with pytest.raises(ValueError):
            BiomechanicsRecorder(user_id="not-a-uuid")

    def test_fault_inserts_cue_event(self):
        recorder = _make_recorder()
        ops = recorder._ops_for("fault", _fault_msg())
        inserts = _ops_by_kind(ops, "insert_cue")
        assert len(inserts) == 1
        assert inserts[0]["fault_type"] == "knee_valgus"
        assert inserts[0]["cue_key"] == "knees_out"
        assert inserts[0]["session_id"] == recorder.session_id

    def test_rep_links_same_rep_cues(self):
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(rep_number=1))
        ops = recorder._ops_for("rep", _rep_msg(rep_number=1))
        links = _ops_by_kind(ops, "link_cues")
        assert len(links) == 1
        assert len(links[0]["cue_ids"]) == 1
        # Same-rep cue must NOT be outcome-evaluated against its own rep
        assert _ops_by_kind(ops, "cue_outcome") == []

    def test_next_rep_backfills_cue_outcome(self):
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(rep_number=1, severity_score=0.6))
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        ops = recorder._ops_for("rep", _rep_msg(rep_number=2))
        outcomes = _ops_by_kind(ops, "cue_outcome")
        assert len(outcomes) == 1
        assert outcomes[0]["present_next_rep"] is False
        assert outcomes[0]["effective"] is True

    def test_cue_outcome_evaluated_once(self):
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(rep_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=2))
        ops = recorder._ops_for("rep", _rep_msg(rep_number=3))
        assert _ops_by_kind(ops, "cue_outcome") == []

    def test_persisting_fault_marked_not_effective(self):
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(rep_number=1, severity_score=0.4))
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        ops = recorder._ops_for(
            "rep",
            _rep_msg(
                rep_number=2,
                faults_detailed=[_fault_detail(severity_score=0.7)],
            ),
        )
        outcomes = _ops_by_kind(ops, "cue_outcome")
        assert outcomes[0]["present_next_rep"] is True
        assert outcomes[0]["effective"] is False

    def test_set_complete_links_reps_and_cues(self):
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(rep_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=2))
        ops = recorder._ops_for("set", _set_msg())

        set_inserts = _ops_by_kind(ops, "insert_set")
        assert len(set_inserts) == 1
        assert set_inserts[0]["rep_count"] == 2
        assert set_inserts[0]["mean_score"] == pytest.approx(
            0.8, abs=SCORE_TOLERANCE
        )

        rep_assigns = _ops_by_kind(ops, "assign_set_to_reps")
        assert len(rep_assigns) == 1
        assert len(rep_assigns[0]["rep_ids"]) == 2
        cue_assigns = _ops_by_kind(ops, "assign_set_to_cues")
        assert len(cue_assigns) == 1

    def test_set_complete_backfills_per_rep_scores(self):
        recorder = _make_recorder()
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        per_rep_scores = [{
            "rep_number": 1,
            "depth_score": 0.9,
            "trunk_control_score": 0.8,
            "knee_tracking_score": 0.7,
            "symmetry_score": 0.85,
            "composite_score": 0.81,
        }]
        ops = recorder._ops_for("set", _set_msg(per_rep_scores=per_rep_scores))
        score_updates = _ops_by_kind(ops, "update_rep_scores")
        assert len(score_updates) == 1
        assert score_updates[0]["composite_score"] == pytest.approx(
            0.81, abs=SCORE_TOLERANCE
        )

    def test_next_set_backfills_severity_next_set(self):
        recorder = _make_recorder()
        # Set 1: cue fires
        recorder._ops_for("fault", _fault_msg(rep_number=1, severity_score=0.6))
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        recorder._ops_for("set", _set_msg(set_number=1))
        # Set 2: fault persists at lower severity
        recorder._ops_for(
            "rep",
            _rep_msg(
                rep_number=2,
                set_number=2,
                faults_detailed=[_fault_detail(severity_score=0.3)],
            ),
        )
        ops = recorder._ops_for("set", _set_msg(set_number=2))
        next_set = _ops_by_kind(ops, "cue_next_set")
        assert len(next_set) == 1
        assert next_set[0]["severity_next_set"] == pytest.approx(
            0.3, abs=SCORE_TOLERANCE
        )

    def test_session_totals_accumulate_across_sets(self):
        recorder = _make_recorder()
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=2))
        recorder._ops_for("set", _set_msg(set_number=1, mean_score=0.8))
        recorder._ops_for("rep", _rep_msg(rep_number=3, set_number=2))
        ops = recorder._ops_for("set", _set_msg(set_number=2, mean_score=0.6))
        updates = _ops_by_kind(ops, "update_session")
        assert len(updates) == 1
        assert updates[0]["total_reps"] == 3
        assert updates[0]["total_sets"] == 2
        assert updates[0]["mean_session_score"] == pytest.approx(
            0.7, abs=1e-3
        )

    def test_session_causes_merged_by_cause_id(self):
        recorder = _make_recorder()
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        recorder._ops_for(
            "set",
            _set_msg(
                set_number=1,
                session_causes=[{"cause_id": "ankle_mobility", "score": 0.5}],
            ),
        )
        recorder._ops_for("rep", _rep_msg(rep_number=2, set_number=2))
        ops = recorder._ops_for(
            "set",
            _set_msg(
                set_number=2,
                session_causes=[{"cause_id": "ankle_mobility", "score": 0.7}],
            ),
        )
        updates = _ops_by_kind(ops, "update_session")
        causes = updates[0]["session_causes"]
        assert len(causes) == 1
        assert causes[0]["score"] == pytest.approx(0.7, abs=SCORE_TOLERANCE)

    def test_rep_diagnosis_updates_matching_rep(self):
        recorder = _make_recorder()
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        ops = recorder._ops_for(
            "rep_diagnosis",
            {
                "rep_number": 1,
                "rep_score": {
                    "rep_number": 1,
                    "depth_score": 0.9,
                    "trunk_control_score": 0.8,
                    "knee_tracking_score": 0.7,
                    "symmetry_score": 0.85,
                    "composite_score": 0.81,
                },
            },
        )
        updates = _ops_by_kind(ops, "update_rep_scores")
        assert len(updates) == 1
        assert updates[0]["depth_score"] == pytest.approx(
            0.9, abs=SCORE_TOLERANCE
        )

    def test_rep_diagnosis_without_matching_rep_is_noop(self):
        recorder = _make_recorder()
        ops = recorder._ops_for(
            "rep_diagnosis", {"rep_number": 5, "rep_score": {"depth_score": 0.9}}
        )
        assert ops == []

    def test_session_end_finalizes(self):
        recorder = _make_recorder()
        ops = recorder._ops_session_end()
        assert [kind for kind, _ in ops] == ["finalize_session"]

    def test_duplicate_fault_same_rep_recorded_once(self):
        recorder = _make_recorder()
        first = recorder._ops_for("fault", _fault_msg(rep_number=1))
        second = recorder._ops_for("fault", _fault_msg(rep_number=1))
        assert len(_ops_by_kind(first, "insert_cue")) == 1
        assert second == []
        # Same fault on a different rep is a new cue event
        third = recorder._ops_for("fault", _fault_msg(rep_number=2))
        assert len(_ops_by_kind(third, "insert_cue")) == 1

    def test_end_of_rep_fault_not_linked_or_evaluated(self):
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(fault_type="lockout", rep_number=2))
        rep1_ops = recorder._ops_for("rep", _rep_msg(rep_number=2))
        assert _ops_by_kind(rep1_ops, "link_cues") == []
        rep2_ops = recorder._ops_for("rep", _rep_msg(rep_number=3))
        assert _ops_by_kind(rep2_ops, "cue_outcome") == []

    @pytest.mark.parametrize("fault_type", ["hip_shoot", "bilateral_asymmetry", "velocity_loss"])
    def test_once_per_rep_squat_fault_is_linked_and_evaluated(self, fault_type):
        """Once-per-rep verdicts carry their own rep's number and appear in
        that rep's faults_detailed, so they link and get an outcome."""
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(fault_type=fault_type, rep_number=2))
        rep2_ops = recorder._ops_for(
            "rep", _rep_msg(rep_number=2, faults_detailed=[_fault_detail(fault_type, 0.6)])
        )
        assert len(_ops_by_kind(rep2_ops, "link_cues")) == 1
        rep3_ops = recorder._ops_for("rep", _rep_msg(rep_number=3))
        outcomes = _ops_by_kind(rep3_ops, "cue_outcome")
        assert len(outcomes) == 1
        assert outcomes[0]["present_next_rep"] is False

    def test_end_of_rep_fault_no_severity_next_set(self):
        recorder = _make_recorder()
        recorder._ops_for("fault", _fault_msg(fault_type="depth", rep_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        recorder._ops_for("set", _set_msg(set_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=2, set_number=2))
        ops = recorder._ops_for("set", _set_msg(set_number=2))
        assert _ops_by_kind(ops, "cue_next_set") == []

    def test_missing_diagnosis_flushes_unscored_set_on_next_rep(self):
        recorder = _make_recorder()
        recorder._ops_for("rep", _rep_msg(rep_number=1, set_number=1))
        recorder._ops_for("rep", _rep_msg(rep_number=2, set_number=1))
        # No diagnosis_complete for set 1 — first rep of set 2 closes it
        ops = recorder._ops_for("rep", _rep_msg(rep_number=3, set_number=2))
        set_inserts = _ops_by_kind(ops, "insert_set")
        assert len(set_inserts) == 1
        assert set_inserts[0]["set_number"] == 1
        assert set_inserts[0]["rep_count"] == 2
        assert "mean_score" not in set_inserts[0]
        totals = _ops_by_kind(ops, "update_session")
        assert totals[0]["total_reps"] == 2
        assert totals[0]["total_sets"] == 1
        # Set 2's later diagnosis only counts its own rep
        set2_ops = recorder._ops_for("set", _set_msg(set_number=2))
        assert _ops_by_kind(set2_ops, "insert_set")[0]["rep_count"] == 1
        assert _ops_by_kind(set2_ops, "update_session")[0]["total_reps"] == 3

    def test_session_end_flushes_leftover_reps(self):
        recorder = _make_recorder()
        recorder._ops_for("rep", _rep_msg(rep_number=1, set_number=1))
        ops = recorder._ops_session_end()
        kinds = [kind for kind, _ in ops]
        assert "insert_set" in kinds
        assert kinds[-1] == "finalize_session"
        assert _ops_by_kind(ops, "update_session")[0]["total_reps"] == 1

    def test_cue_delivered_marks_latest_matching(self):
        recorder = _make_recorder()
        first = _ops_by_kind(
            recorder._ops_for("fault", _fault_msg(rep_number=1)), "insert_cue"
        )[0]
        recorder._ops_for("rep", _rep_msg(rep_number=1))
        second = _ops_by_kind(
            recorder._ops_for("fault", _fault_msg(rep_number=2)), "insert_cue"
        )[0]
        ops = recorder._ops_for(
            "cue_delivered", {"fault_type": "knee_valgus", "cue_key": "knees_out"}
        )
        marks = _ops_by_kind(ops, "mark_delivered")
        assert len(marks) == 1
        assert marks[0]["cue_id"] == second["id"]
        assert marks[0]["cue_id"] != first["id"]

    def test_cue_delivered_unknown_fault_is_noop(self):
        recorder = _make_recorder()
        assert recorder._ops_for("cue_delivered", {"fault_type": "butt_wink"}) == []


class _FakeDbSession:
    """Chainable no-op stand-in for a SQLAlchemy session."""

    def __init__(self, log):
        self._log = log

    def add(self, obj):
        self._log.append(type(obj).__name__)

    def query(self, *args):
        return self

    def filter(self, *args):
        return self

    def update(self, *args, **kwargs):
        return 0

    def flush(self):
        pass

    def commit(self):
        self._log.append("commit")

    def rollback(self):
        pass

    def close(self):
        pass


class TestRecorderWorker:
    def test_full_flow_flushes_and_close_returns_true(self):
        log = []
        recorder = BiomechanicsRecorder(
            user_id=USER_ID, session_factory=lambda: _FakeDbSession(log)
        )
        recorder.start()
        recorder.record_fault(_fault_msg(rep_number=1))
        recorder.record_rep(_rep_msg(rep_number=1))
        recorder.record_rep(_rep_msg(rep_number=2))
        recorder.record_set(_set_msg())
        assert recorder.close(timeout_s=5.0) is True
        assert log.count("BiomechanicsSession") == 1
        assert log.count("BiomechanicsRep") == 2
        assert log.count("CueEvent") == 1
        assert log.count("BiomechanicsSet") == 1
        # One commit per queued message incl. session start and finalize
        assert log.count("commit") == 6


class _FakeCueRow:
    def __init__(self, fault_type, cue_key, effective):
        self.fault_type = fault_type
        self.cue_key = cue_key
        self.effective = effective


class TestAggregateEffectiveness:
    def test_groups_by_fault_and_cue(self):
        rows = [
            _FakeCueRow("knee_valgus", "knees_out", True),
            _FakeCueRow("knee_valgus", "knees_out", True),
            _FakeCueRow("knee_valgus", "knees_out", False),
            _FakeCueRow("butt_wink", "brace_core", True),
        ]
        result = _aggregate_effectiveness(rows)
        assert len(result) == 2
        valgus = next(e for e in result if e["fault_type"] == "knee_valgus")
        assert valgus["n_evaluated"] == 3
        assert valgus["n_effective"] == 2
        assert valgus["effectiveness"] == pytest.approx(0.667, abs=1e-3)

    def test_empty_rows(self):
        assert _aggregate_effectiveness([]) == []


class _FakeSetRow:
    def __init__(self, depth=0.8, trunk=0.75, knee=0.6, symmetry=0.78,
                 tempo=0.9, set_number=1, rep_count=5, mean_score=0.7):
        self.depth_score_avg = depth
        self.trunk_score_avg = trunk
        self.knee_score_avg = knee
        self.symmetry_score_avg = symmetry
        self.tempo_score_avg = tempo
        self.set_number = set_number
        self.rep_count = rep_count
        self.mean_score = mean_score


class _FakeBaselineRep:
    def __init__(self, kinematics=None, faults=None, set_number=1):
        self.kinematics = kinematics
        self.faults = faults
        self.set_number = set_number


class TestBuildBaselineSummary:
    def test_aggregates_dimensions_kinematics_and_faults(self):
        from datetime import datetime, timedelta

        set_rows = [_FakeSetRow(knee=0.6, set_number=1), _FakeSetRow(knee=0.7, set_number=2)]
        rep_rows = [
            _FakeBaselineRep(
                kinematics={
                    "knee_valgus_l": 10.0,
                    "knee_valgus_r": 14.0,
                    "trunk_pitch_at_bottom": 40.0,
                },
                faults=[_fault_detail(severity_score=0.5)],
            ),
            _FakeBaselineRep(
                kinematics={
                    "knee_valgus_l": 16.0,
                    "knee_valgus_r": 12.0,
                    "trunk_pitch_at_bottom": 36.0,
                },
                faults=[
                    _fault_detail(severity_score=0.7),
                    _fault_detail("butt_wink", 0.3),
                ],
            ),
        ]
        summary = build_baseline_summary(
            started_at=datetime.utcnow() - timedelta(days=3),
            mean_session_score=0.72,
            total_reps=2,
            total_sets=2,
            set_rows=set_rows,
            rep_rows=rep_rows,
        )
        assert summary["days_ago"] == 3
        assert summary["mean_score"] == pytest.approx(0.72, abs=SCORE_TOLERANCE)
        assert summary["per_dimension"]["knee_tracking"] == pytest.approx(
            0.65, abs=1e-3
        )
        # Worst-side valgus per rep: max(10,14)=14, max(16,12)=16 → mean 15
        assert summary["avg_knee_valgus_deg"] == pytest.approx(15.0, abs=0.1)
        assert summary["avg_trunk_pitch_deg"] == pytest.approx(38.0, abs=0.1)
        assert summary["top_faults"][0]["fault_type"] == "knee_valgus"
        assert summary["top_faults"][0]["count"] == 2

    def test_empty_session_produces_nulls(self):
        from datetime import datetime

        summary = build_baseline_summary(
            started_at=datetime.utcnow(),
            mean_session_score=None,
            total_reps=0,
            total_sets=0,
            set_rows=[],
            rep_rows=[],
        )
        assert summary["per_dimension"] == {}
        assert summary["avg_knee_valgus_deg"] is None
        assert summary["top_faults"] == []


class _FakeRepRow:
    def __init__(self, created_at, rep_number, faults):
        self.created_at = created_at
        self.rep_number = rep_number
        self.faults = faults


class TestExtractFaultSeries:
    def test_filters_by_fault_type(self):
        reps = [
            _FakeRepRow("t1", 1, [_fault_detail(severity_score=0.5)]),
            _FakeRepRow("t2", 2, [_fault_detail("butt_wink", 0.3)]),
            _FakeRepRow("t3", 3, None),
        ]
        series = _extract_fault_series(reps, "knee_valgus")
        assert len(series) == 1
        assert series[0]["rep_number"] == 1
        assert series[0]["severity_score"] == pytest.approx(
            0.5, abs=SCORE_TOLERANCE
        )
