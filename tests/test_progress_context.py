"""
Tests for progress context formatters: greeting line, recap comparisons,
and the main-menu progress report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.progress_context import (
    FAULT_LABELS,
    build_greeting_progress_line,
    build_progress_comparison_lines,
    build_progress_report,
    build_session_comparison_line,
    build_trend_comparison_lines,
    fault_label,
)

SQUAT_CONTRACT_FAULT_TYPES = [
    "knee_valgus", "hip_shoot", "heel_rise", "balance", "hip_shift",
    "bilateral_asymmetry", "depth", "foot_placement", "lockout", "tempo",
    "depth_drift", "velocity_loss",
]
JARGON_WORDS = ["valgus", "eccentric", "concentric", "dorsiflexion", "asymmetry"]


def _baseline(**overrides):
    baseline = {
        "session_date": "2026-07-15T10:00:00",
        "days_ago": 3,
        "mean_score": 0.72,
        "per_dimension": {
            "depth": 0.80,
            "trunk_control": 0.75,
            "knee_tracking": 0.60,
            "symmetry": 0.78,
        },
        "total_reps": 24,
        "total_sets": 3,
        "top_faults": [{"fault_type": "knee_valgus", "count": 9, "avg_severity": 0.5}],
        "avg_knee_valgus_deg": 14.2,
        "avg_trunk_pitch_deg": 38.0,
    }
    baseline.update(overrides)
    return baseline


def _scoring(mean_score=0.78, **dims):
    per_dimension = {
        "depth": 0.78,
        "trunk_control": 0.76,
        "knee_tracking": 0.74,
        "symmetry": 0.79,
    }
    per_dimension.update(dims)
    return {"mean_score": mean_score, "per_dimension": per_dimension}


class TestGreetingProgressLine:
    def test_none_baseline_returns_none(self):
        assert build_greeting_progress_line(None) is None

    def test_full_baseline_mentions_key_facts(self):
        line = build_greeting_progress_line(_baseline())
        assert "3 days ago" in line
        assert "72 out of 100" in line
        assert "knee tracking" in line
        assert "14.2 degrees" in line
        assert "knees caving in" in line

    def test_same_day_phrasing(self):
        line = build_greeting_progress_line(_baseline(days_ago=0))
        assert "earlier today" in line

    def test_minimal_baseline_still_builds(self):
        line = build_greeting_progress_line(
            _baseline(
                mean_score=None,
                per_dimension={},
                top_faults=[],
                avg_knee_valgus_deg=None,
                total_reps=0,
            )
        )
        assert "last squat session" in line


class TestProgressComparisonLines:
    def test_no_baseline_returns_empty(self):
        assert build_progress_comparison_lines(None, _scoring()) == []

    def test_no_scoring_returns_empty(self):
        assert build_progress_comparison_lines(_baseline(), None) == []

    def test_overall_delta_computed(self):
        lines = build_progress_comparison_lines(_baseline(), _scoring(mean_score=0.78))
        joined = " ".join(lines)
        assert "72 to 78" in joined
        assert "+6 points" in joined

    def test_dimension_deltas_above_threshold_included(self):
        lines = build_progress_comparison_lines(_baseline(), _scoring())
        joined = " ".join(lines)
        # knee_tracking moved 0.60 -> 0.74 (+14): included
        assert "knee tracking: 60 to 74 (+14)" in joined
        # depth moved 0.80 -> 0.78 (-2 < 3-point threshold): excluded
        assert "depth: 80" not in joined

    def test_delta_always_matches_rounded_endpoints(self):
        # 0.716 -> 0.784: raw delta rounds to +7, but spoken endpoints are
        # 72 and 78 — the stated delta must be their difference (+6)
        lines = build_progress_comparison_lines(
            _baseline(mean_score=0.716), _scoring(mean_score=0.784)
        )
        joined = " ".join(lines)
        assert "72 to 78" in joined
        assert "+6 points" in joined

    def test_tempo_change_is_compared(self):
        baseline = _baseline(per_dimension={"tempo": 0.50})
        lines = build_progress_comparison_lines(baseline, _scoring(tempo=0.70))
        assert "tempo: 50 to 70 (+20)" in " ".join(lines)

    def test_missing_dimensions_skipped(self):
        scoring = {"mean_score": 0.7, "per_dimension": {}}
        lines = build_progress_comparison_lines(_baseline(), scoring)
        joined = " ".join(lines)
        assert "72 to 70" in joined
        assert "Changed vs last session" not in joined


class TestSessionComparisonLine:
    def test_improvement(self):
        line = build_session_comparison_line(_baseline(), 0.80)
        assert "8 points better" in line

    def test_decline(self):
        line = build_session_comparison_line(_baseline(), 0.65)
        assert "7 points below" in line

    def test_matched(self):
        line = build_session_comparison_line(_baseline(), 0.72)
        assert "matched" in line

    def test_no_baseline_returns_none(self):
        assert build_session_comparison_line(None, 0.8) is None
        assert build_session_comparison_line(_baseline(mean_score=None), 0.8) is None


def _score_row(mean_score, **dims):
    row = {
        "timestamp": None,
        "set_number": 1,
        "rep_count": 8,
        "mean_score": mean_score,
        "depth": 0.8,
        "trunk_control": 0.75,
        "knee_tracking": 0.65,
        "symmetry": 0.78,
        "trend_slope": 0.0,
    }
    row.update(dims)
    return row


class TestProgressReport:
    def test_empty_rows(self):
        report = build_progress_report([], None)
        assert "No squat sessions recorded" in report

    def test_trend_reported_with_enough_sets(self):
        rows = [
            _score_row(0.60),
            _score_row(0.62),
            _score_row(0.70),
            _score_row(0.74),
            _score_row(0.78),
        ]
        report = build_progress_report(rows, _baseline())
        assert "5 squat sets recorded" in report
        assert "Form score trend" in report
        assert "Biggest opportunity: knee tracking" in report
        assert "knees caving in" in report

    def test_few_sets_skips_trend(self):
        rows = [_score_row(0.60), _score_row(0.70)]
        report = build_progress_report(rows, None)
        assert "Form score trend" not in report
        assert "2 squat sets recorded" in report


class TestFaultLabels:
    def test_every_squat_fault_has_a_plain_label(self):
        for fault_type in SQUAT_CONTRACT_FAULT_TYPES:
            label = FAULT_LABELS[fault_type]
            assert "_" not in label
            assert not any(word in label for word in JARGON_WORDS), label

    def test_dead_keys_are_gone(self):
        for fault_type in ("butt_wink", "shallow_depth", "asymmetric_loading"):
            assert fault_type not in FAULT_LABELS

    def test_unknown_fault_falls_back_to_spaced_name(self):
        assert fault_label("elbow_flare") == "elbow flare"


def _fault_trends(total_occurrences: int = 30, total_reps: int = 60, sessions: int = 3) -> dict:
    return {
        "sessions_analyzed": sessions,
        "total_reps": total_reps,
        "fault_profile": [{"fault_type": "knee_valgus", "total_occurrences": total_occurrences}],
        "chronic_faults": [],
    }


class TestTrendComparisonLines:
    """A set is compared with recent sessions as a per-rep rate: a 5-rep set
    must not look 'improving' just because a session has more reps."""

    def test_same_rate_as_usual_says_nothing(self):
        # Usual rate 30/60 = half the reps; this set 4 of 8 = half
        assert build_trend_comparison_lines(_fault_trends(), {"knee_valgus": 4}, 8) == []

    def test_short_set_at_the_usual_rate_is_not_called_improving(self):
        # Per-session average is 10 occurrences; 3 of 6 reps is still half the reps
        assert build_trend_comparison_lines(_fault_trends(), {"knee_valgus": 3}, 6) == []

    def test_lower_rate_reads_as_improving(self):
        lines = build_trend_comparison_lines(_fault_trends(), {"knee_valgus": 1}, 10)
        assert len(lines) == 1
        assert "CROSS-SESSION TREND" in lines[0]
        assert "knees caving in" in lines[0]
        assert "below" in lines[0]

    def test_higher_rate_reads_as_above_usual(self):
        lines = build_trend_comparison_lines(_fault_trends(total_occurrences=6), {"knee_valgus": 5}, 8)
        assert len(lines) == 1
        assert "more often" in lines[0]

    def test_no_reps_gives_nothing(self):
        assert build_trend_comparison_lines(_fault_trends(), {"knee_valgus": 0}, 0) == []
