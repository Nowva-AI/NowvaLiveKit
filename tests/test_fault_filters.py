"""Tests for the one filter every agent-side reader of rep verdicts goes through."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.fault_filters import (  # noqa: E402
    is_approximate,
    is_recorded_only,
    reportable_faults,
)


def _verdict(fault_type: str, observability: str = "observable", recorded_only: bool = False) -> dict:
    details: dict = {"side": None, "observability": observability}
    if recorded_only:
        details["recorded_only"] = True
    return {"fault_type": fault_type, "severity": "mild", "details": details}


class TestIsRecordedOnly:
    def test_flag_under_details(self) -> None:
        assert is_recorded_only(_verdict("press_uneven", recorded_only=True))

    def test_flag_on_a_fault_message(self) -> None:
        assert is_recorded_only({"fault_type": "press_uneven", "recorded_only": True, "details": {}})

    def test_unflagged_verdict(self) -> None:
        assert not is_recorded_only(_verdict("knee_valgus"))


class TestIsApproximate:
    def test_reads_details_then_the_message(self) -> None:
        assert is_approximate(_verdict("press_lockout", observability="approximate"))
        assert is_approximate({"fault_type": "hip_shift", "observability": "approximate"})
        assert not is_approximate(_verdict("knee_valgus"))


class TestReportableFaults:
    def test_recorded_only_is_always_dropped(self) -> None:
        faults = [_verdict("press_uneven", recorded_only=True), _verdict("knee_valgus")]
        for spoken in (True, False):
            assert [f["fault_type"] for f in reportable_faults(faults, spoken)] == ["knee_valgus"]

    def test_spoken_drops_approximate(self) -> None:
        faults = [_verdict("press_lockout", observability="approximate"), _verdict("knee_valgus")]
        assert [f["fault_type"] for f in reportable_faults(faults, spoken=True)] == ["knee_valgus"]

    def test_unspoken_keeps_approximate_for_hedging(self) -> None:
        faults = [_verdict("press_lockout", observability="approximate")]
        assert reportable_faults(faults, spoken=False) == faults

    def test_recorded_only_plus_approximate_leaves_nothing_to_say(self) -> None:
        faults = [
            _verdict("press_uneven", recorded_only=True),
            _verdict("press_lockout", observability="approximate"),
        ]
        assert reportable_faults(faults, spoken=True) == []

    def test_flat_verdicts_without_details(self) -> None:
        faults = [{"fault_type": "hip_shoot", "side": None, "observability": "approximate"}]
        assert reportable_faults(faults, spoken=True) == []
