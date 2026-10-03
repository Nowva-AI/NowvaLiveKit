"""Checks on the display page's set report for exercises other than the squat:
every press and RDL fault has a focus line, and depth tiles (on the profile's
reports_depth flag, never on a null value) and the form score are hidden when an
exercise has none."""

from __future__ import annotations

import re
from pathlib import Path

DISPLAY_PATH = Path(__file__).parent.parent / "src" / "visual" / "display.html"
PRESS_AND_RDL_FAULTS = [
    "press_layback", "press_bar_forward", "press_lockout", "press_leg_drive",
    "press_uneven", "press_velocity_loss", "rdl_hip_sink", "rdl_bar_drift",
    "rdl_lockout", "rdl_bar_tilt", "rdl_range_drift", "rdl_tempo",
]


def _block(page: str, constant: str) -> str:
    match = re.search(rf"const {constant} = \{{(.*?)\n  \}};", page, re.S)
    assert match, constant
    return match.group(1)


class TestSetReport:
    def test_every_press_and_rdl_fault_has_a_focus_line(self) -> None:
        focus_text = _block(DISPLAY_PATH.read_text(), "FOCUS_TEXT")
        for fault_type in PRESS_AND_RDL_FAULTS:
            assert re.search(rf"\b{fault_type}: \"[^\"]+\"", focus_text), fault_type

    def test_only_the_squat_has_score_dimensions(self) -> None:
        score_dims = _block(DISPLAY_PATH.read_text(), "SCORE_DIMS")
        assert re.findall(r"^    (\w+): \[", score_dims, re.M) == ["squat"]

    def test_depth_tiles_hide_on_the_profile_flag_not_on_null(self) -> None:
        """A squat set with one unmeasured rep sends null depth: its tiles show "—"."""
        page = DISPLAY_PATH.read_text()
        assert page.count('parentElement.classList.toggle("hidden", !workout.reportsDepth)') == 2
        assert "summary.avg_depth == null)" not in page
        assert "summary.depth_consistency == null)" not in page
        assert "#report .tile.hidden { display: none; }" in page

    def test_null_depth_still_reads_as_a_dash(self) -> None:
        page = DISPLAY_PATH.read_text()
        assert 'summary.avg_depth != null ? `${Math.round(summary.avg_depth)}°` : "—"' in page
        assert 'summary.depth_consistency != null ? `±${summary.depth_consistency}°` : "—"' in page

    def test_only_an_explicit_false_hides_depth(self) -> None:
        """An older hub sends no flag; it runs the squat, which has depth."""
        page = DISPLAY_PATH.read_text()
        assert "workout.reportsDepth = msg.reports_depth !== false;" in page
        assert "reportsDepth: true," in page
