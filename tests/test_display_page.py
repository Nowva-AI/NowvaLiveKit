"""Checks on the display page's set report for exercises other than the squat:
every press and RDL fault has a focus line, and depth tiles and the form score
are hidden when an exercise has none."""

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

    def test_null_depth_hides_its_tiles(self) -> None:
        page = DISPLAY_PATH.read_text()
        assert 'parentElement.classList.toggle("hidden", summary.avg_depth == null)' in page
        assert "#report .tile.hidden { display: none; }" in page
