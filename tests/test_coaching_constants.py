"""Cue text for the overhead press and the Romanian deadlift: every key the plans
name has its banner text, report label and praise, and the placeholders are gone."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.coaching_constants import (  # noqa: E402
    BETTER_CUE_TEXT,
    CUE_DISPLAY_LABELS,
    CUE_TEXT_MAP,
    FIXED_CUE_TEXT,
    PREEMPTIVE_TEXT,
)

PRESS_CORRECTIONS = [
    "press_layback", "press_bar_forward", "press_lockout", "press_lockout_left",
    "press_lockout_right", "press_leg_drive", "press_uneven", "press_uneven_left",
    "press_uneven_right", "press_velocity_loss",
]
RDL_CORRECTIONS = [
    "rdl_hips_back", "rdl_bar_close", "rdl_finish_tall", "rdl_level_bar",
    "rdl_same_depth", "rdl_slow_down", "rdl_lower",
]
SETUP_KEYS = ["press_setup", "rdl_setup", "rdl_back_note", "level_camera"]
PRESS_FIXED = [
    "press_layback_fixed", "press_bar_forward_fixed", "press_lockout_fixed",
    "press_leg_drive_fixed", "press_uneven_fixed", "press_velocity_loss_fixed",
]
RDL_FIXED = [
    "rdl_hips_back_fixed", "rdl_bar_close_fixed", "rdl_finish_tall_fixed",
    "rdl_level_bar_fixed", "rdl_same_depth_fixed", "rdl_slow_down_fixed",
]
PLAN_BETTER_KEYS = {
    "press_leg_drive_better", "press_uneven_better", "press_lockout_better",
    "press_bar_forward_better", "press_velocity_loss_better",
    "rdl_hips_back_better", "rdl_bar_close_better", "rdl_finish_tall_better",
}
REPLACED_PLACEHOLDERS = ["press_elbows", "press_bar_path", "press_even", "rdl_flat_back", "rdl_even"]
MAX_EXERCISE_CUE_WORDS = 6


def _spoken_words(text: str) -> int:
    return sum(1 for token in text.split() if any(char.isalnum() for char in token))


class TestExerciseCueText:
    def test_every_plan_key_has_text_and_a_label(self) -> None:
        for key in PRESS_CORRECTIONS + RDL_CORRECTIONS + SETUP_KEYS + PRESS_FIXED + RDL_FIXED:
            assert CUE_TEXT_MAP.get(key), key
            assert CUE_DISPLAY_LABELS.get(key), key

    def test_corrections_fit_the_word_limit(self) -> None:
        for key in PRESS_CORRECTIONS + RDL_CORRECTIONS:
            assert _spoken_words(CUE_TEXT_MAP[key]) <= MAX_EXERCISE_CUE_WORDS, key

    def test_side_variants_name_their_side(self) -> None:
        for base in ("press_lockout", "press_uneven"):
            for side in ("left", "right"):
                assert side in CUE_TEXT_MAP[f"{base}_{side}"].lower()

    def test_fix_praise_exists_for_every_cued_fault(self) -> None:
        for key in PRESS_FIXED + RDL_FIXED:
            assert key in FIXED_CUE_TEXT

    def test_rdl_lower_has_no_fix_praise(self) -> None:
        assert "rdl_lower_fixed" not in FIXED_CUE_TEXT


class TestBetterCueText:
    def test_exactly_the_plans_better_keys(self) -> None:
        assert set(BETTER_CUE_TEXT) == PLAN_BETTER_KEYS

    def test_each_line_says_better_and_is_short(self) -> None:
        for key, text in BETTER_CUE_TEXT.items():
            assert text.startswith("Better"), key
            assert _spoken_words(text) <= MAX_EXERCISE_CUE_WORDS, key

    def test_better_lines_are_spoken_and_labelled(self) -> None:
        for key, text in BETTER_CUE_TEXT.items():
            assert CUE_TEXT_MAP[key] == text
            assert CUE_DISPLAY_LABELS[key] == "Better"

    def test_no_better_key_for_layback(self) -> None:
        assert "press_layback_better" not in BETTER_CUE_TEXT


class TestPreemptiveText:
    def test_keyed_by_a_real_correction(self) -> None:
        for key in PREEMPTIVE_TEXT:
            assert key in PRESS_CORRECTIONS + RDL_CORRECTIONS, key

    def test_squat_has_no_preemptive_lines(self) -> None:
        assert all(key.startswith(("press_", "rdl_")) for key in PREEMPTIVE_TEXT)


class TestReplacedPlaceholders:
    def test_placeholder_keys_are_gone(self) -> None:
        for key in REPLACED_PLACEHOLDERS:
            assert key not in CUE_TEXT_MAP, key
            assert key not in CUE_DISPLAY_LABELS, key
