"""Tests for the durable athlete facts store (pain reports and preferences)."""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.agent_state import AgentState
from agent.services.athlete_facts import (
    MAX_PAIN_FLAGS,
    active_pain_flags,
    add_pain_flag,
    athlete_facts_line,
    resolve_pain_flag,
    safety_line,
    set_preference,
)


@pytest.fixture
def athlete_state(tmp_path: Path) -> AgentState:
    state = AgentState(state_dir=tmp_path)
    state.state["user"]["id"] = "athlete-1"
    return state


class TestPainFlags:
    def test_pain_flag_survives_a_new_session(self, athlete_state: AgentState, tmp_path: Path) -> None:
        add_pain_flag(athlete_state, "Left Knee", "feels off on the way up")
        reloaded = AgentState(state_dir=tmp_path)
        reloaded.state["user"]["id"] = "athlete-1"
        reloaded.load_state("athlete-1")
        flags = active_pain_flags(reloaded)
        assert [f["body_part"] for f in flags] == ["left knee"]
        assert flags[0]["note"] == "feels off on the way up"

    def test_same_body_part_refreshes_instead_of_duplicating(self, athlete_state: AgentState) -> None:
        add_pain_flag(athlete_state, "left knee", "a bit sore")
        add_pain_flag(athlete_state, "left knee", "sharp now")
        flags = active_pain_flags(athlete_state)
        assert len(flags) == 1
        assert flags[0]["note"] == "sharp now"

    def test_resolve_removes_only_that_body_part(self, athlete_state: AgentState) -> None:
        add_pain_flag(athlete_state, "left knee", "")
        add_pain_flag(athlete_state, "lower back", "tight")
        assert resolve_pain_flag(athlete_state, "Left knee") is True
        assert [f["body_part"] for f in active_pain_flags(athlete_state)] == ["lower back"]
        assert resolve_pain_flag(athlete_state, "shoulder") is False

    def test_flag_count_is_bounded(self, athlete_state: AgentState) -> None:
        for i in range(MAX_PAIN_FLAGS + 3):
            add_pain_flag(athlete_state, f"spot {i}", "")
        assert len(active_pain_flags(athlete_state)) == MAX_PAIN_FLAGS


class TestPromptLines:
    def test_no_facts_gives_no_lines(self, athlete_state: AgentState) -> None:
        assert athlete_facts_line(athlete_state) is None
        assert safety_line(athlete_state) is None

    def test_facts_line_names_pain_and_preferences(self, athlete_state: AgentState) -> None:
        add_pain_flag(athlete_state, "left knee", "feels off")
        set_preference(athlete_state, "humor", "none")
        line = athlete_facts_line(athlete_state)
        assert "left knee (feels off, today)" in line
        assert "humor: none" in line

    def test_older_pain_report_shows_its_date(self, athlete_state: AgentState) -> None:
        add_pain_flag(athlete_state, "left knee", "")
        later = date.today() + timedelta(days=3)
        line = athlete_facts_line(athlete_state, today=later)
        reported = datetime.now().date()
        assert f"since {reported.strftime('%b')} {reported.day}" in line

    def test_safety_line_lists_every_open_report(self, athlete_state: AgentState) -> None:
        add_pain_flag(athlete_state, "left knee", "")
        add_pain_flag(athlete_state, "lower back", "")
        line = safety_line(athlete_state)
        assert "left knee, lower back" in line
        assert "pushing through pain" in line

    def test_preference_update_keeps_one_entry_per_topic(self, athlete_state: AgentState) -> None:
        set_preference(athlete_state, "talk", "more")
        set_preference(athlete_state, "Talk", "less")
        assert athlete_facts_line(athlete_state).count("talk:") == 1
        assert "talk: less" in athlete_facts_line(athlete_state)
