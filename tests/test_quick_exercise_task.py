"""
Tests for quick-exercise instruction building with prefilled parameters.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent.agents.quickExerciseAgent as quick_exercise_module
from agent.agents.quickExerciseAgent import CollectExerciseInfoTask, build_task_instructions
from agent.core.agent_state import AgentState


class _StubAgent:
    def __init__(self, state=None, userdata=None):
        self.state = state
        self.userdata = userdata


class _StubWorkoutAgent(_StubAgent):
    pass


class _StubTeachingAgent(_StubAgent):
    pass


def _run_start_workout(
    state: AgentState, calibration_profile: dict | None, exercise_name: str = "squat",
):
    async def _run():
        task = CollectExerciseInfoTask(
            exercise_name=exercise_name,
            user_id="test-user",
            state=state,
            userdata=object(),
        )
        return await task.start_workout(sets=2, reps=4, weight=0.0, rest_seconds=30)

    async def _fake_check_calibration(user_id: str, exercise_name: str):
        return calibration_profile

    return _fake_check_calibration, _run


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentState:
    monkeypatch.setattr(
        AgentState, "_load_user_from_database", lambda self, user_id: None
    )
    return AgentState(state_dir=tmp_path)


class TestStartWorkoutCalibrationFlag:
    def test_stale_calibration_flag_cleared_when_profile_found(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ):
        # Regression: a session killed mid-calibration leaves calibration.active=True
        # in the persisted state. When the next quick exercise finds a calibration
        # profile in the DB, the stale flag must be cleared — otherwise main.py
        # launches the pipeline in assessment mode while the voice side runs a
        # normal workout set.
        state.set("calibration.active", True)
        state.set("calibration.pending_workout", {"type": "quick_exercise"})

        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)
        fake_check, run = _run_start_workout(state, calibration_profile={"depth": {}})
        monkeypatch.setattr(quick_exercise_module, "check_calibration", fake_check)

        result = asyncio.run(run())

        assert isinstance(result, _StubWorkoutAgent)
        assert not state.get("calibration.active")
        assert state.get("workout.calibration_profile") == {"depth": {}}
        assert state.get_mode() == "workout"

    def test_no_profile_enters_calibration_mode(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)
        fake_check, run = _run_start_workout(state, calibration_profile=None)
        monkeypatch.setattr(quick_exercise_module, "check_calibration", fake_check)

        result = asyncio.run(run())

        assert isinstance(result, _StubTeachingAgent)
        assert state.get("calibration.active") is True
        assert state.get_mode() == "workout"

    def test_uncalibrated_exercise_skips_squat_teaching_and_calibration(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ):
        """Calibrating an overhead press would store its reps as the squat calibration."""
        state.set("calibration.active", True)
        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)
        fake_check, run = _run_start_workout(
            state, calibration_profile=None, exercise_name="Barbell Overhead Press",
        )
        monkeypatch.setattr(quick_exercise_module, "check_calibration", fake_check)

        result = asyncio.run(run())

        assert isinstance(result, _StubWorkoutAgent)
        assert not state.get("calibration.active")
        assert state.get("workout.exercise_name") == "Barbell Overhead Press"


class TestBuildTaskInstructions:
    def test_nothing_prefilled_asks_for_everything(self):
        instructions = build_task_instructions("squat", None, None, None, None)
        assert "number of sets" in instructions
        assert "reps per set" in instructions
        assert "weight in lbs" in instructions
        assert "rest between sets" in instructions
        assert "ALREADY provided" not in instructions

    def test_partial_prefill_only_asks_for_missing(self):
        instructions = build_task_instructions("squat", 2, 3, None, 30)
        assert "number of sets = 2" in instructions
        assert "reps per set = 3" in instructions
        assert "rest between sets in seconds = 30" in instructions
        assert "Do NOT ask for these again" in instructions
        assert "Conversationally collect ONLY the remaining details" in instructions
        assert "weight in lbs" in instructions.split("remaining details:")[1]

    def test_all_prefilled_starts_immediately(self):
        instructions = build_task_instructions("squat", 2, 3, 0.0, 30)
        assert "Call the start_workout tool IMMEDIATELY" in instructions
        assert "Conversationally collect" not in instructions

    def test_bodyweight_zero_counts_as_provided(self):
        instructions = build_task_instructions("squat", None, None, 0.0, None)
        assert "weight in lbs (0 for bodyweight) = 0.0" in instructions
        assert "Do NOT ask for these again" in instructions
