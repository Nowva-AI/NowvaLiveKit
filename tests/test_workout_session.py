"""Tests for WorkoutSession: set loads with units, and recording RPE after a set."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.workout_session import WorkoutSession


def _workout_data(**set_overrides: object) -> dict:
    set_data = {
        "set_id": 11, "set_number": 1, "reps": 5, "intensity_percent": 75.0,
        "rpe": 8.0, "rest_seconds": 90,
    }
    set_data.update(set_overrides)
    return {
        "workout_id": 1,
        "workout_name": "Squat day",
        "exercises": [{
            "workout_exercise_id": 7, "exercise_id": 3,
            "exercise_name": "Barbell Back Squat", "order_number": 1,
            "sets": [set_data, {**set_data, "set_id": 12, "set_number": 2}],
        }],
    }


class TestSetLoad:
    def test_target_weight_is_the_load_not_the_intensity(self):
        session = WorkoutSession("u1", 5, _workout_data())
        current = session.get_current_set()
        assert current.target_weight is None
        assert current.intensity_percent == pytest.approx(75.0)

    def test_weight_and_unit_map_into_the_set(self):
        session = WorkoutSession("u1", 5, _workout_data(weight=60.0, weight_unit="kg"))
        current = session.get_current_set()
        assert current.target_weight == pytest.approx(60.0)
        assert current.weight_unit == "kg"

    def test_zero_weight_is_bodyweight(self):
        session = WorkoutSession("u1", 5, _workout_data(weight=0, weight_unit="lb"))
        assert session.get_current_set().target_weight == 0

    def test_round_trip_keeps_the_unit(self):
        session = WorkoutSession("u1", 5, _workout_data(weight=135.0, weight_unit="lb"))
        restored = WorkoutSession.from_dict(session.to_dict())
        assert restored.get_current_set().weight_unit == "lb"
        assert restored.get_current_set().target_weight == pytest.approx(135.0)

    def test_saved_state_without_a_unit_still_loads(self):
        saved = WorkoutSession("u1", 5, _workout_data()).to_dict()
        for set_dict in saved["exercises"][0]["sets"]:
            del set_dict["weight_unit"]
        restored = WorkoutSession.from_dict(saved)
        assert restored.get_current_set().weight_unit is None

    def test_quick_session_carries_the_unit(self):
        session = WorkoutSession.create_quick_session(
            "u1", "Barbell Back Squat", sets=2, reps=5, weight=60.0, weight_unit="kg",
        )
        assert session.get_current_set().weight_unit == "kg"

    def test_quick_session_weight_defaults_to_pounds(self):
        session = WorkoutSession.create_quick_session(
            "u1", "Barbell Back Squat", sets=2, reps=5, weight=45.0,
        )
        assert session.get_current_set().weight_unit == "lb"


class TestLastCompletedSet:
    def test_none_before_any_set_is_done(self):
        assert WorkoutSession("u1", 5, _workout_data()).last_completed_set() is None

    def test_is_the_set_just_finished_after_advancing(self):
        session = WorkoutSession("u1", 5, _workout_data())
        session.mark_set_complete(performed_reps=5)
        session.advance_to_next_set()
        last = session.last_completed_set()
        assert last is not None and last.set_number == 1

    def test_is_the_most_recent_of_several(self):
        session = WorkoutSession("u1", 5, _workout_data())
        session.mark_set_complete(performed_reps=5)
        session.advance_to_next_set()
        session.mark_set_complete(performed_reps=4)
        assert session.last_completed_set().set_number == 2
