"""
Tests for shared agent helpers: sex normalization, service headers,
calibration lookup, and the program generation payload builder.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent.agents.shared.helpers as helpers
from agent.agents.shared.helpers import (
    build_program_generation_payload,
    check_calibration,
    normalize_exercise_name,
    normalize_sex,
    service_headers,
)


def _website_program_params() -> dict:
    """Program params dict as the website agents hold it after apply_defaults."""
    return {
        "height_cm": 188.5,
        "weight_kg": 85.0,
        "age": 28,
        "sex": "male",
        "goal_category": "strength",
        "goal_raw": "get stronger",
        "duration_weeks": 12,
        "days_per_week": 4,
        "session_duration": 60,
        "injury_history": "none",
        "specific_sport": "none",
        "fitness_level": "intermediate",
        "has_vbt_capability": False,
        "training_season": None,
        "games_per_week": 0,
        "equipment_tier": 3,
    }


class TestNormalizeSex:
    def test_male_aliases(self):
        for alias in ("m", "male", "man", "boy", "M", " Male "):
            assert normalize_sex(alias) == "male"

    def test_female_aliases(self):
        for alias in ("f", "female", "woman", "girl", "F", " FEMALE "):
            assert normalize_sex(alias) == "female"

    def test_unclear_returns_none(self):
        assert normalize_sex("yes") is None
        assert normalize_sex("") is None


class TestNormalizeExerciseName:
    def test_plain_squat_defaults_to_bodyweight(self):
        for alias in ("squat", "squats", "Squat ", "bodyweight squat", "air squats", "bw squat"):
            assert normalize_exercise_name(alias) == "Bodyweight Squat"

    def test_bar_mention_maps_to_barbell_back_squat(self):
        for alias in ("back squat", "barbell squat", "barbell back squats"):
            assert normalize_exercise_name(alias) == "Barbell Back Squat"

    def test_bodyweight_squat_is_a_squat_movement_pattern(self):
        # Calibration is stored per movement pattern; a bodyweight squat must find
        # the same calibration as a barbell squat (biomechanics/calibration.py).
        from biomechanics.calibration import get_movement_pattern

        assert get_movement_pattern("Bodyweight Squat") == "squat"


class TestServiceHeaders:
    def test_reads_service_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SERVICE_API_KEY", "secret-key")
        assert service_headers() == {"X-Service-Key": "secret-key"}

    def test_empty_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        assert service_headers() == {"X-Service-Key": ""}


class TestCheckCalibration:
    """check_calibration must resolve casual exercise names ("squat") to the
    same movement pattern the save path uses, or saved calibrations are
    never found and every workout re-enters calibration mode."""

    @pytest.fixture
    def db_spy(self, monkeypatch):
        import db.calibration_utils as calibration_utils

        captured = {}
        sentinel = {"knee_valgus": {"mild": 10.0}}

        def fake_get_user_calibration(db, user_id, movement_pattern):
            captured["user_id"] = user_id
            captured["pattern"] = movement_pattern
            return sentinel

        class _FakeSession:
            def close(self):
                pass

        monkeypatch.setattr(calibration_utils, "get_user_calibration", fake_get_user_calibration)
        monkeypatch.setattr(helpers, "SessionLocal", _FakeSession)
        captured["sentinel"] = sentinel
        return captured

    def test_casual_name_reaches_db_lookup(self, db_spy):
        result = asyncio.run(check_calibration("user-1", "squat"))
        assert result == db_spy["sentinel"]
        assert db_spy["pattern"] == "squat"

    def test_canonical_name_reaches_db_lookup(self, db_spy):
        result = asyncio.run(check_calibration("user-1", "Barbell Back Squat"))
        assert result == db_spy["sentinel"]
        assert db_spy["pattern"] == "squat"

    def test_unknown_exercise_returns_none_without_query(self, db_spy):
        result = asyncio.run(check_calibration("user-1", "juggling"))
        assert result is None
        assert "pattern" not in db_spy

    def test_program_library_squat_name_reaches_db_lookup(self, db_spy):
        asyncio.run(check_calibration("user-1", "Dumbbell Goblet Squat"))
        assert db_spy["pattern"] == "squat"

    def test_uncalibrated_exercise_returns_none_without_query(self, db_spy):
        assert asyncio.run(check_calibration("user-1", "Barbell Romanian Deadlift")) is None
        assert "pattern" not in db_spy


class TestCalibrationExercise:
    def test_first_squat_in_the_workout_is_calibrated(self):
        names = ["Barbell Overhead Press", "Barbell Back Squat", "Barbell Front Squat"]
        assert helpers.calibration_exercise(names) == "Barbell Back Squat"

    def test_workout_without_a_squat_has_no_calibration(self):
        assert helpers.calibration_exercise(["Barbell Overhead Press", "Barbell Bench Press"]) is None


class TestBuildProgramGenerationPayload:
    def test_website_payload_matches_collected_params(self):
        payload = build_program_generation_payload(
            _website_program_params(),
            user_id="uuid-123",
            name="Marwan",
            email="user@example.com",
            send_email=True,
        )
        assert payload["user_id"] == "uuid-123"
        assert payload["name"] == "Marwan"
        assert payload["email"] == "user@example.com"
        assert payload["height_cm"] == pytest.approx(188.5)
        assert payload["weight_kg"] == pytest.approx(85.0)
        assert payload["send_email"] is True
        assert payload["equipment_tier"] == 3
        assert payload["games_per_week"] == 0
        assert payload["user_notes"] is None

    def test_optional_fields_fall_back_to_defaults(self):
        params = _website_program_params()
        for key in (
            "session_duration", "injury_history", "specific_sport",
            "has_vbt_capability", "training_season", "games_per_week",
        ):
            del params[key]
        payload = build_program_generation_payload(
            params, user_id="uuid-123", name=None, email=None, send_email=False,
        )
        assert payload["session_duration"] == 60
        assert payload["injury_history"] == "none"
        assert payload["specific_sport"] == "none"
        assert payload["has_vbt_capability"] is False
        assert payload["training_season"] is None
        assert payload["games_per_week"] == 0
        assert payload["send_email"] is False

    def test_equipment_tier_omitted_when_not_captured(self):
        # ProgramCreationAgent never captures equipment_tier; omitting the key
        # lets the API apply its own default (tier 1).
        params = _website_program_params()
        del params["equipment_tier"]
        payload = build_program_generation_payload(
            params, user_id="uuid-123", name="X", email="x@y.z", send_email=False,
        )
        assert "equipment_tier" not in payload

    def test_none_games_per_week_coerced_to_zero(self):
        params = _website_program_params()
        params["games_per_week"] = None
        payload = build_program_generation_payload(
            params, user_id="uuid-123", name="X", email="x@y.z", send_email=True,
        )
        assert payload["games_per_week"] == 0
