"""
Shared helper functions extracted from the monolithic voice agent.
Used by multiple agent classes.
"""

import logging
import os
from typing import Optional
import asyncio

from db.database import SessionLocal

logger = logging.getLogger(__name__)


def service_headers() -> dict:
    """Headers for authenticated service-to-service API calls."""
    return {"X-Service-Key": os.getenv("SERVICE_API_KEY", "")}


def normalize_sex(raw_sex: str) -> Optional[str]:
    """Normalize a spoken sex value to 'male' or 'female'. Returns None if unclear."""
    sex_lower = raw_sex.lower().strip()
    if sex_lower in ("m", "male", "man", "boy"):
        return "male"
    if sex_lower in ("f", "female", "woman", "girl"):
        return "female"
    return None


def build_program_generation_payload(
    program_params: dict,
    user_id: str,
    name: Optional[str],
    email: Optional[str],
    send_email: bool,
) -> dict:
    """Build the request body for POST /api/programs/generate.

    equipment_tier is only included when the caller captured it, so callers
    that never set it keep the API's own default.
    """
    payload = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "height_cm": program_params["height_cm"],
        "weight_kg": program_params["weight_kg"],
        "age": program_params["age"],
        "sex": program_params["sex"],
        "goal_category": program_params["goal_category"],
        "goal_raw": program_params["goal_raw"],
        "duration_weeks": program_params["duration_weeks"],
        "days_per_week": program_params["days_per_week"],
        "session_duration": program_params.get("session_duration", 60),
        "injury_history": program_params.get("injury_history", "none"),
        "specific_sport": program_params.get("specific_sport", "none"),
        "user_notes": program_params.get("user_notes"),
        "fitness_level": program_params["fitness_level"],
        "has_vbt_capability": program_params.get("has_vbt_capability", False),
        "send_email": send_email,
        "training_season": program_params.get("training_season"),
        "games_per_week": program_params.get("games_per_week") or 0,
    }
    if "equipment_tier" in program_params:
        payload["equipment_tier"] = program_params["equipment_tier"]
    return payload


def normalize_exercise_name(raw_name: str) -> Optional[str]:
    """Normalize a casual exercise name to its formal library name.
    Returns None if the exercise isn't supported for biomechanics tracking.
    """
    name_lower = raw_name.lower().strip()

    EXERCISE_ALIASES = {
        "squat": "Barbell Back Squat",
        "squats": "Barbell Back Squat",
        "back squat": "Barbell Back Squat",
        "back squats": "Barbell Back Squat",
        "barbell squat": "Barbell Back Squat",
        "barbell back squat": "Barbell Back Squat",
        "front squat": "Barbell Front Squat",
        "front squats": "Barbell Front Squat",
        "barbell front squat": "Barbell Front Squat",
        "goblet squat": "Goblet Squat",
        "goblet squats": "Goblet Squat",
        "bodyweight squat": "Barbell Back Squat",
        "bodyweight squats": "Barbell Back Squat",
        "bw squat": "Barbell Back Squat",
        "bw squats": "Barbell Back Squat",
        "air squat": "Barbell Back Squat",
        "air squats": "Barbell Back Squat",
        "deadlift": "Barbell Deadlift",
        "deadlifts": "Barbell Deadlift",
        "barbell deadlift": "Barbell Deadlift",
        "romanian deadlift": "Romanian Deadlift",
        "rdl": "Romanian Deadlift",
        "sumo deadlift": "Sumo Deadlift",
        "bench": "Barbell Bench Press",
        "bench press": "Barbell Bench Press",
        "barbell bench press": "Barbell Bench Press",
        "overhead press": "Barbell Overhead Press",
        "ohp": "Barbell Overhead Press",
        "press": "Barbell Overhead Press",
        "military press": "Barbell Overhead Press",
    }

    return EXERCISE_ALIASES.get(name_lower)


async def check_calibration(user_id: str, exercise_name: str) -> Optional[dict]:
    from biomechanics.calibration import get_movement_pattern
    from db.calibration_utils import get_user_calibration

    canonical = normalize_exercise_name(exercise_name) or exercise_name
    pattern = get_movement_pattern(canonical)
    if not pattern:
        logger.info(
            f"[CALIBRATION] {exercise_name!r} (canonical={canonical!r}) is not "
            f"calibrated — calibration lookup skipped"
        )
        return None

    def _query():
        db = SessionLocal()
        try:
            return get_user_calibration(db, user_id, pattern)
        finally:
            db.close()

    return await asyncio.to_thread(_query)


def calibration_exercise(exercise_names: list[str]) -> Optional[str]:
    """The first exercise in a workout that is calibrated (a squat), or None.

    Its stored calibration, body measurements included, serves the whole
    workout. Assessment only runs when this is also the first exercise.
    """
    from biomechanics.calibration import get_movement_pattern

    return next((name for name in exercise_names if get_movement_pattern(name)), None)


def start_calibration_mode(state, exercise_name: str, pending_workout: dict):
    """Set state keys to trigger calibration in the pipeline subprocess."""
    from biomechanics.calibration import get_movement_pattern

    state.set("calibration.active", True)
    state.set("calibration.movement_pattern", get_movement_pattern(exercise_name))
    state.set("calibration.pending_workout", pending_workout)


def get_recommended_duration(goal_category: str) -> int:
    """Get recommended program duration based on goal category."""
    if goal_category == "power":
        return 6
    elif goal_category == "strength":
        return 10
    else:  # hypertrophy
        return 12


def normalize_fitness_level(level_str: str) -> str:
    """Normalize fitness level to beginner, intermediate, or advanced."""
    level_lower = level_str.lower()

    beginner_keywords = ["beginner", "new", "just starting", "never", "first time", "noob"]
    advanced_keywords = ["advanced", "experienced", "years", "competitive", "athlete", "expert"]

    if any(kw in level_lower for kw in beginner_keywords):
        return "beginner"
    elif any(kw in level_lower for kw in advanced_keywords):
        return "advanced"
    else:
        return "intermediate"


def should_enable_vbt(fitness_level: str, goal_category: str, specific_sport: str) -> bool:
    """Determine if VBT should be enabled based on training parameters.

    VBT is enabled when:
    1. Goal is power or athletic_performance (any fitness level)
    2. Goal is strength AND fitness level is advanced
    3. Sport involves explosiveness (intermediate or advanced)
    """
    # Rule 1: Power/athletic goals get VBT at any level
    if goal_category in ["power", "athletic_performance"]:
        if fitness_level == "beginner":
            logger.info(f"[VBT] Enabled (BASIC only): {goal_category} goal with beginner level — limited to basic power exercises")
        else:
            logger.info(f"[VBT] Enabled: {goal_category} goal with {fitness_level} level")
        return True

    # Rule 2: Hypertrophy-only goals don't use VBT
    if goal_category == "hypertrophy" and specific_sport == "none":
        logger.info("[VBT] Disabled: hypertrophy goal without sport context")
        return False

    # Rule 3: Beginners with non-power goals don't get VBT
    if fitness_level == "beginner":
        logger.info("[VBT] Disabled: beginner level with non-power goal (form mastery priority)")
        return False

    # Rule 4: Advanced strength athletes use VBT
    if goal_category == "strength" and fitness_level == "advanced":
        logger.info("[VBT] Enabled: advanced strength training")
        return True

    # Rule 5: Check if sport involves explosiveness (intermediate/advanced only)
    explosive_sports = [
        "sprinting", "track and field", "olympic weightlifting", "powerlifting",
        "basketball", "football", "volleyball", "jumping", "throwing",
        "baseball", "softball", "rugby", "hockey", "soccer", "lacrosse"
    ]

    sport_lower = specific_sport.lower() if specific_sport else ""
    if any(sport in sport_lower for sport in explosive_sports):
        logger.info(f"[VBT] Enabled: explosive sport ({specific_sport}) with {fitness_level} level")
        return True

    # Default: No VBT
    logger.info(f"[VBT] Disabled: {fitness_level} + {goal_category} + {specific_sport} doesn't warrant VBT")
    return False
