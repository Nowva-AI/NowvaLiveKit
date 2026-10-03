"""
Exercise Profile Registry

Maps exercise names to ExerciseProfile subclasses.
Use @register_profile("name") to register a profile, and get_profile()
to look one up by exercise name.

Dev override: NOWVA_DEV_COACHING_READY=<profile>[,<profile>] marks those
profiles coaching-ready, computed once at import as the class attribute so
every reader (menu, registry, gating) sees the same value.
"""

import logging
import os
import re
from typing import Dict, FrozenSet, List, Optional, Type

from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.untracked import UntrackedProfile

logger = logging.getLogger(__name__)

PROFILE_REGISTRY: Dict[str, Type[ExerciseProfile]] = {}

DEV_COACHING_READY_ENV = "NOWVA_DEV_COACHING_READY"


def dev_coaching_ready_names() -> FrozenSet[str]:
    """Profile names the dev override marks coaching-ready."""
    raw = os.getenv(DEV_COACHING_READY_ENV, "")
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def register_profile(*names: str):
    """Decorator to register a profile class under one or more names.

    Usage:
        @register_profile("squat", "back_squat", "front_squat")
        class SquatProfile(ExerciseProfile):
            ...
    """
    def decorator(cls: Type[ExerciseProfile]) -> Type[ExerciseProfile]:
        for name in names:
            PROFILE_REGISTRY[name] = cls
        if cls.name in dev_coaching_ready_names():
            cls.coaching_ready = True
        return cls
    return decorator


def _normalize(exercise_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", exercise_name.lower()).strip("_")


def find_profile_class(exercise_name: str) -> Optional[Type[ExerciseProfile]]:
    """The profile class for an exercise name, or None when no profile models it.

    An exact registered name wins. Otherwise the longest registered name
    that appears in the exercise name as whole words, so
    "Barbell Romanian Deadlift" is a Romanian deadlift rather than a deadlift
    and "Barbell Bench Press" matches nothing.
    """
    normalized = _normalize(exercise_name)
    if normalized in PROFILE_REGISTRY:
        return PROFILE_REGISTRY[normalized]
    padded = f"_{normalized}_"
    matches = [key for key in PROFILE_REGISTRY if f"_{key}_" in padded]
    if not matches:
        return None
    return PROFILE_REGISTRY[max(matches, key=len)]


def _gated_stand_in(profile_class: Type[ExerciseProfile]) -> UntrackedProfile:
    """Untracked, but keeping the gated profile's safety flags, so its frames
    never feed what that exercise would have kept them out of."""
    stand_in = UntrackedProfile()
    stand_in.allows_camera_refine = profile_class.allows_camera_refine
    stand_in.feeds_body_calibration = profile_class.feeds_body_calibration
    return stand_in


def get_profile(exercise_name: str) -> ExerciseProfile:
    """Instantiate the profile for an exercise.

    An exercise no profile models gets UntrackedProfile: the camera still
    runs, but no reps are counted and no faults fire. It never borrows the
    squat's rules. A profile that gates itself until coaching-ready
    (gate_until_ready) is served the same way, so a scheduled program never
    runs its unvalidated rules.
    """
    profile_class = find_profile_class(exercise_name)
    if profile_class is None:
        logger.warning(
            "[PROFILES] No profile for '%s' — untracked: no rep counting or faults",
            exercise_name,
        )
        return UntrackedProfile()
    if profile_class.gate_until_ready and not profile_class.coaching_ready:
        logger.warning(
            "[PROFILES] '%s' -> %s is not coaching-ready — untracked: no rep counting or faults",
            exercise_name, profile_class.__name__,
        )
        return _gated_stand_in(profile_class)
    logger.info("[PROFILES] '%s' -> %s", exercise_name, profile_class.__name__)
    return profile_class()


def coaching_ready_profiles() -> List[Type[ExerciseProfile]]:
    """Profiles Nova offers for camera coaching, in registration order."""
    ready: List[Type[ExerciseProfile]] = []
    for profile_class in PROFILE_REGISTRY.values():
        if profile_class.coaching_ready and profile_class not in ready:
            ready.append(profile_class)
    return ready
