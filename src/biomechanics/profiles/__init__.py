"""
Exercise Profiles

Modular exercise-specific configuration for the biomechanics pipeline.
Each profile bundles fault rules, rep signal, cues, and calibration
for a specific exercise or movement pattern.
"""

from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import (
    PROFILE_REGISTRY,
    coaching_ready_profiles,
    find_profile_class,
    get_profile,
    register_profile,
)
from biomechanics.profiles.untracked import UntrackedProfile

# Import profile modules to trigger @register_profile decorators
import biomechanics.profiles.squat  # noqa: F401
import biomechanics.profiles.deadlift  # noqa: F401
import biomechanics.profiles.romanian_deadlift  # noqa: F401
import biomechanics.profiles.lunge  # noqa: F401
import biomechanics.profiles.bulgarian_split_squat  # noqa: F401
import biomechanics.profiles.overhead_press  # noqa: F401
import biomechanics.profiles.barbell_row  # noqa: F401
import biomechanics.profiles.barbell_curl  # noqa: F401
import biomechanics.profiles.overhead_tricep_extension  # noqa: F401
import biomechanics.profiles.skull_crusher  # noqa: F401

__all__ = [
    "ExerciseProfile",
    "UntrackedProfile",
    "coaching_ready_profiles",
    "find_profile_class",
    "get_profile",
    "register_profile",
    "PROFILE_REGISTRY",
]
