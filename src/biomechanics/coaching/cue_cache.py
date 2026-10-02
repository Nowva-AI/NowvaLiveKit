"""
Cue Cache for Real-Time Coaching

Manages exercise-specific audio cue lookups with rate-limiting to prevent
overwhelming the lifter. Cues are pre-cached per exercise so the voice agent
can pre-generate TTS and play them with minimal latency on fault detection.
Which cues an exercise has, and which fault triggers which cue, comes from
its exercise profile; the squat profile uses SQUAT_CUES and FAULT_TO_CUE_MAP.
"""

import random
from typing import Dict, Optional

from biomechanics.config import CoachingConfig


# =============================================================================
# CUE DICTIONARIES
# =============================================================================

def _build_cues(**named_cues: str) -> Dict[str, str]:
    """Build a cue dict with named cues + rep_1..rep_20."""
    cues = {k: k for k in named_cues}
    for i in range(1, 21):
        cues[f"rep_{i}"] = f"rep_{i}"
    return cues


def build_cue_dict(*cue_keys: str) -> Dict[str, str]:
    """Cue dict for an exercise profile: the given keys plus rep_1..rep_20."""
    return _build_cues(**{key: key for key in cue_keys})


# Praise that fits any lift. great_depth is squat-only.
GENERIC_POSITIVE_CUE_KEYS = ("good_rep", "strong", "clean", "perfect")


SQUAT_CUES: Dict[str, str] = _build_cues(
    # Corrections. A _left/_right variant names the side; the base key is
    # the fallback when no side is known.
    knees_out="knees_out",
    knees_out_left="knees_out_left",
    knees_out_right="knees_out_right",
    chest_up="chest_up",
    heels_down="heels_down",
    heels_down_left="heels_down_left",
    heels_down_right="heels_down_right",
    whole_foot="whole_foot",
    even_it_out="even_it_out",
    even_it_out_left="even_it_out_left",
    even_it_out_right="even_it_out_right",
    level_bar="level_bar",
    deeper="deeper",
    square_feet="square_feet",
    square_feet_left="square_feet_left",
    square_feet_right="square_feet_right",
    lockout="lockout",
    slow_down="slow_down",
    same_depth="same_depth",
    drive="drive",
    brace="brace",
    # Intra-set stance / toe-out coaching
    stance_explain="stance_explain",
    stance_wider="stance_wider",
    stance_narrower="stance_narrower",
    toe_out_explain="toe_out_explain",
    toe_out_more="toe_out_more",
    toe_out_less="toe_out_less",
    adjust_good="adjust_good",
    # Positive reinforcement
    good_rep="good_rep",
    great_depth="great_depth",
    strong="strong",
    clean="clean",
    perfect="perfect",
)

# The squat profile's fault -> cue map. Other profiles bring their own.
FAULT_TO_CUE_MAP: Dict[str, str] = {
    # Squat
    "knee_valgus": "knees_out",
    "hip_shoot": "chest_up",
    "heel_rise": "heels_down",
    "balance": "whole_foot",
    "hip_shift": "even_it_out",
    "bilateral_asymmetry": "level_bar",
    "depth": "deeper",
    "foot_placement": "square_feet",
    "lockout": "lockout",
    "tempo": "slow_down",
    "depth_drift": "same_depth",
    "velocity_loss": "drive",
    # Other profiles
    "forward_lean": "chest_up",
    "back_rounding": "chest_up",
}

SIDE_CUE_SIDES = ("left", "right")

POSITIVE_CUE_KEYS = frozenset({"good_rep", "great_depth", "strong", "clean", "perfect"})

# Which fault wins when several compete for the same cue slot — lower first.
# Knee cave and hip shoot lead: they are what a coach watching the squat
# corrects first. Foot and balance faults follow, then depth and setup,
# then the fatigue signals. Faults not listed rank after all of these.
FAULT_CUE_PRIORITY: Dict[str, int] = {
    "knee_valgus": 0,
    "hip_shoot": 1,
    "heel_rise": 2,
    "balance": 3,
    "hip_shift": 4,
    "bilateral_asymmetry": 5,
    "depth": 6,
    "foot_placement": 7,
    "lockout": 8,
    "tempo": 9,
    "depth_drift": 10,
    "velocity_loss": 11,
}
DEFAULT_FAULT_CUE_PRIORITY = 12

# A higher-priority fault may jump the cue gap, but must still wait out
# a fraction of it to avoid back-to-back audio. The floor differs by
# caller: the pipeline cue_cache only assigns cue keys (no audio), so
# co-fired faults from the same detection frame should let the highest
# priority win immediately (floor_ratio=0). The orchestrator actually
# plays audio, so it keeps a small floor to space out speech.
PREEMPT_GAP_RATIO_PIPELINE = 0.0
PREEMPT_GAP_RATIO_ORCHESTRATOR = 0.0


def fault_cue_priority(fault_type: str) -> int:
    return FAULT_CUE_PRIORITY.get(fault_type, DEFAULT_FAULT_CUE_PRIORITY)


def base_cue_key(cue_key: str) -> str:
    """The side-neutral key for a side variant (knees_out_left -> knees_out)."""
    for side in SIDE_CUE_SIDES:
        suffix = f"_{side}"
        if cue_key.endswith(suffix):
            return cue_key[:-len(suffix)]
    return cue_key


def can_cue_fault(
    elapsed: float, gap: float, priority: int, last_priority: int,
    preempt_floor_ratio: float = PREEMPT_GAP_RATIO_PIPELINE,
) -> bool:
    """Whether a fault may claim the cue slot this soon after the last one."""
    if elapsed >= gap:
        return True
    if priority >= last_priority:
        return False
    return elapsed >= gap * preempt_floor_ratio


# =============================================================================
# CUE CACHE
# =============================================================================

class CueCache:
    """
    Manages audio cue lookups with rate-limiting.

    Prepares exercise-specific cues and provides fault-to-cue mapping
    with a minimum gap between cues to avoid overwhelming the lifter.
    """

    def __init__(self, config: Optional[CoachingConfig] = None):
        config = config or CoachingConfig()
        self.current_exercise: Optional[str] = None
        self.profile_name: Optional[str] = None
        self.cues: Dict[str, str] = {}
        self.fault_to_cue: Dict[str, str] = {}
        self.last_cue_time: float = 0.0
        self.last_cue_priority: int = DEFAULT_FAULT_CUE_PRIORITY
        self.min_cue_gap: float = config.min_cue_gap_seconds

    def prepare_for_exercise(self, exercise_name: str) -> Dict[str, str]:
        """
        Load cues for an exercise. Returns the cue dict for IPC transmission.

        Args:
            exercise_name: Exercise name (e.g. "Barbell Back Squat")

        Returns:
            Dict mapping cue keys to cue identifiers
        """
        # Profiles import this module for the squat maps, so resolve lazily.
        from biomechanics.profiles import get_profile

        profile = get_profile(exercise_name)
        self.current_exercise = exercise_name.lower().replace(" ", "_")
        self.profile_name = profile.name
        self.cues = profile.get_cue_dict()
        self.fault_to_cue = profile.get_fault_to_cue_map()
        self.last_cue_time = 0.0
        self.last_cue_priority = DEFAULT_FAULT_CUE_PRIORITY
        return dict(self.cues)

    def get_cue_for_fault(
        self, fault_type: str, timestamp: float, side: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a cue key for a detected fault, respecting rate limiting.

        Args:
            fault_type: The fault type string (e.g. "knee_valgus")
            timestamp: Current time in seconds
            side: "left" or "right" picks the side-specific cue when the
                exercise has one; anything else gets the base cue

        Returns:
            Cue key string if available and not rate-limited, else None
        """
        priority = fault_cue_priority(fault_type)
        if not can_cue_fault(
            timestamp - self.last_cue_time,
            self.min_cue_gap,
            priority,
            self.last_cue_priority,
        ):
            return None

        cue_key = self.fault_to_cue.get(fault_type)
        if cue_key is None or cue_key not in self.cues:
            return None
        if side in SIDE_CUE_SIDES and f"{cue_key}_{side}" in self.cues:
            cue_key = f"{cue_key}_{side}"

        self.last_cue_time = timestamp
        self.last_cue_priority = priority
        return cue_key

    def get_rep_cue(self, rep_number: int) -> Optional[str]:
        """Get the cue key for a rep count callout."""
        key = f"rep_{rep_number}"
        return key if key in self.cues else None

    def get_positive_cue(self) -> Optional[str]:
        """Get a random positive reinforcement cue key."""
        available = [k for k in self.cues if k in POSITIVE_CUE_KEYS]
        return random.choice(available) if available else None
