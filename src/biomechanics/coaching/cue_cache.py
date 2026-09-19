"""
Cue Cache for Real-Time Coaching

Manages exercise-specific audio cue lookups with rate-limiting to prevent
overwhelming the lifter. Cues are pre-cached per exercise so the voice agent
can pre-generate TTS and play them with minimal latency on fault detection.
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


SQUAT_CUES: Dict[str, str] = _build_cues(
    # Corrections
    knees_out="knees_out",
    chest_up="chest_up",
    deeper="deeper",
    heels_down="heels_down",
    even_it_out="even_it_out",
    slow_down="slow_down",
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

DEADLIFT_CUES: Dict[str, str] = _build_cues(
    # Corrections
    hips_through="hips_through",
    flat_back="flat_back",
    chest_up="chest_up",
    lockout="lockout",
    slow_down="slow_down",
    # Positive reinforcement
    good_rep="good_rep",
    strong="strong",
)

DEFAULT_CUES: Dict[str, str] = _build_cues(
    good_rep="good_rep",
    chest_up="chest_up",
    brace="brace",
    slow_down="slow_down",
    strong="strong",
)

EXERCISE_CUE_MAP: Dict[str, Dict[str, str]] = {
    "squat": SQUAT_CUES,
    "back_squat": SQUAT_CUES,
    "barbell_back_squat": SQUAT_CUES,
    "barbell_front_squat": SQUAT_CUES,
    "front_squat": SQUAT_CUES,
    "goblet_squat": SQUAT_CUES,
    "bodyweight_squat": SQUAT_CUES,
    "deadlift": DEADLIFT_CUES,
    "romanian_deadlift": DEADLIFT_CUES,
    "sumo_deadlift": DEADLIFT_CUES,
    "barbell_deadlift": DEADLIFT_CUES,
    "barbell_romanian_deadlift": DEADLIFT_CUES,
}

FAULT_TO_CUE_MAP: Dict[str, str] = {
    "knee_valgus": "knees_out",
    "forward_lean": "chest_up",
    "depth": "deeper",
    "bilateral_asymmetry": "even_it_out",
    "back_rounding": "chest_up",
    "heel_rise": "heels_down",
}

POSITIVE_CUE_KEYS = frozenset({"good_rep", "great_depth", "strong", "clean", "perfect"})

# Which fault wins when several compete for the same cue slot — lower first.
# Forward lean leads because it is the root-cause fault: it drives the
# intra-set stance correction, and fixing it usually resolves the knee and
# asymmetry faults downstream. Left flat, the more frequent knee/asymmetry
# faults claim every slot and forward lean is never cued at all.
FAULT_CUE_PRIORITY: Dict[str, int] = {
    "forward_lean": 0,
    "knee_valgus": 1,
    "heel_rise": 2,
    "bilateral_asymmetry": 3,
}
DEFAULT_FAULT_CUE_PRIORITY = 4

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
        self.cues: Dict[str, str] = {}
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
        normalized = exercise_name.lower().replace(" ", "_")
        self.current_exercise = normalized

        # Exact match first, then substring match (e.g. "barbell_back_squat" contains "squat")
        cues = EXERCISE_CUE_MAP.get(normalized)
        if cues is None:
            for key, value in EXERCISE_CUE_MAP.items():
                if key in normalized:
                    cues = value
                    break
        self.cues = dict(cues or DEFAULT_CUES)
        self.last_cue_time = 0.0
        self.last_cue_priority = DEFAULT_FAULT_CUE_PRIORITY
        return dict(self.cues)

    def get_cue_for_fault(self, fault_type: str, timestamp: float) -> Optional[str]:
        """
        Get a cue key for a detected fault, respecting rate limiting.

        Args:
            fault_type: The fault type string (e.g. "knee_valgus")
            timestamp: Current time in seconds

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

        cue_key = FAULT_TO_CUE_MAP.get(fault_type)
        if cue_key is None or cue_key not in self.cues:
            return None

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
