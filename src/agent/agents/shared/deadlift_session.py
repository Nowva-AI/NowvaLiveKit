"""Deadlift session helpers for the voice agents: the coaching-ready gate, the session
metadata (grip, plates, belt, shoes) the pipeline receives, the first-session setup
briefing and the form-check findings (docs/deadlift/PLAN.md §4.4, §4.6;
.claude/deadlift/CONTRACT.md). Every helper answers "not a deadlift" for any other
exercise, so squat sessions never see a deadlift line, tool or message field.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from agent.core.agent_state import AgentState
from biomechanics.faults.observability import OBSERVABLE

DEADLIFT_PROFILE_NAME = "deadlift"
EXERCISE_META_STATE_KEY = "workout.exercise_meta"
DEFAULT_PLATE_DIAMETER_CM = 45.0

GripType = Literal["double", "mixed", "hook"]
ShoeType = Literal["flat", "barefoot", "heeled", "cushioned"]

SETUP_STEPS: tuple[str, ...] = (
    "feet hip-width with the bar over the middle of the foot",
    "grip just outside the legs",
    "shins to the bar",
    "shoulders over the bar",
    "pull the slack out, then push the floor away",
)

SETUP_QUESTION = (
    "Ask in one short question about their deadlift setup: their grip (double overhand, "
    "mixed or hook), belt or no belt, and their shoes. Plates are standard 45 centimetre "
    "plates unless they mention smaller ones. If they don't know or would rather skip it, "
    "go on without it."
)

# Cue tiers from lowest to highest (CONTRACT §2 min_tier); a "recap" fault is never cued.
CUE_TIERS: tuple[str, ...] = ("mild", "moderate", "severe")
RECAP_TIER = "recap"

# What the athlete would call each fault: behaviour, never anatomy or a back-shape claim.
FAULT_LABELS: dict[str, str] = {
    "deadlift_bar_position": "the bar not starting over the middle of the foot",
    "deadlift_setup_hips": "hips set at the wrong height before the pull",
    "deadlift_shoulders_behind": "shoulders behind the bar at the start",
    "deadlift_hips_shoot": "hips rising before the chest off the floor",
    "deadlift_bar_drift": "the bar drifting away from the legs",
    "deadlift_lockout": "not finishing standing tall",
    "deadlift_lean_back": "leaning back at the top",
    "deadlift_hip_shift": "hips sliding to one side",
    "deadlift_bar_tilt": "the bar tilting",
    "deadlift_bent_arms": "arms bending during the pull",
    "deadlift_velocity_loss": "the bar slowing down",
}

# Faults whose details.direction says which way to fix them (CONTRACT §1).
DIRECTION_LABELS: dict[tuple[str, str], str] = {
    ("deadlift_bar_position", "forward"): "the bar starting out past the middle of the foot, so step closer",
    ("deadlift_bar_position", "back"): "the bar starting too close, back toward the shins",
    ("deadlift_setup_hips", "up"): "hips set too low before the pull",
    ("deadlift_setup_hips", "down"): "hips set too high before the pull",
}


class DeadliftSessionMeta(BaseModel):
    grip: GripType | None = None
    plate_diameter_cm: float = DEFAULT_PLATE_DIAMETER_CM
    belt: bool | None = None
    shoes: ShoeType | None = None


def _measured(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _reaches_min_tier(fault: dict) -> bool:
    min_tier = (fault.get("details") or {}).get("min_tier", CUE_TIERS[0])
    if min_tier == RECAP_TIER or fault.get("severity") not in CUE_TIERS:
        return False
    return CUE_TIERS.index(fault["severity"]) >= CUE_TIERS.index(min_tier)


def _fault_phrase(fault: dict) -> str:
    fault_type = fault["fault_type"]
    direction = (fault.get("details") or {}).get("direction")
    return DIRECTION_LABELS.get(
        (fault_type, direction), FAULT_LABELS.get(fault_type, fault_type.replace("_", " "))
    )


def is_coached_deadlift(exercise_name: str | None) -> bool:
    """The exercise is the conventional deadlift and the deadlift is coaching-ready (until
    validation, only under NOWVA_DEV_COACHING_READY). A gated deadlift runs untracked, so
    the agent treats it like any exercise without a profile."""
    if not exercise_name:
        return False
    from biomechanics.profiles import find_profile_class

    profile_class = find_profile_class(exercise_name)
    return (
        profile_class is not None
        and profile_class.name == DEADLIFT_PROFILE_NAME
        and profile_class.coaching_ready
    )


def session_exercise_names(state: AgentState) -> list[str]:
    session_data = state.get("workout.current_session") or {}
    return [exercise.get("exercise_name", "") for exercise in session_data.get("exercises") or []]


def session_includes_coached_deadlift(state: AgentState) -> bool:
    return any(is_coached_deadlift(name) for name in session_exercise_names(state))


def store_exercise_meta(state: AgentState, meta: DeadliftSessionMeta) -> None:
    state.set(EXERCISE_META_STATE_KEY, meta.model_dump(exclude_none=True))


def exercise_meta_for(state: AgentState, exercise_name: str | None) -> dict | None:
    """The session metadata the pipeline gets with this exercise: the deadlift's dict
    (empty if it was never asked), or None for any other exercise, whose messages stay
    unchanged."""
    if not is_coached_deadlift(exercise_name):
        return None
    return state.get(EXERCISE_META_STATE_KEY) or {}


def first_session_briefing_instructions() -> str:
    steps = "; ".join(f"{number}. {step}" for number, step in enumerate(SETUP_STEPS, start=1))
    return (
        "[CONTEXT] This is the athlete's first deadlift session with you.\n\n"
        f"Walk them through the setup in five short steps, in this order: {steps}. "
        "Then suggest doing the first set with just the empty bar, so you can learn their "
        "setup before they load it. Calm and direct, plain words, one short sentence per step."
    )


def deadlift_form_findings(rep_message: dict | None) -> list[str]:
    """What the last deadlift rep showed, from its features and the faults that reached
    their minimum cue tier: what was right first, then what to change. A fault recorded
    below its tier is not relayed, and its measurement is not praised either. Empty
    before the first deadlift rep."""
    features = (rep_message or {}).get("features") or {}
    if "dl_schema" not in features:
        return []
    recorded = rep_message.get("faults_detailed") or []
    recorded_types = {fault.get("fault_type") for fault in recorded}
    faults = [
        fault
        for fault in recorded
        if (fault.get("details") or {}).get("observability", OBSERVABLE) == OBSERVABLE
        and _reaches_min_tier(fault)
    ]

    findings: list[str] = []
    if "deadlift_bar_position" not in recorded_types and _measured(features.get("bar_midfoot_setup_cm")):
        findings.append("the bar started over the middle of their foot")
    if "deadlift_bar_drift" not in recorded_types and _measured(features.get("bar_drift_cm")):
        findings.append("the bar stayed close to their legs on the way up")
    velocity_mps = features.get("concentric_velocity_mps")
    if _measured(velocity_mps):
        findings.append(f"the bar came up at about {velocity_mps:.1f} metres per second")
    seen: set[str] = set()
    for fault in faults:
        if fault["fault_type"] not in seen:
            seen.add(fault["fault_type"])
            findings.append(f"their last rep showed {_fault_phrase(fault)}")
    if not findings:
        findings.append("their last rep had nothing to correct in what the cameras measure")
    return findings
