"""Compact squat knowledge for the voice coach, generated from the diagnosis graph.

The voice LLM answers technique questions ("why do my heels come up?") from this
card instead of free-styling, so it agrees with what the diagnosis engine detects
and prescribes, and it never claims to see what the cameras can't.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from livekit.agents import RunContext, function_tool

from biomechanics.faults.observability import (
    APPROXIMATE,
    NOT_OBSERVABLE,
    OBSERVABLE,
    SINGLE_CAMERA,
    capture_mode_from_env,
    measurement_observability,
)

GRAPH_DIR = Path(__file__).resolve().parents[2] / "biomechanics" / "diagnosis" / "graph"
OBSERVABILITY_TOPIC = "what_i_can_see"
MAX_CAUSES_PER_SYMPTOM = 4
MAX_SECTIONS_PER_ANSWER = 2
MIN_ID_KEYWORD_CHARS = 4

# What the athlete would call each symptom; a symptom missing here falls back to its id.
SYMPTOM_LABELS: dict[str, str] = {
    "excessive_trunk_lean": "leaning forward",
    "hip_shoot": "hips rising before the chest",
    "knee_not_tracking_toes": "knees caving in",
    "hip_shift": "hips sliding to one side",
    "depth_limit": "not reaching depth",
    "heel_rise": "heels lifting",
    "balance_forward": "weight tipping onto the toes",
    "velocity_loss": "reps slowing down",
    "incomplete_lockout": "not standing tall at the top",
    "fast_descent": "dropping too fast",
    "head_cranked_up": "looking up at the ceiling",
    "uneven_setup": "feet set up unevenly",
}

CAUSE_TIER_TAGS: dict[int, str] = {
    0: "build, not a fault",
    1: "fix today",
    2: "load or fatigue",
    3: "takes weeks",
}

# No keypoint model in use has spine or foot-arch landmarks (FINDINGS.md, part C).
NEVER_VISIBLE = (
    "lower-back tuck at the bottom (butt wink)",
    "upper-back rounding",
    "bracing",
    "foot arch collapse",
)

# Short answers to the questions athletes ask most, kept consistent with the graph.
FAQ: dict[str, str] = {
    "butt_wink": (
        "Lower-back tuck at the bottom (butt wink) is invisible to these cameras, so never say you saw it. "
        "A little tuck at the very bottom of a light squat is common; under heavy load, squat only as deep "
        "as the lower back stays neutral. If they feel it: brace before each rep, stop a little higher, and "
        "raise the heels if their ankles are tight. Ask how it feels."
    ),
    "go_heavier": (
        "Add weight when every rep of the last set matched the first (same depth, knees tracking, no big "
        "slowdown) and they had two or more reps left, an effort of eight or under. If form or speed fell "
        "off late in the set, or the effort was nine or ten, stay at this weight. Small jumps. Never add "
        "load while a pain report is open."
    ),
    "knees_past_toes": (
        "Knees traveling past the toes is normal and needed to reach depth; holding them back forces the "
        "chest further forward. What matters is that the knees track in line with the toes, not inward."
    ),
    "how_deep": (
        "Depth is personal: as deep as they can go with heels down, hips centered and the lower back "
        "neutral, which is the target set when their range was measured. Parallel means the hip crease at "
        "knee height. Never force depth past their range."
    ),
    "trunk_lean": (
        "Some forward lean is built in: the trunk angle that keeps the weight over midfoot depends on thigh "
        "and torso length and ankle mobility, so long thighs mean more lean. Only leaning more than their "
        "build needs is a fault, and only the three-camera rig can judge it."
    ),
}

# Words an athlete uses for each topic. A symptom added to the YAML later without an
# entry here is matched on the words of its id.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    OBSERVABILITY_TOPIC: ("see", "detect", "camera", "notice", "measure", "spot", "track", "round"),
    "butt_wink": ("butt wink", "wink", "tuck", "lower back", "tailbone", "pelvis"),
    "go_heavier": ("heavier", "more weight", "add weight", "load", "progress", "increase"),
    "knees_past_toes": ("past my toes", "past the toes", "over my toes", "over the toes"),
    "how_deep": ("deep", "depth", "parallel", "how low"),
    "trunk_lean": ("lean", "chest", "torso", "upright"),
    "excessive_trunk_lean": ("lean", "chest", "forward"),
    "hip_shoot": ("shoot", "hips rise", "hips come up", "good morning"),
    "knee_not_tracking_toes": ("knee", "cave", "caving", "valgus"),
    "hip_shift": ("shift", "one side", "lopsided"),
    "depth_limit": ("deep", "depth", "parallel"),
    "heel_rise": ("heel",),
    "balance_forward": ("toes", "tipping", "balance"),
    "velocity_loss": ("slow", "speed", "grind", "velocity"),
    "incomplete_lockout": ("lockout", "lock out", "stand tall", "top of"),
    "fast_descent": ("too fast", "drop", "control", "tempo", "descent"),
    "head_cranked_up": ("head", "neck", "look up", "gaze", "eyes"),
    "uneven_setup": ("feet", "foot", "stance", "stagger", "uneven"),
}


def _load_graph(filename: str) -> dict:
    with open(GRAPH_DIR / filename) as graph_file:
        return yaml.safe_load(graph_file)


def _label(symptom_id: str) -> str:
    return SYMPTOM_LABELS.get(symptom_id, symptom_id.replace("_", " "))


def _observability_section(symptoms: dict) -> str:
    labels_by_level: dict[str, list[str]] = {OBSERVABLE: [], APPROXIMATE: [], NOT_OBSERVABLE: []}
    for symptom_id, symptom in symptoms.items():
        level = measurement_observability(symptom["measurement"], SINGLE_CAMERA)
        labels_by_level[level].append(_label(symptom_id))
    return (
        f"One camera reads well: {', '.join(labels_by_level[OBSERVABLE])}. "
        f"One camera only roughly reads {', '.join(labels_by_level[APPROXIMATE])}; those are judged on "
        "the three-camera rig, so on one camera say the setup can't judge them. "
        f"Only the three-camera rig sees {', '.join(labels_by_level[NOT_OBSERVABLE])}. "
        f"No setup sees {', '.join(NEVER_VISIBLE)}: never claim to, ask how it feels instead."
    )


def _symptom_section(symptom_id: str, symptom: dict, causes: dict) -> str:
    ranked = sorted(symptom["candidate_causes"], key=lambda entry: entry["prior"], reverse=True)
    cause_texts = [
        f"{causes[entry['cause_id']]['description']} ({CAUSE_TIER_TAGS[causes[entry['cause_id']]['tier']]})"
        for entry in ranked[:MAX_CAUSES_PER_SYMPTOM]
        if entry["cause_id"] in causes
    ]
    rig_only = measurement_observability(symptom["measurement"], SINGLE_CAMERA) != OBSERVABLE
    visibility = " Judged on the three-camera rig only." if rig_only else ""
    return (
        f"{_label(symptom_id).capitalize()}: {symptom['description']}. "
        f"Usual causes: {'; '.join(cause_texts)}.{visibility}"
    )


def build_squat_card() -> dict[str, str]:
    """Topic -> short section, built from symptoms.yaml and causes.yaml."""
    symptoms = _load_graph("symptoms.yaml")
    causes = _load_graph("causes.yaml")
    card = {OBSERVABILITY_TOPIC: _observability_section(symptoms)}
    card.update(FAQ)
    for symptom_id, symptom in symptoms.items():
        card[symptom_id] = _symptom_section(symptom_id, symptom, causes)
    return card


SQUAT_CARD = build_squat_card()


def _topic_keywords(topic: str) -> tuple[str, ...]:
    if topic in TOPIC_KEYWORDS:
        return TOPIC_KEYWORDS[topic]
    return tuple(word for word in topic.split("_") if len(word) >= MIN_ID_KEYWORD_CHARS)


def explain_squat_topic(question: str, capture_mode: str | None = None) -> str:
    """The card sections that best match a free-text question, with the current camera setup."""
    text = question.lower().strip()
    if text in SQUAT_CARD:
        topics = [text]
    else:
        hits = {
            topic: sum(1 for keyword in _topic_keywords(topic) if keyword in text)
            for topic in SQUAT_CARD
        }
        topics = sorted((topic for topic in hits if hits[topic]), key=lambda topic: -hits[topic])
        topics = topics[:MAX_SECTIONS_PER_ANSWER]
    mode = capture_mode or capture_mode_from_env()
    setup = "one camera" if mode == SINGLE_CAMERA else "the three-camera rig"
    sections = [f"This session runs on {setup}."]
    if topics:
        sections += [SQUAT_CARD[topic] for topic in topics]
    else:
        sections.append(f"No notes on that. Topics with notes: {', '.join(SQUAT_CARD)}.")
        sections.append(SQUAT_CARD[OBSERVABILITY_TOPIC])
    return "\n".join(sections)


class ExplainSquatMixin:
    """Gives an agent the explain_squat tool. Mix in before BaseNovaAgent."""

    @function_tool
    async def explain_squat(self, topic: str, context: RunContext = None):
        """
        Look up squat facts before answering any how or why question about squat technique:
        what causes a fault, whether something is bad, how deep to go, forward lean, when to
        add weight, or what the cameras can and can't see.

        Args:
            topic: The subject of the question in a few words, e.g. "heels coming up"
        """
        return (
            f"{explain_squat_topic(topic)}\n"
            "Answer from these notes only, in one to three short spoken sentences, plain words, no jargon."
        )
