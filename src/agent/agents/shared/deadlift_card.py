"""Compact conventional-deadlift knowledge for the voice coach, mirroring squat_card.py.

The voice LLM answers deadlift technique questions from this card instead of
free-styling, so it agrees with what the deadlift rules measure
(.claude/deadlift/CONTRACT.md) and never claims to see what the cameras can't:
no keypoint sits on the spine, so back rounding is never claimed either way.
"""

from __future__ import annotations

from livekit.agents import function_tool

from agent.agents.shared.deadlift_session import SETUP_STEPS
from biomechanics.faults.observability import SINGLE_CAMERA, capture_mode_from_env

OBSERVABILITY_TOPIC = "what_i_can_see"
MAX_SECTIONS_PER_ANSWER = 2
SINGLE_CAMERA_NOTE = (
    "This session runs on one camera: the bar is followed through the hands, so bar position "
    "and path are rough."
)
RIG_NOTE = "This session runs on the three-camera rig."

NEVER_VISIBLE = (
    "back rounding, upper or lower",
    "bracing",
    "head and neck position",
)

DEADLIFT_CARD: dict[str, str] = {
    OBSERVABILITY_TOPIC: (
        "Measured on every rep: where the bar starts against the middle of the foot, the bar "
        "path, hips rising before the chest, hip height and shoulder position at the start, "
        "lockout, leaning back at the top, hips sliding sideways, bar tilt, bent arms and bar "
        "speed. Bar position and path come from the bar seen by all three cameras; without it "
        "they are read from the hands, more loosely. "
        f"No setup sees {', '.join(NEVER_VISIBLE)}: never claim to, ask how it feels instead."
    ),
    "setup": (
        "The setup, in order: "
        + "; ".join(f"{number}. {step}" for number, step in enumerate(SETUP_STEPS, start=1))
        + ". Same setup every rep; reset it if it slips."
    ),
    "bar_over_midfoot": (
        "The bar starts over the middle of the foot, about where the laces are. Too far out and "
        "the bar swings forward off the floor: step closer. Shins pushing it back over the heels: "
        "step back a touch. Standing at the bar, you guide the feet in before they set up."
    ),
    "bar_path": (
        "The bar travels straight up, brushing the legs. Drifting away from the legs makes the "
        "same weight heavier and pulls the shoulders forward. Fix: bar close, push it back into "
        "the legs on the way up. Often starts with the bar too far out at setup."
    ),
    "hips_shoot": (
        "Hips and chest rise together off the floor. Hips rising first turns the pull into a "
        "stiff-leg pull. Usual causes: hips set too low, slack not pulled out before the pull, "
        "weak off the floor, or the load is too heavy today."
    ),
    "hip_height": (
        "Hip height at the start depends on build: long thighs or short arms put the hips higher. "
        "Set them where the shins touch the bar and the shoulders sit just in front of it. Don't "
        "squat the deadlift, and don't start with straight legs."
    ),
    "shoulders": (
        "Shoulders start just in front of the bar, not behind it, with the arms hanging straight down."
    ),
    "lockout": (
        "Finish standing tall: hips and knees straight, then stop. Leaning back at the top adds "
        "nothing; stand tall and squeeze the glutes."
    ),
    "symmetry": (
        "Push evenly through both feet and keep the bar level. Hips sliding to one side or a "
        "tilted bar usually comes from an uneven stance or grip, sometimes a weaker side. A mixed "
        "grip twists the trunk a little, which is expected."
    ),
    "bent_arms": (
        "Arms stay long, like ropes; the legs and hips move the bar. Bending the arms doesn't make "
        "the pull faster."
    ),
    "back_rounding": (
        "Back rounding is invisible to these cameras: no point is tracked on the spine, so never "
        "say the back rounded or stayed flat. What the cameras see that goes with losing position: "
        "hips rising first, the bar drifting forward, shoulders behind the bar. If they feel it "
        "round or it hurts: lighter weight, brace before each rep, set up before pulling. Ask how "
        "it felt."
    ),
    "grip": (
        "Double overhand while it holds; mixed or hook grip once the grip gives out. With a mixed "
        "grip, swap which hand faces up from session to session."
    ),
    "belt_and_shoes": (
        "Belt: optional, for heavy sets; brace into it before each pull. Shoes: flat, thin soles or "
        "socks are best; cushioned running shoes squash and make balance harder; raised-heel squat "
        "shoes tip the knees forward into the bar."
    ),
    "touch_and_go": (
        "Every rep counts, from a dead stop or touch and go. Cues wait until the bar is resting on "
        "the floor, so nothing is said in the middle of a pull."
    ),
    "go_heavier": (
        "Add weight when every rep of the last set matched the first: bar close, hips and chest "
        "rising together, bar speed holding, and two or more reps left, an effort of eight or "
        "under. If bar speed dropped a lot or form slipped late, stay at this weight. Small jumps. "
        "Never add load while a pain report is open."
    ),
    "warm_up": (
        "First deadlift session: the first set with just the empty bar, so the setup can be "
        "learned before the bar is loaded."
    ),
}

# Words an athlete uses for each topic.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    OBSERVABILITY_TOPIC: ("see", "detect", "camera", "notice", "measure", "spot", "track"),
    "setup": ("set up", "setup", "start position", "starting position", "steps", "how do i"),
    "bar_over_midfoot": ("midfoot", "mid foot", "middle of", "how far", "how close", "step closer"),
    "bar_path": ("bar path", "drift", "away from", "swing", "straight up", "close to"),
    "hips_shoot": ("shoot", "hips rise", "hips come up", "hips first", "stiff"),
    "hip_height": ("hip height", "hips too", "how low", "how high", "squat it"),
    "shoulders": ("shoulder",),
    "lockout": ("lockout", "lock out", "at the top", "top of", "finish", "stand tall", "lean back"),
    "symmetry": ("shift", "one side", "uneven", "tilt", "level", "lopsided"),
    "bent_arms": ("arms", "elbow", "bend"),
    "back_rounding": ("round", "spine", "flat back", "lower back", "upper back", "neutral"),
    "grip": ("grip", "mixed", "hook", "overhand", "straps", "chalk"),
    "belt_and_shoes": ("belt", "shoe", "barefoot", "socks", "slipper"),
    "touch_and_go": ("touch and go", "reset", "bounce", "dead stop", "count"),
    "go_heavier": ("heavier", "more weight", "add weight", "load", "progress", "increase"),
    "warm_up": ("warm", "empty bar", "first time", "beginner"),
}


def explain_deadlift_topic(question: str, capture_mode: str | None = None) -> str:
    """The card sections that best match a free-text question, with the current camera setup."""
    text = question.lower().strip()
    if text in DEADLIFT_CARD:
        topics = [text]
    else:
        hits = {
            topic: sum(1 for keyword in TOPIC_KEYWORDS[topic] if keyword in text)
            for topic in DEADLIFT_CARD
        }
        topics = sorted((topic for topic in hits if hits[topic]), key=lambda topic: -hits[topic])
        topics = topics[:MAX_SECTIONS_PER_ANSWER]
    mode = capture_mode or capture_mode_from_env()
    sections = [SINGLE_CAMERA_NOTE if mode == SINGLE_CAMERA else RIG_NOTE]
    if topics:
        sections += [DEADLIFT_CARD[topic] for topic in topics]
    else:
        sections.append(f"No notes on that. Topics with notes: {', '.join(DEADLIFT_CARD)}.")
        sections.append(DEADLIFT_CARD[OBSERVABILITY_TOPIC])
    return "\n".join(sections)


@function_tool
async def explain_deadlift(topic: str) -> str:
    """
    Look up deadlift facts before answering any how or why question about deadlift technique:
    the setup, how close to stand, the bar path, hips rising first, lockout, grip, belt or shoes,
    back rounding, when to add weight, or what the cameras can and can't see.

    Args:
        topic: The subject of the question in a few words, e.g. "bar drifting forward"
    """
    return (
        f"{explain_deadlift_topic(topic)}\n"
        "Answer from these notes only, in one to three short spoken sentences, plain words, no jargon."
    )
