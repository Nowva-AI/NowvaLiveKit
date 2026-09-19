"""Shared coaching constants: the Nova persona and the cue key registry.

Cue keys are emitted by biomechanics.coaching.cue_cache. CUE_TEXT_MAP holds
the exact spoken strings that pre-generated TTS audio is cached against —
do not reword them without regenerating the audio. CUE_DISPLAY_LABELS holds
the short labels shown in set reports.
"""

from __future__ import annotations

from agent.agents.prompts.base_prompt import NOVA_IDENTITY, SPOKEN_OUTPUT_RULES

# Persona prepended to all coaching LLM instructions. Derived from the same
# identity as BASE_PROMPT so the mid-workout voice and the conversational
# voice are one person.
COACHING_PERSONA = (
    f"{NOVA_IDENTITY} "
    "You are mid-workout, coaching your athlete through their sets. "
    "Calm, direct, and specific — your energy rises when something is actually good, not before. "
    "Honest about what needs work, warm about what improved. "
    "Dry humor only between sets, only when the context notes say humor fits, never two replies in a row. "
    "Almost never say the athlete's name. "
    "SHORT responses only — obey the word and sentence limits you are given exactly. "
    "Never reuse a phrase you have already said this session; any lines listed as already said are off limits. "
    "Example sentences in instructions show the vibe, not the words — never copy them verbatim. "
    "If an ATHLETE STATE note says they sound strained or frustrated, or are near their limit, be briefer and calmer "
    "and lead with what went right; never say the state out loud.\n"
    f"{SPOKEN_OUTPUT_RULES}"
)

# Number words for rep cues
_NUMBER_WORDS = {
    1: "One!", 2: "Two!", 3: "Three!", 4: "Four!", 5: "Five!",
    6: "Six!", 7: "Seven!", 8: "Eight!", 9: "Nine!", 10: "Ten!",
    11: "Eleven!", 12: "Twelve!", 13: "Thirteen!", 14: "Fourteen!", 15: "Fifteen!",
    16: "Sixteen!", 17: "Seventeen!", 18: "Eighteen!", 19: "Nineteen!", 20: "Twenty!",
}

# Cue key → spoken text mapping (cached TTS audio exists for these exact strings)
CUE_TEXT_MAP: dict[str, str] = {
    # Squat corrections
    "knees_out": "Knees out!",
    "chest_up": "Chest up!",
    "deeper": "Get deeper!",
    "heels_down": "Heels down!",
    "even_it_out": "Even it out!",
    "slow_down": "Slow down!",
    "brace": "Brace your core!",
    # Deadlift corrections
    "hips_through": "Hips through!",
    "flat_back": "Flat back!",
    "lockout": "Lock it out!",
    # Intra-set stance / toe-out coaching
    "stance_explain": "That lean's coming from your stance — step your feet out wider.",
    "stance_wider": "A little wider.",
    "stance_narrower": "Bring it in a touch.",
    "toe_out_explain": "That lean's coming from your feet — turn your toes out more.",
    "toe_out_more": "More toe-out.",
    "toe_out_less": "Ease them back in.",
    "adjust_good": "Right there — hold that.",
    # Positive reinforcement
    "good_rep": "Good rep!",
    "great_depth": "Great depth!",
    "strong": "Strong!",
    "clean": "Clean!",
    "perfect": "Perfect!",
    # Rep counts
    **{f"rep_{i}": _NUMBER_WORDS[i] for i in range(1, 21)},
}

# Cue key → human-readable label for set reports (rep_* labels are built inline)
CUE_DISPLAY_LABELS: dict[str, str] = {
    "knees_out": "Knees out!",
    "chest_up": "Chest up!",
    "deeper": "Go deeper!",
    "heels_down": "Heels down!",
    "even_it_out": "Even it out!",
    "slow_down": "Slow down!",
    "brace": "Brace core!",
    "hips_through": "Hips through!",
    "flat_back": "Flat back!",
    "lockout": "Lock it out!",
    "stance_explain": "Stance is the cause",
    "stance_wider": "Wider",
    "stance_narrower": "Narrower",
    "toe_out_explain": "Foot angle is the cause",
    "toe_out_more": "More toe-out",
    "toe_out_less": "Less toe-out",
    "adjust_good": "On target",
    "good_rep": "Good rep!",
    "great_depth": "Great depth!",
    "strong": "Strong!",
    "clean": "Clean!",
    "perfect": "Perfect!",
}

# Preemptive outcome text: cue_key → (positive_text, negative_text)
# Positive plays when the fault is fixed on the next rep; negative if it persists.
PREEMPTIVE_TEXT: dict[str, tuple[str, str]] = {
    "knees_out": ("Good, knees are tracking better!", "Still caving in, push those knees out!"),
    "chest_up": ("Nice, chest is up!", "Still leaning forward, keep that chest up!"),
    "deeper": ("Great depth that time!", "Still a bit shallow, try to get lower!"),
    "heels_down": ("Heels are planted!", "Heels are still coming up!"),
    "even_it_out": ("Looking more even!", "Still favoring one side!"),
    "slow_down": ("Better tempo!", "Still rushing, slow it down!"),
    "brace": ("Good brace!", "Don't forget to brace your core!"),
}

# Intra-set stance/toe-out coaching (Feature 2). These play from the cue
# cache, not the LLM: the corrections fire every 1.5s between reps, and a
# generate_reply round-trip lands after the lifter has already moved on.
# Spoken once when the monitor arms — carries the why and the fix.
ADJUSTMENT_EXPLAIN_CUES: dict[str, str] = {
    "stance_width": "stance_explain",
    "toe_out": "toe_out_explain",
}

# Spoken on each poll, keyed by which way the lifter needs to move.
ADJUSTMENT_CUES: dict[str, dict[str, str]] = {
    "stance_width": {"more": "stance_wider", "less": "stance_narrower"},
    "toe_out": {"more": "toe_out_more", "less": "toe_out_less"},
}

ADJUSTMENT_ON_TARGET_CUE = "adjust_good"

# Only used when a cue has no pre-generated audio on disk.
ADJUSTMENT_SYSTEM_PROMPT = (
    f"{NOVA_IDENTITY} Mid-set. Give a 2-5 word stance adjustment cue. "
    "No filler words, no humor, plain spoken text only. Examples: "
    "'A little wider', 'Right there, perfect', 'Too wide, bring it in'."
)

ADJUSTMENT_PARAM_LABELS: dict[str, str] = {
    "stance_width": "stance width",
    "toe_out": "toe-out angle",
}
