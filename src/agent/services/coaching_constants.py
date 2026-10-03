"""Shared coaching constants: the Nova persona and the cue key registry.

Cue keys are emitted by biomechanics.coaching.cue_cache. The spoken lines for
the pre-generated cue audio live in src/assets/cue_text/cues.json; CUE_TEXT_MAP
holds one plain line per key for the on-screen banner and for runtime TTS when
a cue has no audio on disk. CUE_DISPLAY_LABELS holds the short labels shown in
set reports.
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

# Fault-specific praise, played once a cued fault stays gone for two reps.
# Keyed "<base cue key>_fixed"; external focus, like the cues themselves.
FIXED_CUE_TEXT: dict[str, str] = {
    "knees_out_fixed": "That's it, spreading the floor.",
    "chest_up_fixed": "Better, bar and hips rose together.",
    "heels_down_fixed": "Good, whole foot stayed down.",
    "whole_foot_fixed": "Better, balanced over mid-foot.",
    "even_it_out_fixed": "Good, staying centered.",
    "level_bar_fixed": "Good, bar stayed level.",
    "deeper_fixed": "There's your depth.",
    "square_feet_fixed": "Good, feet are even.",
    "lockout_fixed": "Good, all the way up.",
    "slow_down_fixed": "Better, controlled on the way down.",
    "same_depth_fixed": "Right back to your depth.",
    "drive_fixed": "That one moved, good drive.",
    # Deadlift
    "deadlift_bar_midfoot_fixed": "Good, bar over midfoot.",
    "deadlift_hips_fixed": "Good, hips set right.",
    "deadlift_hips_up_fixed": "Good, hips set higher.",
    "deadlift_hips_down_fixed": "Good, hips set lower.",
    "deadlift_shoulders_over_fixed": "Good, shoulders over the bar.",
    "deadlift_chest_with_hips_fixed": "Better, chest and hips together.",
    "deadlift_bar_close_fixed": "Good, bar stayed close.",
    "deadlift_lockout_fixed": "Good, all the way up.",
    "deadlift_finish_neutral_fixed": "Good, tall with no lean.",
    "deadlift_even_feet_fixed": "Good, staying centered.",
    "deadlift_level_bar_fixed": "Good, bar stayed level.",
    "deadlift_long_arms_fixed": "Good, arms stayed long.",
}

# Played once when the pipeline loses sight of the athlete mid-set.
TRACKING_LOST_CUE = "tracking_lost"

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
    "knees_out_left": "Left knee out!",
    "knees_out_right": "Right knee out!",
    "chest_up": "Chest up!",
    "heels_down": "Heels down!",
    "heels_down_left": "Left heel down!",
    "heels_down_right": "Right heel down!",
    "whole_foot": "Whole foot!",
    "even_it_out": "Even it out!",
    "even_it_out_left": "Drifting left, stay centered!",
    "even_it_out_right": "Drifting right, stay centered!",
    "level_bar": "Keep the bar level!",
    "deeper": "Get deeper!",
    "square_feet": "Square your feet!",
    "square_feet_left": "Left foot even!",
    "square_feet_right": "Right foot even!",
    "lockout": "Stand tall up top!",
    "slow_down": "Control the way down!",
    "same_depth": "Match your first rep!",
    "drive": "Drive up hard!",
    "brace": "Brace your core!",
    # Deadlift corrections (.claude/deadlift/CONTRACT.md §4)
    "deadlift_bar_midfoot": "Bar over midfoot!",
    "deadlift_hips": "Set your hip height!",
    "deadlift_hips_up": "Hips a bit higher!",
    "deadlift_hips_down": "Hips a bit lower!",
    "deadlift_shoulders_over": "Shoulders over the bar!",
    "deadlift_chest_with_hips": "Chest and hips together!",
    "deadlift_bar_close": "Keep the bar close!",
    "deadlift_lockout": "Stand tall!",
    "deadlift_finish_neutral": "Stand tall, no lean!",
    "deadlift_even_feet": "Push evenly!",
    "deadlift_even_feet_left": "Drifting left, push evenly!",
    "deadlift_even_feet_right": "Drifting right, push evenly!",
    "deadlift_level_bar": "Keep the bar level!",
    "deadlift_long_arms": "Long arms!",
    # Deadlift closed-loop foot guidance, while standing at the bar
    "deadlift_step_closer": "Step closer.",
    "deadlift_closer": "A bit closer.",
    "deadlift_back": "Back a little.",
    # Other exercises, keyed <exercise>_<cue> (their profiles' fault -> cue maps)
    "rdl_hips_back": "Push your hips back!",
    "rdl_flat_back": "Flat back!",
    "rdl_even": "Hinge evenly!",
    "lunge_deeper": "Sink a little deeper!",
    "lunge_knee_out": "Front knee out!",
    "lunge_chest_up": "Chest up!",
    "lunge_steady": "Stay steady!",
    "lunge_even": "Even out both legs!",
    "press_lockout": "Lock it out overhead!",
    "press_elbows": "Elbows under the bar!",
    "press_bar_path": "Press straight up!",
    "press_even": "Press evenly!",
    "row_higher": "Pull higher!",
    "row_flat_back": "Flat back!",
    "row_steady": "Keep your torso still!",
    "row_even": "Pull evenly!",
    "curl_extend": "All the way down!",
    "curl_higher": "Curl it higher!",
    "curl_elbows": "Elbows pinned!",
    "curl_even": "Even it out!",
    "triceps_lockout": "Extend all the way!",
    "triceps_deeper": "Lower it deeper!",
    "triceps_elbows": "Keep your elbows still!",
    "triceps_even": "Even it out!",
    # Intra-set stance / toe-out coaching
    "stance_explain": "That's coming from your stance — step your feet out wider.",
    "stance_wider": "A little wider.",
    "stance_narrower": "Bring it in a touch.",
    "toe_out_explain": "That's coming from your feet — turn your toes out more.",
    "toe_out_more": "More toe-out.",
    "toe_out_less": "Ease them back in.",
    "adjust_good": "Right there — hold that.",
    # Positive reinforcement
    "good_rep": "Good rep!",
    "great_depth": "Great depth!",
    "strong": "Strong!",
    "clean": "Clean!",
    "perfect": "Perfect!",
    **FIXED_CUE_TEXT,
    TRACKING_LOST_CUE: "I can't see you fully, step back into view.",
    # Rep counts
    **{f"rep_{i}": _NUMBER_WORDS[i] for i in range(1, 21)},
}

# Cue key → human-readable label for set reports (rep_* labels are built inline)
CUE_DISPLAY_LABELS: dict[str, str] = {
    "knees_out": "Knees out!",
    "knees_out_left": "Left knee out!",
    "knees_out_right": "Right knee out!",
    "chest_up": "Chest up!",
    "heels_down": "Heels down!",
    "heels_down_left": "Left heel down!",
    "heels_down_right": "Right heel down!",
    "whole_foot": "Whole foot!",
    "even_it_out": "Even it out!",
    "even_it_out_left": "Drifting left",
    "even_it_out_right": "Drifting right",
    "level_bar": "Level the bar",
    "deeper": "Go deeper!",
    "square_feet": "Square feet",
    "square_feet_left": "Left foot even",
    "square_feet_right": "Right foot even",
    "lockout": "Stand tall",
    "slow_down": "Control the descent",
    "same_depth": "Same depth",
    "drive": "Drive up!",
    "brace": "Brace core!",
    "deadlift_bar_midfoot": "Bar over midfoot",
    "deadlift_hips": "Hip height",
    "deadlift_hips_up": "Hips higher",
    "deadlift_hips_down": "Hips lower",
    "deadlift_shoulders_over": "Shoulders over bar",
    "deadlift_chest_with_hips": "Chest with hips",
    "deadlift_bar_close": "Bar close",
    "deadlift_lockout": "Stand tall",
    "deadlift_finish_neutral": "No lean back",
    "deadlift_even_feet": "Push evenly",
    "deadlift_even_feet_left": "Drifting left",
    "deadlift_even_feet_right": "Drifting right",
    "deadlift_level_bar": "Level the bar",
    "deadlift_long_arms": "Long arms",
    "deadlift_step_closer": "Step closer",
    "deadlift_closer": "Closer",
    "deadlift_back": "Step back",
    "rdl_hips_back": "Hips back",
    "rdl_flat_back": "Flat back",
    "rdl_even": "Hinge evenly",
    "lunge_deeper": "Deeper",
    "lunge_knee_out": "Front knee out",
    "lunge_chest_up": "Chest up",
    "lunge_steady": "Stay steady",
    "lunge_even": "Even legs",
    "press_lockout": "Lock out",
    "press_elbows": "Elbows under bar",
    "press_bar_path": "Straight up",
    "press_even": "Press evenly",
    "row_higher": "Pull higher",
    "row_flat_back": "Flat back",
    "row_steady": "Torso still",
    "row_even": "Pull evenly",
    "curl_extend": "Full extension",
    "curl_higher": "Curl higher",
    "curl_elbows": "Elbows pinned",
    "curl_even": "Even it out",
    "triceps_lockout": "Full extension",
    "triceps_deeper": "Deeper",
    "triceps_elbows": "Elbows still",
    "triceps_even": "Even it out",
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
    **{key: "Fixed" for key in FIXED_CUE_TEXT},
    TRACKING_LOST_CUE: "Out of view",
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

# Closed-loop deadlift foot guidance: while the lifter stands at the bar, Nova
# moves the feet until the bar is over the midfoot, then says ADJUSTMENT_ON_TARGET_CUE.
DEADLIFT_CLOSED_LOOP = "deadlift_bar_midfoot"
DEADLIFT_STEP_CLOSER_CUE = "deadlift_step_closer"
DEADLIFT_CLOSER_CUE = "deadlift_closer"
DEADLIFT_BACK_CUE = "deadlift_back"

# Only used when a cue has no pre-generated audio on disk.
ADJUSTMENT_SYSTEM_PROMPT = (
    f"{NOVA_IDENTITY} Mid-set. Give one 2-5 word stance cue: which way to move the feet, "
    "or a confirmation when they're on target. No filler words, no humor, plain spoken "
    "text only. Vary the wording."
)

ADJUSTMENT_PARAM_LABELS: dict[str, str] = {
    "stance_width": "stance width",
    "toe_out": "toe-out angle",
}
