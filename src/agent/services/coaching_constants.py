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
    # Overhead press
    "press_layback_fixed": "Stayed stacked, good.",
    "press_bar_forward_fixed": "Bar right over you, good.",
    "press_lockout_fixed": "Locked out tall, that's it.",
    "press_leg_drive_fixed": "Legs stayed locked, good.",
    "press_uneven_fixed": "Bar stayed level, nice.",
    "press_velocity_loss_fixed": "That one moved, good.",
    # Romanian deadlift
    "rdl_hips_back_fixed": "That's it, hips went back.",
    "rdl_bar_close_fixed": "Good, bar stayed on your legs.",
    "rdl_finish_tall_fixed": "Good, clean finish up top.",
    "rdl_level_bar_fixed": "Good, bar stayed level.",
    "rdl_same_depth_fixed": "Right back to your depth.",
    "rdl_slow_down_fixed": "Better, nice and controlled.",
}

# Praise for a cued fault that improved but isn't proven gone: a hold rep still
# showed it below its cue start tier. Keyed "<base cue key>_better". Drafted for
# every fault whose start tier can sit above mild; without a clip nothing plays.
BETTER_CUE_TEXT: dict[str, str] = {
    "press_leg_drive_better": "Better, knees staying straighter.",
    "press_uneven_better": "Better, bar's leveling out.",
    "press_lockout_better": "Better, almost locked out.",
    "press_bar_forward_better": "Better, bar closer over you.",
    "press_velocity_loss_better": "Better, that one moved more.",
    "rdl_hips_back_better": "Better, hips went further back.",
    "rdl_bar_close_better": "Better, bar stayed closer.",
    "rdl_finish_tall_better": "Better, finishing taller.",
}

# The line that names a cue's fix before a set starts, keyed by base cue key.
# The rest-complete prompt carries it when that fault is the set's focus.
PREEMPTIVE_TEXT: dict[str, str] = {
    "press_layback": "Glutes tight, ribs down this set.",
    "press_bar_forward": "Get your head through at the top.",
    "press_lockout": "Finish every rep reaching tall.",
    "press_leg_drive": "Knees stay locked the whole set.",
    "press_uneven": "Keep the bar level all the way up.",
    "rdl_hips_back": "This set: hips back, not down.",
    "rdl_bar_close": "This set: bar stays on your legs.",
    "rdl_finish_tall": "This set: finish tall at the top.",
    "rdl_level_bar": "This set: keep the bar level.",
    "rdl_same_depth": "This set: same depth every rep.",
    "rdl_slow_down": "This set: two seconds down.",
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
    # Other exercises, keyed <exercise>_<cue> (their profiles' fault -> cue maps)
    "deadlift_lockout": "Finish tall, hips through!",
    "deadlift_flat_back": "Flat back!",
    "deadlift_bar_close": "Keep the bar close!",
    "deadlift_even": "Pull evenly!",
    "rdl_hips_back": "Hips back to the wall!",
    "rdl_bar_close": "Drag the bar down your legs!",
    "rdl_finish_tall": "Finish tall, hips under shoulders!",
    "rdl_level_bar": "Keep the bar level!",
    "rdl_same_depth": "Same depth every rep!",
    "rdl_slow_down": "Two seconds on the way down!",
    "rdl_lower": "Take the bar lower!",
    "rdl_setup": "Hooks at mid-thigh. Unrack, one step back, soft knees, then push your hips back.",
    "rdl_back_note": "I can't see your lower back, so stop before it rounds.",
    "level_camera": "Check the camera — it looks tilted. Keep it straight for better feedback.",
    "lunge_deeper": "Sink a little deeper!",
    "lunge_knee_out": "Front knee out!",
    "lunge_chest_up": "Chest up!",
    "lunge_steady": "Stay steady!",
    "lunge_even": "Even out both legs!",
    "press_layback": "Glutes tight, ribs down!",
    "press_bar_forward": "Head through at the top!",
    "press_lockout": "Reach the bar to the ceiling!",
    "press_lockout_left": "Straighten your left arm up top!",
    "press_lockout_right": "Straighten your right arm up top!",
    "press_leg_drive": "Keep your knees locked!",
    "press_uneven": "Push both hands up evenly!",
    "press_uneven_left": "Drive your left hand up!",
    "press_uneven_right": "Drive your right hand up!",
    "press_velocity_loss": "Fast off your shoulders!",
    "press_setup": "Unrack, step back, hold it on your shoulders.",
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
    **BETTER_CUE_TEXT,
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
    "deadlift_lockout": "Finish tall",
    "deadlift_flat_back": "Flat back",
    "deadlift_bar_close": "Bar close",
    "deadlift_even": "Pull evenly",
    "rdl_hips_back": "Hips back",
    "rdl_bar_close": "Bar close",
    "rdl_finish_tall": "Finish tall",
    "rdl_level_bar": "Level bar",
    "rdl_same_depth": "Same depth",
    "rdl_slow_down": "Control the descent",
    "rdl_lower": "Lower to count",
    "rdl_setup": "Setup",
    "rdl_back_note": "Back note",
    "level_camera": "Level camera",
    "lunge_deeper": "Deeper",
    "lunge_knee_out": "Front knee out",
    "lunge_chest_up": "Chest up",
    "lunge_steady": "Stay steady",
    "lunge_even": "Even legs",
    "press_layback": "Ribs down",
    "press_bar_forward": "Head through",
    "press_lockout": "Lock out",
    "press_lockout_left": "Lock out left arm",
    "press_lockout_right": "Lock out right arm",
    "press_leg_drive": "Knees locked",
    "press_uneven": "Press evenly",
    "press_uneven_left": "Left hand up",
    "press_uneven_right": "Right hand up",
    "press_velocity_loss": "Drive",
    "press_setup": "Setup",
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
    **{key: "Better" for key in BETTER_CUE_TEXT},
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
