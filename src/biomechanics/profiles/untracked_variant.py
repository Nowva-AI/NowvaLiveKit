"""
Untracked exercise variants

Variants of the deadlift, Romanian deadlift and overhead press that their
profiles do not model (sumo, single-leg, seated, push press...). Whole-word
longest matching sends them here instead of to the profile they contain, so
they count no reps, fire no faults, and are never offered for coaching: Nova
says the variant isn't coached yet (SHARED-PLUMBING.md §7).
"""

from __future__ import annotations

from biomechanics.profiles.registry import register_profile
from biomechanics.profiles.untracked import UntrackedProfile

DEADLIFT_VARIANTS = (
    "sumo_deadlift", "trap_bar_deadlift", "hex_bar_deadlift", "deficit_deadlift",
    "snatch_grip_deadlift", "single_leg_deadlift", "rack_pull",
)
RDL_VARIANTS = (
    "single_leg_rdl", "single_leg_romanian_deadlift",
    "snatch_grip_rdl", "snatch_grip_romanian_deadlift",
    "stiff_leg_deadlift",
    "dumbbell_rdl", "dumbbell_romanian_deadlift",
    "kettlebell_rdl", "kettlebell_romanian_deadlift",
    "deficit_rdl", "deficit_romanian_deadlift",
    "b_stance_rdl", "b_stance_romanian_deadlift",
)
PRESS_VARIANTS = (
    "push_press", "seated_press", "seated_shoulder_press", "dumbbell_shoulder_press",
    "dumbbell_press", "behind_the_neck_press", "landmine_press",
    # Longer than the press's own names, so they win the longest match (ohp.md §7.1 Step 4).
    "seated_overhead_press", "seated_barbell_overhead_press", "seated_military_press",
    "dumbbell_overhead_press", "kettlebell_overhead_press", "single_arm_overhead_press",
    "smith_machine_overhead_press",
)


@register_profile(*DEADLIFT_VARIANTS, *RDL_VARIANTS, *PRESS_VARIANTS)
class UntrackedVariantProfile(UntrackedProfile):
    """Carries the hinge exercises' safety flags: variant frames never refine the
    rig's camera calibration or feed the session's body measurement."""

    allows_camera_refine = False
    feeds_body_calibration = False
