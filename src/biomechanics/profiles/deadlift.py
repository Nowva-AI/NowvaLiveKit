"""
Conventional Deadlift Exercise Profile

The bar drives the rep: a state machine over the bar's height (deadlift.analyzer)
counts dead-stop and touch-and-go reps, and ten whole-rep rules judge setup,
coordination, bar path, lockout and symmetry (docs/deadlift/PLAN.md §2.6). The
profile is stateless; all per-frame state lives in the analyser.

Until the deadlift is validated on real lifts it is unreachable: get_profile()
hands out GatedDeadliftProfile, which counts nothing and fires nothing, unless
NOWVA_DEV_COACHING_READY lists "deadlift". Sumo and other variants are never
this profile (registry: UntrackedVariantProfile).
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.deadlift_bar_drift import DeadliftBarDriftRule
from biomechanics.faults.rules.deadlift_bar_position import DeadliftBarPositionRule
from biomechanics.faults.rules.deadlift_bar_tilt import DeadliftBarTiltRule
from biomechanics.faults.rules.deadlift_bent_arms import DeadliftBentArmsRule
from biomechanics.faults.rules.deadlift_hip_shift import DeadliftHipShiftRule
from biomechanics.faults.rules.deadlift_hips_shoot import DeadliftHipsShootRule
from biomechanics.faults.rules.deadlift_lean_back import DeadliftLeanBackRule
from biomechanics.faults.rules.deadlift_lockout import DeadliftLockoutRule
from biomechanics.faults.rules.deadlift_setup_hips import DeadliftSetupHipsRule
from biomechanics.faults.rules.deadlift_shoulders_behind import DeadliftShouldersBehindRule
from biomechanics.faults.rules.deadlift_velocity_loss import DeadliftVelocityLossRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.profiles.untracked import UntrackedProfile, UntrackedVariantProfile
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import JointAngles, Skeleton3D

if TYPE_CHECKING:
    from biomechanics.deadlift.analyzer import DeadliftRepAnalyzer
    from biomechanics.deadlift.diagnosis import DeadliftSetDiagnosis
    from biomechanics.deadlift.rep_counter import DeadliftRepCounter
    from biomechanics.deadlift.session_reference import DeadliftSessionReference

DEV_COACHING_READY_ENV = "NOWVA_DEV_COACHING_READY"
DEADLIFT_SET_IDLE_TIMEOUT_S = 30.0
DEADLIFT_CLOSED_LOOP_CUE = "deadlift_bar_midfoot"
# Cues stay on while these are seen: plates may hide the shins and ankles during
# the pull, but the midfoot is locked at setup by then.
DEADLIFT_TRACKING_KEYPOINTS = (
    CK.LEFT_HIP, CK.RIGHT_HIP,
    CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER,
    CK.LEFT_WRIST, CK.RIGHT_WRIST,
)

# fault_type -> cue base key (side / direction variants live in the cue dict).
DEADLIFT_FAULT_TO_CUE: dict[str, str] = {
    "deadlift_bar_position": "deadlift_bar_midfoot",
    "deadlift_shoulders_behind": "deadlift_shoulders_over",
    "deadlift_setup_hips": "deadlift_hips",
    "deadlift_hips_shoot": "deadlift_chest_with_hips",
    "deadlift_bar_drift": "deadlift_bar_close",
    "deadlift_lockout": "deadlift_lockout",
    "deadlift_lean_back": "deadlift_finish_neutral",
    "deadlift_hip_shift": "deadlift_even_feet",
    "deadlift_bar_tilt": "deadlift_level_bar",
    "deadlift_bent_arms": "deadlift_long_arms",
}
DEADLIFT_CUE_VARIANTS = (
    "deadlift_hips_up", "deadlift_hips_down",
    "deadlift_even_feet_left", "deadlift_even_feet_right",
)
# The closed-loop foot guidance before the first pull (CONTRACT.md §4).
DEADLIFT_GUIDANCE_CUES = ("deadlift_step_closer", "deadlift_closer", "deadlift_back", "adjust_good")


def _dev_coaching_ready(exercise: str) -> bool:
    listed = os.environ.get(DEV_COACHING_READY_ENV, "")
    return exercise in {name.strip().lower() for name in listed.split(",")}


class GatedDeadliftProfile(UntrackedProfile):
    """What a deadlift resolves to until it is coaching-ready: nothing counted,
    nothing judged, but the deadlift's safety flags, so its frames never touch the
    squat's camera refines or body measurements."""

    allows_camera_refine = False
    feeds_body_calibration = False


@register_profile(
    "deadlift",
    "deadlifts",
    "conventional_deadlift",
    "barbell_deadlift",
    "barbell_conventional_deadlift",
)
class DeadliftProfile(ExerciseProfile):
    """Profile for the conventional barbell deadlift."""

    name = "deadlift"
    movement_pattern = "deadlift"
    # The squat's assessment, calibration and causal engine model squats only.
    uses_diagnosis_engine = False
    uses_bilstm_counter = False
    coaching_ready = _dev_coaching_ready("deadlift")
    display_name = "conventional deadlift"

    gate_until_ready = True
    needs_bar_3d = True
    allows_camera_refine = False
    feeds_body_calibration = False
    set_idle_timeout_s = DEADLIFT_SET_IDLE_TIMEOUT_S
    waits_for_diagnosis = True
    closed_loop_cue = DEADLIFT_CLOSED_LOOP_CUE
    tracking_keypoints = DEADLIFT_TRACKING_KEYPOINTS
    # Only these words may stand beside "deadlift": every other word names a
    # variant the conventional rules do not model (sumo, band, suitcase,
    # deficit, paused, ...), which resolves to UntrackedVariantProfile.
    name_qualifiers = frozenset({"barbell", "conventional", "touch", "and", "go"})

    @classmethod
    def gated_profile(cls) -> ExerciseProfile:
        return GatedDeadliftProfile()

    def create_fault_rules(self, config: BiomechanicsConfig) -> list[FaultRule]:
        """D1-D10, in cue priority order. No DepthRule: one would turn on
        depth-gated counting."""
        fc = config.faults
        scale = fc.deadlift_wrist_proxy_threshold_scale
        return [
            DeadliftBarPositionRule(fc.deadlift_bar_position, scale),
            DeadliftShouldersBehindRule(fc.deadlift_shoulders_behind, scale),
            DeadliftSetupHipsRule(fc.deadlift_setup_hips, scale),
            DeadliftHipsShootRule(fc.deadlift_hips_shoot, scale),
            DeadliftBarDriftRule(fc.deadlift_bar_drift, scale),
            DeadliftLockoutRule(fc.deadlift_lockout, scale),
            DeadliftLeanBackRule(fc.deadlift_lean_back, scale),
            DeadliftHipShiftRule(fc.deadlift_hip_shift, scale),
            DeadliftBarTiltRule(fc.deadlift_bar_tilt, scale),
            DeadliftBentArmsRule(fc.deadlift_bent_arms, scale),
            DeadliftVelocityLossRule(fc.deadlift_velocity_loss, scale),
        ]

    def create_rep_analyzer(self, config: BiomechanicsConfig) -> DeadliftRepAnalyzer:
        from biomechanics.deadlift.analyzer import DeadliftRepAnalyzer

        return DeadliftRepAnalyzer(config.deadlift)

    def create_rep_counter(self, config: BiomechanicsConfig) -> DeadliftRepCounter:
        """The counter is a view of an analyser; the pipeline builds both through
        create_rep_analyzer, this standalone pair is for tools and tests."""
        from biomechanics.deadlift.rep_counter import DeadliftRepCounter

        return DeadliftRepCounter(self.create_rep_analyzer(config))

    def create_session_reference(self) -> DeadliftSessionReference:
        from biomechanics.deadlift.session_reference import DeadliftSessionReference

        return DeadliftSessionReference()

    def create_set_diagnosis(self, capture_mode: str) -> DeadliftSetDiagnosis:
        from biomechanics.deadlift.diagnosis import DeadliftSetDiagnosis

        return DeadliftSetDiagnosis(capture_mode=capture_mode)

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: JointAngles | None = None
    ) -> float:
        # Never called: the analyser supplies the signal (bar height above its rest).
        return math.nan

    def get_fault_to_cue_map(self) -> dict[str, str]:
        return dict(DEADLIFT_FAULT_TO_CUE)

    def get_cue_dict(self) -> dict[str, str]:
        from biomechanics.coaching.cue_cache import GENERIC_POSITIVE_CUE_KEYS, build_cue_dict

        return build_cue_dict(
            *DEADLIFT_FAULT_TO_CUE.values(),
            *DEADLIFT_CUE_VARIANTS,
            *DEADLIFT_GUIDANCE_CUES,
            *GENERIC_POSITIVE_CUE_KEYS,
        )

    def min_cue_tiers(self, config: BiomechanicsConfig) -> dict[str, str]:
        fc = config.faults
        return {
            fault_type: getattr(fc, fault_type).min_tier
            for fault_type in (
                "deadlift_bar_position",
                "deadlift_shoulders_behind",
                "deadlift_setup_hips",
                "deadlift_hips_shoot",
                "deadlift_bar_drift",
                "deadlift_lockout",
                "deadlift_lean_back",
                "deadlift_hip_shift",
                "deadlift_bar_tilt",
                "deadlift_bent_arms",
                "deadlift_velocity_loss",
            )
        }


# A partial deadlift without "deadlift" in its name: untracked, never DeadliftProfile.
# Names with "deadlift" and a word outside name_qualifiers are variants already.
register_profile("rack_pull", "rack_pulls")(UntrackedVariantProfile)
