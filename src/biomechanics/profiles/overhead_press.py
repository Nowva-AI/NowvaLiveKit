"""
Overhead Press Exercise Profile

The standing barbell overhead press, taken from the rack (docs/exercises/ohp.md).
A PressRepAnalyzer owns rack readiness, the rep signal (the bar above the
shoulders, cm) and every per-rep feature; the rules judge each counted rep once.
Untracked until validated on real recordings (gate_until_ready).
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

from biomechanics.analysis.press_features import (
    PITCH_BAND_LEVEL,
    REJECT_NO_LOCKOUT,
    PressRepAnalyzer,
    PressRepFeatures,
    PressSessionReference,
    missed_rep_event,
    pitch_band,
)
from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule, FaultType
from biomechanics.faults.observability import TRIANGULATED, capture_mode_from_env
from biomechanics.faults.rules.press_bar_forward import PressBarForwardRule
from biomechanics.faults.rules.press_grip import PressGripRule
from biomechanics.faults.rules.press_layback import PressLaybackRule
from biomechanics.faults.rules.press_leg_drive import PressLegDriveRule
from biomechanics.faults.rules.press_lockout import PressLockoutRule
from biomechanics.faults.rules.press_uneven import PressUnevenRule
from biomechanics.faults.rules.velocity_loss import VelocityLossRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.frame_context import FrameContext
from biomechanics.utils.types import LEG_KEYPOINTS, CocoKeypoints, FaultEvent, FaultSeverity, JointAngles, Skeleton3D

MILD = FaultSeverity.MILD
MODERATE = FaultSeverity.MODERATE
SEVERE = FaultSeverity.SEVERE

# Static start tiers (ohp.md §5.0) per capture mode; a missing fault starts at mild.
# Single camera: the level-band tiers (lockout stays approximate until promoted;
# side-view faults are not observable). Rig: pre-validation estimates.
SINGLE_CAMERA_START_TIERS: Mapping[str, FaultSeverity] = MappingProxyType({
    FaultType.PRESS_LEG_DRIVE.value: MODERATE,
    FaultType.PRESS_UNEVEN.value: SEVERE,
    FaultType.PRESS_GRIP.value: MODERATE,
})
RIG_START_TIERS: Mapping[str, FaultSeverity] = MappingProxyType({
    FaultType.PRESS_BAR_FORWARD.value: MODERATE,
    FaultType.PRESS_LOCKOUT.value: MODERATE,
    FaultType.PRESS_LEG_DRIVE.value: MODERATE,
    FaultType.PRESS_UNEVEN.value: MODERATE,
    FaultType.PRESS_GRIP.value: MODERATE,
    FaultType.PRESS_VELOCITY_LOSS.value: MODERATE,
})
# Single-camera faults whose start tier follows the mount's pitch band.
PITCH_BANDED_FAULTS = frozenset({FaultType.PRESS_LEG_DRIVE.value, FaultType.PRESS_UNEVEN.value})

# Cue keys (agent/services/coaching_constants.py); faults with a side get _left/_right.
FAULT_TO_CUE: Mapping[str, str] = MappingProxyType({
    FaultType.PRESS_LAYBACK.value: "press_layback",
    FaultType.PRESS_BAR_FORWARD.value: "press_bar_forward",
    FaultType.PRESS_LOCKOUT.value: "press_lockout",
    FaultType.PRESS_LEG_DRIVE.value: "press_leg_drive",
    FaultType.PRESS_UNEVEN.value: "press_uneven",
    FaultType.PRESS_VELOCITY_LOSS.value: "press_velocity_loss",
})
SIDED_CUES = ("press_lockout", "press_uneven")
SETUP_CUE = "press_setup"

# Clipped wrists must read as tracking lost (ohp.md P1).
PRESS_TRACKING_KEYPOINTS = (*LEG_KEYPOINTS, CocoKeypoints.LEFT_WRIST, CocoKeypoints.RIGHT_WRIST)


@register_profile(
    "overhead_press", "ohp", "military_press", "standing_press", "strict_press",
    "barbell_overhead_press",
)
class OverheadPressProfile(ExerciseProfile):
    """Standing barbell overhead press from the rack."""

    name = "overhead_press"
    movement_pattern = "press"
    display_name = "the standing barbell overhead press, from the rack"
    # Untracked until the press is validated (SHARED-PLUMBING.md §9); the
    # NOWVA_DEV_COACHING_READY override marks it ready at registration.
    gate_until_ready = True
    reports_depth = False
    # Arms overhead migrate the shoulder keypoints, which would bias the torso
    # length and shoulder width the squat and the press normaliser read.
    feeds_body_calibration = False
    readiness_hint = "BAR ON SHOULDERS"
    tracking_keypoints = PRESS_TRACKING_KEYPOINTS

    def __init__(self) -> None:
        self.min_cue_tiers = RIG_START_TIERS if capture_mode_from_env() == TRIANGULATED else SINGLE_CAMERA_START_TIERS

    def get_feature_extractor(self):
        from biomechanics.ml.upper_body_feature_extractor import UpperBodyFeatureExtractor
        return UpperBodyFeatureExtractor()

    def create_fault_rules(self, config: BiomechanicsConfig) -> list[FaultRule]:
        press = config.faults.press
        tilted_validated = press.tilted_band_validated_faults
        return [
            PressLaybackRule(press.layback),
            PressBarForwardRule(press.bar_forward),
            PressLockoutRule(press.lockout, press.lockout_side_min_difference_deg),
            PressLegDriveRule(press.leg_drive, FaultType.PRESS_LEG_DRIVE.value in tilted_validated),
            PressUnevenRule(press.uneven, FaultType.PRESS_UNEVEN.value in tilted_validated),
            VelocityLossRule(
                mild_pct=press.velocity_loss_pct.mild,
                moderate_pct=press.velocity_loss_pct.moderate,
                severe_pct=press.velocity_loss_pct.severe,
                fault_type=FaultType.PRESS_VELOCITY_LOSS,
                phase="press",
            ),
            PressGripRule(press.grip),
        ]

    def create_rep_analyzer(self, config: BiomechanicsConfig) -> PressRepAnalyzer:
        return PressRepAnalyzer(config.faults.press)

    def create_session_reference(self, config: BiomechanicsConfig) -> SessionReference:
        return PressSessionReference()

    def get_rep_signal(self, skeleton_3d: Skeleton3D, angles: JointAngles | None = None) -> float:
        """The bar above the shoulders, cm (the analyser's signal applies the rack floor)."""
        if angles is None:
            return math.nan
        return angles.avg_wrist_y

    def create_rep_counter_config(self, config: BiomechanicsConfig) -> HipPositionCounterConfig:
        return config.faults.press.counter

    def get_depth_metric(self, angles: JointAngles) -> float:
        """Elbow angle (180 = straight): what the press's lockout reaches."""
        return 180.0 - angles.avg_elbow_flexion

    def get_asymmetry_metrics(self, angles: JointAngles) -> dict[str, float]:
        return {"wrist_height": abs(angles.wrist_y_l - angles.wrist_y_r)}

    def on_rep_rejected(
        self, reason: str, features: PressRepFeatures, angles: JointAngles, config: BiomechanicsConfig,
    ) -> list[FaultEvent]:
        if reason != REJECT_NO_LOCKOUT:
            return []
        event = missed_rep_event(features, angles, config.faults.press.acceptance.missed_min_travel_cm)
        return [event] if event is not None else []

    def get_hedged_recap_faults(self, config: BiomechanicsConfig) -> frozenset[str]:
        return frozenset(config.faults.press.hedged_recap_faults)

    def cue_start_tier(self, fault_type: str, ctx: FrameContext, config: BiomechanicsConfig) -> FaultSeverity:
        """Rig: the static rig tiers. Single camera: leg drive and uneven follow the
        mount's pitch band; outside a validated band their rules also cap
        observability at approximate, so they stay recap-only."""
        if ctx.capture_mode == TRIANGULATED:
            return RIG_START_TIERS.get(fault_type, MILD)
        if fault_type in PITCH_BANDED_FAULTS:
            press = config.faults.press
            if pitch_band(ctx.pitch_certified, ctx.mount_pitch_deg, press.camera) != PITCH_BAND_LEVEL:
                return SEVERE
            if fault_type == FaultType.PRESS_UNEVEN.value and not press.uneven_promoted:
                return SEVERE
            return MODERATE
        return SINGLE_CAMERA_START_TIERS.get(fault_type, MILD)

    def get_fault_to_cue_map(self) -> dict[str, str]:
        return dict(FAULT_TO_CUE)

    def get_cue_dict(self) -> dict[str, str]:
        from biomechanics.coaching.cue_cache import GENERIC_POSITIVE_CUE_KEYS, SIDE_CUE_SIDES, build_cue_dict

        sided = [f"{cue}_{side}" for cue in SIDED_CUES for side in SIDE_CUE_SIDES]
        return build_cue_dict(*FAULT_TO_CUE.values(), *sided, SETUP_CUE, *GENERIC_POSITIVE_CUE_KEYS)
