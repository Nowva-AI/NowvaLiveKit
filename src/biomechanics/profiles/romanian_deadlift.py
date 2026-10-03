"""
Barbell Romanian deadlift from the rack (docs/exercises/rdl.md).

A rep analyser (analysis.hinge_features) owns the shoulder-height rep signal,
the athlete's standing setups and every per-rep feature. The rules judge hip
sink, the rig-only bar path and finish, bar tilt, range against the athlete's
own and lowering tempo, once per rep. Spinal rounding is not judged: one
camera cannot tell hip flexion from spinal flexion.
"""

from __future__ import annotations

from types import MappingProxyType

from biomechanics.analysis.hinge_features import (
    REJECT_PARTIAL,
    HingeRepFeatures,
    RdlRepAnalyzer,
    hip_sink_start_tier,
    skeleton_signal_cm,
)
from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig, RdlConfig
from biomechanics.faults.fault_types import FaultRule, FaultType
from biomechanics.faults.hinge_reference import HingeSessionReference
from biomechanics.faults.rules.rdl_bar_drift import RdlBarDriftRule
from biomechanics.faults.rules.rdl_bar_tilt import RdlBarTiltRule
from biomechanics.faults.rules.rdl_hip_sink import RdlHipSinkRule
from biomechanics.faults.rules.rdl_lockout import RdlLockoutRule
from biomechanics.faults.rules.rdl_partial import build_rejected_faults
from biomechanics.faults.rules.rdl_range_drift import RdlRangeDriftRule
from biomechanics.faults.rules.rdl_tempo import RdlTempoRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.frame_context import FrameContext
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles, Skeleton3D

# Fault type -> cue key (agent CUE_TEXT_MAP), in cue priority order (40-46).
RDL_FAULT_TO_CUE = MappingProxyType({
    FaultType.RDL_HIP_SINK.value: "rdl_hips_back",
    FaultType.RDL_BAR_DRIFT.value: "rdl_bar_close",
    FaultType.RDL_LOCKOUT.value: "rdl_finish_tall",
    FaultType.RDL_BAR_TILT.value: "rdl_level_bar",
    FaultType.RDL_RANGE_DRIFT.value: "rdl_same_depth",
    FaultType.RDL_TEMPO.value: "rdl_slow_down",
    FaultType.RDL_PARTIAL.value: "rdl_lower",
})
# Along the camera axis: they start at the rig start tier until measured
# gravity and the rig validation are in.
RIG_ONLY_FAULTS = frozenset({FaultType.RDL_BAR_DRIFT.value, FaultType.RDL_LOCKOUT.value})


@register_profile("romanian_deadlift", "rdl")
class RomanianDeadliftProfile(ExerciseProfile):
    """The thresholds the session-aware hooks read (start tiers, partial cues,
    the session reference) come from the config the pipeline builds the rules
    with; until then, the defaults."""

    name = "romanian_deadlift"
    movement_pattern = "hip_hinge"
    display_name = "Romanian deadlifts (barbell, from the rack)"
    # Untracked until the RDL is validated (SHARED-PLUMBING.md §9).
    gate_until_ready = True
    reports_depth = False
    # Hinged poses with plates near the shins would degrade the rig calibration
    # and the standing body measurement.
    allows_camera_refine = False
    feeds_body_calibration = False
    readiness_hint = "HANDS IN VIEW"

    def __init__(self) -> None:
        self._rdl = RdlConfig()

    def create_fault_rules(self, config: BiomechanicsConfig) -> list[FaultRule]:
        """RDL rules in cue priority order."""
        self._rdl = config.faults.rdl
        rdl = self._rdl
        return [
            RdlHipSinkRule(rdl.hip_sink),
            RdlBarDriftRule(rdl.bar_drift),
            RdlLockoutRule(rdl.lockout),
            RdlBarTiltRule(rdl.bar_tilt),
            RdlRangeDriftRule(rdl.range_drift),
            RdlTempoRule(rdl.tempo),
        ]

    def create_rep_analyzer(self, config: BiomechanicsConfig) -> RdlRepAnalyzer:
        self._rdl = config.faults.rdl
        return RdlRepAnalyzer(config.faults.rdl, config.readiness_gate)

    def create_session_reference(self) -> HingeSessionReference:
        return HingeSessionReference(self._rdl.rom)

    def create_rep_counter_config(self, config: BiomechanicsConfig) -> HipPositionCounterConfig:
        return config.faults.rdl.counter

    def get_rep_signal(self, skeleton_3d: Skeleton3D, angles: JointAngles | None = None) -> float:
        """Minus the shoulder height above the ankles (cm): rises through the
        descent. The pipeline reads the analyser's levelled rep_signal instead."""
        return skeleton_signal_cm(skeleton_3d)

    def get_asymmetry_metrics(self, angles: JointAngles) -> dict[str, float]:
        return {"hip": angles.hip_asymmetry}

    def get_fault_to_cue_map(self) -> dict[str, str]:
        return dict(RDL_FAULT_TO_CUE)

    def get_hedged_recap_faults(self, config: BiomechanicsConfig) -> frozenset[str]:
        return frozenset(config.faults.rdl.hedged_recap_faults)

    def cue_start_tier(self, fault_type: str, ctx: FrameContext) -> FaultSeverity:
        """Hip sink starts at mild only on a certified mount whose lens checks
        pass (rdl.md §3.2); the rig-only faults at the rig start tier."""
        if fault_type == FaultType.RDL_HIP_SINK.value:
            return hip_sink_start_tier(ctx, self._rdl.camera)
        if fault_type in RIG_ONLY_FAULTS:
            return FaultSeverity(self._rdl.rig_min_cue_severity)
        return super().cue_start_tier(fault_type, ctx)

    def on_rep_rejected(
        self, reason: str, features: HingeRepFeatures, angles: JointAngles,
    ) -> list[FaultEvent]:
        if reason == REJECT_PARTIAL:
            return build_rejected_faults(features, angles)
        return []

    def shallow_rep_cue(self, fault: FaultEvent, streak: int, cued_this_set: int) -> str | None:
        """The fault's cue from the second partial in a row, at most twice a set:
        a lone unrack or re-rack dip is never cued."""
        rom = self._rdl.rom
        if streak < rom.partial_cue_streak or cued_this_set >= rom.max_partial_cues_per_set:
            return None
        return RDL_FAULT_TO_CUE.get(fault.fault_type)
