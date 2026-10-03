"""
RDL fault rules, the rejected-rep fault builder, the shallow-rep cue and the
session-aware start tiers (docs/exercises/rdl.md §3.2, §3.6, §5).
"""

from __future__ import annotations

import math

import pytest

from biomechanics.analysis.hinge_features import HIP_SOURCE_MEASURED, HIP_SOURCE_SETUP, HingeRepFeatures
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultType
from biomechanics.faults.hinge_reference import DRIFT_HIP_SINK, HingeSessionReference
from biomechanics.faults.observability import APPROXIMATE, OBSERVABLE, SINGLE_CAMERA, TRIANGULATED
from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.faults.rules.rdl_bar_drift import RdlBarDriftRule
from biomechanics.faults.rules.rdl_bar_tilt import RdlBarTiltRule
from biomechanics.faults.rules.rdl_hip_sink import RdlHipSinkRule
from biomechanics.faults.rules.rdl_lockout import RdlLockoutRule
from biomechanics.faults.rules.rdl_partial import build_rejected_faults
from biomechanics.faults.rules.rdl_range_drift import RdlRangeDriftRule
from biomechanics.faults.rules.rdl_tempo import RdlTempoRule
from biomechanics.profiles.romanian_deadlift import RomanianDeadliftProfile
from biomechanics.utils.frame_context import MOUNT_INCLINOMETER, MOUNT_NONE, MOUNT_RACK, FrameContext
from biomechanics.utils.segment_lengths import BodyProportions
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles, Skeleton3D
from test_rdl_hinge_features import rdl_features
from test_rdl_poses import STANDING

LENS_IN_BAND_M = 0.95
LENS_TOO_LOW_M = 0.5
LENS_TOO_HIGH_M = 1.3
TALL_FEMUR_M = 0.48
TALL_TIBIA_M = 0.475
SHORT_FEMUR_M = 0.39
SHORT_TIBIA_M = 0.39
MOUNT_PITCH_DEG = 1.5
STEEP_INCLINOMETER_DEG = 3.0
LENS_CHECK_LOW_M = 0.8
# Past the engine's same-fault dedup interval.
LATER_REP_S = 3.0


def _rdl():
    return BiomechanicsConfig().faults.rdl


def _angles() -> JointAngles:
    return JointAngles(timestamp=1.0, frame_index=10)


def _reference() -> HingeSessionReference:
    return HingeSessionReference(_rdl().rom)


def _measured(**overrides) -> HingeRepFeatures:
    return rdl_features(hinge_hip_source=HIP_SOURCE_MEASURED, **overrides)


def _proportions(femur_m: float, tibia_m: float) -> BodyProportions:
    return BodyProportions(
        hip_width=0.24, femur_length_avg=femur_m, tibia_length_avg=tibia_m, torso_length_avg=0.52,
        shoulder_width=0.38, foot_length_avg=0.18, hip_to_femur_ratio=0.5, tibia_to_reference_ratio=1.0,
        forward_lean_scale=1.0,
    )


def _ctx(
    capture_mode: str = SINGLE_CAMERA,
    pitch_certified: bool = False,
    mount_source: str = MOUNT_NONE,
    mount_pitch_deg: float = math.nan,
    mount_height_m: float = math.nan,
    body_proportions: BodyProportions | None = None,
) -> FrameContext:
    return FrameContext(
        skeleton_3d=Skeleton3D.from_numpy(STANDING.points()),
        angles=None,
        timestamp=0.0,
        phase="idle",
        in_rep=False,
        capture_mode=capture_mode,
        pitch_certified=pitch_certified,
        mount_source=mount_source,
        mount_pitch_deg=mount_pitch_deg,
        mount_height_m=mount_height_m,
        body_proportions=body_proportions,
    )


def _certified(height_m: float = LENS_IN_BAND_M, **fields) -> FrameContext:
    return _ctx(
        pitch_certified=True, mount_source=MOUNT_INCLINOMETER, mount_pitch_deg=MOUNT_PITCH_DEG,
        mount_height_m=height_m, **fields,
    )


def _fault(fault_type: str, severity: FaultSeverity = FaultSeverity.MODERATE) -> FaultEvent:
    return FaultEvent(fault_type=fault_type, severity=severity, severity_score=1.5, message="m", details={"shallow_rep": True})


class TestHipSinkRule:
    @pytest.mark.parametrize(
        "hip_sink, severity",
        [(0.119, None), (0.12, FaultSeverity.MILD), (0.15, FaultSeverity.MODERATE), (0.19, FaultSeverity.SEVERE)],
    )
    def test_band_edges(self, hip_sink: float, severity: FaultSeverity | None):
        fault = RdlHipSinkRule(_rdl().hip_sink).judge_rep(_measured(hip_sink_ratio=hip_sink), _reference(), _angles())
        assert (fault.severity if fault else None) == severity

    def test_fields_and_drift(self):
        reference = _reference()
        reference.update(_measured(hip_sink_ratio=0.05, hinge_deg=70.0))
        fault = RdlHipSinkRule(_rdl().hip_sink).judge_rep(
            _measured(hip_sink_ratio=0.16, hinge_deg=62.0, knee_flexion_bottom_deg=48.0), reference, _angles(),
        )
        assert fault.fault_type == FaultType.RDL_HIP_SINK.value
        assert fault.details["side"] == "both"
        assert fault.details["phase"] == "bottom"
        assert fault.details["is_drift"]
        assert fault.details["hinge_deg"] == 62.0
        assert fault.details["knee_flexion_bottom_deg"] == 48.0
        assert reference.is_drift(DRIFT_HIP_SINK, _rdl().hip_sink.mild)

    def test_not_judged_without_measured_bottom_hips(self):
        rule = RdlHipSinkRule(_rdl().hip_sink)
        assert rule.judge_rep(rdl_features(hinge_hip_source=HIP_SOURCE_SETUP, hip_sink_ratio=0.3), _reference(), _angles()) is None

    def test_the_gross_pitch_guard_downgrades_to_approximate(self):
        engine = RuleEngine(rules=[RdlHipSinkRule(_rdl().hip_sink)], capture_mode=SINGLE_CAMERA, reference=_reference())
        untrusted = engine.finish_rep(_angles(), 1, _measured(hip_sink_ratio=0.2, camera_pitch_untrusted=True))
        later = JointAngles(timestamp=_angles().timestamp + LATER_REP_S)
        trusted = engine.finish_rep(later, 2, _measured(rep_number=2, hip_sink_ratio=0.2))
        assert untrusted[0].details["observability"] == APPROXIMATE
        assert trusted[0].details["observability"] == OBSERVABLE

    def test_the_rule_never_stamps_recorded_only(self):
        fault = RdlHipSinkRule(_rdl().hip_sink).judge_rep(_measured(hip_sink_ratio=0.13), _reference(), _angles())
        assert "recorded_only" not in fault.details


class TestRigRules:
    @pytest.mark.parametrize("offset, severity", [(0.24, None), (0.25, FaultSeverity.MILD), (0.6, FaultSeverity.SEVERE)])
    def test_bar_drift_bands_and_phase(self, offset: float, severity: FaultSeverity | None):
        fault = RdlBarDriftRule(_rdl().bar_drift).judge_rep(
            rdl_features(bar_offset_ratio_p80=offset, bar_offset_peak_phase="bottom"), _reference(), _angles(),
        )
        assert (fault.severity if fault else None) == severity
        if fault:
            assert fault.details["phase"] == "bottom"

    def test_bar_drift_is_rig_only(self):
        rules = [RdlBarDriftRule(_rdl().bar_drift), RdlLockoutRule(_rdl().lockout)]
        features = rdl_features(bar_offset_ratio_p80=0.5, lockout_pitch_deg=20.0)
        mono = RuleEngine(rules=rules, capture_mode=SINGLE_CAMERA, reference=_reference())
        rig = RuleEngine(rules=rules, capture_mode=TRIANGULATED, reference=_reference())
        assert mono.finish_rep(_angles(), 1, features) == []
        assert {fault.details["observability"] for fault in rig.finish_rep(_angles(), 1, features)} == {APPROXIMATE}

    @pytest.mark.parametrize(
        "pitch_deg, severity, direction",
        [
            (9.9, None, None), (10.0, FaultSeverity.MILD, "forward"), (22.0, FaultSeverity.SEVERE, "forward"),
            (-7.9, None, None), (-8.0, FaultSeverity.MILD, "back"), (-12.0, FaultSeverity.MODERATE, "back"),
        ],
    )
    def test_lockout_bands_by_direction(self, pitch_deg: float, severity: FaultSeverity | None, direction: str | None):
        fault = RdlLockoutRule(_rdl().lockout).judge_rep(rdl_features(lockout_pitch_deg=pitch_deg), _reference(), _angles())
        assert (fault.severity if fault else None) == severity
        assert (fault.details["direction"] if fault else None) == direction

    def test_lockout_is_not_judged_on_a_turnaround(self):
        rule = RdlLockoutRule(_rdl().lockout)
        assert rule.judge_rep(rdl_features(lockout_pitch_deg=30.0, started_at_turnaround=True), _reference(), _angles()) is None
        assert rule.judge_rep(rdl_features(), _reference(), _angles()) is None


class TestFrontalRules:
    @pytest.mark.parametrize("tilt, severity", [(3.9, None), (4.0, FaultSeverity.MILD), (7.0, FaultSeverity.MODERATE)])
    def test_bar_tilt_bands_and_side(self, tilt: float, severity: FaultSeverity | None):
        fault = RdlBarTiltRule(_rdl().bar_tilt).judge_rep(
            rdl_features(bar_tilt_deg=tilt, bar_tilt_side="left"), _reference(), _angles(),
        )
        assert (fault.severity if fault else None) == severity
        if fault:
            assert fault.details["side"] == "left"

    def test_range_drift_waits_for_three_counted_reps(self):
        rule = RdlRangeDriftRule(_rdl().range_drift)
        reference = _reference()
        short = _measured(hinge_deg=55.0, range_label="at the knee")
        for _ in range(2):
            reference.update(_measured(hinge_deg=75.0))
            assert rule.judge_rep(short, reference, _angles()) is None
        reference.update(_measured(hinge_deg=75.0))
        fault = rule.judge_rep(short, reference, _angles())
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["is_drift"]
        assert fault.details["reference_deg"] == 75.0
        assert fault.details["range_label"] == "at the knee"

    def test_range_drift_needs_measured_hips(self):
        reference = _reference()
        for _ in range(3):
            reference.update(_measured(hinge_deg=75.0))
        features = rdl_features(hinge_hip_source=HIP_SOURCE_SETUP, hinge_deg=40.0)
        assert RdlRangeDriftRule(_rdl().range_drift).judge_rep(features, reference, _angles()) is None

    @pytest.mark.parametrize(
        "descent_s, severity",
        [(0.67, None), (0.66, FaultSeverity.MILD), (0.50, FaultSeverity.MODERATE), (0.33, FaultSeverity.SEVERE)],
    )
    def test_tempo_bands(self, descent_s: float, severity: FaultSeverity | None):
        fault = RdlTempoRule(_rdl().tempo).judge_rep(rdl_features(descent_time_s=descent_s), _reference(), _angles())
        assert (fault.severity if fault else None) == severity
        if fault:
            assert fault.details["kind"] == "fast_descent"
            assert fault.details["phase"] == "descent"


class TestRejectedFaults:
    def test_a_partial_gets_one_shallow_rdl_partial_fault(self):
        faults = build_rejected_faults(_measured(bar_travel_ratio=0.35, hip_sink_ratio=0.05, range_label="at the knee"), _angles())
        assert [fault.fault_type for fault in faults] == [FaultType.RDL_PARTIAL.value]
        details = faults[0].details
        assert details["shallow_rep"] and details["counted"] is False
        assert details["range_label"] == "at the knee"
        assert faults[0].severity == FaultSeverity.MODERATE

    @pytest.mark.parametrize("travel, severity", [(0.44, FaultSeverity.MILD), (0.38, FaultSeverity.MODERATE), (0.31, FaultSeverity.SEVERE)])
    def test_partial_severity_is_the_shortfall(self, travel: float, severity: FaultSeverity):
        assert build_rejected_faults(_measured(bar_travel_ratio=travel), _angles())[0].severity == severity

    def test_a_squatted_partial_is_a_hip_sink_fault(self):
        faults = build_rejected_faults(_measured(bar_travel_ratio=0.35, hip_sink_ratio=0.16), _angles())
        assert [fault.fault_type for fault in faults] == [FaultType.RDL_HIP_SINK.value]
        assert faults[0].details["shallow_rep"]

    def test_a_hip_sink_below_the_session_start_tier_stays_a_partial(self):
        features = _measured(bar_travel_ratio=0.35, hip_sink_ratio=0.16, hip_sink_start_tier="severe")
        assert [fault.fault_type for fault in build_rejected_faults(features, _angles())] == [FaultType.RDL_PARTIAL.value]

    def test_an_approximate_hip_sink_stays_a_partial(self):
        features = _measured(bar_travel_ratio=0.35, hip_sink_ratio=0.2, camera_pitch_untrusted=True)
        assert [fault.fault_type for fault in build_rejected_faults(features, _angles())] == [FaultType.RDL_PARTIAL.value]

    def test_profile_routes_partials_and_ignores_the_rest(self):
        profile = RomanianDeadliftProfile()
        assert len(profile.on_rep_rejected("partial", _measured(bar_travel_ratio=0.35), _angles())) == 1
        assert profile.on_rep_rejected("ignore", _measured(bar_travel_ratio=0.1), _angles()) == []


class TestShallowRepCue:
    @pytest.mark.parametrize(
        "fault_type, streak, cued, cue",
        [
            ("rdl_partial", 1, 0, None),
            ("rdl_partial", 2, 0, "rdl_lower"),
            ("rdl_partial", 3, 1, "rdl_lower"),
            ("rdl_hip_sink", 2, 1, "rdl_hips_back"),
            ("rdl_partial", 2, 2, None),
            ("rdl_hip_sink", 4, 2, None),
        ],
    )
    def test_cue_matrix(self, fault_type: str, streak: int, cued: int, cue: str | None):
        assert RomanianDeadliftProfile().shallow_rep_cue(_fault(fault_type), streak, cued) == cue


class TestStartTier:
    @pytest.mark.parametrize(
        "ctx, tier",
        [
            pytest.param(_certified(), FaultSeverity.MILD, id="certified-lens-ok"),
            pytest.param(_ctx(mount_height_m=LENS_IN_BAND_M), FaultSeverity.MODERATE, id="uncertified-lens-ok"),
            pytest.param(_certified(LENS_TOO_LOW_M), FaultSeverity.SEVERE, id="certified-lens-0.5m"),
            pytest.param(_ctx(mount_height_m=LENS_TOO_LOW_M), FaultSeverity.SEVERE, id="uncertified-lens-0.5m"),
            pytest.param(_certified(LENS_TOO_HIGH_M), FaultSeverity.SEVERE, id="certified-lens-1.3m"),
            pytest.param(_certified(math.nan), FaultSeverity.SEVERE, id="certified-height-unknown"),
            pytest.param(_ctx(), FaultSeverity.SEVERE, id="no-mount"),
            pytest.param(
                _certified(LENS_CHECK_LOW_M, body_proportions=_proportions(TALL_FEMUR_M, TALL_TIBIA_M)),
                FaultSeverity.SEVERE, id="certified-tall-lifter-0.8m",
            ),
            pytest.param(
                _certified(LENS_CHECK_LOW_M, body_proportions=_proportions(SHORT_FEMUR_M, SHORT_TIBIA_M)),
                FaultSeverity.MILD, id="certified-short-lifter-0.8m",
            ),
            pytest.param(
                _ctx(pitch_certified=True, mount_source=MOUNT_INCLINOMETER, mount_pitch_deg=STEEP_INCLINOMETER_DEG,
                     mount_height_m=LENS_IN_BAND_M),
                FaultSeverity.MODERATE, id="inclinometer-beyond-2deg",
            ),
            pytest.param(
                _ctx(pitch_certified=True, mount_source=MOUNT_RACK, mount_pitch_deg=STEEP_INCLINOMETER_DEG,
                     mount_height_m=LENS_IN_BAND_M),
                FaultSeverity.MILD, id="rack-mount",
            ),
            pytest.param(_ctx(capture_mode=TRIANGULATED), FaultSeverity.MILD, id="rig"),
        ],
    )
    def test_hip_sink_start_tier_truth_table(self, ctx: FrameContext, tier: FaultSeverity):
        assert RomanianDeadliftProfile().cue_start_tier(FaultType.RDL_HIP_SINK.value, ctx) == tier

    def test_rig_only_faults_start_at_the_rig_tier(self):
        profile = RomanianDeadliftProfile()
        for fault_type in ("rdl_bar_drift", "rdl_lockout"):
            assert profile.cue_start_tier(fault_type, _ctx(capture_mode=TRIANGULATED)) == FaultSeverity.MODERATE

    def test_the_rig_tier_follows_the_config(self):
        config = BiomechanicsConfig()
        config.faults.rdl.rig_min_cue_severity = "mild"
        profile = RomanianDeadliftProfile()
        profile.create_fault_rules(config)
        assert profile.cue_start_tier("rdl_bar_drift", _ctx(capture_mode=TRIANGULATED)) == FaultSeverity.MILD

    @pytest.mark.parametrize("fault_type", ["rdl_bar_tilt", "rdl_range_drift", "rdl_tempo", "rdl_partial"])
    def test_other_faults_start_at_mild(self, fault_type: str):
        assert RomanianDeadliftProfile().cue_start_tier(fault_type, _ctx()) == FaultSeverity.MILD
