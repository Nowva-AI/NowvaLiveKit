"""
Press fault rules (docs/exercises/ohp.md §5): one verdict per rep from the press
features — tiers, side, phase, drift, NaN -> no verdict, the measured-gravity and
pitch-band observability caps, and grip judged once per set on the first valid rep.
"""

from __future__ import annotations

import math

import pytest

from biomechanics.analysis.press_features import (
    DRIFT_LEG_DRIVE,
    PITCH_BAND_LEVEL,
    PITCH_BAND_TILTED,
    PITCH_BAND_UNKNOWN,
    PressRepFeatures,
    PressSessionReference,
)
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.observability import APPROXIMATE, SINGLE_CAMERA, TRIANGULATED
from biomechanics.faults.rules.press_bar_forward import PressBarForwardRule
from biomechanics.faults.rules.press_grip import PressGripRule
from biomechanics.faults.rules.press_layback import PressLaybackRule
from biomechanics.faults.rules.press_leg_drive import PressLegDriveRule
from biomechanics.faults.rules.press_lockout import PressLockoutRule
from biomechanics.faults.rules.press_uneven import PressUnevenRule
from biomechanics.profiles.overhead_press import OverheadPressProfile
from biomechanics.utils.frame_context import GRAVITY_BODY, GRAVITY_MEASURED
from biomechanics.utils.types import FaultSeverity, JointAngles

PRESS = BiomechanicsConfig().faults.press
REP = 3


def _features(**overrides) -> PressRepFeatures:
    values = {
        "rep_number": REP, "capture_mode": TRIANGULATED, "pitch_band": PITCH_BAND_UNKNOWN,
        "gravity_source": GRAVITY_MEASURED, "facing_ok": True,
    }
    values.update(overrides)
    return PressRepFeatures(**values)


def _angles() -> JointAngles:
    return JointAngles(timestamp=12.5)


def _judge(rule, features: PressRepFeatures, reference: PressSessionReference | None = None):
    return rule.judge_rep(features, reference or PressSessionReference(), _angles())


def _rules() -> dict[str, object]:
    return {rule.fault_type.value: rule for rule in OverheadPressProfile().create_fault_rules(BiomechanicsConfig())}


class TestLayback:
    @pytest.mark.parametrize(("value", "severity"), [
        (11.9, None), (12.0, FaultSeverity.MILD), (18.0, FaultSeverity.MODERATE), (25.0, FaultSeverity.SEVERE),
    ])
    def test_tiers(self, value: float, severity):
        fault = _judge(PressLaybackRule(PRESS.layback), _features(layback_deg=value))
        assert (fault.severity if fault else None) == severity

    def test_details_and_phase(self):
        fault = _judge(PressLaybackRule(PRESS.layback), _features(layback_deg=20.0))
        assert fault.details["phase"] == "press"
        assert fault.details["side"] is None
        assert fault.details["gravity_source"] == GRAVITY_MEASURED
        assert "observability" not in fault.details
        assert fault.rep_number == REP

    def test_body_gravity_is_capped_approximate(self):
        fault = _judge(PressLaybackRule(PRESS.layback), _features(layback_deg=30.0, gravity_source=GRAVITY_BODY))
        assert fault.details["observability"] == APPROXIMATE

    def test_nan_gives_no_verdict(self):
        assert _judge(PressLaybackRule(PRESS.layback), _features()) is None


class TestBarForward:
    def test_measured_gravity_reads_the_midfoot_metric(self):
        fault = _judge(PressBarForwardRule(PRESS.bar_forward), _features(
            bar_over_midfoot_ratio=0.18, bar_forward_shoulder_ratio=0.0,
        ))
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["metric"] == "midfoot"
        assert fault.details["phase"] == "lockout"
        assert "observability" not in fault.details

    def test_body_gravity_falls_back_to_the_shoulder_metric_capped(self):
        fault = _judge(PressBarForwardRule(PRESS.bar_forward), _features(
            gravity_source=GRAVITY_BODY, bar_over_midfoot_ratio=0.30, bar_forward_shoulder_ratio=0.11,
        ))
        assert fault.severity == FaultSeverity.MILD
        assert fault.details["metric"] == "shoulder"
        assert fault.details["observability"] == APPROXIMATE

    def test_bar_behind_is_no_fault(self):
        assert _judge(PressBarForwardRule(PRESS.bar_forward), _features(bar_over_midfoot_ratio=-0.3)) is None


class TestLockout:
    @pytest.mark.parametrize(("left", "right", "severity", "side"), [
        (10.0, 12.0, None, None),
        (16.0, 14.0, FaultSeverity.MILD, "both"),
        (26.0, 10.0, FaultSeverity.MODERATE, "left"),
        (5.0, 36.0, FaultSeverity.SEVERE, "right"),
        (math.nan, 27.0, FaultSeverity.MODERATE, "both"),
    ])
    def test_worse_side_and_its_name(self, left: float, right: float, severity, side):
        rule = PressLockoutRule(PRESS.lockout, PRESS.lockout_side_min_difference_deg)
        fault = _judge(rule, _features(lockout_elbow_l_deg=left, lockout_elbow_r_deg=right))
        assert (fault.severity if fault else None) == severity
        if fault:
            assert fault.details["side"] == side
            assert fault.details["phase"] == "lockout"

    def test_both_arms_unmeasured_gives_no_verdict(self):
        rule = PressLockoutRule(PRESS.lockout, PRESS.lockout_side_min_difference_deg)
        assert _judge(rule, _features()) is None


class TestLegDrive:
    def test_tiers_and_phase(self):
        rule = PressLegDriveRule(PRESS.leg_drive, tilted_band_validated=False)
        assert _judge(rule, _features(leg_drive_ratio=0.03)) is None
        fault = _judge(rule, _features(leg_drive_ratio=0.07, leg_drive_phase="setup"))
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["phase"] == "setup"
        assert fault.details["side"] is None

    def test_drift_only_after_a_clean_best_rep(self):
        rule = PressLegDriveRule(PRESS.leg_drive, tilted_band_validated=False)
        reference = PressSessionReference()
        assert not _judge(rule, _features(leg_drive_ratio=0.07), reference).details["is_drift"]
        reference.update(_features(leg_drive_ratio=0.01))
        assert reference.best(DRIFT_LEG_DRIVE) == pytest.approx(0.01)
        assert _judge(rule, _features(leg_drive_ratio=0.07), reference).details["is_drift"]

    @pytest.mark.parametrize(("band", "validated", "capped"), [
        (PITCH_BAND_LEVEL, False, False),
        (PITCH_BAND_TILTED, False, True),
        (PITCH_BAND_TILTED, True, False),
        (PITCH_BAND_UNKNOWN, True, True),
    ])
    def test_single_camera_band_cap(self, band: str, validated: bool, capped: bool):
        rule = PressLegDriveRule(PRESS.leg_drive, tilted_band_validated=validated)
        fault = _judge(rule, _features(leg_drive_ratio=0.11, capture_mode=SINGLE_CAMERA, pitch_band=band))
        assert (fault.details.get("observability") == APPROXIMATE) == capped
        assert fault.details["pitch_band"] == band

    def test_rig_is_never_band_capped(self):
        rule = PressLegDriveRule(PRESS.leg_drive, tilted_band_validated=False)
        fault = _judge(rule, _features(leg_drive_ratio=0.11, pitch_band=PITCH_BAND_UNKNOWN))
        assert "observability" not in fault.details


class TestUneven:
    @pytest.mark.parametrize(("tilt", "severity", "side"), [
        (3.9, None, None),
        (4.5, FaultSeverity.MILD, "right"),
        (-7.0, FaultSeverity.MODERATE, "left"),
        (9.5, FaultSeverity.SEVERE, "right"),
    ])
    def test_tiers_and_the_lagging_hand(self, tilt: float, severity, side):
        rule = PressUnevenRule(PRESS.uneven, tilted_band_validated=False)
        fault = _judge(rule, _features(bar_tilt_deg=tilt, bar_tilt_phase="press"))
        assert (fault.severity if fault else None) == severity
        if fault:
            assert fault.details["side"] == side
            assert fault.details["phase"] == "press"
            assert fault.details["value"] == pytest.approx(abs(tilt))

    def test_nan_gives_no_verdict(self):
        assert _judge(PressUnevenRule(PRESS.uneven, tilted_band_validated=False), _features()) is None


class TestGrip:
    def test_judged_once_per_set_on_the_first_valid_rep(self):
        rule = PressGripRule(PRESS.grip)
        assert _judge(rule, _features(grip_tilt_deg=math.nan)) is None
        assert _judge(rule, _features(grip_tilt_deg=30.0, facing_ok=False)) is None
        first = _judge(rule, _features(grip_tilt_deg=4.0))
        assert first is None
        assert _judge(rule, _features(grip_tilt_deg=30.0)) is None
        rule.reset()
        fault = _judge(rule, _features(grip_tilt_deg=16.0))
        assert fault.severity == FaultSeverity.MODERATE

    def test_never_judged_on_a_touch_and_go_rep(self):
        rule = PressGripRule(PRESS.grip)
        assert _judge(rule, _features(grip_tilt_deg=25.0, piston=True)) is None
        assert _judge(rule, _features(grip_tilt_deg=25.0)).severity == FaultSeverity.SEVERE

    @pytest.mark.parametrize(("tilt", "direction"), [(16.0, "narrower"), (-16.0, "wider")])
    def test_direction_is_the_correction(self, tilt: float, direction: str):
        fault = _judge(PressGripRule(PRESS.grip), _features(grip_tilt_deg=tilt))
        assert fault.details["direction"] == direction
        assert fault.details["phase"] == "setup"
        assert fault.details["signed_value"] == pytest.approx(tilt)

    def test_the_verdict_is_a_setup_note(self):
        fault = _judge(PressGripRule(PRESS.grip), _features(grip_tilt_deg=25.0))
        assert fault.details["setup_note"] is True


class TestVelocityLoss:
    def test_press_velocity_loss_is_its_own_fault_and_phase(self):
        rule = _rules()["press_velocity_loss"]
        fault = _judge(rule, _features(velocity_loss_pct=32.0, concentric_velocity_mps=0.3))
        assert fault.fault_type == "press_velocity_loss"
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["phase"] == "press"
        assert fault.details["is_drift"]


class TestProfileRuleSet:
    def test_rules_in_priority_order_with_config_tiers(self):
        rules = OverheadPressProfile().create_fault_rules(BiomechanicsConfig())
        assert [rule.fault_type.value for rule in rules] == [
            "press_layback", "press_bar_forward", "press_lockout", "press_leg_drive",
            "press_uneven", "press_velocity_loss", "press_grip",
        ]

    def test_thresholds_come_from_config(self):
        config = BiomechanicsConfig()
        config.faults.press.uneven.mild = 5.0
        rule = {r.fault_type.value: r for r in OverheadPressProfile().create_fault_rules(config)}["press_uneven"]
        assert _judge(rule, _features(bar_tilt_deg=4.5)) is None
