"""
Shared registrations for the overhead press and Romanian deadlift: fault
types and messages, observability per capture mode (ohp.md §5.9, rdl.md §5),
and their config sub-models under faults.press / faults.rdl.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biomechanics.config import BiomechanicsConfig, load_pipeline_config
from biomechanics.faults.fault_types import FAULT_MESSAGES, FaultType
from biomechanics.faults.observability import (
    APPROXIMATE,
    FAULT_MEASUREMENT,
    NOT_OBSERVABLE,
    OBSERVABLE,
    SINGLE_CAMERA,
    TRIANGULATED,
    fault_observability,
)

PRESS_FAULTS = (
    "press_layback", "press_bar_forward", "press_lockout", "press_leg_drive",
    "press_uneven", "press_grip", "press_velocity_loss", "press_missed",
)
RDL_FAULTS = (
    "rdl_hip_sink", "rdl_bar_drift", "rdl_lockout", "rdl_bar_tilt",
    "rdl_range_drift", "rdl_tempo", "rdl_partial",
)
SEVERITY_TIERS = ("mild", "moderate", "severe")

# (single camera, triangulated), from the plans' observability tables.
EXPECTED_OBSERVABILITY = {
    "press_leg_drive": (OBSERVABLE, OBSERVABLE),
    "press_missed": (OBSERVABLE, OBSERVABLE),
    "press_uneven": (OBSERVABLE, OBSERVABLE),
    "press_grip": (OBSERVABLE, OBSERVABLE),
    "press_lockout": (APPROXIMATE, APPROXIMATE),
    "press_layback": (NOT_OBSERVABLE, APPROXIMATE),
    "press_bar_forward": (NOT_OBSERVABLE, APPROXIMATE),
    "press_velocity_loss": (NOT_OBSERVABLE, APPROXIMATE),
    "rdl_hip_sink": (OBSERVABLE, OBSERVABLE),
    "rdl_range_drift": (OBSERVABLE, OBSERVABLE),
    "rdl_partial": (OBSERVABLE, OBSERVABLE),
    "rdl_tempo": (OBSERVABLE, OBSERVABLE),
    "rdl_bar_tilt": (OBSERVABLE, OBSERVABLE),
    "rdl_bar_drift": (NOT_OBSERVABLE, APPROXIMATE),
    "rdl_lockout": (NOT_OBSERVABLE, APPROXIMATE),
}
SQUAT_MEASUREMENT = {
    "depth": "vertical", "depth_drift": "vertical", "lockout": "vertical", "tempo": "vertical",
    "velocity_loss": "side_view", "knee_valgus": "frontal", "hip_shift": "lateral_travel",
    "hip_shoot": "side_view", "balance": "side_view", "foot_placement": "sagittal",
    "heel_rise": "foot_contact", "bilateral_asymmetry": "bar",
}


class TestFaultTypes:
    @pytest.mark.parametrize("fault_type", PRESS_FAULTS + RDL_FAULTS)
    def test_every_plan_fault_is_a_fault_type_with_three_plain_messages(self, fault_type: str):
        member = FaultType(fault_type)
        assert set(FAULT_MESSAGES[member]) == set(SEVERITY_TIERS)
        assert all(FAULT_MESSAGES[member][tier] for tier in SEVERITY_TIERS)

    @pytest.mark.parametrize("fault_type", PRESS_FAULTS + RDL_FAULTS)
    def test_every_plan_fault_declares_what_it_is_measured_from(self, fault_type: str):
        single, triangulated = EXPECTED_OBSERVABILITY[fault_type]
        assert fault_type in FAULT_MEASUREMENT
        assert fault_observability(fault_type, SINGLE_CAMERA) == single
        assert fault_observability(fault_type, TRIANGULATED) == triangulated

    def test_squat_measurements_are_unchanged(self):
        assert {key: FAULT_MEASUREMENT[key] for key in SQUAT_MEASUREMENT} == SQUAT_MEASUREMENT


class TestExerciseConfig:
    def test_press_defaults_follow_the_plan(self):
        press = BiomechanicsConfig().faults.press
        assert press.counter.velocity_window_s == pytest.approx(0.40)
        assert press.counter.entry_vel_threshold == pytest.approx(12.0)
        assert press.counter.min_depth_cm == pytest.approx(20.0)
        assert (press.uneven.mild, press.uneven.moderate, press.uneven.severe) == (4.0, 6.0, 9.0)
        assert press.acceptance.top_height_min_ratio == pytest.approx(0.80)
        assert press.camera.tripwire_deg == pytest.approx(3.0)
        assert press.hedged_recap_faults == set()
        assert not press.uneven_promoted and not press.tilted_band_validated

    def test_rdl_defaults_follow_the_plan(self):
        rdl = BiomechanicsConfig().faults.rdl
        assert rdl.counter.min_depth_cm == pytest.approx(8.0)
        assert rdl.counter.velocity_window_s is None
        assert (rdl.hip_sink.mild, rdl.hip_sink.moderate, rdl.hip_sink.severe) == (0.12, 0.15, 0.19)
        assert rdl.rom.count_floor_travel == pytest.approx(0.45)
        assert rdl.rig_min_cue_severity == "moderate"
        assert rdl.camera.delta_tripwire_deg == pytest.approx(2.5)
        assert rdl.tempo.mild_seconds == pytest.approx(0.67)

    def test_the_shipped_yaml_parses_to_the_defaults(self):
        config = load_pipeline_config()
        defaults = BiomechanicsConfig()
        assert config.faults.press == defaults.faults.press
        assert config.faults.rdl == defaults.faults.rdl

    def test_yaml_blocks_override_the_defaults(self, tmp_path: Path):
        path = tmp_path / "biomechanics.yaml"
        path.write_text(
            "faults:\n"
            "  press:\n"
            "    uneven: {mild: 5.0, moderate: 7.0, severe: 10.0}\n"
            "    hedged_recap_faults: [press_uneven]\n"
            "    uneven_promoted: true\n"
            "  rdl:\n"
            "    rig_min_cue_severity: mild\n"
            "    counter: {min_depth_cm: 9.0}\n"
        )
        config = load_pipeline_config(str(path))
        assert config.faults.press.uneven.mild == pytest.approx(5.0)
        assert config.faults.press.hedged_recap_faults == {"press_uneven"}
        assert config.faults.press.uneven_promoted
        assert config.faults.rdl.rig_min_cue_severity == "mild"
        assert config.faults.rdl.counter.min_depth_cm == pytest.approx(9.0)
        assert config.faults.knee_valgus == BiomechanicsConfig().faults.knee_valgus

    def test_an_unknown_start_tier_is_rejected(self, tmp_path: Path):
        path = tmp_path / "biomechanics.yaml"
        path.write_text("faults:\n  rdl:\n    rig_min_cue_severity: extreme\n")
        with pytest.raises(ValueError):
            load_pipeline_config(str(path))
