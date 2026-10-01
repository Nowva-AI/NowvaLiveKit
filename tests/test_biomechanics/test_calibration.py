"""Tests for calibration: athlete capacities and the depth target, never fault thresholds."""

from __future__ import annotations

import pytest

from biomechanics.calibration import (
    CalibrationTracker,
    apply_calibration_to_rule_engine,
    build_calibration_profile,
    depth_target_ratio,
)
from biomechanics.config import BiomechanicsConfig, DepthFaultConfig
from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.profiles.squat import SquatProfile

RATIO_TOLERANCE = 1e-9


def _rep_features(depth: float, dorsi: float = 30.0, hip: float = 120.0) -> dict:
    return {
        "depth_ratio": depth,
        "dorsiflexion_max_l": dorsi,
        "dorsiflexion_max_r": dorsi - 2.0,
        "hip_flexion_max_l": hip,
        "hip_flexion_max_r": hip - 3.0,
    }


class TestDepthTarget:
    def test_athlete_who_reaches_parallel_is_held_to_parallel(self):
        assert depth_target_ratio(-0.1, DepthFaultConfig()) == pytest.approx(0.0)

    def test_limited_athlete_is_held_to_their_own_range(self):
        assert depth_target_ratio(0.3, DepthFaultConfig()) == pytest.approx(0.3)

    def test_target_never_shallower_than_a_half_squat(self):
        assert depth_target_ratio(0.8, DepthFaultConfig()) == pytest.approx(0.5)

    def test_unmeasured_capacity_falls_back_to_the_default(self):
        assert depth_target_ratio(None, DepthFaultConfig()) == pytest.approx(0.0)
        assert depth_target_ratio(float("nan"), DepthFaultConfig()) == pytest.approx(0.0)


class TestCalibrationTracker:
    def test_capacities_are_the_median_rep(self):
        tracker = CalibrationTracker(target_reps=3)
        for depth, dorsi in ((0.1, 22.0), (0.3, 26.0), (0.2, 40.0)):
            tracker.on_rep_complete(0.0, _rep_features(depth, dorsi=dorsi))
        peaks = tracker.get_peaks()
        assert peaks["depth_capacity_ratio"] == pytest.approx(0.2)
        assert peaks["peak_dorsiflexion"] == pytest.approx(26.0)
        assert peaks["peak_hip_flexion"] == pytest.approx(120.0)
        assert tracker.is_complete

    def test_reps_without_features_leave_capacities_unmeasured(self):
        tracker = CalibrationTracker(target_reps=1)
        tracker.on_rep_complete(110.0)
        peaks = tracker.get_peaks()
        assert peaks["depth_capacity_ratio"] is None
        assert peaks["peak_dorsiflexion"] is None


class TestCalibrationProfile:
    def test_profile_carries_only_the_depth_target(self):
        profile = build_calibration_profile({"depth_capacity_ratio": 0.25}, BiomechanicsConfig())
        assert profile == {"depth_capacity_ratio": 0.25, "depth_target_ratio": pytest.approx(0.25)}

    def test_applying_a_profile_sets_the_target_and_no_thresholds(self):
        rules = SquatProfile().create_fault_rules(BiomechanicsConfig())
        engine = RuleEngine(rules=rules)
        knee = rules[0]
        before = (knee.mild_threshold, knee.moderate_threshold, knee.severe_threshold)

        apply_calibration_to_rule_engine(engine, {"depth_target_ratio": 0.2})

        assert engine.depth_target_ratio == pytest.approx(0.2, abs=RATIO_TOLERANCE)
        assert (knee.mild_threshold, knee.moderate_threshold, knee.severe_threshold) == before

    def test_old_threshold_profiles_are_ignored(self):
        """Stored per-athlete thresholds came from observed reps — the bug A2 removed."""
        rules = SquatProfile().create_fault_rules(BiomechanicsConfig())
        engine = RuleEngine(rules=rules)
        knee = rules[0]
        before = knee.mild_threshold

        apply_calibration_to_rule_engine(engine, {"knee_valgus": {"mild": 24.0, "moderate": 29.0, "severe": 34.0}})

        assert knee.mild_threshold == before
