"""
HingeSessionReference: the athlete's own RDL range and best reps this session
(docs/exercises/rdl.md §3.6, §4.2).
"""

from __future__ import annotations

import math

import pytest

from biomechanics.analysis.hinge_features import HIP_SOURCE_MEASURED, HIP_SOURCE_SETUP
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.hinge_reference import (
    DRIFT_BAR_OFFSET,
    DRIFT_DESCENT_TIME,
    DRIFT_HIP_SINK,
    DRIFT_LOCKOUT_ABS,
    DRIFT_TILT,
    HingeSessionReference,
)
from test_rdl_hinge_features import rdl_features

TEMPO_MILD_S = 0.67


def _reference() -> HingeSessionReference:
    return HingeSessionReference(BiomechanicsConfig().faults.rdl.rom)


def _counted(hinge_deg: float, **overrides):
    return rdl_features(hinge_hip_source=HIP_SOURCE_MEASURED, hinge_deg=hinge_deg, **overrides)


class TestHingeReference:
    def test_no_reference_before_three_counted_reps(self):
        reference = _reference()
        for hinge in (70.0, 72.0):
            reference.update(_counted(hinge))
            assert math.isnan(reference.hinge_reference())

    def test_reference_is_the_median_of_the_three_deepest(self):
        reference = _reference()
        for hinge in (60.0, 74.0, 70.0, 78.0, 65.0):
            reference.update(_counted(hinge))
        assert reference.hinge_reference() == pytest.approx(74.0)

    def test_reference_is_capped(self):
        reference = _reference()
        for _ in range(3):
            reference.update(_counted(110.0))
        assert reference.hinge_reference() == pytest.approx(BiomechanicsConfig().faults.rdl.rom.reference_cap_deg)

    def test_setup_source_hinges_are_not_folded(self):
        reference = _reference()
        for _ in range(3):
            reference.update(rdl_features(hinge_hip_source=HIP_SOURCE_SETUP, hinge_deg=80.0))
        assert math.isnan(reference.hinge_reference())

    def test_range_and_bests_survive_set_boundaries(self):
        reference = _reference()
        for _ in range(3):
            reference.update(_counted(75.0, hip_sink_ratio=0.05))
        reference.start_new_set()
        assert reference.hinge_reference() == pytest.approx(75.0)
        assert reference.best(DRIFT_HIP_SINK) == pytest.approx(0.05)

    def test_every_rep_reaches_the_absent_depth_target(self):
        assert _reference().reaches_depth_target(math.nan)


class TestHighlight:
    def test_before_a_reference_the_best_rep_is_the_most_hips_back(self):
        reference = _reference()
        features = _counted(60.0, hip_sink_ratio=0.04)
        reference.annotate(features)
        assert features.highlight_score == pytest.approx(-0.04)
        assert math.isnan(features.hinge_reference_deg)

    def test_reps_short_of_the_athletes_range_score_nothing(self):
        reference = _reference()
        for _ in range(3):
            reference.update(_counted(75.0))
        within = _counted(68.0, hip_sink_ratio=0.03)
        short = _counted(66.0, hip_sink_ratio=0.01)
        reference.annotate(within)
        reference.annotate(short)
        assert within.highlight_score == pytest.approx(-0.03)
        assert within.hinge_reference_deg == pytest.approx(75.0)
        assert math.isnan(short.highlight_score)

    def test_no_score_without_a_measured_hip_sink(self):
        features = rdl_features(hinge_hip_source=HIP_SOURCE_SETUP, hinge_deg=75.0)
        _reference().annotate(features)
        assert math.isnan(features.highlight_score)


class TestDriftBests:
    def test_lower_is_better_metrics_keep_the_minimum(self):
        reference = _reference()
        reference.update(_counted(70.0, hip_sink_ratio=0.08, bar_offset_ratio_p80=0.3, bar_tilt_deg=5.0, lockout_pitch_deg=-9.0))
        reference.update(_counted(70.0, hip_sink_ratio=0.03, bar_offset_ratio_p80=0.1, bar_tilt_deg=2.0, lockout_pitch_deg=4.0))
        assert reference.best(DRIFT_HIP_SINK) == pytest.approx(0.03)
        assert reference.best(DRIFT_BAR_OFFSET) == pytest.approx(0.1)
        assert reference.best(DRIFT_TILT) == pytest.approx(2.0)
        assert reference.best(DRIFT_LOCKOUT_ABS) == pytest.approx(4.0)
        assert reference.is_drift(DRIFT_TILT, 4.0)
        assert not reference.is_drift(DRIFT_HIP_SINK, 0.02)

    def test_descent_time_drift_means_an_earlier_rep_was_controlled(self):
        reference = _reference()
        assert not reference.is_drift(DRIFT_DESCENT_TIME, TEMPO_MILD_S)
        reference.update(_counted(70.0, descent_time_s=0.5))
        assert not reference.is_drift(DRIFT_DESCENT_TIME, TEMPO_MILD_S)
        reference.update(_counted(70.0, descent_time_s=1.4))
        reference.update(_counted(70.0, descent_time_s=0.4))
        assert reference.best(DRIFT_DESCENT_TIME) == pytest.approx(1.4)
        assert reference.is_drift(DRIFT_DESCENT_TIME, TEMPO_MILD_S)
