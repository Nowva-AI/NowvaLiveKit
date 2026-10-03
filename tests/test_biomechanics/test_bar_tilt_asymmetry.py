"""Tests for the barbell-tilt bilateral asymmetry rule."""

from __future__ import annotations

from collections import deque

import pytest

from biomechanics.faults.rules.bar_tilt_asymmetry import BarTiltAsymmetryRule
from biomechanics.utils.types import (
    BarbellDetection,
    FaultSeverity,
    JointAngles,
    Keypoint2D,
)


def _make_angles(frame_index: int) -> JointAngles:
    return JointAngles(frame_index=frame_index, timestamp=frame_index * (1.0 / 30.0))


def _detection(left_y: float, right_y: float, bar_len_px: float = 500.0) -> BarbellDetection:
    """Build a detection with a known bar pixel length and per-endpoint y-values."""
    return BarbellDetection(
        left_end=Keypoint2D(x=100.0, y=left_y, confidence=0.9),
        right_end=Keypoint2D(x=100.0 + bar_len_px, y=right_y, confidence=0.9),
        bbox_conf=0.9,
    )


def _run_rep(rule: BarTiltAsymmetryRule, detections, rep_number: int = 1):
    """Feed a rep's detections with in_rep=True, then complete it the way the engine does."""
    history = deque(maxlen=90)
    for i, det in enumerate(detections):
        rule.set_frame_context(bar_detection=det)
        rule.evaluate(_make_angles(i), history, in_rep=True, rep_number=rep_number)
    return rule.finish_rep(_make_angles(len(detections)), rep_number)


def test_no_detections_returns_none():
    rule = BarTiltAsymmetryRule()
    history = deque(maxlen=90)

    # Whole rep with no detections at all
    for i in range(10):
        rule.set_frame_context(bar_detection=None)
        rule.evaluate(_make_angles(i), history, in_rep=True, rep_number=1)

    fault = rule.finish_rep(_make_angles(10), rep_number=1)
    assert fault is None


def test_fault_lands_on_the_rep_that_tilted():
    """The verdict used to arrive on the next frame tagged with the next rep's number."""
    rule = BarTiltAsymmetryRule()
    fault = _run_rep(rule, [_detection(300.0, 340.0)] * 10, rep_number=3)
    assert fault is not None
    assert fault.rep_number == 3
    assert fault.details["side"] == "right"


def test_evaluate_never_emits_on_its_own():
    rule = BarTiltAsymmetryRule()
    history = deque(maxlen=90)
    for i in range(10):
        rule.set_frame_context(bar_detection=_detection(300.0, 340.0))
        assert rule.evaluate(_make_angles(i), history, in_rep=True, rep_number=1) is None
    rule.set_frame_context(bar_detection=None)
    assert rule.evaluate(_make_angles(10), history, in_rep=False, rep_number=2) is None


def test_horizontal_bar_no_fault():
    rule = BarTiltAsymmetryRule()
    # Perfectly horizontal bar across the whole rep
    dets = [_detection(left_y=300.0, right_y=300.0) for _ in range(20)]
    fault = _run_rep(rule, dets)
    assert fault is None


def test_progressive_left_drop_triggers_severe_fault():
    """Left end dropping (higher y) → heavier_side should be 'left'."""
    rule = BarTiltAsymmetryRule(
        bar_length_m=2.2,
        mild_deg=2.0, moderate_deg=4.0, severe_deg=7.0,
        mild_cm=3.0, moderate_cm=6.0, severe_cm=10.0,
    )

    # Bar length 500 px represents 2.2 m → cm_per_px = 0.44
    # To hit severe (>10 cm height diff), we need > ~22.7 px differential.
    # Progress from 0 → 30 px left-drop over 15 frames.
    dets = []
    for i in range(15):
        left_drop = i * 2.0  # 0, 2, 4, ..., 28
        dets.append(_detection(left_y=300.0 + left_drop, right_y=300.0))

    fault = _run_rep(rule, dets)
    assert fault is not None
    assert fault.severity == FaultSeverity.SEVERE
    assert fault.details["heavier_side"] == "left"
    assert fault.details["source"] == "barbell_tilt"
    assert fault.details["peak_height_diff_cm"] > 10.0


def test_right_drop_attributes_to_right_side():
    rule = BarTiltAsymmetryRule()
    # Right side drops (right_y > left_y)
    dets = [_detection(left_y=300.0, right_y=300.0 + i * 1.5) for i in range(15)]
    fault = _run_rep(rule, dets)
    assert fault is not None
    assert fault.details["heavier_side"] == "right"


def test_mild_tilt_produces_mild_severity():
    # Thresholds chosen so drop=10px on a 500px bar (≈1.15° tilt, ≈4.4cm
    # height diff) lands cleanly in the mild band on both axes.
    rule = BarTiltAsymmetryRule(
        mild_deg=1.0, moderate_deg=3.0, severe_deg=5.0,
        mild_cm=2.0, moderate_cm=10.0, severe_cm=20.0,
    )
    dets = [_detection(left_y=300.0 + 10.0, right_y=300.0) for _ in range(10)]
    fault = _run_rep(rule, dets)
    assert fault is not None
    assert fault.severity == FaultSeverity.MILD


def test_rep_metrics_reset_between_reps():
    """A severe first rep must not leak into a clean second rep."""
    rule = BarTiltAsymmetryRule()

    # Rep 1: severe
    bad = [_detection(left_y=300.0 + i * 2.0, right_y=300.0) for i in range(15)]
    fault1 = _run_rep(rule, bad, rep_number=1)
    assert fault1 is not None

    # Rep 2: horizontal
    good = [_detection(left_y=300.0, right_y=300.0) for _ in range(15)]
    fault2 = _run_rep(rule, good, rep_number=2)
    assert fault2 is None
