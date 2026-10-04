"""One deadlift frame measured in the sagittal frame (deadlift/measure.py): planted
feet behind the plates, a bar carried over a tracking gap, the subject's left end
of the bar, and the hands-on-bar band."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.config import DeadliftConfig
from biomechanics.deadlift.frame import default_up, height_m
from biomechanics.deadlift.measure import (
    FOOT_KEYPOINTS,
    DeadliftFrameInput,
    MeasureContext,
    feet_measured,
    hands_on_bar,
    measure_frame,
)
from biomechanics.deadlift.simulator import RepScript, Scenario, SimFrame, simulate
from biomechanics.deadlift.types import BAR_SOURCE_BAR, BAR_SOURCE_WRIST_PROXY, BarState3D
from biomechanics.utils.types import CocoKeypoints as CK

POSITION_TOLERANCE_M = 1e-9
GRIP_OFFSET_M = 0.075


def _setup_frame() -> SimFrame:
    """A frame of the lifter set up on the bar, hands on it."""
    sim = simulate(Scenario(reps=[RepScript()]))
    return next(frame for frame in sim.frames if frame.timestamp >= sim.reps[0].liftoff_time - 1e-6)


def _context(**overrides) -> MeasureContext:
    values = dict(
        up=default_up(), rest_source=None, rest_axis=None, grip_offset_m=GRIP_OFFSET_M,
        carried_bar=None, planted_feet=None,
    )
    values.update(overrides)
    return MeasureContext(**values)


def _input(frame: SimFrame, confidences: np.ndarray | None = None, bar: BarState3D | None = None) -> DeadliftFrameInput:
    return DeadliftFrameInput(
        frame.timestamp, frame.frame_index, frame.points,
        frame.confidences if confidences is None else confidences, bar,
    )


class TestPlantedFeet:
    def test_hidden_feet_without_planted_ones_lose_the_frame(self):
        frame = _setup_frame()
        confidences = frame.confidences.copy()
        confidences[list(FOOT_KEYPOINTS)] = 0.0
        assert measure_frame(_input(frame, confidences, frame.bar), _context(), DeadliftConfig()) is None

    def test_planted_feet_stand_in_for_hidden_ones(self):
        frame = _setup_frame()
        confidences = frame.confidences.copy()
        confidences[list(FOOT_KEYPOINTS)] = 0.0
        planted = {index: frame.points[index].copy() for index in FOOT_KEYPOINTS}
        measured = measure_frame(_input(frame, confidences, frame.bar), _context(planted_feet=planted), DeadliftConfig())
        expected_ankle_mid = (frame.points[CK.LEFT_ANKLE] + frame.points[CK.RIGHT_ANKLE]) / 2.0
        assert measured.ankle_mid == pytest.approx(expected_ankle_mid, abs=POSITION_TOLERANCE_M)
        assert measured.midfoot is not None
        assert not measured.feet_measured

    def test_feet_measured_reads_each_foot_keypoint(self):
        confidences = np.full(21, 0.9)
        confidences[CK.LEFT_FOOT_INDEX] = 0.0
        measured = feet_measured(confidences)
        assert not measured[CK.LEFT_FOOT_INDEX]
        assert measured[CK.RIGHT_ANKLE]


class TestBarSources:
    def test_a_carried_bar_gives_geometry_but_no_height(self):
        frame = _setup_frame()
        measured = measure_frame(_input(frame, bar=None), _context(carried_bar=frame.bar), DeadliftConfig())
        assert measured.bar_source == BAR_SOURCE_BAR
        assert measured.bar_centre == pytest.approx(frame.bar.centre, abs=POSITION_TOLERANCE_M)
        assert math.isnan(measured.bar_up)
        assert not measured.bar_measured
        assert measured.hands_on_bar

    def test_a_predicted_bar_keeps_its_geometry_but_has_no_height(self):
        frame = _setup_frame()
        predicted = frame.bar.model_copy(update={"predicted": True})
        measured = measure_frame(_input(frame, bar=predicted), _context(), DeadliftConfig())
        assert math.isnan(measured.bar_up)
        assert not measured.bar_measured
        assert measured.bar_centre is not None
        assert measured.hands_on_bar

    def test_without_a_bar_the_wrists_stand_in(self):
        frame = _setup_frame()
        measured = measure_frame(_input(frame, bar=None), _context(), DeadliftConfig())
        wrist_mid = (frame.points[CK.LEFT_WRIST] + frame.points[CK.RIGHT_WRIST]) / 2.0
        assert measured.bar_source == BAR_SOURCE_WRIST_PROXY
        assert measured.bar_up == pytest.approx(height_m(measured.frame, wrist_mid, np.zeros(3)) - GRIP_OFFSET_M, abs=1e-9)

    def test_a_set_on_the_tracked_bar_never_falls_back_to_the_wrists(self):
        frame = _setup_frame()
        measured = measure_frame(_input(frame, bar=None), _context(rest_source=BAR_SOURCE_BAR), DeadliftConfig())
        assert measured.bar_centre is None
        assert measured.bar_source == BAR_SOURCE_BAR

    def test_the_subjects_left_end_whatever_the_tracker_calls_it(self):
        frame = _setup_frame()
        swapped = frame.bar.model_copy(update={"left_end_m": frame.bar.right_end_m, "right_end_m": frame.bar.left_end_m})
        measured = measure_frame(_input(frame, bar=swapped), _context(), DeadliftConfig())
        # The simulator's subject's left is +X.
        assert measured.bar_left[0] > measured.bar_right[0]


class TestHandsOnBar:
    @pytest.mark.parametrize(
        ("wrist_above_bar_m", "on_bar"),
        [(0.075, True), (0.15, True), (-0.05, True), (0.20, False), (-0.08, False)],
    )
    def test_the_band_around_the_bar(self, wrist_above_bar_m: float, on_bar: bool):
        frame = _setup_frame()
        measured = measure_frame(_input(frame, bar=frame.bar), _context(), DeadliftConfig())
        wrist = measured.bar_centre + wrist_above_bar_m * measured.frame.up
        result = hands_on_bar(measured.frame, BAR_SOURCE_BAR, measured.bar_centre, [wrist], measured.knee_mid, 60.0, DeadliftConfig())
        assert result is on_bar

    def test_on_the_proxy_it_is_a_hinge_with_the_hands_below_the_knees(self):
        frame = _setup_frame()
        measured = measure_frame(_input(frame, bar=None), _context(), DeadliftConfig())
        wrists = [measured.l_wrist, measured.r_wrist]
        config = DeadliftConfig()
        assert hands_on_bar(measured.frame, BAR_SOURCE_WRIST_PROXY, measured.bar_centre, wrists, measured.knee_mid, 60.0, config)
        assert not hands_on_bar(measured.frame, BAR_SOURCE_WRIST_PROXY, measured.bar_centre, wrists, measured.knee_mid, 10.0, config)
