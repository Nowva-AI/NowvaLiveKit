"""Per-rep deadlift features (deadlift/features.py): the event times fitted as the
change point of the bar leaving or reaching a level, and the hands carried out to
the plate hubs when the wrists stand in for the bar."""

from __future__ import annotations

import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.analyzer import DeadliftFrameInput, DeadliftRepAnalyzer
from biomechanics.deadlift.features import RepTrack, floor_time, liftoff_time, top_time
from biomechanics.deadlift.frame import default_up
from biomechanics.deadlift.measure import FrameMeasure, MeasureContext, measure_frame
from biomechanics.deadlift.simulator import RepScript, Scenario, simulate
from biomechanics.deadlift.types import GRAVITY_SOURCE_MEASURED

FRAME_DT_S = 1.0 / 30.0
EVENT_TOLERANCE_S = 0.01
BAND_M = 0.005
REST_UP = 0.0
# Half the bar's acceleration (m/s^2): a slow grind, whose first 0.5 cm takes 0.22 s.
CURVATURE = 0.1


def _template() -> FrameMeasure:
    sim = simulate(Scenario(reps=[RepScript()]))
    frame = sim.frames[0]
    context = MeasureContext(default_up(), None, None, 0.075, None, None, False)
    return measure_frame(
        DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar),
        context, BiomechanicsConfig().deadlift,
    )


TEMPLATE = _template()


def _frames(heights: list[tuple[float, float]]) -> list[FrameMeasure]:
    return [TEMPLATE._replace(t=t, bar_up=height) for t, height in heights]


def _rep(heights: list[tuple[float, float]], liftoff_index: int) -> RepTrack:
    frames = _frames(heights)
    rep = RepTrack(frames[liftoff_index], touch_and_go=False, setup={})
    rep.lead_in = frames[:liftoff_index]
    for frame in frames[liftoff_index + 1:]:
        rep.append(frame, 0.0)
    rep.top_time = frames[-1].t
    return rep


class TestEventTimes:
    def test_a_slow_departure_is_dated_where_it_began_not_where_it_cleared_the_band(self):
        onset = 1.0
        times = [index * FRAME_DT_S for index in range(60)]
        heights = [(t, REST_UP + CURVATURE * max(0.0, t - onset) ** 2) for t in times]
        liftoff_index = max(index for index, (_, height) in enumerate(heights) if height <= BAND_M)
        rep = _rep(heights, liftoff_index)
        assert heights[liftoff_index][0] - onset > 0.1
        assert liftoff_time(rep, REST_UP, BAND_M) == pytest.approx(onset, abs=EVENT_TOLERANCE_S)

    def test_a_fast_departure_keeps_its_detecting_frame(self):
        heights = [(0.0, 0.0), (FRAME_DT_S, 0.0), (2 * FRAME_DT_S, 0.04), (3 * FRAME_DT_S, 0.10)]
        rep = _rep(heights, 1)
        assert liftoff_time(rep, REST_UP, BAND_M) == pytest.approx(FRAME_DT_S, abs=1e-9)

    def test_the_top_is_the_arrival_at_the_level_held_there(self):
        arrival = 1.0
        top_up = 0.5
        times = [index * FRAME_DT_S for index in range(60)]
        heights = [(t, top_up - CURVATURE * max(0.0, arrival - t) ** 2) for t in times]
        rep = _rep(heights, 0)
        rep.top_up = top_up
        rep.top_frames = [frame for frame in rep.frames if frame.t >= arrival]
        rep.top_time = next(frame.t for frame in rep.frames if frame.bar_up >= top_up - BAND_M)
        assert rep.top_time < arrival - 0.05
        assert top_time(rep, BAND_M) == pytest.approx(arrival, abs=EVENT_TOLERANCE_S)

    def test_touchdown_is_where_the_lowering_reached_the_rest(self):
        touchdown = 1.0
        times = [index * FRAME_DT_S for index in range(60)]
        heights = [(t, REST_UP + CURVATURE * max(0.0, touchdown - t) ** 2) for t in times]
        rep = _rep(heights, 0)
        rep.lower_start = 0.0
        rep.floor_time = next(frame.t for frame in rep.frames if frame.bar_up <= BAND_M)
        assert touchdown - rep.floor_time > 0.05
        assert floor_time(rep, REST_UP, BAND_M) == pytest.approx(touchdown, abs=EVENT_TOLERANCE_S)

    def test_a_touch_and_go_floor_is_its_low_point(self):
        rep = _rep([(0.0, 0.1), (FRAME_DT_S, 0.0), (2 * FRAME_DT_S, 0.05)], 0)
        rep.lower_start = 0.0
        rep.floor_time = FRAME_DT_S
        rep.touch_and_go_out = True
        assert floor_time(rep, REST_UP, BAND_M) == FRAME_DT_S


class TestWristProxyTilt:
    def _tilt(self, bar_tilt_m: float) -> tuple[float, str]:
        sim = simulate(Scenario(track_bar=False, reps=[RepScript(bar_tilt_m=bar_tilt_m)]))
        analyzer = DeadliftRepAnalyzer(BiomechanicsConfig().deadlift)
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        features = []
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, None))
            while analyzer.take_completed_rep() is not None:
                features.append(analyzer.finish_rep())
        return features[0].bar_tilt_cm, features[0].bar_low_side

    def test_the_hands_tilt_is_carried_out_to_the_plate_hubs(self):
        tilt_cm, low_side = self._tilt(0.06)
        assert tilt_cm == pytest.approx(6.0, abs=0.6)
        assert low_side == "left"

    def test_level_hands_read_a_level_bar(self):
        tilt_cm, _ = self._tilt(0.0)
        assert tilt_cm == pytest.approx(0.0, abs=0.2)
