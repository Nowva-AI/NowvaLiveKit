"""
Press then squat (docs/exercises/ohp.md §8.4.1): a session that starts on the
squat, switches to a coaching-ready overhead press for real press reps (latch,
counted reps, verdicts, a missed rep) and switches back must reproduce the fresh
squat golden exactly. The press frames stand on the squat session's feet: the
rig's foot-contact anchors are session-scoped and carry over any stance change.
"""

from __future__ import annotations

import pytest

from biomechanics.profiles.overhead_press import OverheadPressProfile
from biomechanics.utils.types import CocoKeypoints as CK
from conftest import SYNTHETIC_TOE_FORWARD_M
from test_press_poses import FPS_RIG, hands_down, hold, points_for, press_rep
from test_squat_golden_master import (
    CAPTURE_FRAME,
    CLOCK_START_S,
    FRAME_DT_S,
    OTHER_EXERCISE,
    OTHER_EXERCISE_LEAD_S,
    SQUAT,
    _check_golden,
    _Session,
    _squat_session,
    _start,
    _switch,
)

# A whole number of frame_data intervals that ends before the squat's clock starts.
PRESS_FRAMES = 240
MISSED_TOP_M = 0.30
SEVERE_TILT_DEG = 10.0


def _press_frames(session: _Session) -> list:
    poses = hands_down(0.5, FPS_RIG) + hold(1.0, FPS_RIG)
    poses += press_rep(FPS_RIG, hold_s=0.5, tilt_top_deg=SEVERE_TILT_DEG) + press_rep(FPS_RIG, hold_s=0.5, top_m=MISSED_TOP_M)
    poses += hold((PRESS_FRAMES - len(poses)) / FPS_RIG, FPS_RIG)
    assert len(poses) == PRESS_FRAMES
    events_before = len(session.events)
    for pose in poses:
        points = points_for(pose)
        points[[CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX], 2] = -SYNTHETIC_TOE_FORWARD_M
        if session.multi_camera:
            session.provider.push(points)
        else:
            session.pose.push(points)
            session.pipeline._publish_frame(CAPTURE_FRAME, session.clock.time())
        session._deliver(session.pipeline.process_frame())
        session.clock.advance(FRAME_DT_S)
    return session.events[events_before:]


@pytest.mark.parametrize(("multi_camera", "golden"), [
    pytest.param(True, "squat_hip_counter_athlete_params", id="rig"),
    pytest.param(False, "squat_single_camera_athlete_params", id="mono"),
])
def test_squat_after_real_press_reps_matches_the_fresh_golden(monkeypatch, multi_camera: bool, golden: str):
    monkeypatch.setattr(OverheadPressProfile, "coaching_ready", True)
    session = _Session(monkeypatch, multi_camera, False, True, CLOCK_START_S - OTHER_EXERCISE_LEAD_S)
    _start(session.pipeline, session.tracker, session.bridge, SQUAT)
    _switch(session.pipeline, session.tracker, session.bridge, OTHER_EXERCISE)
    assert session.pipeline.profile.name == "overhead_press"

    press_events = _press_frames(session)
    reps = [event["rep_data"] for event in press_events if event["rep_data"]]
    assert [rep["rep_number"] for rep in reps] == [1]
    fault_types = {fault["fault_type"] for event in press_events for fault in event["faults"]}
    assert {"press_uneven", "press_missed"} <= fault_types

    _switch(session.pipeline, session.tracker, session.bridge, SQUAT)
    session.rewind(CLOCK_START_S)
    ipc_from = len(session.client.messages)
    _squat_session(session)
    _check_golden(golden, session.snapshot(ipc_from))
