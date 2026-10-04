"""The deadlift simulator (deadlift/simulator.py): reps whose scripts differ
follow one another without the lifter jumping between frames."""

from __future__ import annotations

import numpy as np
import pytest

from biomechanics.deadlift.simulator import RepScript, Scenario, simulate

# 3 m/s at 30 fps: no keypoint of a lifter moves further between two frames (a
# fast hips-first pull peaks at 7.7 cm; a setup swapped in one frame moved 25 cm).
MAX_KEYPOINT_STEP_M = 0.10
HIPS_FIRST = RepScript(setup_trunk_deg=40.0, knee_pass_trunk_deg=55.0, shoulder_ahead_m=0.06)


def _largest_step_m(scenario: Scenario) -> float:
    frames = simulate(scenario).frames
    return max(
        float(np.max(np.linalg.norm(after.points - before.points, axis=1)))
        for before, after in zip(frames, frames[1:])
    )


class TestScriptChanges:
    @pytest.mark.parametrize("floor_hold_s", [1.2, 0.0], ids=["dead_stop", "touch_and_go"])
    def test_a_new_setup_is_reached_over_the_lowering(self, floor_hold_s: float):
        reps = [RepScript(floor_hold_s=floor_hold_s)] * 2 + [HIPS_FIRST.model_copy(update={"floor_hold_s": floor_hold_s})] * 2
        reps.append(RepScript())
        assert _largest_step_m(Scenario(reps=reps)) <= MAX_KEYPOINT_STEP_M

    def test_a_failed_rep_settles_into_the_next_setup(self):
        reps = [RepScript(fail_rise_m=0.25), HIPS_FIRST]
        assert _largest_step_m(Scenario(reps=reps)) <= MAX_KEYPOINT_STEP_M


class TestStickingPoint:
    def test_the_climb_out_of_a_stall_takes_finish_s(self):
        grind = RepScript(pull_s=2.0, stall_fraction=0.95, stall_s=0.8)
        slow_finish = grind.model_copy(update={"finish_s": 1.0})
        scenario = Scenario(reps=[grind])
        frame_s = 1.0 / scenario.fps
        quick = simulate(scenario).reps[0]
        slow = simulate(Scenario(reps=[slow_finish])).reps[0]
        added_s = 1.0 - grind.pull_s * (1.0 - grind.stall_fraction)
        slower_by_s = (slow.top_time - slow.liftoff_time) - (quick.top_time - quick.liftoff_time)
        assert slower_by_s == pytest.approx(added_s, abs=frame_s)
