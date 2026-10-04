"""Smoke tests for scripts/tools/deadlift_envelope.py, the script that re-measures
the deadlift's simulated envelope in docs/deadlift/IMPLEMENTATION.md: its noise
draws, its per-set bookkeeping and one sweep end to end.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "tools" / "deadlift_envelope.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

from biomechanics.deadlift.simulator import RepScript, Scenario, simulate  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402

DRAW_SIZE = 8
MAX_CLEAN_TOP_ERROR_S = 0.1
SETTLE_RISE_M = 0.01
SETTLE_AFTER_TOP_S = 0.3
SETTLE_RAMP_S = 0.3
HEIGHT_TOLERANCE_M = 1e-9
COMPUTE_ROWS = 4
SAG_M = 0.015
LOST_BEFORE_TOP_S = 0.3
LOST_AFTER_TOP_S = 0.5
FRAME_S = 1.0 / 30.0
SCRIPT_TIMEOUT_S = 300


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("deadlift_envelope", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


envelope = _load_script()


@pytest.fixture
def unsalted() -> Iterator[None]:
    envelope.SALT["value"] = 0
    yield
    envelope.SALT["value"] = 0


class TestNoiseDraws:
    def test_the_same_row_draws_the_same_noise(self, unsalted: None):
        first = envelope._rng("clean", 0.02, "default", 3).normal(size=DRAW_SIZE)
        again = envelope._rng("clean", 0.02, "default", 3).normal(size=DRAW_SIZE)
        assert first == pytest.approx(again, abs=0.0)

    @pytest.mark.parametrize("other_key", [
        ("clean", 0.024, "default", 3), ("clean", 0.02, "short", 3), ("clean", 0.02, "default", 4),
        ("no_pause", 0.02, "default", 3),
    ], ids=["noise_level", "body", "seed", "sweep"])
    def test_every_key_draws_its_own_noise(self, unsalted: None, other_key: tuple[object, ...]):
        first = envelope._rng("clean", 0.02, "default", 3).normal(size=DRAW_SIZE)
        other = envelope._rng(*other_key).normal(size=DRAW_SIZE)
        assert not np.allclose(first, other)

    def test_a_salt_draws_every_row_afresh(self, unsalted: None):
        docs_draw = envelope._rng("clean", 0.02, "default", 3).normal(size=DRAW_SIZE)
        envelope.SALT["value"] = 1
        salted = envelope._rng("clean", 0.02, "default", 3).normal(size=DRAW_SIZE)
        assert not np.allclose(docs_draw, salted)

    def test_no_noise_leaves_the_frames_alone(self):
        assert envelope._noise("ar", 0.0, "clean") is None


class TestOneSet:
    def test_a_clean_set_is_counted_and_timed(self):
        sim = simulate(Scenario(reps=[RepScript()] * 2, bar_noise_m=envelope.TRACKED_BAR_NOISE_M))
        errors = envelope._top_errors(sim, envelope._analyse(sim))
        assert len(errors) == len(sim.reps)
        assert max(map(abs, errors)) <= MAX_CLEAN_TOP_ERROR_S
        assert envelope._summary(errors).endswith(f"of {len(sim.reps)}")

    def test_a_miscounted_set_has_no_top_errors(self):
        sim = simulate(Scenario(reps=[RepScript()] * 2))
        assert envelope._top_errors(sim, envelope._analyse(sim)[:1]) == []
        assert envelope._summary([]) == "no rep counted right"

    def test_reps_not_counted_are_reported(self):
        assert envelope._not_counted(5, [set()] * 4) == "; 1 of 5 reps not counted"
        assert envelope._not_counted(5, [set()] * 5) == ""

    def test_a_settle_lifts_the_bar_and_shoulders_until_the_lowering(self):
        sim = simulate(Scenario(reps=[RepScript(top_hold_s=1.2)] * 2))
        mutate = envelope._settled_up(sim, SETTLE_RISE_M, SETTLE_AFTER_TOP_S, SETTLE_RAMP_S, rep_indices=(1,))
        top = sim.reps[1].top_time
        before = next(frame for frame in sim.frames if frame.timestamp >= top)
        raised = next(frame for frame in sim.frames if frame.timestamp >= top + SETTLE_AFTER_TOP_S + SETTLE_RAMP_S)
        assert mutate(before).points == pytest.approx(before.points, abs=HEIGHT_TOLERANCE_M)
        lifted = mutate(raised)
        # Y-down: up is -y.
        assert raised.points[CK.LEFT_SHOULDER, 1] - lifted.points[CK.LEFT_SHOULDER, 1] == pytest.approx(
            SETTLE_RISE_M, abs=HEIGHT_TOLERANCE_M)
        assert raised.bar.left_end_m[1] - lifted.bar.left_end_m[1] == pytest.approx(SETTLE_RISE_M, abs=HEIGHT_TOLERANCE_M)
        assert lifted.points[CK.LEFT_HIP] == pytest.approx(raised.points[CK.LEFT_HIP], abs=HEIGHT_TOLERANCE_M)


    def test_a_sag_lowers_the_bar_and_shoulders_then_comes_back(self):
        sim = simulate(Scenario(reps=[RepScript(top_hold_s=1.2)] * 2))
        mutate = envelope._sagged(sim, SAG_M, rep_indices=(1,))
        top = sim.reps[1].top_time
        sagged = next(frame for frame in sim.frames if frame.timestamp >= top + envelope.SAG_START_S + envelope.SAG_RAMP_S)
        back = next(frame for frame in sim.frames if frame.timestamp >= top + envelope.SAG_UP_S + envelope.SAG_RAMP_S)
        # Y-down: down is +y.
        assert mutate(sagged).bar.left_end_m[1] - sagged.bar.left_end_m[1] == pytest.approx(SAG_M, abs=HEIGHT_TOLERANCE_M)
        assert mutate(back).points == pytest.approx(back.points, abs=HEIGHT_TOLERANCE_M)

    def test_a_lost_bar_coasts_as_the_tracker_reports_it_then_is_gone(self):
        sim = simulate(Scenario(reps=[RepScript()] * 2))
        top = sim.reps[0].top_time
        start, end = top - LOST_BEFORE_TOP_S, top + LOST_AFTER_TOP_S
        mutate = envelope._bar_lost([(start, end)])
        states = [(frame.timestamp, mutate(frame).bar) for frame in sim.frames]
        in_gap = [(t, bar) for t, bar in states if start <= t < end]
        coasting = [(t, bar) for t, bar in in_gap if bar is not None]
        assert coasting and all(bar.predicted for _, bar in coasting)
        assert max(t for t, _ in coasting) - start <= envelope.TRACKER_CONFIG.max_prediction_s + FRAME_S
        assert in_gap[-1][1] is None


class TestCommandLine:
    def test_the_compute_sweep_runs_end_to_end(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "compute"], capture_output=True, text=True,
                                timeout=SCRIPT_TIMEOUT_S, check=False)
        assert result.returncode == 0, result.stderr
        rows = [line for line in result.stdout.splitlines() if "ms per frame" in line]
        assert len(rows) == COMPUTE_ROWS

    def test_an_unknown_sweep_is_refused(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "squats"], capture_output=True, text=True,
                                timeout=SCRIPT_TIMEOUT_S, check=False)
        assert result.returncode != 0
