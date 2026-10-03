"""
Synthetic press sessions through the real pipeline (docs/exercises/ohp.md §7.2):
rack -> press -> lockout -> lower, touch-and-go, misses, re-racks, bare-hand
reaches, clipped wrists and merged turnarounds at 12 and 30 fps; first-rep timing
pinned in frames; each injected fault at its tier only; and single-camera pitch
leak and levelling with anisotropic noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.analysis.press_features import PITCH_BAND_UNKNOWN
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.observability import APPROXIMATE, OBSERVABLE
from biomechanics.utils.frame_context import GRAVITY_MEASURED
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultSeverity
from test_press_poses import (
    FPS_MONO,
    FPS_RIG,
    AnalyzerHarness,
    bare_hand_reach,
    camera_pitched,
    counted_reps,
    faults_of,
    hands_down,
    hold,
    press_rep,
    press_session,
    run_session,
    world_press_points,
    wrists_clipped_above,
)

MODES = [pytest.param(False, FPS_MONO, id="mono-12fps"), pytest.param(True, FPS_RIG, id="rig-30fps")]
MISSED_TOP_M = 0.30
CLIP_HIGH_M = 0.45
CHIN_PROGRESS = 0.2
ABOVE_CHIN_PROGRESS = 0.6
RE_RACK_S = 3.0
# Rising frames still inside the rack band at 12 fps with the 0.8 s cosine press.
IN_BAND_RISING_FRAMES = 3
PITCH_DEG = 15.0
HIP_FORWARD_M = 0.05
BAR_YAW_M = 0.04
LEVELLED_LEG_DRIVE_MAX = 0.003
LEVELLED_TILT_MAX_DEG = 0.3
LEAK_LEG_DRIVE = 0.014
LEAK_LEG_DRIVE_TOLERANCE = 0.002
LEAK_TILT_DEG = 1.1
LEAK_TILT_TOLERANCE_DEG = 0.15
# Budget noise (ohp.md §5.0, central values): sigma_z = 3 sigma_xy.
HIP_SIGMA_XY_M = 0.008
HIP_SIGMA_Z_M = 0.024
WRIST_SIGMA_XY_M = 0.010
WRIST_SIGMA_Z_M = 0.030
DEPTH_NOISE_RATIO = 3.0
NOISE_REPS = 200
NOISE_REPS_PER_SET = 5
NOISE_SEED = 11
BUDGET_PITCHES_DEG = (0.0, 3.0, 10.0, 15.0)
SCALING_TOLERANCE = 0.10
# Depth-bias change within a rep (assumed, §5.0) and the inclinometer residual.
DEPTH_BIAS_CM = 3.0
GRIP_SPACING_CM = 55.0
RESIDUAL_LEG_DRIVE_CM = 0.1
RESIDUAL_TILT_DEG = 0.09
LEVEL_MODERATE_ALLOWANCE_CM = 1.9
SEVERE_ALLOWANCE_CM = 3.1
SEVERE_ALLOWANCE_DEG = 3.0


def _unrack(fps: float, hold_s: float = 1.5, **pose) -> list[dict]:
    return hands_down(1.0, fps) + hold(hold_s, fps, **pose)


def _clean_set(fps: float, reps: int = 3) -> list[dict]:
    poses = _unrack(fps)
    for _ in range(reps):
        poses += press_rep(fps)
    return poses + hold(1.0, fps)


def _rep_faults(results: list) -> dict[int, list]:
    by_rep: dict[int, list] = {}
    for fault in faults_of(results):
        by_rep.setdefault(fault.rep_number, []).append(fault)
    return by_rep


class TestSequences:
    @pytest.mark.parametrize(("multi_camera", "fps"), MODES)
    def test_clean_set_counts_every_rep_with_no_fault(self, monkeypatch, multi_camera: bool, fps: float):
        results = run_session(press_session(monkeypatch, multi_camera), _clean_set(fps), fps)
        assert [rep.rep_number for rep in counted_reps(results)] == [1, 2, 3]
        assert faults_of(results) == []
        assert all(rep.is_clean for rep in counted_reps(results))

    @pytest.mark.parametrize(("multi_camera", "fps"), MODES)
    def test_touch_and_go_at_the_chin_splits_into_reps(self, monkeypatch, multi_camera: bool, fps: float):
        poses = _unrack(fps) + press_rep(fps, end_progress=CHIN_PROGRESS)
        poses += press_rep(fps, hold_s=0.0, start_progress=CHIN_PROGRESS, end_progress=CHIN_PROGRESS)
        poses += press_rep(fps, hold_s=0.0, start_progress=CHIN_PROGRESS) + hold(1.0, fps)
        results = run_session(press_session(monkeypatch, multi_camera), poses, fps)
        assert [rep.rep_number for rep in counted_reps(results)] == [1, 2, 3]
        assert faults_of(results) == []

    @pytest.mark.parametrize(("multi_camera", "fps"), MODES)
    def test_turnaround_above_the_chin_merges_into_one_rep(self, monkeypatch, multi_camera: bool, fps: float):
        poses = _unrack(fps) + press_rep(fps, end_progress=ABOVE_CHIN_PROGRESS)
        poses += press_rep(fps, hold_s=0.0, start_progress=ABOVE_CHIN_PROGRESS) + hold(1.0, fps)
        [rep] = counted_reps(run_session(press_session(monkeypatch, multi_camera), poses, fps))
        assert rep.features["partial_bottom"]

    @pytest.mark.parametrize(("multi_camera", "fps"), MODES)
    def test_missed_rep_is_recorded_not_counted(self, monkeypatch, multi_camera: bool, fps: float):
        poses = _unrack(fps) + press_rep(fps) + press_rep(fps, top_m=MISSED_TOP_M) + press_rep(fps) + hold(1.0, fps)
        results = run_session(press_session(monkeypatch, multi_camera), poses, fps)
        assert [rep.rep_number for rep in counted_reps(results)] == [1, 2]
        [missed] = faults_of(results)
        assert missed.fault_type == "press_missed"
        assert missed.details["observability"] == OBSERVABLE

    @pytest.mark.parametrize(("multi_camera", "fps"), MODES)
    def test_re_rack_with_the_hands_down_mid_set(self, monkeypatch, multi_camera: bool, fps: float):
        poses = _unrack(fps) + press_rep(fps) + hands_down(RE_RACK_S, fps) + hold(1.5, fps) + press_rep(fps)
        results = run_session(press_session(monkeypatch, multi_camera), poses + hold(1.0, fps), fps)
        assert [rep.rep_number for rep in counted_reps(results)] == [1, 2]
        assert faults_of(results) == []

    @pytest.mark.parametrize(("multi_camera", "fps"), MODES)
    def test_bare_hand_reach_is_never_a_rep(self, monkeypatch, multi_camera: bool, fps: float):
        poses = _unrack(fps) + press_rep(fps) + hands_down(1.0, fps) + bare_hand_reach(fps) + hands_down(1.5, fps)
        results = run_session(press_session(monkeypatch, multi_camera), poses, fps)
        assert [rep.rep_number for rep in counted_reps(results)] == [1]
        assert faults_of(results) == []

    def test_wrists_clipped_at_the_frame_top_count_and_read_as_tracking_lost(self, monkeypatch):
        clip = wrists_clipped_above(CLIP_HIGH_M)
        results = run_session(press_session(monkeypatch), _clean_set(FPS_MONO, reps=2), FPS_MONO, confidence=clip)
        reps = counted_reps(results)
        assert [rep.rep_number for rep in reps] == [1, 2]
        for rep in reps:
            assert rep.features["wrists_clipped"]
            assert rep.features["lockout_elbow_l_deg"] is None or math.isnan(rep.features["lockout_elbow_l_deg"])
        lost = [result for result in results if "left_wrist" in result.missing_keypoints]
        assert lost and all("right_wrist" in result.missing_keypoints for result in lost)


class TestFirstRepTiming:
    """A hold is consecutive frames passing the rack predicate, still or rising
    (ohp.md §3.7); the readiness gate needs five of them."""

    @staticmethod
    def _first_press(still_frames: int) -> list[dict]:
        return hands_down(1.0, FPS_MONO) + press_rep(FPS_MONO, hold_s=still_frames / FPS_MONO)

    def _reps(self, monkeypatch, still_frames: int) -> list:
        poses = self._first_press(still_frames) + hold(1.0, FPS_MONO)
        return counted_reps(run_session(press_session(monkeypatch), poses, FPS_MONO))

    def test_five_in_band_frames_count_the_first_rep(self, monkeypatch):
        [rep] = self._reps(monkeypatch, 5 - IN_BAND_RISING_FRAMES)
        assert rep.features["first_rep_after_latch"]

    def test_four_in_band_frames_miss_the_first_rep(self, monkeypatch):
        assert self._reps(monkeypatch, 4 - IN_BAND_RISING_FRAMES) == []

    def test_short_still_hold_plus_the_rise_counts_without_a_press_start(self, monkeypatch):
        """0.30 s of stillness (3 frames at 12 fps) plus the rise latches mid-rise."""
        [rep] = self._reps(monkeypatch, int(0.30 * FPS_MONO))
        assert not rep.features["press_start_observed"]
        velocity = rep.features["concentric_velocity_mps"]
        assert velocity is None or math.isnan(velocity)

    def test_rising_frames_inside_the_band_are_what_the_count_assumes(self):
        poses = press_rep(FPS_MONO, hold_s=0.0)
        rising = [pose["bar_height_m"] for pose in poses]
        band_top_m = BiomechanicsConfig().faults.press.rack.bar_height_max_ratio * 1.15 * 0.543
        assert sum(1 for height in rising[:IN_BAND_RISING_FRAMES + 1] if height <= band_top_m) == IN_BAND_RISING_FRAMES

    def test_y_up_skeleton_fails_the_rack_predicate(self):
        harness = AnalyzerHarness()
        upright = world_press_points()
        flipped = upright * np.array([1.0, -1.0, 1.0])
        assert harness.analyzer.is_ready(harness.context(upright, raw=True))
        assert not harness.analyzer.is_ready(harness.context(flipped, raw=True))


class TestInjectedFaultTiers:
    def test_rig_each_fault_fires_its_tier_only(self, monkeypatch):
        session = press_session(monkeypatch, multi_camera=True)
        monkeypatch.setattr(session.pipeline, "_gravity", lambda: (WORLD_UP.copy(), GRAVITY_MEASURED))
        injected = [
            ({}, None),
            ({"dip_m": 0.07}, ("press_leg_drive", FaultSeverity.MODERATE)),
            ({"tilt_top_deg": 7.5}, ("press_uneven", FaultSeverity.MODERATE)),
            ({"lockout_flex_deg": 30.0}, ("press_lockout", FaultSeverity.MODERATE)),
            ({"layback_top_deg": 22.0}, ("press_layback", FaultSeverity.MODERATE)),
            ({"bar_forward_top_m": 0.13}, ("press_bar_forward", FaultSeverity.MODERATE)),
        ]
        poses = _unrack(FPS_RIG)
        for rep, _ in injected:
            poses += press_rep(FPS_RIG, **rep)
        results = run_session(session, poses + hold(1.0, FPS_RIG), FPS_RIG)
        assert len(counted_reps(results)) == len(injected)
        by_rep = _rep_faults(results)
        for rep_number, (_, expected) in enumerate(injected, start=1):
            observed = [(fault.fault_type, fault.severity) for fault in by_rep.get(rep_number, [])]
            assert observed == ([expected] if expected else []), rep_number

    def test_single_camera_level_mount_faults_and_observability(self, monkeypatch):
        config = BiomechanicsConfig()
        config.capture.mount_source = "inclinometer"
        config.capture.mount_pitch_deg = 0.0
        session = press_session(monkeypatch, config=config)
        injected = [
            ({"dip_m": 0.07}, "press_leg_drive", FaultSeverity.MODERATE, OBSERVABLE),
            ({"tilt_top_deg": 10.0}, "press_uneven", FaultSeverity.SEVERE, OBSERVABLE),
            ({"lockout_flex_deg": 30.0}, "press_lockout", FaultSeverity.MODERATE, APPROXIMATE),
        ]
        poses = _unrack(FPS_MONO, grip_tilt_deg=17.0)
        for rep, *_ in injected:
            poses += press_rep(FPS_MONO, grip_tilt_deg=17.0, **rep)
        results = run_session(session, poses + hold(1.0, FPS_MONO, grip_tilt_deg=17.0), FPS_MONO)
        by_rep = _rep_faults(results)
        for rep_number, (_, fault_type, severity, observability) in enumerate(injected, start=1):
            faults = {fault.fault_type: fault for fault in by_rep[rep_number] if fault.fault_type != "press_grip"}
            assert list(faults) == [fault_type]
            assert faults[fault_type].severity == severity
            assert faults[fault_type].details["observability"] == observability
            assert not faults[fault_type].details.get("recorded_only")
        [grip] = faults_of(results, "press_grip")
        assert grip.rep_number == 1
        assert grip.severity == FaultSeverity.MODERATE
        assert grip.details["direction"] == "narrower"


def _budget_noise(rng: np.random.Generator):
    """Central-budget noise in the camera frame: hips move as one (pelvis), wrists apart."""
    hip_sigma = np.array([HIP_SIGMA_XY_M, HIP_SIGMA_XY_M, HIP_SIGMA_Z_M])
    wrist_sigma = np.array([WRIST_SIGMA_XY_M, WRIST_SIGMA_XY_M, WRIST_SIGMA_Z_M])

    def noise(shape: tuple[int, ...]) -> np.ndarray:
        offsets = np.zeros(shape)
        hip = rng.normal(0.0, 1.0, 3) * hip_sigma
        offsets[CK.LEFT_HIP] += hip
        offsets[CK.RIGHT_HIP] += hip
        for wrist in (CK.LEFT_WRIST, CK.RIGHT_WRIST):
            offsets[wrist] += rng.normal(0.0, 1.0, 3) * wrist_sigma
        return offsets

    return noise


def _levelling_factor(pitch_deg: float) -> float:
    pitch = math.radians(pitch_deg)
    return math.sqrt(math.cos(pitch) ** 2 + DEPTH_NOISE_RATIO ** 2 * math.sin(pitch) ** 2)


def _clean_rep_p95(pitch_deg: float) -> tuple[float, float, list]:
    """Clean-rep P95 of the leg-drive hip drop (cm) and |bar tilt| (deg) on a
    certified mount at this pitch, levelled; common random numbers across pitches."""
    rng = np.random.default_rng(NOISE_SEED)
    noise = _budget_noise(rng)
    drops, tilts, features = [], [], []
    while len(drops) < NOISE_REPS:
        harness = AnalyzerHarness(pitch_certified=True, mount_pitch_deg=pitch_deg)
        poses = hands_down(1.0, FPS_MONO) + press_rep(FPS_MONO, hold_s=2.0)
        for _ in range(NOISE_REPS_PER_SET - 1):
            poses += press_rep(FPS_MONO)
        harness.run(
            poses + hold(1.0, FPS_MONO),
            transform=lambda points: camera_pitched(points, pitch_deg) + noise(points.shape),
        )
        for rep in harness.counted:
            drops.append(100.0 * rep.leg_drive_ratio * rep.leg_ref_m)
            tilts.append(abs(rep.bar_tilt_deg))
            features.append(rep)
    return float(np.percentile(drops[:NOISE_REPS], 95)), float(np.percentile(tilts[:NOISE_REPS], 95)), features


@pytest.fixture(scope="module")
def clean_rep_budget() -> dict[float, tuple[float, float, list]]:
    return {pitch: _clean_rep_p95(pitch) for pitch in BUDGET_PITCHES_DEG}


class TestPitchLeakAndLevelling:
    @staticmethod
    def _leak_rep(certified: bool) -> object:
        harness = AnalyzerHarness(pitch_certified=certified, mount_pitch_deg=-PITCH_DEG if certified else math.nan)
        poses = hands_down(1.0, FPS_MONO) + press_rep(FPS_MONO, hold_s=2.0)
        poses += press_rep(FPS_MONO, hip_forward_m=HIP_FORWARD_M, bar_yaw_top_m=BAR_YAW_M) + hold(1.0, FPS_MONO)
        # Lens tilted down 15°: the hips' forward shift reads as a drop when not levelled.
        harness.run(poses, transform=lambda points: camera_pitched(points, -PITCH_DEG))
        return harness.counted[-1]

    def test_levelled_geometry_removes_the_leak(self):
        rep = self._leak_rep(certified=True)
        assert abs(rep.leg_drive_ratio) < LEVELLED_LEG_DRIVE_MAX
        assert abs(rep.bar_tilt_deg) < LEVELLED_TILT_MAX_DEG
        assert rep.pitch_band == PITCH_BAND_UNKNOWN

    def test_unlevelled_geometry_shows_the_leak(self):
        rep = self._leak_rep(certified=False)
        assert rep.leg_drive_ratio == pytest.approx(LEAK_LEG_DRIVE, abs=LEAK_LEG_DRIVE_TOLERANCE)
        assert abs(rep.bar_tilt_deg) == pytest.approx(LEAK_TILT_DEG, abs=LEAK_TILT_TOLERANCE_DEG)

    @pytest.mark.parametrize("pitch_deg", BUDGET_PITCHES_DEG[1:])
    def test_noise_p95_scales_with_the_levelling_factor(self, clean_rep_budget, pitch_deg: float):
        level_drop, level_tilt, _ = clean_rep_budget[0.0]
        drop, tilt, _ = clean_rep_budget[pitch_deg]
        factor = _levelling_factor(pitch_deg)
        assert drop / level_drop == pytest.approx(factor, rel=SCALING_TOLERANCE)
        assert tilt / level_tilt == pytest.approx(factor, rel=SCALING_TOLERANCE)

    @pytest.mark.parametrize("pitch_deg", [0.0, 3.0])
    def test_level_band_leg_drive_fits_the_moderate_allowance(self, clean_rep_budget, pitch_deg: float):
        drop, _, _ = clean_rep_budget[pitch_deg]
        bias_cm = DEPTH_BIAS_CM * math.sin(math.radians(pitch_deg))
        assert drop + bias_cm + RESIDUAL_LEG_DRIVE_CM <= LEVEL_MODERATE_ALLOWANCE_CM

    @pytest.mark.parametrize("pitch_deg", [0.0, 3.0])
    def test_level_band_uneven_fits_its_severe_start_tier(self, clean_rep_budget, pitch_deg: float):
        _, tilt, _ = clean_rep_budget[pitch_deg]
        bias_deg = math.degrees(math.atan(DEPTH_BIAS_CM * math.sin(math.radians(pitch_deg)) / GRIP_SPACING_CM))
        assert tilt + bias_deg + RESIDUAL_TILT_DEG <= SEVERE_ALLOWANCE_DEG

    def test_tilted_band_leg_drive_fits_the_severe_allowance(self, clean_rep_budget):
        drop, _, _ = clean_rep_budget[10.0]
        bias_cm = DEPTH_BIAS_CM * math.sin(math.radians(10.0))
        assert drop + bias_cm + RESIDUAL_LEG_DRIVE_CM <= SEVERE_ALLOWANCE_CM

    def test_fifteen_degrees_is_the_unknown_band_and_recap_only(self, clean_rep_budget):
        _, _, features = clean_rep_budget[15.0]
        assert {rep.pitch_band for rep in features} == {PITCH_BAND_UNKNOWN}
