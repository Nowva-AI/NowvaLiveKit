"""
Press per-rep features (docs/exercises/ohp.md §4): production-frame signs, the
single camera's image-X lateral axis, windows and their minimum samples, the
hip-anchored acceptance, normalizer fallbacks, the facing check, and the rig's
gravity: ctx.up used as is (sign trap), measured vs body-axis vertical.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.faults.observability import APPROXIMATE, SINGLE_CAMERA, TRIANGULATED
from biomechanics.utils.frame_context import GRAVITY_BODY, GRAVITY_MEASURED
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.segment_lengths import BodyProportions
from biomechanics.utils.types import CocoKeypoints as CK
from test_press_poses import (
    ARM_REF_M,
    FPS_MONO,
    FPS_RIG,
    LEG_M,
    AnalyzerHarness,
    hands_down,
    hold,
    lockout_bar_height,
    press_rep,
    tilted_up,
)

SEED = 7
ANGLE_TOLERANCE_DEG = 0.5
RATIO_TOLERANCE = 0.01
UNBIASED_ANGLE_DEG = 0.2
UNBIASED_OFFSET_M = 0.003
VERTICAL_CHANGE_FRACTION = 0.01
LAYBACK_DEG = 10.0
BAR_FORWARD_M = 0.08
WORLD_TILT_DEG = 3.0
FACING_REPS = 200
FACING_REPS_PER_SET = 5
FACING_MIN_PASS = 0.95
OVERHEAD_SHOULDER_NARROWING = 0.8
OVERHEAD_WRIST_ABOVE_SHOULDER_M = 0.3
YAW_DEG = 30.0
ARM_DEPTH_NOISE_M = 0.05
SHOULDER_LIFT_M = 0.05
BODY_TORSO_M = 0.50
BODY_FEMUR_M = 0.44
BODY_TIBIA_M = 0.42
MED3_CUSP_FLOOR = 0.8
QUICK_LOWER_S = 0.42
SHORT_HOLD_S = 2.0 / FPS_MONO


def _set(fps: float, reps: int = 3, **rep) -> list[dict]:
    """Walk in hands down, unrack and hold (the latch), reps, then hold in the rack."""
    first = press_rep(fps, hold_s=2.0, **rep)
    poses = hands_down(1.0, fps) + first
    for _ in range(reps - 1):
        poses += press_rep(fps, **rep)
    return poses + [dict(first[0]) for _ in range(int(fps))]


def _rig(fps: float = FPS_RIG, **kwargs) -> AnalyzerHarness:
    return AnalyzerHarness(capture_mode=TRIANGULATED, fps=fps, **kwargs)


def _proportions(torso_m: float, femur_m: float, tibia_m: float) -> BodyProportions:
    return BodyProportions(
        hip_width=0.24, femur_length_avg=femur_m, tibia_length_avg=tibia_m, torso_length_avg=torso_m,
        shoulder_width=0.38, foot_length_avg=0.26, hip_to_femur_ratio=0.5, tibia_to_reference_ratio=1.0,
        forward_lean_scale=1.0,
    )


class TestProductionFrameSigns:
    def test_clean_press_reads_upward_and_reaches_overhead(self):
        harness = AnalyzerHarness().run(_set(FPS_MONO))
        features = harness.counted[-1]
        assert features.rom_cm == pytest.approx(100.0 * (lockout_bar_height() - 0.04), abs=1.0)
        assert features.top_height_ratio == pytest.approx(lockout_bar_height() / ARM_REF_M, abs=RATIO_TOLERANCE)
        assert features.concentric_velocity_mps > 0.0
        assert features.max_depth_angle == pytest.approx(180.0, abs=ANGLE_TOLERANCE_DEG)

    def test_left_hand_higher_is_positive_tilt(self):
        features = AnalyzerHarness().run(_set(FPS_MONO, tilt_top_deg=8.0)).counted[-1]
        assert features.bar_tilt_deg > 0.0

    def test_wrists_outside_the_elbows_is_positive_grip(self):
        features = AnalyzerHarness().run(_set(FPS_MONO, grip_tilt_deg=12.0)).counted[0]
        assert features.grip_tilt_deg == pytest.approx(12.0, abs=ANGLE_TOLERANCE_DEG)

    def test_hips_dropping_is_positive_leg_drive(self):
        features = AnalyzerHarness().run(_set(FPS_MONO, dip_m=0.05)).counted[-1]
        # The dip's one-frame bottom at 12 fps is shaved by the 3-sample median.
        assert MED3_CUSP_FLOOR * 0.05 / LEG_M <= features.leg_drive_ratio <= 0.05 / LEG_M + 0.003
        assert features.leg_drive_phase == "setup"


class TestMonoLateralIsImageX:
    def test_depth_noise_on_the_arms_leaves_frontal_features_unchanged(self):
        rng = np.random.default_rng(SEED)

        def arm_depth_noise(points: np.ndarray) -> np.ndarray:
            noisy = points.copy()
            for index in (CK.LEFT_WRIST, CK.RIGHT_WRIST, CK.LEFT_ELBOW, CK.RIGHT_ELBOW):
                noisy[index, 2] += rng.normal(0.0, ARM_DEPTH_NOISE_M)
            return noisy

        poses = _set(FPS_MONO, tilt_top_deg=6.0, grip_tilt_deg=8.0, lockout_flex_deg=20.0, dip_m=0.04)
        clean = AnalyzerHarness().run(poses).counted
        perturbed = AnalyzerHarness().run(poses, transform=arm_depth_noise).counted
        assert len(clean) == len(perturbed) == 3
        for first, second in zip(clean, perturbed):
            for name in (
                "bar_tilt_deg", "grip_tilt_deg", "lockout_elbow_l_deg", "lockout_elbow_r_deg",
                "top_height_ratio", "leg_drive_ratio", "width_ratio", "rom_cm",
            ):
                assert getattr(second, name) == pytest.approx(getattr(first, name), abs=1e-9, nan_ok=True), name


class TestWindows:
    def test_hip_anchored_acceptance_ignores_shoulder_keypoint_migration(self):
        clean = AnalyzerHarness().run(_set(FPS_MONO)).counted
        lifted = AnalyzerHarness().run(_set(FPS_MONO, shoulder_lift_top_m=SHOULDER_LIFT_M)).counted
        assert [f.top_height_ratio for f in lifted] == pytest.approx([f.top_height_ratio for f in clean], abs=1e-9)
        assert lifted[-1].rom_cm < clean[-1].rom_cm

    def test_too_few_press_samples_leave_layback_unmeasured(self):
        quick = _rig(fps=FPS_MONO).run(_set(FPS_MONO, press_s=0.25)).counted
        slow = _rig(fps=FPS_MONO).run(_set(FPS_MONO)).counted
        assert quick and slow
        assert all(math.isnan(f.layback_deg) for f in quick)
        assert all(math.isfinite(f.layback_deg) for f in slow)

    def test_a_touch_and_go_top_still_has_a_lockout_window(self):
        features = AnalyzerHarness().run(_set(FPS_MONO, lock_s=0.0)).counted
        assert features and all(math.isfinite(f.lockout_elbow_l_deg) for f in features)

    def test_short_setup_counts_but_leaves_setup_features_unmeasured(self):
        # A quick turnaround in the rack: two still frames plus the press's first,
        # slow frame make three setup samples (criterion 1 needs three, setup
        # features need four).
        poses = hands_down(1.0, FPS_MONO) + hold(1.0, FPS_MONO) + press_rep(FPS_MONO, lower_s=QUICK_LOWER_S)
        poses += press_rep(FPS_MONO, hold_s=SHORT_HOLD_S) + hold(1.0, FPS_MONO)
        harness = AnalyzerHarness().run(poses)
        assert len(harness.counted) == 2
        second = harness.counted[1]
        assert second.rack_start
        assert math.isnan(second.grip_tilt_deg)
        assert math.isnan(second.rack_height_ratio)
        assert math.isfinite(second.leg_drive_ratio)


class TestNormalizers:
    def test_in_set_medians_without_a_body_measurement(self):
        features = AnalyzerHarness().run(_set(FPS_MONO)).counted[0]
        assert features.arm_ref_m == pytest.approx(ARM_REF_M, abs=1e-6)
        assert features.leg_ref_m == pytest.approx(LEG_M, abs=0.005)

    def test_body_measurement_wins_when_present(self):
        proportions = _proportions(BODY_TORSO_M, BODY_FEMUR_M, BODY_TIBIA_M)
        features = AnalyzerHarness(body_proportions=proportions).run(_set(FPS_MONO)).counted[0]
        assert features.arm_ref_m == pytest.approx(1.15 * BODY_TORSO_M)
        assert features.leg_ref_m == pytest.approx(BODY_FEMUR_M + BODY_TIBIA_M)


class TestFacing:
    @pytest.mark.parametrize("sigma_m", [0.010, 0.014])
    def test_frontal_lifter_passes_through_keypoint_noise(self, sigma_m: float):
        rng = np.random.default_rng(SEED)
        noise = lambda shape: rng.normal(0.0, sigma_m, shape)  # noqa: E731
        passed = total = 0
        while total < FACING_REPS:
            harness = AnalyzerHarness().run(_set(FPS_MONO, reps=FACING_REPS_PER_SET), noise=noise)
            passed += sum(1 for features in harness.features if features.facing_ok)
            total += len(harness.features)
        assert passed / total >= FACING_MIN_PASS

    def test_shoulders_narrowing_overhead_never_fail_facing(self):
        def narrow_overhead(points: np.ndarray) -> np.ndarray:
            shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
            wrist_mid = (points[CK.LEFT_WRIST] + points[CK.RIGHT_WRIST]) / 2.0
            if shoulder_mid[1] - wrist_mid[1] < OVERHEAD_WRIST_ABOVE_SHOULDER_M:
                return points
            narrowed = points.copy()
            for index in (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER):
                narrowed[index, 0] = shoulder_mid[0] + OVERHEAD_SHOULDER_NARROWING * (points[index, 0] - shoulder_mid[0])
            return narrowed

        harness = AnalyzerHarness().run(_set(FPS_MONO), transform=narrow_overhead)
        assert len(harness.counted) == 3
        assert all(features.facing_ok for features in harness.counted)

    def test_rotated_set_after_a_frontal_one_loses_frontal_features(self):
        harness = AnalyzerHarness().run(_set(FPS_MONO, grip_tilt_deg=12.0))
        assert all(features.facing_ok for features in harness.counted)
        harness.reset_set()
        harness.run(_set(FPS_MONO, grip_tilt_deg=12.0, tilt_top_deg=8.0, yaw_deg=YAW_DEG))
        rotated = harness.counted[3:]
        assert len(rotated) == 3
        for features in rotated:
            assert not features.facing_ok
            assert math.isnan(features.bar_tilt_deg)
            assert math.isnan(features.grip_tilt_deg)
            assert math.isnan(features.lockout_elbow_l_deg)
            assert math.isfinite(features.top_height_ratio)

    def test_rotated_from_the_first_set_is_caught_by_the_anthropometric_guard(self):
        harness = AnalyzerHarness().run(_set(FPS_MONO, yaw_deg=YAW_DEG))
        assert harness.counted
        assert not any(features.facing_ok for features in harness.counted)

    def test_rig_lateral_axis_follows_the_hips(self):
        features = _rig().run(_set(FPS_RIG, tilt_top_deg=8.0, yaw_deg=YAW_DEG)).counted
        assert features and all(f.facing_ok for f in features)
        assert all(f.bar_tilt_deg == pytest.approx(8.0, abs=ANGLE_TOLERANCE_DEG) for f in features)


class TestGravity:
    """Rig sagittal metrics read FrameContext.up as is (SHARED-PLUMBING.md §3)."""

    @staticmethod
    def _scenes() -> list[tuple[str, dict, dict]]:
        return [
            ("body", {"gravity_source": GRAVITY_BODY, "up": WORLD_UP}, {}),
            (
                "measured_tilted",
                {"gravity_source": GRAVITY_MEASURED, "up": tilted_up(WORLD_TILT_DEG)},
                {"gravity_tilt_deg": WORLD_TILT_DEG},
            ),
        ]

    @pytest.mark.parametrize("scene", range(2), ids=["body", "measured_tilted"])
    def test_upright_back_lean_and_bar_ahead_read_true(self, scene: int):
        _, context, world = self._scenes()[scene]
        upright = _rig(**context).run(_set(FPS_RIG, **world)).counted[-1]
        assert upright.layback_deg == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)
        leaning = _rig(**context).run(_set(FPS_RIG, layback_deg=LAYBACK_DEG, **world)).counted[-1]
        assert leaning.layback_deg == pytest.approx(LAYBACK_DEG, abs=ANGLE_TOLERANCE_DEG)
        ahead = _rig(**context).run(_set(FPS_RIG, bar_forward_m=BAR_FORWARD_M, **world)).counted[-1]
        assert ahead.bar_over_midfoot_ratio == pytest.approx(BAR_FORWARD_M / ARM_REF_M, abs=RATIO_TOLERANCE)
        assert ahead.bar_over_midfoot_ratio == pytest.approx(0.13, abs=RATIO_TOLERANCE)

    def test_measured_up_is_unbiased_in_a_tilted_world(self):
        context = {"gravity_source": GRAVITY_MEASURED, "up": tilted_up(WORLD_TILT_DEG)}
        poses = _set(FPS_RIG, layback_deg=LAYBACK_DEG, bar_forward_m=BAR_FORWARD_M, gravity_tilt_deg=WORLD_TILT_DEG)
        features = _rig(**context).run(poses).counted[-1]
        assert features.layback_deg == pytest.approx(LAYBACK_DEG, abs=UNBIASED_ANGLE_DEG)
        assert features.bar_over_midfoot_ratio * features.arm_ref_m == pytest.approx(BAR_FORWARD_M, abs=UNBIASED_OFFSET_M)

    def test_body_axis_in_a_tilted_world_falls_back_and_caps_both_faults(self):
        poses = _set(
            FPS_RIG, layback_deg=2.0 * LAYBACK_DEG, bar_forward_m=2.0 * BAR_FORWARD_M, gravity_tilt_deg=WORLD_TILT_DEG,
        )
        harness = _rig(gravity_source=GRAVITY_BODY, up=WORLD_UP).run(poses)
        faults = {fault.fault_type: fault for fault in harness.faults if fault.rep_number == harness.counted[-1].rep_number}
        assert faults["press_layback"].details["observability"] == APPROXIMATE
        assert faults["press_bar_forward"].details["observability"] == APPROXIMATE
        assert faults["press_bar_forward"].details["metric"] == "shoulder"

    def test_negated_up_fails_the_upright_trunk_check(self):
        """The round-3 bug (up = -gravity under "body"): an upright trunk reads ~180°."""
        upright = _rig(gravity_source=GRAVITY_BODY, up=-WORLD_UP).run(_set(FPS_RIG)).counted[-1]
        assert upright.layback_deg != pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)

    def test_vertical_metrics_do_not_depend_on_the_gravity_source(self):
        poses = _set(FPS_RIG, dip_m=0.06, tilt_top_deg=7.0, gravity_tilt_deg=WORLD_TILT_DEG)
        body = _rig(gravity_source=GRAVITY_BODY, up=WORLD_UP).run(poses).counted
        measured = _rig(gravity_source=GRAVITY_MEASURED, up=tilted_up(WORLD_TILT_DEG)).run(poses).counted
        for first, second in zip(body, measured, strict=True):
            for name in ("top_height_ratio", "leg_drive_ratio", "bar_tilt_deg"):
                assert getattr(second, name) == pytest.approx(
                    getattr(first, name), rel=VERTICAL_CHANGE_FRACTION, abs=1e-9,
                ), name


class TestCaptureModeFields:
    def test_highlight_score_is_rig_only(self):
        mono = AnalyzerHarness().run(_set(FPS_MONO)).counted[-1]
        rig = _rig().run(_set(FPS_RIG)).counted[-1]
        assert math.isnan(mono.highlight_score)
        assert rig.highlight_score == pytest.approx(rig.concentric_velocity_mps)
        assert mono.capture_mode == SINGLE_CAMERA and rig.capture_mode == TRIANGULATED

    def test_timing_feeds_rep_data(self):
        features = AnalyzerHarness().run(_set(FPS_MONO)).counted[-1]
        assert features.concentric_time_s == pytest.approx(features.press_time_s)
        assert features.eccentric_time_s == pytest.approx(features.lower_time_s)
        assert 0.4 < features.press_time_s < 0.9
