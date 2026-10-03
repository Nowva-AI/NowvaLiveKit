"""Tests for standing pose gate."""

import numpy as np
import pytest

from biomechanics.utils.standing_gate import PROFILE_CHECK_FAILURE, StandingPoseGate, REQUIRED_KEYPOINTS
from biomechanics.utils.types import Skeleton3D, CocoKeypoints as CK


def _make_skeleton(
    positions: np.ndarray,
    confidences: np.ndarray | None = None,
    timestamp: float = 0.0,
    frame_index: int = 0,
) -> Skeleton3D:
    if confidences is None:
        confidences = np.ones(len(positions))
    return Skeleton3D.from_numpy(
        positions, confidences=confidences,
        timestamp=timestamp, frame_index=frame_index,
    )


def _standing_skeleton() -> np.ndarray:
    """Standing skeleton with knees extended, torso upright, plausible distance."""
    points = np.zeros((17, 3))
    points[CK.NOSE] = [0.0, 1.70, 0.0]
    points[CK.LEFT_EYE] = [0.03, 1.72, -0.02]
    points[CK.RIGHT_EYE] = [-0.03, 1.72, -0.02]
    points[CK.LEFT_EAR] = [0.07, 1.70, 0.0]
    points[CK.RIGHT_EAR] = [-0.07, 1.70, 0.0]
    points[CK.LEFT_SHOULDER] = [0.20, 1.50, 0.0]
    points[CK.RIGHT_SHOULDER] = [-0.20, 1.50, 0.0]
    points[CK.LEFT_HIP] = [0.10, 1.00, 0.0]
    points[CK.RIGHT_HIP] = [-0.10, 1.00, 0.0]
    points[CK.LEFT_ELBOW] = [0.25, 1.25, 0.0]
    points[CK.LEFT_WRIST] = [0.25, 1.00, 0.0]
    points[CK.RIGHT_ELBOW] = [-0.25, 1.25, 0.0]
    points[CK.RIGHT_WRIST] = [-0.25, 1.00, 0.0]
    points[CK.LEFT_KNEE] = [0.10, 0.55, 0.0]
    points[CK.LEFT_ANKLE] = [0.10, 0.10, 0.0]
    points[CK.RIGHT_KNEE] = [-0.10, 0.55, 0.0]
    points[CK.RIGHT_ANKLE] = [-0.10, 0.10, 0.0]
    return points


def _standing_skeleton_with_feet(heel_y: float = 0.0) -> np.ndarray:
    """Standing skeleton extended with toe and heel keypoints.

    Ankles sit at y=0.10 with the toes on the floor ahead of them. Raising
    ``heel_y`` lifts both heels, which is what standing on tiptoe looks like.
    """
    points = np.zeros((21, 3))
    points[:17] = _standing_skeleton()
    points[CK.LEFT_FOOT_INDEX] = [0.10, 0.0, 0.16]
    points[CK.RIGHT_FOOT_INDEX] = [-0.10, 0.0, 0.16]
    points[CK.LEFT_HEEL] = [0.10, heel_y, -0.06]
    points[CK.RIGHT_HEEL] = [-0.10, heel_y, -0.06]
    return points


class TestFlatFootCheck:

    def test_flat_feet_pass(self):
        gate = StandingPoseGate(required_consecutive_frames=1)
        assert gate.check(_make_skeleton(_standing_skeleton_with_feet()))
        assert gate.last_failure is None

    def test_raised_heels_fail(self):
        gate = StandingPoseGate(required_consecutive_frames=1)
        # 10 cm of heel rise over a ~24 cm foot: ratio 0.41, well past 0.30.
        assert not gate.check(_make_skeleton(_standing_skeleton_with_feet(heel_y=0.10)))
        assert gate.last_failure == "flat_foot"

    def test_slight_heel_rise_still_passes(self):
        # 4 cm reads as ratio 0.18 — inside the noise-tolerant threshold, so
        # normal landmark jitter must not lock the athlete out of calibration.
        gate = StandingPoseGate(required_consecutive_frames=1)
        assert gate.check(_make_skeleton(_standing_skeleton_with_feet(heel_y=0.04)))

    def test_one_raised_heel_fails(self):
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = _standing_skeleton_with_feet()
        pts[CK.LEFT_HEEL][1] = 0.12
        assert not gate.check(_make_skeleton(pts))
        assert gate.last_failure == "flat_foot"

    def test_detects_heel_rise_in_y_down_frame(self):
        # MediaPipe world landmarks are Y-down; the check derives "up" from
        # the torso rather than assuming a sign.
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = _standing_skeleton_with_feet(heel_y=0.10)
        pts[:, 1] *= -1.0
        assert not gate.check(_make_skeleton(pts))
        assert gate.last_failure == "flat_foot"

    def test_untracked_heels_skip_the_check(self):
        # Heels are frequently occluded. A zero-confidence heel must not fail
        # the gate, or those users can never finish calibrating.
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = _standing_skeleton_with_feet(heel_y=0.12)
        confidences = np.ones(len(pts))
        confidences[CK.LEFT_HEEL] = 0.0
        confidences[CK.RIGHT_HEEL] = 0.0
        assert gate.check(_make_skeleton(pts, confidences=confidences))

    def test_skeleton_without_heel_keypoints_passes(self):
        # Poses captured before heel tracking are narrower than 21 keypoints.
        gate = StandingPoseGate(required_consecutive_frames=1)
        assert gate.check(_make_skeleton(_standing_skeleton()))


class TestStandingPoseGate:

    def test_passes_after_consecutive_frames(self):
        gate = StandingPoseGate(required_consecutive_frames=5)
        pts = _standing_skeleton()

        for i in range(4):
            assert not gate.check(_make_skeleton(pts, frame_index=i))
        assert gate.check(_make_skeleton(pts, frame_index=4))
        assert gate.is_ready

    def test_low_confidence_fails(self):
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = _standing_skeleton()
        confs = np.ones(17)
        confs[CK.LEFT_SHOULDER] = 0.1  # below 0.5 threshold

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts, confidences=confs))
        assert not gate.is_ready

    def test_bent_knees_fails(self):
        gate = StandingPoseGate(required_consecutive_frames=1, max_knee_flexion_deg=20.0)
        pts = _standing_skeleton()
        # Push knees out sideways to create large frontal-plane flexion
        pts[CK.LEFT_KNEE] = [0.35, 0.60, 0.0]
        pts[CK.RIGHT_KNEE] = [-0.35, 0.60, 0.0]

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts))
        assert not gate.is_ready

    def test_leaning_torso_fails(self):
        gate = StandingPoseGate(required_consecutive_frames=1, max_trunk_flexion_deg=25.0)
        pts = _standing_skeleton()
        # Lean shoulders forward heavily
        pts[CK.LEFT_SHOULDER] = [0.20, 1.30, -0.40]
        pts[CK.RIGHT_SHOULDER] = [-0.20, 1.30, -0.40]

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts))
        assert not gate.is_ready

    def test_too_close_fails(self):
        gate = StandingPoseGate(required_consecutive_frames=1, max_torso_length_m=0.80)
        pts = _standing_skeleton()
        # Scale everything up to simulate being very close to camera
        pts *= 3.0

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts))
        assert not gate.is_ready

    def test_too_far_fails(self):
        gate = StandingPoseGate(required_consecutive_frames=1, min_torso_length_m=0.25)
        pts = _standing_skeleton()
        # Scale down to simulate being very far
        pts *= 0.1

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts))
        assert not gate.is_ready

    def test_consecutive_resets_on_failure(self):
        gate = StandingPoseGate(required_consecutive_frames=5)
        good_pts = _standing_skeleton()
        bad_pts = _standing_skeleton()
        bad_pts[CK.LEFT_KNEE] = [0.35, 0.60, 0.0]  # bent knee (frontal plane)

        # 4 good frames, then 1 bad → resets counter
        for i in range(4):
            gate.check(_make_skeleton(good_pts, frame_index=i))
        gate.check(_make_skeleton(bad_pts, frame_index=4))

        # Need another 5 consecutive good frames
        for i in range(4):
            assert not gate.check(_make_skeleton(good_pts, frame_index=5 + i))
        assert gate.check(_make_skeleton(good_pts, frame_index=9))

    def test_latch_stays_true(self):
        gate = StandingPoseGate(required_consecutive_frames=1)
        good_pts = _standing_skeleton()
        bad_pts = _standing_skeleton()
        bad_pts[CK.LEFT_KNEE] = [0.35, 0.60, 0.0]

        assert gate.check(_make_skeleton(good_pts))
        assert gate.is_ready

        # Even with a bad frame, gate stays open
        assert gate.check(_make_skeleton(bad_pts))
        assert gate.is_ready

    def test_reset(self):
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = _standing_skeleton()

        gate.check(_make_skeleton(pts))
        assert gate.is_ready

        gate.reset()
        assert not gate.is_ready

    def test_depth_folded_legs_fail(self):
        # Regression (session 2026-07-22_11-39-49): MediaPipe hallucinated
        # legs folded in depth — frontally straight, so the frontal knee
        # check passes, but the ankles sit far too close to hip height.
        # Y-down hip-centered coords like live MediaPipe world landmarks.
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = np.zeros((19, 3))
        pts[CK.LEFT_SHOULDER] = [-0.18, -0.50, 0.0]
        pts[CK.RIGHT_SHOULDER] = [0.18, -0.50, 0.0]
        pts[CK.LEFT_HIP] = [-0.10, 0.0, 0.0]
        pts[CK.RIGHT_HIP] = [0.10, 0.0, 0.0]
        # Frontally collinear hip→knee→ankle (flexion ≈ 0) with the leg
        # length consumed in depth: vertical span 0.55m vs ~1.0m of leg.
        pts[CK.LEFT_KNEE] = [-0.10, 0.30, -0.45]
        pts[CK.RIGHT_KNEE] = [0.10, 0.30, -0.45]
        pts[CK.LEFT_ANKLE] = [-0.10, 0.55, -0.85]
        pts[CK.RIGHT_ANKLE] = [0.10, 0.55, -0.85]

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts))
        assert not gate.is_ready
        assert gate.last_failure == "leg_extension"

    def test_inverted_legs_fail(self):
        # Ankles on the same vertical side of the hips as the shoulders —
        # a large span, frontally straight, but anatomically impossible.
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = _standing_skeleton()
        pts[CK.LEFT_KNEE] = [0.10, 1.45, -0.30]
        pts[CK.RIGHT_KNEE] = [-0.10, 1.45, -0.30]
        pts[CK.LEFT_ANKLE] = [0.10, 1.90, -0.60]
        pts[CK.RIGHT_ANKLE] = [-0.10, 1.90, -0.60]

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts))
        assert not gate.is_ready
        assert gate.last_failure == "leg_extension"

    def test_corrupted_landmarks_report_tracking_lost(self):
        # Regression (session 2026-07-27_15-55-59): raw MediaPipe world
        # landmarks folded the legs frontally — ankles hallucinated back up
        # near hip height, putting hip and ankle on the same side of the
        # knee (flexion reads 135-180°). That is a tracking failure, not
        # bent knees, and must not be labelled "knee_extension".
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = np.zeros((19, 3))
        pts[CK.LEFT_SHOULDER] = [-0.18, -0.50, 0.0]
        pts[CK.RIGHT_SHOULDER] = [0.18, -0.50, 0.0]
        pts[CK.LEFT_HIP] = [-0.10, 0.0, 0.0]
        pts[CK.RIGHT_HIP] = [0.10, 0.0, 0.0]
        pts[CK.LEFT_KNEE] = [-0.10, 0.45, -0.30]
        pts[CK.RIGHT_KNEE] = [0.10, 0.45, -0.30]
        pts[CK.LEFT_ANKLE] = [-0.10, 0.05, -0.60]
        pts[CK.RIGHT_ANKLE] = [0.10, 0.05, -0.60]

        for _ in range(10):
            assert not gate.check(_make_skeleton(pts))
        assert not gate.is_ready
        assert gate.last_failure == "tracking_lost"

    def test_y_down_standing_passes(self):
        # Live MediaPipe world landmarks are Y-down hip-centered; the gate
        # must accept a proper standing pose in that convention too.
        gate = StandingPoseGate(required_consecutive_frames=1)
        pts = np.zeros((19, 3))
        pts[CK.LEFT_SHOULDER] = [-0.18, -0.50, 0.0]
        pts[CK.RIGHT_SHOULDER] = [0.18, -0.50, 0.0]
        pts[CK.LEFT_HIP] = [-0.10, 0.0, 0.0]
        pts[CK.RIGHT_HIP] = [0.10, 0.0, 0.0]
        pts[CK.LEFT_KNEE] = [-0.10, 0.45, 0.0]
        pts[CK.RIGHT_KNEE] = [0.10, 0.45, 0.0]
        pts[CK.LEFT_ANKLE] = [-0.10, 0.90, 0.0]
        pts[CK.RIGHT_ANKLE] = [0.10, 0.90, 0.0]

        assert gate.check(_make_skeleton(pts))
        assert gate.is_ready

    def test_last_failure_tracks_failing_check(self):
        gate = StandingPoseGate(required_consecutive_frames=5)
        pts = _standing_skeleton()

        confs = np.ones(17)
        confs[CK.LEFT_ANKLE] = 0.1
        gate.check(_make_skeleton(pts, confidences=confs))
        assert gate.last_failure == "visibility"

        bent = _standing_skeleton()
        bent[CK.LEFT_KNEE] = [0.35, 0.60, 0.0]
        bent[CK.RIGHT_KNEE] = [-0.35, 0.60, 0.0]
        gate.check(_make_skeleton(bent))
        assert gate.last_failure == "knee_extension"

        gate.check(_make_skeleton(pts))
        assert gate.last_failure is None

        gate.reset()
        assert gate.last_failure is None


class TestProfileReadinessCheck:
    """An exercise's setup predicate is ANDed with the gate's own checks over its window."""

    def test_a_failing_profile_check_blocks_the_latch(self):
        gate = StandingPoseGate(required_consecutive_frames=3)
        skeleton = _make_skeleton(_standing_skeleton())

        for _ in range(10):
            gate.check(skeleton, profile_check=False)

        assert not gate.is_ready
        assert gate.last_failure == PROFILE_CHECK_FAILURE

    def test_the_profile_check_must_hold_over_the_whole_window(self):
        gate = StandingPoseGate(required_consecutive_frames=3)
        skeleton = _make_skeleton(_standing_skeleton())

        for profile_check in (True, True, False, True, True):
            gate.check(skeleton, profile_check=profile_check)
        assert not gate.is_ready

        gate.check(skeleton, profile_check=True)
        assert gate.is_ready

    def test_a_passing_profile_check_never_replaces_the_gates_checks(self):
        gate = StandingPoseGate(required_consecutive_frames=3)
        crouched = _standing_skeleton()
        crouched[CK.LEFT_KNEE] = [0.40, 0.55, 0.0]
        crouched[CK.RIGHT_KNEE] = [-0.40, 0.55, 0.0]

        for _ in range(10):
            gate.check(_make_skeleton(crouched), profile_check=True)

        assert not gate.is_ready
        assert gate.last_failure != PROFILE_CHECK_FAILURE
