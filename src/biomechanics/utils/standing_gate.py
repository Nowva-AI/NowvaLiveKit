"""
Standing Pose Gate

Validates that the user is standing properly before allowing calibration
to begin. Checks keypoint visibility, knee extension, torso uprightness,
and distance plausibility.

Requires several consecutive passing frames to filter single-frame flukes.
Once satisfied, the gate latches and stays open.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from biomechanics.utils.types import Skeleton3D, CocoKeypoints as CK
from biomechanics.utils.geometry import angle_between_vectors, joint_angle_3_points

logger = logging.getLogger(__name__)

# Frontal knee flexion beyond this is anatomically impossible for someone
# trying to stand — it means the 3D landmarks are corrupted (e.g. the
# folded-leg hallucination puts hip and ankle on the same side of the knee,
# reading 135-180°), not that the user's knees are bent.
TRACKING_LOST_KNEE_FLEXION_DEG = 60.0

# Keypoints that MUST be visible for a valid standing pose.
REQUIRED_KEYPOINTS = [
    CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER,
    CK.LEFT_HIP, CK.RIGHT_HIP,
    CK.LEFT_KNEE, CK.RIGHT_KNEE,
    CK.LEFT_ANKLE, CK.RIGHT_ANKLE,
]

# (heel, toe) per side, for the flat-foot check.
FOOT_CONTACT_PAIRS = [
    (CK.LEFT_HEEL, CK.LEFT_FOOT_INDEX),
    (CK.RIGHT_HEEL, CK.RIGHT_FOOT_INDEX),
]

# Below this, heel and toe have collapsed onto each other and their height
# difference is noise rather than foot inclination.
MIN_FOOT_LENGTH_M = 0.10


def _frontal_knee_flexion(points: np.ndarray) -> tuple:
    # Monocular depth (z) is too noisy for a 3D knee angle — straight legs
    # read 5-30° flexion frame to frame. The frontal (x, y) projection is
    # stable and sufficient to confirm the user is standing.
    frontal = points.copy()
    frontal[:, 2] = 0.0
    left_flexion = 180.0 - joint_angle_3_points(
        frontal[CK.LEFT_HIP], frontal[CK.LEFT_KNEE], frontal[CK.LEFT_ANKLE],
    )
    right_flexion = 180.0 - joint_angle_3_points(
        frontal[CK.RIGHT_HIP], frontal[CK.RIGHT_KNEE], frontal[CK.RIGHT_ANKLE],
    )
    return (left_flexion, right_flexion)


def _heel_rise_ratios(
    points: np.ndarray, confidences: np.ndarray, min_confidence: float,
) -> list[float]:
    # Heel height above the toe as a fraction of foot length. Positive means
    # the heel is raised. Sides whose heel or toe is untracked are skipped
    # rather than failed — heels are often occluded by the camera angle, and
    # a hard requirement would deadlock the gate for those users.
    if len(points) <= CK.RIGHT_HEEL:
        return []

    shoulder_mid_y = (
        points[CK.LEFT_SHOULDER, 1] + points[CK.RIGHT_SHOULDER, 1]
    ) / 2.0
    hip_mid_y = (points[CK.LEFT_HIP, 1] + points[CK.RIGHT_HIP, 1]) / 2.0
    up_sign = 1.0 if shoulder_mid_y > hip_mid_y else -1.0

    ratios: list[float] = []
    for heel_idx, toe_idx in FOOT_CONTACT_PAIRS:
        if min(confidences[heel_idx], confidences[toe_idx]) < min_confidence:
            continue
        foot_length = float(np.linalg.norm(points[heel_idx] - points[toe_idx]))
        if foot_length < MIN_FOOT_LENGTH_M:
            continue
        rise = float(points[heel_idx, 1] - points[toe_idx, 1]) * up_sign
        ratios.append(rise / foot_length)
    return ratios


class StandingPoseGate:
    """
    Validates that the skeleton represents a standing person before
    allowing downstream calibration to begin.

    Checks per frame:
      1. All 8 major keypoints visible with confidence >= min_confidence
      2. Knees nearly extended in the frontal plane (flexion < max_knee_flexion_deg)
      3. Torso roughly upright (trunk angle from vertical < max_trunk_flexion_deg)
      4. Legs vertically extended away from the shoulders
         (ankle-to-hip span >= min_leg_extension_ratio of leg length)
      5. Person at reasonable distance (torso length in plausible range)
      6. Feet flat on the floor (heel not raised above the toe by more than
         max_heel_rise_ratio of foot length)

    Requires ``required_consecutive_frames`` consecutive passing frames
    before latching ``is_ready = True``.
    """

    def __init__(
        self,
        min_confidence: float = 0.25,
        max_knee_flexion_deg: float = 20.0,
        max_trunk_flexion_deg: float = 25.0,
        min_torso_length_m: float = 0.25,
        max_torso_length_m: float = 0.80,
        min_leg_extension_ratio: float = 0.6,
        max_heel_rise_ratio: float = 0.30,
        required_consecutive_frames: int = 5,
    ):
        self.min_confidence = min_confidence
        self.max_knee_flexion_deg = max_knee_flexion_deg
        self.max_trunk_flexion_deg = max_trunk_flexion_deg
        self.min_torso_length_m = min_torso_length_m
        self.max_torso_length_m = max_torso_length_m
        self.min_leg_extension_ratio = min_leg_extension_ratio
        # ~17 degrees of foot inclination. Deliberately lenient: heel and toe
        # depth are the noisiest landmarks the pose model emits, and a flat
        # foot's heel-toe height difference is only a couple of centimetres.
        self.max_heel_rise_ratio = max_heel_rise_ratio
        self.required_consecutive_frames = required_consecutive_frames

        self._consecutive_passes: int = 0
        self._passed: bool = False
        self.last_failure: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """Whether the standing pose gate has been satisfied."""
        return self._passed

    @property
    def progress(self) -> tuple:
        """Return (consecutive_passes, required_frames) for status tracking."""
        return (self._consecutive_passes, self.required_consecutive_frames)

    def check(self, skeleton: Skeleton3D) -> bool:
        """
        Check if the current skeleton passes the standing pose gate.

        Returns True once the gate is satisfied (and stays True forever).
        """
        if self._passed:
            return True

        if self._check_single_frame(skeleton):
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        if self._consecutive_passes > 0 and self._consecutive_passes % 10 == 0:
            logger.info(
                "[STANDING GATE] Progress: %d/%d frames",
                self._consecutive_passes,
                self.required_consecutive_frames,
            )

        if self._consecutive_passes >= self.required_consecutive_frames:
            self._passed = True
            logger.info("[STANDING GATE] Passed — ready")

        return self._passed

    def _check_single_frame(self, skeleton: Skeleton3D) -> bool:
        """Run all standing-pose checks on a single frame."""
        points = skeleton.to_numpy()  # (17, 3)
        confidences = np.array([kp.confidence for kp in skeleton.keypoints])

        if not self._check_keypoint_visibility(confidences):
            self.last_failure = "visibility"
            self._log_failure("visibility", confidences)
            return False
        if not self._check_knee_extension(points):
            left_flexion, right_flexion = _frontal_knee_flexion(points)
            if max(left_flexion, right_flexion) > TRACKING_LOST_KNEE_FLEXION_DEG:
                self.last_failure = "tracking_lost"
            else:
                self.last_failure = "knee_extension"
            self._log_failure(self.last_failure, points)
            return False
        if not self._check_torso_upright(points):
            self.last_failure = "torso_upright"
            self._log_failure("torso_upright", points)
            return False
        # After the torso check so a bent-over user is reported as
        # "torso_upright", not blamed on their (correct) legs.
        if not self._check_leg_extension(points):
            self.last_failure = "leg_extension"
            self._log_failure("leg_extension", points)
            return False
        if not self._check_distance(points):
            self.last_failure = "distance"
            self._log_failure("distance", points)
            return False
        # Last, because passing the gate on raised heels silently poisons the
        # session: the session-scoped foot contact model anchors that ankle
        # height as the planted floor for every rep that follows.
        if not self._check_flat_feet(points, confidences):
            self.last_failure = "flat_foot"
            self._log_failure("flat_foot", (points, confidences))
            return False

        self.last_failure = None
        return True

    def _log_failure(self, check_name: str, data: np.ndarray) -> None:
        """Log diagnostic info about why a check failed (every 30 frames)."""
        if not hasattr(self, "_diag_counter"):
            self._diag_counter = 0
        self._diag_counter += 1
        if self._diag_counter % 30 != 1:
            return

        if check_name == "visibility":
            confs = data
            low = {
                REQUIRED_KEYPOINTS[i]: f"{confs[REQUIRED_KEYPOINTS[i]]:.2f}"
                for i in range(len(REQUIRED_KEYPOINTS))
                if confs[REQUIRED_KEYPOINTS[i]] < self.min_confidence
            }
            logger.warning(
                "[GATE DIAG] FAIL visibility — low confidence keypoints: %s (threshold=%.2f)",
                low, self.min_confidence,
            )
        elif check_name in ("knee_extension", "tracking_lost"):
            l_flexion, r_flexion = _frontal_knee_flexion(data)
            logger.warning(
                "[GATE DIAG] FAIL %s — frontal L flexion=%.1f° R flexion=%.1f° (max=%.1f°)",
                check_name, l_flexion, r_flexion, self.max_knee_flexion_deg,
            )
        elif check_name == "leg_extension":
            pts = data
            hip_mid_y = (pts[CK.LEFT_HIP][1] + pts[CK.RIGHT_HIP][1]) / 2.0
            logger.warning(
                "[GATE DIAG] FAIL leg_extension — ankle-to-hip span "
                "L=%.2fm R=%.2fm (min ratio=%.2f of leg length)",
                abs(pts[CK.LEFT_ANKLE][1] - hip_mid_y),
                abs(pts[CK.RIGHT_ANKLE][1] - hip_mid_y),
                self.min_leg_extension_ratio,
            )
        elif check_name == "torso_upright":
            pts = data
            shoulder_mid = (pts[CK.LEFT_SHOULDER] + pts[CK.RIGHT_SHOULDER]) / 2.0
            hip_mid = (pts[CK.LEFT_HIP] + pts[CK.RIGHT_HIP]) / 2.0
            trunk_vec = shoulder_mid - hip_mid
            raw = angle_between_vectors(trunk_vec, np.array([0.0, 1.0, 0.0]))
            adjusted = min(raw, 180.0 - raw)
            logger.warning(
                "[GATE DIAG] FAIL torso — raw_angle=%.1f° adjusted=%.1f° (max=%.1f°) trunk_vec=%s",
                raw, adjusted, self.max_trunk_flexion_deg, trunk_vec,
            )
        elif check_name == "flat_foot":
            pts, confs = data
            ratios = _heel_rise_ratios(pts, confs, self.min_confidence)
            logger.warning(
                "[GATE DIAG] FAIL flat_foot — heel rise %s of foot length "
                "(max=%.2f). User is on their toes.",
                [f"{ratio:.2f}" for ratio in ratios], self.max_heel_rise_ratio,
            )
        elif check_name == "distance":
            pts = data
            shoulder_mid = (pts[CK.LEFT_SHOULDER] + pts[CK.RIGHT_SHOULDER]) / 2.0
            hip_mid = (pts[CK.LEFT_HIP] + pts[CK.RIGHT_HIP]) / 2.0
            length = float(np.linalg.norm(shoulder_mid - hip_mid))
            logger.warning(
                "[GATE DIAG] FAIL distance — torso_length=%.3fm (range=%.2f–%.2f)",
                length, self.min_torso_length_m, self.max_torso_length_m,
            )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_keypoint_visibility(self, confidences: np.ndarray) -> bool:
        for idx in REQUIRED_KEYPOINTS:
            if confidences[idx] < self.min_confidence:
                return False
        return True

    def _check_knee_extension(self, points: np.ndarray) -> bool:
        left_flexion, right_flexion = _frontal_knee_flexion(points)
        return (
            left_flexion < self.max_knee_flexion_deg
            and right_flexion < self.max_knee_flexion_deg
        )

    def _check_leg_extension(self, points: np.ndarray) -> bool:
        # A depth-folded hallucination (ankles near hip height) is frontally
        # straight, so it slips past the frontal knee check. Require the
        # ankles to extend vertically away from the shoulders by most of the
        # leg's own length. Sign product handles Y-up and Y-down frames.
        hip_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
        torso_direction = shoulder_mid[1] - hip_mid[1]

        for hip_idx, knee_idx, ankle_idx in (
            (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE),
            (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
        ):
            leg_length = float(
                np.linalg.norm(points[hip_idx] - points[knee_idx])
                + np.linalg.norm(points[knee_idx] - points[ankle_idx])
            )
            vertical_span = points[ankle_idx][1] - hip_mid[1]
            if vertical_span * torso_direction >= 0.0:
                return False
            if abs(vertical_span) < self.min_leg_extension_ratio * leg_length:
                return False
        return True

    def _check_torso_upright(self, points: np.ndarray) -> bool:
        shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
        hip_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        trunk_vec = shoulder_mid - hip_mid
        vertical = np.array([0.0, 1.0, 0.0])
        trunk_flexion = angle_between_vectors(trunk_vec, vertical)
        # Handle either Y-up or Y-down coordinate systems (e.g. MediaPipe
        # world landmarks use Y-down).  We only care about deviation from
        # the vertical axis, not which direction along it.
        trunk_flexion = min(trunk_flexion, 180.0 - trunk_flexion)
        return trunk_flexion < self.max_trunk_flexion_deg

    def _check_flat_feet(
        self, points: np.ndarray, confidences: np.ndarray,
    ) -> bool:
        ratios = _heel_rise_ratios(points, confidences, self.min_confidence)
        return all(ratio <= self.max_heel_rise_ratio for ratio in ratios)

    def _check_distance(self, points: np.ndarray) -> bool:
        shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
        hip_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
        return self.min_torso_length_m <= torso_length <= self.max_torso_length_m

    def reset(self) -> None:
        """Reset the gate state."""
        self._consecutive_passes = 0
        self._passed = False
        self.last_failure = None
