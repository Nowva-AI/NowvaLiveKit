"""
Analytical inverse kinematics: joint angles from 3D skeleton landmarks by vector geometry.

Frame (triangulated world and MediaPipe world agree): X = subject's left, Y = down,
+Z = subject's back, so forward is -Z. Any angle whose keypoints fall below the
confidence floor is NaN, never a silent 0.0.
"""

from __future__ import annotations

import logging

import numpy as np

from biomechanics.kinematics.base import IKSolver
from biomechanics.utils.geometry import (
    WORLD_UP,
    angle_between_vectors,
    joint_angle_3_points,
    midpoint,
)
from biomechanics.utils.types import JointAngles, Skeleton3D

logger = logging.getLogger(__name__)

NAN = float("nan")
MIN_CONFIDENCE = 0.1
# Pelvis tilt is typically 30-55 % of trunk flexion in squats. Without ASIS/PSIS
# markers a fixed coupling is used; the per-user coupling was dead code (F7).
PELVIS_TILT_COUPLING = 0.4
MIN_SEGMENT_LENGTH_M = 1e-6
MIN_HIP_WIDTH_M = 0.01
M_TO_CM = 100.0

# Fixed reference-direction vectors, allocated once at import to avoid
# re-creating them every frame in the IK hot path. Treated as read-only.
# Vertical comes from geometry.WORLD_UP.
_AXIS_X = np.array([1.0, 0.0, 0.0])        # subject's left / sagittal-plane normal

_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_foot_index", "right_foot_index",
]


class AnalyticalIKSolver(IKSolver):
    """
    Analytical inverse kinematics solver using vector geometry.

    Computes joint angles directly from 3D keypoint positions without
    requiring a musculoskeletal model.

    Coordinate system (triangulated world frame and MediaPipe world
    landmarks agree; MediaPipe z is smaller closer to the camera the
    subject faces):
    - X-axis points to the subject's left
    - Y-axis points downward (gravity direction)
    - Z-axis points to the subject's BACK; forward is -Z
    """

    # Minimum confidence threshold for keypoints
    MIN_CONFIDENCE = MIN_CONFIDENCE

    def __init__(self, min_confidence: float = MIN_CONFIDENCE):
        super().__init__()
        self.min_confidence = min_confidence
        self._initialized = True  # No initialization needed for analytical solver

    def solve(self, skeleton: Skeleton3D) -> JointAngles:
        """
        Compute joint angles from a 3D skeleton.

        Every angle whose keypoints are missing (confidence below the
        solver minimum) is NaN.
        """
        kpts = self._extract_keypoints(skeleton)

        angles = JointAngles(
            timestamp=skeleton.timestamp,
            frame_index=skeleton.frame_index,
        )

        # Hip flexion (angle between trunk and thigh)
        angles.hip_flexion_l = self._compute_hip_flexion(kpts, side="left")
        angles.hip_flexion_r = self._compute_hip_flexion(kpts, side="right")

        # Hip adduction (medial/lateral deviation)
        angles.hip_adduction_l = self._compute_hip_adduction(kpts, side="left")
        angles.hip_adduction_r = self._compute_hip_adduction(kpts, side="right")

        # Knee valgus + foot_confidence + hip_rotation: written by the
        # pipeline's ValgusEstimator after solve() returns (mode-aware).

        # Knee flexion
        angles.knee_flexion_l = self._compute_knee_flexion(kpts, side="left")
        angles.knee_flexion_r = self._compute_knee_flexion(kpts, side="right")

        # Ankle dorsiflexion
        angles.ankle_dorsiflexion_l = self._compute_ankle_dorsiflexion(kpts, side="left")
        angles.ankle_dorsiflexion_r = self._compute_ankle_dorsiflexion(kpts, side="right")

        # Trunk angles
        angles.trunk_flexion = self._compute_trunk_flexion(kpts)
        angles.trunk_lateral_flexion = self._compute_trunk_lateral_flexion(kpts)
        angles.trunk_rotation = self._compute_trunk_rotation(kpts)

        # Pelvis angles
        angles.pelvis_tilt = self._compute_pelvis_tilt(kpts)
        angles.pelvis_list = self._compute_pelvis_list(kpts)
        angles.pelvis_rotation = self._compute_pelvis_rotation(kpts)

        # Upper-body angles
        angles.elbow_flexion_l = self._compute_elbow_flexion(kpts, side="left")
        angles.elbow_flexion_r = self._compute_elbow_flexion(kpts, side="right")
        angles.shoulder_flexion_l = self._compute_shoulder_flexion(kpts, side="left")
        angles.shoulder_flexion_r = self._compute_shoulder_flexion(kpts, side="right")
        angles.shoulder_abduction_l = self._compute_shoulder_abduction(kpts, side="left")
        angles.shoulder_abduction_r = self._compute_shoulder_abduction(kpts, side="right")

        # Wrist positions relative to shoulder midpoint
        wrist_pos = self._compute_wrist_positions(kpts)
        angles.wrist_y_l = wrist_pos["wrist_y_l"]
        angles.wrist_y_r = wrist_pos["wrist_y_r"]
        angles.wrist_x_l = wrist_pos["wrist_x_l"]
        angles.wrist_x_r = wrist_pos["wrist_x_r"]

        return angles

    def _extract_keypoints(self, skeleton: Skeleton3D) -> dict[str, tuple[np.ndarray, float]]:
        kpts: dict[str, tuple[np.ndarray, float]] = {}
        for i, name in enumerate(_KEYPOINT_NAMES):
            if i < len(skeleton.keypoints):
                kp = skeleton.keypoints[i]
                kpts[name] = (np.array([kp.x, kp.y, kp.z]), kp.confidence)
            else:
                kpts[name] = (np.zeros(3), 0.0)
        return kpts

    def _get_point(self, kpts: dict[str, tuple[np.ndarray, float]], name: str) -> np.ndarray | None:
        pos, conf = kpts.get(name, (np.zeros(3), 0.0))
        if conf >= self.min_confidence:
            return pos
        return None

    def _compute_hip_flexion(self, kpts: dict, side: str) -> float:
        """
        Hip flexion: sagittal-plane angle between the trunk and thigh vectors.
        0 degrees = standing upright (thigh aligned with trunk), increases with flexion.
        """
        hip = self._get_point(kpts, f"{side}_hip")
        knee = self._get_point(kpts, f"{side}_knee")
        shoulder = self._get_point(kpts, f"{side}_shoulder")

        if hip is None or knee is None or shoulder is None:
            return NAN

        trunk_vec = shoulder - hip
        thigh_vec = knee - hip

        if np.linalg.norm(trunk_vec) < MIN_SEGMENT_LENGTH_M or np.linalg.norm(thigh_vec) < MIN_SEGMENT_LENGTH_M:
            return NAN

        raw_angle = angle_between_vectors(trunk_vec, thigh_vec)
        return 180.0 - raw_angle

    def _compute_hip_adduction(self, kpts: dict, side: str) -> float:
        """
        Hip adduction: medial/lateral deviation of the thigh from the sagittal plane.
        Positive = adduction (knee toward the midline), negative = abduction.
        """
        hip = self._get_point(kpts, f"{side}_hip")
        knee = self._get_point(kpts, f"{side}_knee")

        if hip is None or knee is None:
            return NAN

        thigh_vec = knee - hip

        left_hip = self._get_point(kpts, "left_hip")
        right_hip = self._get_point(kpts, "right_hip")

        if left_hip is None or right_hip is None:
            return NAN

        # Project thigh vector onto the frontal plane (remove X component)
        thigh_frontal = np.array([0, thigh_vec[1], thigh_vec[2]])

        if np.linalg.norm(thigh_frontal) < MIN_SEGMENT_LENGTH_M:
            return NAN

        angle = angle_between_vectors(thigh_vec, thigh_frontal)

        # Adduction is the knee travelling toward the midline. +X is the
        # subject's left, so that is -X for the left leg and +X for the right.
        if side == "left":
            sign = 1.0 if thigh_vec[0] < 0 else -1.0
        else:
            sign = 1.0 if thigh_vec[0] > 0 else -1.0

        return angle * sign

    def _compute_knee_flexion(self, kpts: dict, side: str) -> float:
        """Knee flexion: 0 degrees = fully extended, increases with bending."""
        hip = self._get_point(kpts, f"{side}_hip")
        knee = self._get_point(kpts, f"{side}_knee")
        ankle = self._get_point(kpts, f"{side}_ankle")

        if hip is None or knee is None or ankle is None:
            return NAN

        # 180 degrees at the joint = 0 degrees flexion (straight)
        return 180.0 - joint_angle_3_points(hip, knee, ankle)

    def _compute_ankle_dorsiflexion(self, kpts: dict, side: str) -> float:
        """
        Ankle dorsiflexion as shank tilt from vertical.
        0 = shank vertical (neutral). Does not use foot keypoints.
        """
        knee = self._get_point(kpts, f"{side}_knee")
        ankle = self._get_point(kpts, f"{side}_ankle")

        if knee is None or ankle is None:
            return NAN

        shank = knee - ankle
        if np.linalg.norm(shank) < MIN_SEGMENT_LENGTH_M:
            return NAN

        return angle_between_vectors(shank, WORLD_UP)

    def _compute_trunk_flexion(self, kpts: dict) -> float:
        """
        Trunk forward flexion (sagittal plane).
        180 degrees = upright, decreases with forward lean.
        """
        left_shoulder = self._get_point(kpts, "left_shoulder")
        right_shoulder = self._get_point(kpts, "right_shoulder")
        left_hip = self._get_point(kpts, "left_hip")
        right_hip = self._get_point(kpts, "right_hip")

        if any(p is None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return NAN

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        trunk_vec = shoulder_mid - hip_mid

        if np.linalg.norm(trunk_vec) < MIN_SEGMENT_LENGTH_M:
            return NAN

        return 180.0 - angle_between_vectors(trunk_vec, WORLD_UP)

    def _compute_trunk_lateral_flexion(self, kpts: dict) -> float:
        """
        Trunk lateral flexion (frontal plane).
        0 degrees = upright, positive = lean to the left, negative = lean to the right.
        """
        left_shoulder = self._get_point(kpts, "left_shoulder")
        right_shoulder = self._get_point(kpts, "right_shoulder")
        left_hip = self._get_point(kpts, "left_hip")
        right_hip = self._get_point(kpts, "right_hip")

        if any(p is None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return NAN

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)

        # Project trunk to the frontal plane (XY plane, remove Z)
        trunk_vec = shoulder_mid - hip_mid
        trunk_frontal = np.array([trunk_vec[0], trunk_vec[1], 0])

        if np.linalg.norm(trunk_frontal) < MIN_SEGMENT_LENGTH_M:
            return NAN

        angle = angle_between_vectors(trunk_frontal, WORLD_UP)

        # Positive = lean left; +X is the subject's left
        if trunk_vec[0] < 0:
            angle = -angle

        return angle

    def _compute_trunk_rotation(self, kpts: dict) -> float:
        """
        Trunk axial rotation (transverse plane).
        0 degrees = facing forward, positive = rotated to the left, negative = to the right.
        """
        left_shoulder = self._get_point(kpts, "left_shoulder")
        right_shoulder = self._get_point(kpts, "right_shoulder")

        if left_shoulder is None or right_shoulder is None:
            return NAN

        shoulder_vec = left_shoulder - right_shoulder
        shoulder_xz = np.array([shoulder_vec[0], 0, shoulder_vec[2]])

        if np.linalg.norm(shoulder_xz) < MIN_SEGMENT_LENGTH_M:
            return NAN

        angle = angle_between_vectors(shoulder_xz, _AXIS_X)

        # Rotating left moves the left shoulder back (+Z = subject's back)
        if shoulder_vec[2] < 0:
            angle = -angle

        return angle

    def _compute_pelvis_tilt(self, kpts: dict) -> float:
        """
        Pelvis anterior/posterior tilt, estimated as a fixed fraction of
        sagittal trunk inclination (true pelvis tilt needs ASIS/PSIS markers).
        0 degrees = neutral, positive = anterior tilt (forward lean), negative = posterior.
        """
        left_hip = self._get_point(kpts, "left_hip")
        right_hip = self._get_point(kpts, "right_hip")
        left_shoulder = self._get_point(kpts, "left_shoulder")
        right_shoulder = self._get_point(kpts, "right_shoulder")

        if any(p is None for p in [left_hip, right_hip, left_shoulder, right_shoulder]):
            return NAN

        hip_mid = midpoint(left_hip, right_hip)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        trunk_vec = shoulder_mid - hip_mid

        # Project to the sagittal plane (YZ)
        trunk_sagittal = np.array([0, trunk_vec[1], trunk_vec[2]])

        if np.linalg.norm(trunk_sagittal) < MIN_SEGMENT_LENGTH_M:
            return NAN

        trunk_angle = angle_between_vectors(trunk_sagittal, WORLD_UP)

        # Forward is -Z: a forward lean puts the shoulders at a smaller Z than the hips
        if trunk_vec[2] > 0:
            trunk_angle = -trunk_angle

        return trunk_angle * PELVIS_TILT_COUPLING

    def _compute_pelvis_list(self, kpts: dict) -> float:
        """
        Pelvis lateral list (hiking).
        0 degrees = level, positive = left hip higher, negative = right hip higher.
        """
        left_hip = self._get_point(kpts, "left_hip")
        right_hip = self._get_point(kpts, "right_hip")

        if left_hip is None or right_hip is None:
            return NAN

        # Height difference between hips (Y-down: higher means smaller y)
        height_diff = right_hip[1] - left_hip[1]
        hip_width = np.linalg.norm(left_hip - right_hip)

        if hip_width < MIN_HIP_WIDTH_M:
            return NAN

        return float(np.degrees(np.arctan2(height_diff, hip_width)))

    def _compute_pelvis_rotation(self, kpts: dict) -> float:
        """
        Pelvis axial rotation.
        0 degrees = hips facing forward, positive = rotated to the left, negative = to the right.
        """
        left_hip = self._get_point(kpts, "left_hip")
        right_hip = self._get_point(kpts, "right_hip")

        if left_hip is None or right_hip is None:
            return NAN

        hip_vec = left_hip - right_hip
        hip_xz = np.array([hip_vec[0], 0, hip_vec[2]])

        if np.linalg.norm(hip_xz) < MIN_SEGMENT_LENGTH_M:
            return NAN

        angle = angle_between_vectors(hip_xz, _AXIS_X)

        # Rotating left moves the left hip back (+Z = subject's back)
        if hip_vec[2] < 0:
            angle = -angle

        return angle

    # ------------------------------------------------------------------
    # Upper-body angle computations
    # ------------------------------------------------------------------

    def _compute_elbow_flexion(self, kpts: dict, side: str) -> float:
        """Elbow flexion: 0 degrees = straight arm, increases with bending."""
        shoulder = self._get_point(kpts, f"{side}_shoulder")
        elbow = self._get_point(kpts, f"{side}_elbow")
        wrist = self._get_point(kpts, f"{side}_wrist")

        if shoulder is None or elbow is None or wrist is None:
            return NAN

        return 180.0 - joint_angle_3_points(shoulder, elbow, wrist)

    def _compute_shoulder_flexion(self, kpts: dict, side: str) -> float:
        """
        Shoulder flexion (sagittal plane): angle between the upper arm and the trunk.
        0 = arm hanging at the side, 90 = horizontal in front, 180 = overhead.
        """
        shoulder = self._get_point(kpts, f"{side}_shoulder")
        elbow = self._get_point(kpts, f"{side}_elbow")
        hip = self._get_point(kpts, f"{side}_hip")

        if shoulder is None or elbow is None or hip is None:
            return NAN

        trunk_vec = shoulder - hip
        upper_arm_vec = elbow - shoulder

        # A hanging arm is roughly opposite to the trunk (~180), so flexion = 180 - raw
        return 180.0 - angle_between_vectors(trunk_vec, upper_arm_vec)

    def _compute_shoulder_abduction(self, kpts: dict, side: str) -> float:
        """
        Shoulder abduction (frontal plane): lateral deviation of the upper arm from the trunk.
        0 = arm hanging at the side, 90 = horizontal out to the side.
        """
        shoulder = self._get_point(kpts, f"{side}_shoulder")
        elbow = self._get_point(kpts, f"{side}_elbow")
        hip = self._get_point(kpts, f"{side}_hip")

        if shoulder is None or elbow is None or hip is None:
            return NAN

        trunk_vec = shoulder - hip
        upper_arm_vec = elbow - shoulder

        # Project both vectors onto the frontal plane (XY, remove Z/depth)
        trunk_frontal = np.array([trunk_vec[0], trunk_vec[1], 0.0])
        arm_frontal = np.array([upper_arm_vec[0], upper_arm_vec[1], 0.0])

        if np.linalg.norm(trunk_frontal) < MIN_SEGMENT_LENGTH_M or np.linalg.norm(arm_frontal) < MIN_SEGMENT_LENGTH_M:
            return NAN

        return 180.0 - angle_between_vectors(trunk_frontal, arm_frontal)

    def _compute_wrist_positions(self, kpts: dict) -> dict[str, float]:
        """
        Wrist positions relative to the shoulder midpoint, in cm.
        wrist_y: positive = above the shoulder. wrist_x: positive = in front of the shoulder (-Z).
        """
        left_shoulder = self._get_point(kpts, "left_shoulder")
        right_shoulder = self._get_point(kpts, "right_shoulder")
        left_wrist = self._get_point(kpts, "left_wrist")
        right_wrist = self._get_point(kpts, "right_wrist")

        result = {"wrist_y_l": NAN, "wrist_y_r": NAN, "wrist_x_l": NAN, "wrist_x_r": NAN}

        if left_shoulder is None or right_shoulder is None:
            return result

        shoulder_mid = midpoint(left_shoulder, right_shoulder)

        # Y is down, so shoulder_mid_y - wrist_y is the height above the shoulder;
        # forward is -Z, so shoulder_mid_z - wrist_z is the distance in front.
        if left_wrist is not None:
            result["wrist_y_l"] = float(shoulder_mid[1] - left_wrist[1]) * M_TO_CM
            result["wrist_x_l"] = float(shoulder_mid[2] - left_wrist[2]) * M_TO_CM

        if right_wrist is not None:
            result["wrist_y_r"] = float(shoulder_mid[1] - right_wrist[1]) * M_TO_CM
            result["wrist_x_r"] = float(shoulder_mid[2] - right_wrist[2]) * M_TO_CM

        return result
