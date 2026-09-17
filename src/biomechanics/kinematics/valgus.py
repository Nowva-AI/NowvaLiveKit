"""
Mode-aware knee valgus estimation.

Two estimators write the same JointAngles fields (knee_valgus in degrees,
positive = knee medial/valgus; a bilateral knee-to-ankle separation ratio) but
from different data depending on capture mode:

- SingleCameraValgusEstimator: frontal-plane projection angle (FPPA) from the
  image-plane 2D skeleton, plus an x-only knee-to-ankle separation ratio. Avoids
  the unreliable monocular depth axis entirely.
- TriangulatedValgusEstimator: 3D knee deviation from the plane the knee tracks
  in when it follows the toes (spanned by the hip-ankle line and the foot's
  forward direction), so knees-over-toes reads neutral at any toe-out angle and
  any stance width. Also exposes a signed hip internal-rotation estimate.

Any measurement whose keypoints are missing is NaN, never a neutral 0.0.
Frame: X = subject's left, Y = down, +Z = subject's back (forward = -Z).
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np

from biomechanics.utils.geometry import WORLD_UP, angle_between_vectors, normalize_vector
from biomechanics.utils.types import (
    CocoKeypoints as CK,
    Skeleton2D,
    Skeleton3D,
)

NAN = float("nan")
# Minimum limb-segment length (m) in 3D for a stable angle.
_MIN_SEGMENT_M = 0.05
# Minimum horizontal foot length (m) for a usable foot direction.
_MIN_FOOT_M = 0.05
# Minimum sine of the angle between the hip-ankle line and the foot direction
# for a well-conditioned knee-tracking plane.
_MIN_PLANE_SINE = 0.1
# Minimum horizontal ankle separation (px) for a stable 2D ratio.
_MIN_ANKLE_SEP_PX = 1.0
# Minimum hip-to-ankle vertical span (px) for a stable FPPA. A near-zero span
# means a mistracked keypoint (ankle at hip height).
_MIN_LEG_SPAN_PX = 10.0
# Minimum pelvis width (px) for a stable FPPA. It sits in the arctan2
# denominator, so small values turn pixel noise into ~90 degree spikes.
_MIN_PELVIS_WIDTH_PX = 10.0
# Minimum 3D ankle separation (m) for a stable ratio.
_MIN_ANKLE_SEP_M = 0.02
# Facing squareness (horizontal hip separation / torso length) at or above which
# single-camera valgus confidence saturates to full. Below it, confidence decays
# linearly toward zero as the subject rotates out of the frontal plane.
_FRONTAL_NOMINAL = 0.35
# Keypoint confidence floor for a usable measurement.
_MIN_CONFIDENCE = 0.1
# Medial direction along the pelvis ML axis (left hip - right hip) per side.
_MEDIAL_SIGN_LEFT = -1.0
_MEDIAL_SIGN_RIGHT = 1.0


class ValgusResult(NamedTuple):
    """Per-frame frontal/transverse knee kinematics from a valgus estimator."""
    valgus_l: float
    valgus_r: float
    foot_confidence_l: float
    foot_confidence_r: float
    kasr: float
    hip_rotation_l: float = NAN
    hip_rotation_r: float = NAN


class ValgusEstimator(Protocol):
    def estimate(
        self,
        skeleton_2d: Skeleton2D | None,
        skeleton_3d: Skeleton3D | None = None,
    ) -> ValgusResult:
        ...


_MISSING_RESULT = ValgusResult(NAN, NAN, 0.0, 0.0, NAN, NAN, NAN)


def _xy(skeleton_2d: Skeleton2D, index: int) -> tuple[np.ndarray | None, float]:
    kp = skeleton_2d.get_keypoint(index)
    if kp is None:
        return None, 0.0
    if kp.confidence < _MIN_CONFIDENCE:
        return None, kp.confidence
    return np.array([kp.x, kp.y], dtype=np.float64), kp.confidence


def _xyz(skeleton_3d: Skeleton3D, index: int) -> tuple[np.ndarray | None, float]:
    if index >= len(skeleton_3d.keypoints):
        return None, 0.0
    pt = skeleton_3d.keypoints[index]
    if pt.confidence < _MIN_CONFIDENCE:
        return None, pt.confidence
    return np.array([pt.x, pt.y, pt.z], dtype=np.float64), pt.confidence


class SingleCameraValgusEstimator:
    """
    Frontal-plane projection angle (FPPA) from a single frontal camera.

    Works entirely in the image plane (x, y), never touching the monocular depth
    axis — the least reliable coordinate in a single-camera pose. Valgus is the
    knee's horizontal deviation from the hip-ankle line, normalized by pelvis
    width, signed positive when the knee moves medially (toward the midline
    between the feet).

    Pelvis width is the denominator because it is the only stable reference
    available every frame. The deviation itself is horizontal and does not
    foreshorten, but every vertical or diagonal reference does: the previous
    denominator was the live hip-to-ankle span, which collapses on descent and
    inflated identical knee cave by ~1.4x at parallel and ~1.8x at the bottom
    (the angle at the knee is worse still, ~2.4x, because both limb segments
    foreshorten). Pelvis width is frontal, so it holds across depth — and
    since both terms are in pixels, the ratio is also invariant to how far the
    athlete stands from the camera.
    """

    def estimate(
        self,
        skeleton_2d: Skeleton2D | None,
        skeleton_3d: Skeleton3D | None = None,
    ) -> ValgusResult:
        if skeleton_2d is None:
            return _MISSING_RESULT

        l_hip, c_lh = _xy(skeleton_2d, CK.LEFT_HIP)
        r_hip, c_rh = _xy(skeleton_2d, CK.RIGHT_HIP)
        l_knee, c_lk = _xy(skeleton_2d, CK.LEFT_KNEE)
        r_knee, c_rk = _xy(skeleton_2d, CK.RIGHT_KNEE)
        l_ankle, c_la = _xy(skeleton_2d, CK.LEFT_ANKLE)
        r_ankle, c_ra = _xy(skeleton_2d, CK.RIGHT_ANKLE)

        if l_ankle is None or r_ankle is None:
            return _MISSING_RESULT
        midline_x = (l_ankle[0] + r_ankle[0]) / 2.0

        pelvis_width = (
            abs(l_hip[0] - r_hip[0])
            if l_hip is not None and r_hip is not None
            else 0.0
        )

        valgus_l = self._fppa(l_hip, l_knee, l_ankle, midline_x, pelvis_width)
        valgus_r = self._fppa(r_hip, r_knee, r_ankle, midline_x, pelvis_width)

        facing = self._facing_confidence(skeleton_2d, l_hip, r_hip)
        conf_l = min(c_lh, c_lk, c_la) * facing
        conf_r = min(c_rh, c_rk, c_ra) * facing

        kasr = self._kasr_2d(l_knee, r_knee, l_ankle, r_ankle)

        return ValgusResult(valgus_l, valgus_r, conf_l, conf_r, kasr)

    @staticmethod
    def _fppa(
        hip: np.ndarray | None,
        knee: np.ndarray | None,
        ankle: np.ndarray | None,
        midline_x: float,
        pelvis_width: float,
    ) -> float:
        if hip is None or knee is None or ankle is None:
            return NAN

        if abs(ankle[1] - hip[1]) < _MIN_LEG_SPAN_PX:
            return NAN
        if pelvis_width < _MIN_PELVIS_WIDTH_PX:
            return NAN

        # Expected knee x if it tracked the hip-ankle line at the knee's height.
        # Clamp to the hip-ankle segment: a knee's y should fall between hip and
        # ankle, so a t far outside [0, 1] signals a mistracked keypoint rather
        # than real geometry — extrapolating from it would spike the angle.
        t = np.clip((knee[1] - hip[1]) / (ankle[1] - hip[1]), 0.0, 1.0)
        expected_x = hip[0] + t * (ankle[0] - hip[0])
        deviation_x = knee[0] - expected_x

        # Medial direction for this leg: toward the midline between the feet.
        medial_dir = 1.0 if midline_x >= ankle[0] else -1.0
        signed_deviation = deviation_x * medial_dir

        return float(np.degrees(np.arctan2(signed_deviation, pelvis_width)))

    @staticmethod
    def _kasr_2d(
        l_knee: np.ndarray | None,
        r_knee: np.ndarray | None,
        l_ankle: np.ndarray | None,
        r_ankle: np.ndarray | None,
    ) -> float:
        if l_knee is None or r_knee is None or l_ankle is None or r_ankle is None:
            return NAN
        knee_sep = abs(l_knee[0] - r_knee[0])
        ankle_sep = abs(l_ankle[0] - r_ankle[0])
        if ankle_sep < _MIN_ANKLE_SEP_PX:
            return NAN
        return float(knee_sep / ankle_sep)

    @staticmethod
    def _facing_confidence(
        skeleton_2d: Skeleton2D,
        l_hip: np.ndarray | None,
        r_hip: np.ndarray | None,
    ) -> float:
        # Rotation out of the frontal plane collapses the horizontal hip
        # separation. Normalize it by torso length (rotation-stable) so the
        # measure is scale-invariant; saturate to 1.0 for any roughly frontal
        # stance, decay toward 0 only when clearly turned.
        l_sh, _ = _xy(skeleton_2d, CK.LEFT_SHOULDER)
        r_sh, _ = _xy(skeleton_2d, CK.RIGHT_SHOULDER)
        if l_hip is None or r_hip is None or l_sh is None or r_sh is None:
            return 1.0

        hip_sep_x = abs(l_hip[0] - r_hip[0])
        shoulder_mid = (l_sh + r_sh) / 2.0
        hip_mid = (l_hip + r_hip) / 2.0
        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        if torso_len < 1e-6:
            return 1.0

        ratio = hip_sep_x / torso_len
        return float(np.clip(ratio / _FRONTAL_NOMINAL, 0.0, 1.0))


class TriangulatedValgusEstimator:
    """
    3D knee deviation from the knee-tracking plane, from triangulated keypoints.

    The plane spanned by the hip-ankle line and the foot's horizontal forward
    direction is where the knee sits when it tracks over the toes. Valgus is the
    femur's tilt out of that plane, signed positive when the knee lies medial
    to it (knee cave) and negative when lateral (varus). Magnitude and sign
    come from the same signed projection, and the reading is invariant to
    toe-out and stance width by construction. The earlier Grood-Suntay
    variant took its magnitude from the pelvic ML axis and its sign from the
    hip-ankle line, which read a 10 degree knee cave as varus at 15 degrees of
    toe-out.
    """

    def estimate(
        self,
        skeleton_2d: Skeleton2D | None,
        skeleton_3d: Skeleton3D | None = None,
    ) -> ValgusResult:
        if skeleton_3d is None:
            return _MISSING_RESULT

        l_hip, c_lh = _xyz(skeleton_3d, CK.LEFT_HIP)
        r_hip, c_rh = _xyz(skeleton_3d, CK.RIGHT_HIP)
        l_knee, c_lk = _xyz(skeleton_3d, CK.LEFT_KNEE)
        r_knee, c_rk = _xyz(skeleton_3d, CK.RIGHT_KNEE)
        l_ankle, c_la = _xyz(skeleton_3d, CK.LEFT_ANKLE)
        r_ankle, c_ra = _xyz(skeleton_3d, CK.RIGHT_ANKLE)
        l_toe, c_lt = _xyz(skeleton_3d, CK.LEFT_FOOT_INDEX)
        r_toe, c_rt = _xyz(skeleton_3d, CK.RIGHT_FOOT_INDEX)

        if l_hip is None or r_hip is None:
            return _MISSING_RESULT
        ml_pelvis = normalize_vector(l_hip - r_hip)

        valgus_l = self._knee_deviation(l_hip, l_knee, l_ankle, l_toe, ml_pelvis, _MEDIAL_SIGN_LEFT)
        valgus_r = self._knee_deviation(r_hip, r_knee, r_ankle, r_toe, ml_pelvis, _MEDIAL_SIGN_RIGHT)

        hip_rot_l = self._hip_internal_rotation(l_hip, l_knee, l_ankle, l_toe, ml_pelvis, _MEDIAL_SIGN_LEFT)
        hip_rot_r = self._hip_internal_rotation(r_hip, r_knee, r_ankle, r_toe, ml_pelvis, _MEDIAL_SIGN_RIGHT)

        # Valgus confidence tracks every keypoint the metric needs, toe included.
        conf_l = min(c_lh, c_lk, c_la, c_lt)
        conf_r = min(c_rh, c_rk, c_ra, c_rt)

        kasr = self._kasr_3d(l_knee, r_knee, l_ankle, r_ankle)

        return ValgusResult(valgus_l, valgus_r, conf_l, conf_r, kasr, hip_rot_l, hip_rot_r)

    @staticmethod
    def _knee_deviation(
        hip: np.ndarray | None,
        knee: np.ndarray | None,
        ankle: np.ndarray | None,
        toe: np.ndarray | None,
        ml_pelvis: np.ndarray,
        medial_sign: float,
    ) -> float:
        if hip is None or knee is None or ankle is None or toe is None:
            return NAN

        leg_line = ankle - hip
        femur = knee - hip
        foot = toe - ankle
        foot_horizontal = foot - np.dot(foot, WORLD_UP) * WORLD_UP

        leg_len = float(np.linalg.norm(leg_line))
        femur_len = float(np.linalg.norm(femur))
        foot_len = float(np.linalg.norm(foot_horizontal))
        if leg_len < _MIN_SEGMENT_M or femur_len < _MIN_SEGMENT_M or foot_len < _MIN_FOOT_M:
            return NAN

        # Normal of the plane the knee tracks in when it follows the toes.
        plane_normal = np.cross(leg_line, foot_horizontal)
        if float(np.linalg.norm(plane_normal)) < _MIN_PLANE_SINE * leg_len * foot_len:
            return NAN
        plane_normal = normalize_vector(plane_normal)
        # Orient the normal toward the midline so medial deviation is positive.
        if np.dot(plane_normal, medial_sign * ml_pelvis) < 0:
            plane_normal = -plane_normal

        sine = float(np.dot(femur, plane_normal)) / femur_len
        return float(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))

    @staticmethod
    def _hip_internal_rotation(
        hip: np.ndarray | None,
        knee: np.ndarray | None,
        ankle: np.ndarray | None,
        foot: np.ndarray | None,
        ml_pelvis: np.ndarray,
        medial_sign: float,
    ) -> float:
        # Transverse-plane yaw of the foot relative to the pelvis forward axis,
        # about the thigh long axis. Diagnostic only (needs real 3D depth, so it
        # is meaningful in triangulated mode). Positive = internal rotation
        # (toes turned toward the midline), negative = external rotation.
        if hip is None or knee is None or ankle is None or foot is None:
            return NAN

        thigh_axis = normalize_vector(knee - hip)
        if np.linalg.norm(thigh_axis) < 1e-6:
            return NAN

        # Pelvis forward (-Z when standing) = thigh (down) axis crossed with the ML axis.
        forward = normalize_vector(np.cross(thigh_axis, ml_pelvis))
        foot_vec = foot - ankle
        # Project the foot and reference onto the transverse plane (normal = thigh).
        foot_t = foot_vec - np.dot(foot_vec, thigh_axis) * thigh_axis
        fwd_t = forward - np.dot(forward, thigh_axis) * thigh_axis
        if np.linalg.norm(foot_t) < 1e-6 or np.linalg.norm(fwd_t) < 1e-6:
            return NAN
        rotation = angle_between_vectors(fwd_t, foot_t)
        if np.dot(foot_t, medial_sign * ml_pelvis) < 0:
            rotation = -rotation
        return rotation

    @staticmethod
    def _kasr_3d(
        l_knee: np.ndarray | None,
        r_knee: np.ndarray | None,
        l_ankle: np.ndarray | None,
        r_ankle: np.ndarray | None,
    ) -> float:
        if l_knee is None or r_knee is None or l_ankle is None or r_ankle is None:
            return NAN
        knee_sep = float(np.linalg.norm(l_knee - r_knee))
        ankle_sep = float(np.linalg.norm(l_ankle - r_ankle))
        if ankle_sep < _MIN_ANKLE_SEP_M:
            return NAN
        return knee_sep / ankle_sep


def build_valgus_estimator(multi_camera: bool) -> ValgusEstimator:
    """Select the valgus estimator for the active capture mode."""
    if multi_camera:
        return TriangulatedValgusEstimator()
    return SingleCameraValgusEstimator()
