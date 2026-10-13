"""
Geometric Math Utilities for Biomechanics Pipeline

Provides common geometric operations used across the pipeline,
particularly for joint angle calculations and 3D analysis.
"""

import numpy as np
from typing import Tuple, Optional


# Single source of truth for the production coordinate frame.
# 3D keypoints come from MediaPipe world landmarks, which are Y-DOWN: a
# keypoint that is physically higher has a SMALLER y. The world's "up"
# direction is therefore -Y. No module may write a bare [0, 1, 0] for
# vertical — import WORLD_UP.
WORLD_UP = np.array([0.0, -1.0, 0.0])


def is_above(point_a: np.ndarray, point_b: np.ndarray) -> bool:
    """True if point_a is physically higher than point_b in the Y-down frame."""
    return bool(point_a[1] < point_b[1])


def height_above(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Signed height of point_a above point_b, positive when a is higher."""
    return float(point_b[1] - point_a[1])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors in degrees.

    Args:
        v1: First vector (any dimension)
        v2: Second vector (same dimension as v1)

    Returns:
        Angle between vectors in degrees (0-180)
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-10 or v2_norm < 1e-10:
        return 0.0

    # Normalize vectors
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm

    # Compute dot product and clamp to valid range for arccos
    dot = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)

    # Return angle in degrees
    return float(np.degrees(np.arccos(dot)))


def signed_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the signed angle from v1 to v2 in 2D.

    Positive angle means counterclockwise rotation from v1 to v2.

    Args:
        v1: First 2D vector
        v2: Second 2D vector

    Returns:
        Signed angle in degrees (-180 to 180)
    """
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    return float(np.degrees(angle))


def project_to_plane(
    point: np.ndarray, plane_normal: np.ndarray, plane_point: np.ndarray
) -> np.ndarray:
    """
    Project a 3D point onto a plane.

    Args:
        point: 3D point to project
        plane_normal: Normal vector of the plane (will be normalized)
        plane_point: A point on the plane

    Returns:
        Projected point on the plane
    """
    # Normalize the plane normal
    normal = plane_normal / np.linalg.norm(plane_normal)

    # Vector from plane point to the point
    v = point - plane_point

    # Distance from point to plane along the normal
    dist = np.dot(v, normal)

    # Project by subtracting the normal component
    projected = point - dist * normal

    return projected


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculate the midpoint between two points.

    Args:
        p1: First point (any dimension)
        p2: Second point (same dimension as p1)

    Returns:
        Midpoint as numpy array
    """
    return (p1 + p2) / 2.0


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 3D points.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Distance as float
    """
    return float(np.linalg.norm(p1 - p2))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Unit vector (or zero vector if input is zero)
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate cross product of two 3D vectors.

    Args:
        v1: First 3D vector
        v2: Second 3D vector

    Returns:
        Cross product vector
    """
    return np.cross(v1, v2)


def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate rotation matrix that rotates v1 to v2.

    Args:
        v1: Source vector (will be normalized)
        v2: Target vector (will be normalized)

    Returns:
        3x3 rotation matrix
    """
    a = normalize_vector(v1)
    b = normalize_vector(v2)

    # Cross product gives rotation axis
    v = np.cross(a, b)
    c = np.dot(a, b)

    # Handle parallel vectors
    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation - find perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            perp = normalize_vector(perp - np.dot(perp, a) * a)
            return 2 * np.outer(perp, perp) - np.eye(3)

    # Rodrigues' rotation formula
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + np.dot(vx, vx) * (1 / (1 + c))

    return R


def joint_angle_3_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """
    Calculate the angle at p2 formed by points p1-p2-p3.

    This is commonly used for calculating joint angles where:
    - p1 is the proximal joint
    - p2 is the joint of interest
    - p3 is the distal joint

    Args:
        p1: First point (e.g., hip)
        p2: Middle point / vertex (e.g., knee)
        p3: Third point (e.g., ankle)

    Returns:
        Angle at p2 in degrees (0-180)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    return angle_between_vectors(v1, v2)


def flexion_angle(
    proximal: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray,
    invert: bool = False
) -> float:
    """
    Calculate flexion angle at a joint.

    For most joints, 0 degrees is full extension and positive values
    indicate flexion. Use invert=True for joints where this is reversed.

    Args:
        proximal: Proximal joint position
        joint: Current joint position
        distal: Distal joint position
        invert: If True, invert the angle (180 - angle)

    Returns:
        Flexion angle in degrees
    """
    angle = joint_angle_3_points(proximal, joint, distal)

    if invert:
        return 180.0 - angle

    # Standard flexion: 180 is extended, less is flexed
    # Convert so 0 is extended, positive is flexed
    return 180.0 - angle


def calculate_trunk_angle(
    shoulder_mid: np.ndarray,
    hip_mid: np.ndarray,
    vertical: np.ndarray = None
) -> float:
    """
    Calculate trunk forward lean angle.

    Measures the angle between the trunk line (hip to shoulder) and vertical.

    Args:
        shoulder_mid: Midpoint of shoulders
        hip_mid: Midpoint of hips
        vertical: Vertical direction (default: WORLD_UP, the production frame)

    Returns:
        Trunk angle in degrees (0 = perfectly upright)
    """
    if vertical is None:
        vertical = WORLD_UP

    trunk_vector = shoulder_mid - hip_mid
    return angle_between_vectors(trunk_vector, vertical)


def point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> float:
    """
    Calculate perpendicular distance from a point to a line.

    Args:
        point: The point
        line_start: Start of the line
        line_end: End of the line

    Returns:
        Distance from point to line
    """
    line_vec = line_end - line_start
    point_vec = point - line_start

    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return float(np.linalg.norm(point_vec))

    line_unit = line_vec / line_len

    # Project point onto line
    proj_length = np.dot(point_vec, line_unit)
    proj_point = line_start + proj_length * line_unit

    return float(np.linalg.norm(point - proj_point))


def signed_distance_from_plane(
    point: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray
) -> float:
    """
    Calculate signed distance from a point to a plane.

    Positive distance means the point is on the side the normal points to.

    Args:
        point: The point
        plane_normal: Normal vector of the plane
        plane_point: A point on the plane

    Returns:
        Signed distance from point to plane
    """
    normal = normalize_vector(plane_normal)
    v = point - plane_point
    return float(np.dot(v, normal))


def knee_valgus_angle(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    sagittal_normal: np.ndarray = None
) -> float:
    """
    Calculate knee valgus/varus angle.

    Valgus (knock-kneed) is positive, varus (bow-legged) is negative.
    This measures how much the knee deviates medially/laterally from
    the hip-ankle line in the frontal plane.

    Args:
        hip: Hip position
        knee: Knee position
        ankle: Ankle position
        sagittal_normal: Normal to the sagittal plane (default: [1, 0, 0])

    Returns:
        Valgus angle in degrees (positive = valgus, negative = varus)
    """
    if sagittal_normal is None:
        sagittal_normal = np.array([1, 0, 0])

    # Project all points onto frontal plane
    frontal_hip = project_to_plane(hip, sagittal_normal, np.zeros(3))
    frontal_knee = project_to_plane(knee, sagittal_normal, np.zeros(3))
    frontal_ankle = project_to_plane(ankle, sagittal_normal, np.zeros(3))

    # Vector from hip to ankle (ideal knee path)
    ideal_vec = frontal_ankle - frontal_hip

    # Vector from hip to knee (actual position)
    actual_vec = frontal_knee - frontal_hip

    # Calculate signed angle between them
    # Use cross product to determine sign (valgus vs varus)
    cross = np.cross(
        np.append(ideal_vec[:2], 0),
        np.append(actual_vec[:2], 0)
    )

    angle = angle_between_vectors(ideal_vec, actual_vec)

    # Sign based on which side of ideal line the knee is
    if cross[2] < 0:
        angle = -angle

    return angle


def level_camera_frame(points: np.ndarray, pitch_deg: float) -> np.ndarray:
    """
    Rotate camera-frame points about the camera X axis so a camera pitched by
    pitch_deg (positive = lens tilted up, the sign of the standing-pitch
    estimate) reads true heights: the world's up maps back onto WORLD_UP.

    Levelling is not free: each levelled height mixes in the camera's depth
    axis at sin(pitch), so monocular depth noise and bias enter with it.
    """
    pitch = np.radians(pitch_deg)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_pitch, -sin_pitch],
        [0.0, sin_pitch, cos_pitch],
    ])
    return np.asarray(points, dtype=np.float64) @ rotation.T


def sagittal_forward(up: np.ndarray, lateral_right: np.ndarray) -> np.ndarray:
    """
    Unit forward axis of a lifter whose right-pointing lateral axis is given,
    perpendicular to the physically-up vector `up` (never negated). With
    up = WORLD_UP and lateral_right = (-1, 0, 0), forward is (0, 0, -1).
    """
    up_unit = normalize_vector(up)
    lateral = lateral_right - np.dot(lateral_right, up_unit) * up_unit
    return normalize_vector(np.cross(up_unit, lateral))


def lean_back_deg(vector: np.ndarray, up: np.ndarray, forward: np.ndarray) -> float:
    """
    Signed sagittal tilt of `vector` (e.g. hip -> shoulder) from `up`, in
    degrees: positive leaning back, negative leaning forward, 0 when parallel to up.
    """
    up_unit = normalize_vector(up)
    return float(np.degrees(np.arctan2(-np.dot(vector, forward), np.dot(vector, up_unit))))


def forward_offset(vector: np.ndarray, up: np.ndarray, forward: np.ndarray) -> float:
    """Horizontal (perpendicular to up) component of `vector` along forward; + = ahead."""
    up_unit = normalize_vector(up)
    horizontal = vector - np.dot(vector, up_unit) * up_unit
    return float(np.dot(horizontal, forward))
