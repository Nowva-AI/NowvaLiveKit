"""
Bone Length Constraint Enforcement

Calibrates segment lengths from the first N frames, then enforces
them as hard constraints. If a distal keypoint violates the calibrated
distance from its proximal joint, it is projected back onto the sphere
of allowed radius.

Operates on Skeleton3D objects, BEFORE the IK solver.

Bone pairs follow the COCO 17 keypoint ordering and are defined
as (proximal_index, distal_index) — the distal point gets corrected.
"""

import logging
from dataclasses import dataclass

import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from biomechanics.utils.types import Skeleton3D, CocoKeypoints as CK

if TYPE_CHECKING:
    from biomechanics.utils.standing_gate import StandingPoseGate

logger = logging.getLogger(__name__)

# Population-average reference values for proportion scaling.
REFERENCE_HIP_TO_FEMUR_RATIO = 0.556  # ~0.25 m hip width / ~0.45 m femur
REFERENCE_TIBIA_LENGTH_M = 0.45
REFERENCE_FEMUR_TO_TORSO = 0.90  # ~0.45 m femur / ~0.50 m torso


@dataclass
class BodyProportions:
    """Body proportion ratios derived from calibrated bone lengths.

    Used to scale fault-detection thresholds to the user's anatomy.
    """
    hip_width: float            # metres
    femur_length_avg: float     # average of L/R, metres
    tibia_length_avg: float     # average of L/R, metres
    torso_length_avg: float     # average of L/R shoulder-hip, metres

    # Derived ratios
    hip_to_femur_ratio: float
    tibia_to_reference_ratio: float

    # Pre-computed, clamped scale factors for rules
    valgus_scale: float         # multiply valgus thresholds by this
    forward_lean_scale: float   # multiply forward-lean thresholds by this
    pelvis_tilt_coupling: float # pelvis tilt as fraction of trunk flexion


# Ordered from proximal to distal so corrections cascade properly.
# Each tuple is (proximal_keypoint_index, distal_keypoint_index).
BONE_PAIRS: List[Tuple[int, int]] = [
    # Torso
    (CK.LEFT_SHOULDER, CK.LEFT_HIP),
    (CK.RIGHT_SHOULDER, CK.RIGHT_HIP),
    (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER),
    (CK.LEFT_HIP, CK.RIGHT_HIP),
    # Left leg
    (CK.LEFT_HIP, CK.LEFT_KNEE),
    (CK.LEFT_KNEE, CK.LEFT_ANKLE),
    # Right leg
    (CK.RIGHT_HIP, CK.RIGHT_KNEE),
    (CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
    # Left arm
    (CK.LEFT_SHOULDER, CK.LEFT_ELBOW),
    (CK.LEFT_ELBOW, CK.LEFT_WRIST),
    # Right arm
    (CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW),
    (CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
    # Feet (only calibrated/enforced when foot landmarks are visible)
    (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX),
    (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX),
    (CK.LEFT_ANKLE, CK.LEFT_HEEL),
    (CK.RIGHT_ANKLE, CK.RIGHT_HEEL),
]


def _pairs_present(num_keypoints: int) -> list[Tuple[int, int]]:
    # Heel pairs are absent from skeletons produced before heels were tracked.
    return [
        (proximal_idx, distal_idx)
        for proximal_idx, distal_idx in BONE_PAIRS
        if proximal_idx < num_keypoints and distal_idx < num_keypoints
    ]


class BoneLengthConstraints:
    """
    Calibrates and enforces fixed bone lengths on Skeleton3D.

    During calibration (first `calibration_frames` frames), bone lengths
    are measured each frame and the median is stored. After calibration,
    any keypoint that violates its bone length by more than `tolerance`
    percent is projected back to the correct distance.

    Args:
        calibration_frames: Number of frames to observe before locking
            bone lengths. Default 30 (~1 second at 30fps).
        tolerance: Fractional tolerance before correction kicks in (pre-calibration
            only; after calibration, lengths are always locked). Default 0.0.
    """

    def __init__(
        self,
        calibration_frames: int = 30,
        tolerance: float = 0.0,
        standing_gate: Optional["StandingPoseGate"] = None,
    ):
        self.calibration_frames = calibration_frames
        self.tolerance = tolerance
        self._standing_gate = standing_gate

        self._calibrated = False
        self._frame_count = 0

        # During calibration: bone_pair_key -> list of observed lengths
        self._length_observations: Dict[Tuple[int, int], List[float]] = {
            pair: [] for pair in BONE_PAIRS
        }

        # After calibration: bone_pair_key -> median length
        self._calibrated_lengths: Dict[Tuple[int, int], float] = {}

        # Computed after calibration from bone lengths
        self._body_proportions: Optional[BodyProportions] = None

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def progress(self) -> Tuple[int, int]:
        """Return (calibration_frames_collected, calibration_frames_required)."""
        return (min(self._frame_count, self.calibration_frames), self.calibration_frames)

    def enforce(self, skeleton: Skeleton3D) -> Skeleton3D:
        """
        Enforce bone length constraints.

        During calibration: records bone lengths, returns skeleton unchanged.
        After calibration: projects violating keypoints back to allowed radius.

        Args:
            skeleton: Current frame's Skeleton3D

        Returns:
            Skeleton3D with bone lengths enforced (or unchanged during calibration).
        """
        points = skeleton.to_numpy()  # (17, 3)
        confidences = np.array([kp.confidence for kp in skeleton.keypoints])

        if not self._calibrated:
            # Wait for the standing pose gate before recording calibration data.
            # Read-only: the pipeline advances the gate once per frame. Calling
            # check() here too would advance it once per enforce() call, and
            # enforce() runs twice a frame — required_consecutive_frames would
            # latch in a third of the frames it asks for, on skeletons at
            # different stages of the filter chain.
            if self._standing_gate is not None and not self._standing_gate.is_ready:
                return skeleton

            self._record_calibration(points, confidences)
            self._frame_count += 1

            if self._frame_count >= self.calibration_frames:
                self._finalize_calibration()

            return skeleton

        # --- Enforcement pass ---
        corrected = points.copy()

        for proximal_idx, distal_idx in _pairs_present(len(points)):
            # Skip pairs where either keypoint has insufficient confidence
            if confidences[proximal_idx] < 0.1 or confidences[distal_idx] < 0.1:
                continue

            p_prox = corrected[proximal_idx]
            p_dist = corrected[distal_idx]

            current_length = np.linalg.norm(p_dist - p_prox)
            target_length = self._calibrated_lengths.get(
                (proximal_idx, distal_idx)
            )

            if target_length is None or target_length < 1e-6:
                continue

            # After calibration, bone lengths are rigid (no drift allowed).
            length_error = abs(current_length - target_length)
            enforce_threshold = max(self.tolerance * target_length, 1e-6)

            if length_error > enforce_threshold:
                # Project distal point onto sphere of radius target_length
                # centered at proximal point
                if current_length < 1e-10:
                    # Degenerate: distal is on top of proximal.
                    # Push it in the +Y direction (upward) by target_length.
                    direction = np.array([0.0, 1.0, 0.0])
                else:
                    direction = (p_dist - p_prox) / current_length

                corrected[distal_idx] = p_prox + direction * target_length

        return Skeleton3D.from_numpy(
            corrected,
            confidences=confidences,
            timestamp=skeleton.timestamp,
            frame_index=skeleton.frame_index,
        )

    def _record_calibration(self, points: np.ndarray, confidences: np.ndarray) -> None:
        """Record bone lengths for one frame during calibration."""
        for proximal_idx, distal_idx in _pairs_present(len(points)):
            # Skip pairs where either keypoint has insufficient confidence
            if confidences[proximal_idx] < 0.1 or confidences[distal_idx] < 0.1:
                continue
            length = float(np.linalg.norm(
                points[distal_idx] - points[proximal_idx]
            ))
            if length > 1e-6:  # Skip degenerate frames
                self._length_observations[(proximal_idx, distal_idx)].append(length)

    def _finalize_calibration(self) -> None:
        """Compute median bone lengths, lock calibration, derive proportions."""
        for pair, lengths in self._length_observations.items():
            if lengths:
                self._calibrated_lengths[pair] = float(np.median(lengths))

        self._calibrated = True

        # Free observation memory
        self._length_observations.clear()

        # Derive body proportions from calibrated bone lengths
        self._compute_body_proportions()

    def _compute_body_proportions(self) -> None:
        """Derive body-proportion ratios from calibrated bone lengths."""
        hip_width = self._calibrated_lengths.get((CK.LEFT_HIP, CK.RIGHT_HIP), 0.25)

        femur_l = self._calibrated_lengths.get((CK.LEFT_HIP, CK.LEFT_KNEE), 0.45)
        femur_r = self._calibrated_lengths.get((CK.RIGHT_HIP, CK.RIGHT_KNEE), 0.45)
        femur_avg = (femur_l + femur_r) / 2.0

        tibia_l = self._calibrated_lengths.get((CK.LEFT_KNEE, CK.LEFT_ANKLE), 0.45)
        tibia_r = self._calibrated_lengths.get((CK.RIGHT_KNEE, CK.RIGHT_ANKLE), 0.45)
        tibia_avg = (tibia_l + tibia_r) / 2.0

        torso_l = self._calibrated_lengths.get((CK.LEFT_SHOULDER, CK.LEFT_HIP), 0.50)
        torso_r = self._calibrated_lengths.get((CK.RIGHT_SHOULDER, CK.RIGHT_HIP), 0.50)
        torso_avg = (torso_l + torso_r) / 2.0

        hip_to_femur = hip_width / max(femur_avg, 0.01)
        tibia_ratio = tibia_avg / REFERENCE_TIBIA_LENGTH_M

        # Wider hips → higher valgus thresholds (more lenient)
        valgus_scale = float(np.clip(
            hip_to_femur / REFERENCE_HIP_TO_FEMUR_RATIO, 0.7, 1.3,
        ))

        # Longer femurs relative to torso → more forward lean needed (more lenient)
        femur_to_torso = femur_avg / max(torso_avg, 0.01)
        forward_lean_scale = float(np.clip(
            femur_to_torso / REFERENCE_FEMUR_TO_TORSO, 0.8, 1.3,
        ))

        # Pelvis-trunk coupling: wider hips → more pelvis involvement
        hip_to_torso = hip_width / max(torso_avg, 0.01)
        pelvis_tilt_coupling = float(np.clip(
            0.35 + 0.15 * (hip_to_torso / 0.50), 0.30, 0.55,
        ))

        self._body_proportions = BodyProportions(
            hip_width=hip_width,
            femur_length_avg=femur_avg,
            tibia_length_avg=tibia_avg,
            torso_length_avg=torso_avg,
            hip_to_femur_ratio=hip_to_femur,
            tibia_to_reference_ratio=tibia_ratio,
            valgus_scale=valgus_scale,
            forward_lean_scale=forward_lean_scale,
            pelvis_tilt_coupling=pelvis_tilt_coupling,
        )

        logger.info(
            "[BONE CONSTRAINTS] Body proportions: hip_w=%.3fm, femur=%.3fm, "
            "tibia=%.3fm, torso=%.3fm, valgus_scale=%.2f, "
            "fwd_lean_scale=%.2f, pelvis_coupling=%.2f",
            hip_width, femur_avg, tibia_avg, torso_avg,
            valgus_scale, forward_lean_scale, pelvis_tilt_coupling,
        )

    @property
    def body_proportions(self) -> Optional[BodyProportions]:
        """Body proportions derived after calibration. None if not yet calibrated."""
        return self._body_proportions

    def reset(self):
        """Reset calibration state."""
        self._calibrated = False
        self._frame_count = 0
        self._calibrated_lengths.clear()
        self._body_proportions = None
        self._length_observations = {pair: [] for pair in BONE_PAIRS}
        if self._standing_gate is not None:
            self._standing_gate.reset()
