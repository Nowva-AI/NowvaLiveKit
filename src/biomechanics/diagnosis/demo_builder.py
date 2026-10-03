"""Builds choreographed-demo data (corrected pose stack + cue metadata) from a diagnosis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from .graph.parameter_deltas import (
    magnitude_center_weight,
    magnitude_knees_out,
    magnitude_widen_foot_angle,
    magnitude_widen_stance,
)
from .keypoint_corrector import (
    KeypointCorrector,
    HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R,
)
from .pose_validation import validate_pose
from .types import DiagnosisResult, HypothesizedCause

logger = logging.getLogger(__name__)

CORRECTOR_CAUSE_ORDER: tuple[str, ...] = (
    "narrow_stance",
    "narrow_foot_angle",
    "knee_track_cue",
    "weight_shift_cue",
    "depth_cue_unfamiliar",
)

GENERIC_MAGNITUDE_TEXT = "a small adjustment"


class DemoCue(BaseModel):
    cue_index: int
    cause_id: str
    explanation: str
    magnitude_text: str


@dataclass
class DemoData:
    pose_stack: np.ndarray
    cues: list[DemoCue]


# Mirrors the cause_id -> parameter_delta_fn pairing in graph/causes.yaml
_MAGNITUDE_FN_BY_CAUSE = {
    "narrow_stance": magnitude_widen_stance,
    "narrow_foot_angle": magnitude_widen_foot_angle,
    "stance_toe_mismatch": magnitude_widen_foot_angle,
    "knee_track_cue": magnitude_knees_out,
    "weight_shift_cue": magnitude_center_weight,
}


def order_demo_causes(diagnosis: DiagnosisResult) -> list[HypothesizedCause]:
    by_id = {cause.cause_id: cause for cause in diagnosis.immediate_causes}
    return [by_id[cause_id] for cause_id in CORRECTOR_CAUSE_ORDER if cause_id in by_id]


def summarize_cue_magnitude(
    cause_id: str,
    parameter_delta: dict | None,
    actual_shift_m: float | None = None,
) -> str:
    if cause_id == "depth_cue_unfamiliar":
        return "hips down to parallel depth"
    if cause_id == "weight_shift_cue" and actual_shift_m is not None:
        # Speak the shift that was actually rendered, not the engine estimate.
        text = magnitude_center_weight({"pelvis.tx": actual_shift_m})
        return text if text is not None else GENERIC_MAGNITUDE_TEXT
    if parameter_delta is None:
        return GENERIC_MAGNITUDE_TEXT
    magnitude_fn = _MAGNITUDE_FN_BY_CAUSE.get(cause_id)
    if magnitude_fn is None:
        return GENERIC_MAGNITUDE_TEXT
    text = magnitude_fn(parameter_delta)
    return text if text is not None else GENERIC_MAGNITUDE_TEXT


def _lateral_hip_shift(before: np.ndarray, after: np.ndarray) -> float:
    """Hip-midpoint displacement projected on the L->R hip axis, in meters."""
    lat = np.array(before[HIP_R], dtype=np.float64) - np.array(before[HIP_L], dtype=np.float64)
    lat[1] = 0.0
    lat_len = np.linalg.norm(lat)
    if lat_len < 1e-9:
        return 0.0
    lat /= lat_len
    mid_delta = (
        (after[HIP_L] + after[HIP_R]) - (before[HIP_L] + before[HIP_R])
    ) / 2.0
    return float(np.dot(mid_delta, lat))


def _prepare_observed_kpts(kpts: list[list[float]]) -> list[list[float]]:
    """Ground and center — match visualize_video_squats preprocessing."""
    arr = np.array(kpts, dtype=np.float64)
    # Ground lowest foot keypoint to Y=0, keeping real foot geometry.
    # Index 15 onward is every foot keypoint — ankles, toes, and heels — so
    # slicing stays correct for both 19- and 21-keypoint poses.
    foot_y = arr[15:, 1].min()
    arr[:, 1] -= foot_y
    # Center hips at origin
    hip_mid_x = (arr[11, 0] + arr[12, 0]) / 2
    hip_mid_z = (arr[11, 2] + arr[12, 2]) / 2
    arr[:, 0] -= hip_mid_x
    arr[:, 2] -= hip_mid_z
    return arr.tolist()


def build_pose_stack(
    observed_kpts: list[list[float]],
    diagnosis: DiagnosisResult,
    anthro: dict | None = None,
    rom: dict | None = None,
) -> np.ndarray | None:
    """Prefix-corrected poses: stack[k] applies the first k causes to the observed pose.

    The base pose (stack[0]) is the CANONICAL (symmetrized) form of the
    observation, and bone lengths are captured from it — after symmetry, never
    before. Symmetry averages L/R bone vectors and is the only pass allowed to
    change segment lengths, so it must run once, up front, on the base pose.
    Every stack entry is then computed from that same canonical pose with the
    same locked lengths: bone lengths are identical across the whole stack and
    the choreographer's lerps between adjacent poses can never stretch a bone
    endpoint-to-endpoint. The last entry is by construction the all-causes
    correction, identical to VVS --diagnose.
    """
    ordered = order_demo_causes(diagnosis)
    if not ordered:
        return None

    corrector = KeypointCorrector()

    prepared = _prepare_observed_kpts(observed_kpts)
    canonical = corrector.canonicalize(prepared, rom=rom)

    canon_arr = np.asarray(canonical, dtype=np.float64)
    bone_lengths = (
        float(np.linalg.norm(canon_arr[KNEE_L] - canon_arr[HIP_L])),
        float(np.linalg.norm(canon_arr[ANKLE_L] - canon_arr[KNEE_L])),
        float(np.linalg.norm(canon_arr[KNEE_R] - canon_arr[HIP_R])),
        float(np.linalg.norm(canon_arr[ANKLE_R] - canon_arr[KNEE_R])),
    )

    pose_stack = np.empty(
        (len(ordered) + 1, canon_arr.shape[0], 3), dtype=np.float32,
    )
    pose_stack[0] = np.asarray(canonical, dtype=np.float32)

    for k in range(1, len(ordered) + 1):
        prefix = diagnosis.model_copy(
            update={"immediate_causes": ordered[:k]}
        )
        corrected = corrector.correct(
            canonical, prefix, anthro=anthro, rom=rom, bone_lengths=bone_lengths,
        )
        if corrected is None:
            return None
        pose_stack[k] = np.asarray(corrected, dtype=np.float32)

    # Hard gate: these poses are shown to the athlete as a movement target, so
    # an anatomically impossible one is dropped rather than rendered. Bail on
    # the whole stack — a partial stack would leave the narration describing
    # corrections the athlete never sees.
    for index in range(len(pose_stack)):
        validation = validate_pose(pose_stack[index].astype(np.float64))
        if not validation.is_valid:
            label = "observed" if index == 0 else ordered[index - 1].cause_id
            logger.warning(
                "[DEMO BUILDER] Rejected pose stack — %s pose is not "
                "anatomically valid: %s",
                label, "; ".join(validation.violations),
            )
            return None

    return pose_stack


def build_demo_data(
    observed_kpts: list[list[float]],
    diagnosis: DiagnosisResult,
    anthro: dict | None = None,
    rom: dict | None = None,
) -> DemoData | None:
    pose_stack = build_pose_stack(observed_kpts, diagnosis, anthro=anthro, rom=rom)
    if pose_stack is None:
        return None

    ordered = order_demo_causes(diagnosis)
    cues = []
    for index, cause in enumerate(ordered):
        actual_shift_m = None
        if cause.cause_id == "weight_shift_cue":
            actual_shift_m = _lateral_hip_shift(
                pose_stack[index].astype(np.float64),
                pose_stack[index + 1].astype(np.float64),
            )
        cues.append(
            DemoCue(
                cue_index=index,
                cause_id=cause.cause_id,
                explanation=cause.explanation,
                magnitude_text=summarize_cue_magnitude(
                    cause.cause_id, cause.parameter_delta,
                    actual_shift_m=actual_shift_m,
                ),
            )
        )
    return DemoData(pose_stack=pose_stack, cues=cues)
