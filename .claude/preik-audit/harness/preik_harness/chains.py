"""Adapters that run the pre-overhaul production pre-IK filter classes (byte-copies in ./legacy) as pluggable chains.

A chain is any object with process(skeleton3d, context) -> skeleton3d and reset(); optional calibrated()
is polled for calibration-completion timing. Production stages are constructed lazily on the first frame
because BoneLengthConstraints/GroundClamp need the pipeline's standing gate (context.standing_gate), exactly
as BiomechanicsPipeline wires them. Stage names: blend, vclamp, bone, ground, smooth. A repeated "bone" reuses
the same instance (the production second pass).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from biomechanics.utils.types import Skeleton3D

from .legacy.bone_constraints import BoneLengthConstraints
from .legacy.confidence_blend import ConfidenceBlender
from .legacy.ground_clamp import GroundClamp
from .legacy.position_filter import KeypointPositionSmoother
from .legacy.velocity_clamp import VelocityClamp

from .runner import FrameContext, pipeline_config

STAGE_NAMES = ("blend", "vclamp", "bone", "ground", "smooth")
# Parameters the pre-overhaul config carried for these stages (config.py defaults before wave 2 removed the
# sections; identical to the values used for baseline_results.json). Read from the live config when still present.
LEGACY_PARAMS = {
    "confidence_blend": {"min_confidence": 0.1, "max_confidence": 0.9},
    "velocity_clamp": {"max_velocity_m_per_s": 2.5},
    "pipeline": {"target_fps": 30},
    "bone_constraints": {"calibration_frames": 30, "tolerance": 0.0},
    "ground_clamp": {"calibration_frames": 30, "stance_width_tolerance_m": 0.02, "ankle_y_tolerance_m": 0.01,
                     "min_leg_extension_ratio": 0.75},
    "position_filter": {"min_cutoff": 0.8, "beta": 4.0, "d_cutoff": 1.0},
}


def _param(cfg: object, section: str, name: str) -> float:
    section_obj = getattr(cfg, section, None)
    return getattr(section_obj, name, LEGACY_PARAMS[section][name]) if section_obj is not None else \
        LEGACY_PARAMS[section][name]


class ProductionChain:
    def __init__(self, stages: list[str]):
        unknown = [s for s in stages if s not in STAGE_NAMES]
        if unknown:
            raise ValueError(f"unknown stages {unknown}; choose from {STAGE_NAMES}")
        self.stages = list(stages)
        self._objects: dict[str, object] | None = None

    def _build(self, context: FrameContext) -> dict[str, object]:
        cfg = pipeline_config()
        return {
            "blend": ConfidenceBlender(min_confidence=_param(cfg, "confidence_blend", "min_confidence"),
                                       max_confidence=_param(cfg, "confidence_blend", "max_confidence")),
            "vclamp": VelocityClamp(max_velocity_m_per_s=_param(cfg, "velocity_clamp", "max_velocity_m_per_s"),
                                    target_fps=_param(cfg, "pipeline", "target_fps")),
            "bone": BoneLengthConstraints(calibration_frames=_param(cfg, "bone_constraints", "calibration_frames"),
                                          tolerance=_param(cfg, "bone_constraints", "tolerance"),
                                          standing_gate=context.standing_gate),
            "ground": GroundClamp(calibration_frames=_param(cfg, "ground_clamp", "calibration_frames"),
                                  stance_width_tolerance_m=_param(cfg, "ground_clamp", "stance_width_tolerance_m"),
                                  ankle_y_tolerance_m=_param(cfg, "ground_clamp", "ankle_y_tolerance_m"),
                                  min_leg_extension_ratio=_param(cfg, "ground_clamp", "min_leg_extension_ratio"),
                                  standing_gate=context.standing_gate),
            "smooth": KeypointPositionSmoother(min_cutoff=_param(cfg, "position_filter", "min_cutoff"),
                                               beta=_param(cfg, "position_filter", "beta"),
                                               d_cutoff=_param(cfg, "position_filter", "d_cutoff")),
        }

    def process(self, skeleton: Skeleton3D, context: FrameContext) -> Skeleton3D:
        if self._objects is None:
            self._objects = self._build(context)
        for stage in self.stages:
            obj = self._objects[stage]
            if stage == "blend":
                skeleton = obj.blend(skeleton)
            elif stage == "vclamp":
                skeleton = obj.clamp(skeleton)
            elif stage == "bone":
                skeleton = obj.enforce(skeleton)
            elif stage == "ground":
                skeleton = obj.clamp(skeleton)
            else:
                skeleton = obj.smooth(skeleton)
        return skeleton

    def reset(self) -> None:
        # mirrors BiomechanicsPipeline.reset_readiness_gate (bone reset also resets the standing gate)
        if self._objects is None:
            return
        for obj in self._objects.values():
            obj.reset()

    def calibrated(self) -> bool | None:
        needs = [s for s in ("bone", "ground") if s in self.stages]
        if not needs:
            return None
        if self._objects is None:
            return False
        return all(self._objects[s].is_calibrated for s in needs)


def production_chain(stages: list[str]) -> Callable[[], ProductionChain]:
    def factory() -> ProductionChain:
        return ProductionChain(stages)
    factory.__name__ = "production_chain_" + "_".join(stages or ["raw"])
    return factory


BASELINE_CHAINS: dict[str, list[str]] = {
    "raw": [],
    "current": ["blend", "vclamp"],
    "full_original": ["blend", "vclamp", "bone", "ground", "smooth", "bone"],
    "blend_only": ["blend"],
    "vclamp_only": ["vclamp"],
    "bone_only": ["bone"],
    "ground_only": ["ground"],
    "smooth_only": ["smooth"],
}


def baseline_factories() -> dict[str, Callable[[], ProductionChain]]:
    return {name: production_chain(stages) for name, stages in BASELINE_CHAINS.items()}


class DiagnosticConfidenceFloor:
    """NOT a proposed filter: leaves positions raw but floors non-zero confidences at 0.11 so AnalyticalIKSolver /
    TriangulatedValgusEstimator never drop a joint (they return 0.0 below 0.1). Quantifies how much error comes from
    the triangulator's conf*0.1 (reprojection >= ~12.6 px) interacting with IK min_confidence."""

    def reset(self) -> None:
        return None

    def process(self, skeleton: Skeleton3D, context: FrameContext) -> Skeleton3D:
        points = skeleton.to_numpy()
        conf = np.array([kp.confidence for kp in skeleton.keypoints])
        conf = np.where(conf > 0, np.maximum(conf, 0.11), 0.0)
        return Skeleton3D.from_numpy(points, confidences=conf, timestamp=skeleton.timestamp,
                                     frame_index=skeleton.frame_index)
