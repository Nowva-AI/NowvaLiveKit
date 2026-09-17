"""
IPC Bridge for Voice Agent Communication

Translates pipeline events (faults, reps, frames) into throttled,
deduplicated JSON messages sent over the existing IPC socket.
Maintains backward compatibility with the legacy rep_count message format.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from biomechanics.coaching.cue_cache import CueCache
from biomechanics.config import CoachingConfig, IPCConfig
from biomechanics.diagnosis.bridge import (
    build_anthro_dict,
    build_rom_dict,
    compute_foot_direction_angle,
    compute_stance_width_ratio,
    mediapipe_to_viewer_coords,
)
from biomechanics.diagnosis.graph.parameter_deltas import dorsi_driven_targets
from biomechanics.diagnosis.types import DiagnosisResult, RepKinematicSummary, RepScore, SetScoreSummary
from biomechanics.utils.json_safe import nan_to_none
from biomechanics.utils.types import (
    DEPTH_CLASS_NAMES,
    FaultEvent,
    PipelineFrame,
    RepData,
    depth_category,
)

logger = logging.getLogger(__name__)


class IPCBridge:
    """
    Translates pipeline events into IPC messages for the voice agent.

    Handles cue caching, frame throttling, fault deduplication,
    rep completion with depth categorization, and set summaries.

    The ipc_client parameter is duck-typed — any object with a
    ``send_message(dict)`` method works (IPCClient or a mock).
    """

    def __init__(
        self,
        ipc_client: Any,
        coaching_config: Optional[CoachingConfig] = None,
        ipc_config: Optional[IPCConfig] = None,
    ):
        self.ipc_client = ipc_client
        ipc_config = ipc_config or IPCConfig()
        self.cue_cache = CueCache(coaching_config)
        self.last_fault_send: Dict[str, float] = defaultdict(float)
        self.fault_cooldown: float = ipc_config.fault_cooldown_seconds
        self.frame_send_interval: int = ipc_config.frame_send_interval
        self.frame_counter: int = 0
        self.shoulder_width_m: float = 0.0
        self._target_stance_ratio: float = 0.0
        self._target_toe_out_deg: float = 0.0

    # ------------------------------------------------------------------
    # Athlete parameters
    # ------------------------------------------------------------------

    def set_athlete_params(self, params: dict, baseline: dict | None = None) -> None:
        self.shoulder_width_m = params.get("shoulder_width_m", 0.0)
        anthro = build_anthro_dict(params)
        rom = build_rom_dict(params, baseline or {})
        dorsi = rom.get("peak_dorsiflexion", 35.0)
        self._target_stance_ratio, self._target_toe_out_deg = dorsi_driven_targets(
            dorsi, anthro,
        )

    # ------------------------------------------------------------------
    # Exercise preparation
    # ------------------------------------------------------------------

    def prepare_exercise(self, exercise_name: str) -> Dict[str, str]:
        """Cache cues for an exercise and notify the voice agent."""
        cues = self.cue_cache.prepare_for_exercise(exercise_name)
        self._send({
            "type": "cache_cues",
            "exercise_name": exercise_name,
            "cues": cues,
        })
        return cues

    # ------------------------------------------------------------------
    # Frame data (throttled)
    # ------------------------------------------------------------------

    def send_frame_data(self, frame: PipelineFrame, rep_phase: str = "") -> None:
        """Send frame data every N frames. Skips if no joint angles."""
        self.frame_counter += 1
        if self.frame_counter % self.frame_send_interval != 0:
            return
        if frame.joint_angles is None:
            return

        total = frame.total_latency_ms
        fps = round(1000.0 / total, 1) if total > 0 else 0.0

        msg: dict[str, Any] = {
            "type": "frame_data",
            "joint_angles": frame.joint_angles.as_dict(),
            "fps": fps,
            "frame_index": frame.frame_index,
            "rep_phase": rep_phase,
        }

        if frame.skeleton_3d is not None and self.shoulder_width_m > 0:
            try:
                # Same transform the diagnosis path applies before measuring:
                # compute_foot_direction_angle's forward axis is viewer coords.
                kpts = mediapipe_to_viewer_coords(frame.skeleton_3d.to_numpy())
                msg["stance_width_ratio"] = compute_stance_width_ratio(
                    kpts, self.shoulder_width_m,
                )
                msg["foot_direction_angle_l"] = compute_foot_direction_angle(
                    kpts, ankle_idx=15, foot_idx=17,
                )
                msg["foot_direction_angle_r"] = compute_foot_direction_angle(
                    kpts, ankle_idx=16, foot_idx=18,
                )
                if self._target_stance_ratio > 0:
                    msg["target_stance_ratio"] = self._target_stance_ratio
                    msg["target_toe_out_deg"] = self._target_toe_out_deg
            except Exception:
                logger.debug("Stance metric computation failed", exc_info=True)

        self._send(msg)

    # ------------------------------------------------------------------
    # Fault events (deduplicated per fault type)
    # ------------------------------------------------------------------

    def send_fault(self, fault: FaultEvent) -> None:
        """Send a fault message, rate-limited per fault type."""
        now = fault.timestamp or time.time()

        if now - self.last_fault_send[fault.fault_type] < self.fault_cooldown:
            return

        cue_key = self.cue_cache.get_cue_for_fault(fault.fault_type, now)

        self._send({
            "type": "fault",
            "fault_type": fault.fault_type,
            "severity": fault.severity.value,
            "severity_score": fault.severity_score,
            "message": fault.message,
            "cue": cue_key,
            "rep_number": fault.rep_number,
        })

        self.last_fault_send[fault.fault_type] = now

    # ------------------------------------------------------------------
    # Rep completion
    # ------------------------------------------------------------------

    def send_rep_complete(
        self,
        rep: RepData,
        bottom_kpts: Optional[List] = None,
        bottom_angles: Optional[Dict[str, float]] = None,
        standing_kpts: list | None = None,
        rep_kinematic_summary: RepKinematicSummary | None = None,
        set_number: int | None = None,
    ) -> None:
        """Send rep data and legacy rep_count.

        Note: rep count and positive reinforcement cues are now dispatched
        by the CoachingOrchestrator on the voice agent side based on the
        rep_complete message data (no more play_cue messages from here).
        """
        msg: Dict[str, Any] = {
            "type": "rep_complete",
            "rep_number": rep.rep_number,
            "max_depth_angle": round(rep.max_depth_angle, 1),
            "depth_category": self._depth_category(rep.max_depth_angle),
            # Deduplicated: a fault can be detected on several frames of the
            # same rep, and this list is used for membership tests and for
            # fault counts in the recap, where repeats inflate the totals.
            # faults_detailed below keeps every individual detection.
            "faults_in_rep": list(dict.fromkeys(f.fault_type for f in rep.faults)),
            "rep_duration_ms": round(rep.duration * 1000),
            "is_clean": rep.is_clean,
            "descent_time_s": round(rep.descent_time, 3),
            "ascent_time_s": round(rep.ascent_time, 3),
            "depth_class_int": rep.depth_class,
            "depth_class_name": rep.depth_class_name,
            "faults_detailed": [
                {
                    "fault_type": f.fault_type,
                    "severity": f.severity.value,
                    "severity_score": round(f.severity_score, 2),
                    "message": f.message,
                    "details": f.details,
                }
                for f in rep.faults
            ],
        }
        if bottom_kpts is not None:
            msg["bottom_kpts"] = bottom_kpts
        if bottom_angles is not None:
            msg["bottom_angles"] = bottom_angles
        if standing_kpts is not None:
            msg["standing_kpts"] = standing_kpts
        if rep_kinematic_summary is not None:
            msg["rep_kinematic_summary"] = rep_kinematic_summary.model_dump()
        if set_number is not None:
            msg["set_number"] = set_number
        self._send(msg)

        # Backward compatibility
        self._send({
            "type": "rep_count",
            "value": rep.rep_number,
        })

    def send_shallow_rep(
        self,
        depth_class: int,
        fault: Optional[FaultEvent] = None,
        set_number: int | None = None,
    ) -> None:
        """Send a rep that was rejected for insufficient depth.

        Deliberately not routed through send_fault: this replaces the rep
        callout the lifter would otherwise have heard, so it is not subject
        to the fault cooldown. Carries the depth fault's fields so the
        persistence layer can log it like any other depth cue.
        """
        msg: Dict[str, Any] = {
            "type": "shallow_rep",
            "depth_class": depth_class,
            "depth_class_name": DEPTH_CLASS_NAMES.get(depth_class, "Unknown"),
            "cue": "deeper",
        }
        if fault is not None:
            msg.update({
                "fault_type": fault.fault_type,
                "severity": fault.severity.value,
                "severity_score": round(fault.severity_score, 2),
                "message": fault.message,
                "rep_number": fault.rep_number,
                "max_knee_flexion": round(
                    fault.details.get("max_knee_flexion", 0.0), 1
                ),
            })
        if set_number is not None:
            msg["set_number"] = set_number
        self._send(msg)

    def send_rep_diagnosis(
        self,
        rep_number: int,
        diagnosis_result: DiagnosisResult,
        rep_score: RepScore | None = None,
    ) -> None:
        """Send per-rep rolling-window diagnosis during assessment mode."""
        msg: dict[str, Any] = {
            "type": "rep_diagnosis",
            "rep_number": rep_number,
            "diagnosis": {
                "confidence": diagnosis_result.confidence,
                "detected_symptoms": [
                    {
                        "symptom_id": s.symptom_id,
                        "severity": s.severity,
                        "contributing_reps": s.contributing_reps,
                    }
                    for s in diagnosis_result.detected_symptoms
                ],
                "immediate_causes": [
                    {
                        "cause_id": c.cause_id,
                        "score": c.score,
                        "explanation": c.explanation,
                        "parameter_delta": c.parameter_delta,
                        "implicated_by": c.implicated_by,
                    }
                    for c in diagnosis_result.immediate_causes
                ],
            },
        }
        if rep_score is not None:
            msg["rep_score"] = rep_score.model_dump()
        self._send(msg)

    # ------------------------------------------------------------------
    # Set completion
    # ------------------------------------------------------------------

    def send_diagnosis_complete(
        self,
        set_number: int,
        diagnosis_result: DiagnosisResult,
        score_summary: SetScoreSummary,
    ) -> None:
        """Send structured diagnosis and scoring results for a completed set."""
        per_rep = score_summary.per_rep_scores
        n = len(per_rep)
        self._send({
            "type": "diagnosis_complete",
            "set_number": set_number,
            "diagnosis": {
                "confidence": diagnosis_result.confidence,
                "detected_symptoms": [
                    {"symptom_id": s.symptom_id, "severity": s.severity, "contributing_reps": s.contributing_reps}
                    for s in diagnosis_result.detected_symptoms
                ],
                "immediate_causes": [
                    {"cause_id": c.cause_id, "score": c.score, "explanation": c.explanation, "parameter_delta": c.parameter_delta}
                    for c in diagnosis_result.immediate_causes
                ],
                "session_causes": [
                    {"cause_id": c.cause_id, "score": c.score, "explanation": c.explanation}
                    for c in diagnosis_result.session_causes
                ],
                "combined_perturbation": diagnosis_result.combined_perturbation,
            },
            "scoring": {
                "mean_score": score_summary.mean_score,
                "per_dimension": {
                    "depth": round(sum(r.depth_score for r in per_rep) / n, 3),
                    "trunk_control": round(sum(r.trunk_control_score for r in per_rep) / n, 3),
                    "knee_tracking": round(sum(r.knee_tracking_score for r in per_rep) / n, 3),
                    "symmetry": round(sum(r.symmetry_score for r in per_rep) / n, 3),
                    "tempo": round(sum(r.tempo_score for r in per_rep) / n, 3),
                },
                "best_rep": score_summary.best_rep_number,
                "worst_rep": score_summary.worst_rep_number,
                "trend_slope": score_summary.trend_slope,
                "per_rep_scores": [r.model_dump() for r in per_rep],
            },
        })

    # ------------------------------------------------------------------
    # Set completion
    # ------------------------------------------------------------------

    def send_set_complete(self, set_number: int, reps: List[RepData]) -> None:
        """Compute set summary stats and send."""
        if not reps:
            return

        depths = [r.max_depth_angle for r in reps]
        avg_depth = statistics.mean(depths)
        depth_consistency = statistics.stdev(depths) if len(depths) > 1 else 0.0

        # Build per-fault-type summary
        fault_summary: Dict[str, Dict[str, Any]] = {}
        for rep in reps:
            for fault in rep.faults:
                key = fault.fault_type
                if key not in fault_summary:
                    fault_summary[key] = {"count": 0, "total_severity": 0.0}
                fault_summary[key]["count"] += 1
                fault_summary[key]["total_severity"] += fault.severity_score

        for data in fault_summary.values():
            data["avg_severity"] = round(data["total_severity"] / data["count"], 2)
            data["total_severity"] = round(data["total_severity"], 2)

        self._send({
            "type": "set_complete",
            "set_number": set_number,
            "total_reps": len(reps),
            "avg_depth": round(avg_depth, 1),
            "depth_consistency": round(depth_consistency, 1),
            "clean_reps": sum(1 for r in reps if r.is_clean),
            "fault_summary": fault_summary,
        })

    # ------------------------------------------------------------------
    # Pipeline status
    # ------------------------------------------------------------------

    def send_pipeline_status(self, status: str, latency: Dict[str, float]) -> None:
        """Broadcast pipeline health status."""
        self._send({
            "type": "pipeline_status",
            "status": status,
            "latency_ms": latency,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send(self, message: Dict[str, Any]) -> None:
        """Every IPC message passes here: NaN/inf (missing angles) become None."""
        self.ipc_client.send_message(nan_to_none(message))

    @staticmethod
    def _depth_category(angle: float) -> str:
        """Categorize squat depth. Delegates to types.depth_category."""
        return depth_category(angle)
