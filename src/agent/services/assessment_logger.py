"""
Assessment Data Persistence

Saves structured per-rep assessment data during the teaching phase.
Each rep captures the full closed-loop coaching cycle: faults detected,
recommendations given, whether corrections were applied, and Bayesian
cause analysis from a rolling-window diagnosis.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from biomechanics.utils.json_safe import nan_to_none


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

class CauseRecord(BaseModel):
    cause_id: str
    confidence: float
    explanation: str


class FaultRecord(BaseModel):
    fault_type: str
    severity: str
    severity_score: float
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    probable_causes: list[CauseRecord] = Field(default_factory=list)


class RecommendationRecord(BaseModel):
    fault_type: str
    recommendation: str
    source: str = "teaching_agent_llm"


class IncomingRecommendation(BaseModel):
    fault_type: str
    recommendation: str
    was_corrected: bool = False


class DepthRecord(BaseModel):
    max_knee_flexion_deg: float = 0.0
    hip_below_knee_plane: bool = False
    depth_class: str | None = None
    depth_class_int: int | None = None


class KinematicsRecord(BaseModel):
    trunk_pitch_at_bottom_deg: float = 0.0
    knee_valgus_l_deg: float = 0.0
    knee_valgus_r_deg: float = 0.0
    ankle_df_l_max_deg: float = 0.0
    ankle_df_r_max_deg: float = 0.0
    hip_y_at_bottom_cm: float = 0.0
    knee_y_at_bottom_cm: float = 0.0
    stance_width_ratio: float = 0.0


class TimingRecord(BaseModel):
    descent_s: float = 0.0
    ascent_s: float = 0.0
    total_s: float = 0.0
    tempo_ratio: float = 0.0


class RepRecord(BaseModel):
    rep_number: int
    timestamp: float
    is_clean: bool
    consecutive_clean_streak: int = 0
    depth: DepthRecord = Field(default_factory=DepthRecord)
    kinematics: KinematicsRecord = Field(default_factory=KinematicsRecord)
    timing: TimingRecord = Field(default_factory=TimingRecord)
    faults_detected: list[FaultRecord] = Field(default_factory=list)
    incoming_recommendations: list[IncomingRecommendation] = Field(default_factory=list)
    outgoing_recommendations: list[RecommendationRecord] = Field(default_factory=list)
    cues_played: list[str] = Field(default_factory=list)


class AssessmentLog(BaseModel):
    session_id: str
    exercise: str = "squat"
    target_consecutive_clean: int = 1
    user_height_cm: float = 0.0
    started_at: str = ""
    completed_at: str | None = None
    passed: bool = False
    total_reps_to_pass: int = 0
    best_rep_number: int | None = None
    reps: list[RepRecord] = Field(default_factory=list)


# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------

def _deduplicate_faults(faults: list[FaultRecord]) -> list[FaultRecord]:
    worst: dict[str, FaultRecord] = {}
    for f in faults:
        prev = worst.get(f.fault_type)
        if prev is None or f.severity_score > prev.severity_score:
            worst[f.fault_type] = f
    return list(worst.values())


class AssessmentLogger:

    def __init__(
        self,
        session_dir: Path,
        session_id: str,
        user_height_cm: float,
        target_reps: int = 1,
    ) -> None:
        self._assessment_dir = session_dir / "assessment"
        self._assessment_dir.mkdir(parents=True, exist_ok=True)
        # One session can run several assessments (the user loops back
        # through the flow) — never clobber a previous assessment's files.
        run_suffix = ""
        run_index = 2
        while (self._assessment_dir / f"assessment_log{run_suffix}.json").exists():
            run_suffix = f"_{run_index}"
            run_index += 1
        self._log_path = self._assessment_dir / f"assessment_log{run_suffix}.json"
        self._keypoints_dir = self._assessment_dir / f"keypoints{run_suffix}"
        self._keypoints_dir.mkdir(exist_ok=True)

        self._log = AssessmentLog(
            session_id=session_id,
            target_consecutive_clean=target_reps,
            user_height_cm=user_height_cm,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._consecutive_clean: int = 0
        self._best_composite_score: float = -1.0
        self._rep_sequence: int = 0
        # Pipeline rep numbers restart at 1 each assessment round; map the
        # most recent pipeline number to this logger's monotonic sequence.
        self._pipeline_rep_to_seq: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Per-rep ingestion
    # ------------------------------------------------------------------

    def on_rep_complete(self, msg: dict[str, Any]) -> None:
        """Ingest an enriched rep_complete IPC message."""
        self._rep_sequence += 1
        rep_number: int = self._rep_sequence
        self._pipeline_rep_to_seq[msg.get("rep_number", rep_number)] = rep_number
        is_clean: bool = msg.get("is_clean", False)

        if is_clean:
            self._consecutive_clean += 1
        else:
            self._consecutive_clean = 0

        kin_summary = msg.get("rep_kinematic_summary") or {}

        hip_y_avg = (kin_summary.get("hip_y_l_at_bottom", 0.0) + kin_summary.get("hip_y_r_at_bottom", 0.0)) / 2
        knee_y_avg = (kin_summary.get("knee_y_l_at_bottom", 0.0) + kin_summary.get("knee_y_r_at_bottom", 0.0)) / 2
        hip_below_knee = hip_y_avg < knee_y_avg if kin_summary else False

        descent_s = msg.get("descent_time_s", 0.0)
        ascent_s = msg.get("ascent_time_s", 0.0)
        total_s = descent_s + ascent_s
        tempo_ratio = descent_s / ascent_s if ascent_s > 0 else 0.0

        faults = _deduplicate_faults([
            FaultRecord(
                fault_type=f["fault_type"],
                severity=f["severity"],
                severity_score=f["severity_score"],
                message=f["message"],
                details=f.get("details", {}),
            )
            for f in msg.get("faults_detailed", [])
        ])

        current_fault_types = {f.fault_type for f in faults}
        incoming = self._build_incoming_recommendations(current_fault_types)

        rep = RepRecord(
            rep_number=rep_number,
            timestamp=time.time(),
            is_clean=is_clean,
            consecutive_clean_streak=self._consecutive_clean,
            depth=DepthRecord(
                max_knee_flexion_deg=msg.get("max_depth_angle", 0.0),
                hip_below_knee_plane=hip_below_knee,
                depth_class=msg.get("depth_class_name"),
                depth_class_int=msg.get("depth_class_int"),
            ),
            kinematics=KinematicsRecord(
                trunk_pitch_at_bottom_deg=kin_summary.get("trunk_pitch_at_bottom", 0.0),
                knee_valgus_l_deg=kin_summary.get("knee_valgus_l", 0.0),
                knee_valgus_r_deg=kin_summary.get("knee_valgus_r", 0.0),
                ankle_df_l_max_deg=kin_summary.get("ankle_df_l_max", 0.0),
                ankle_df_r_max_deg=kin_summary.get("ankle_df_r_max", 0.0),
                hip_y_at_bottom_cm=round(hip_y_avg, 1),
                knee_y_at_bottom_cm=round(knee_y_avg, 1),
                stance_width_ratio=kin_summary.get("stance_width_ratio", 0.0),
            ),
            timing=TimingRecord(
                descent_s=round(descent_s, 3),
                ascent_s=round(ascent_s, 3),
                total_s=round(total_s, 3),
                tempo_ratio=round(tempo_ratio, 2),
            ),
            faults_detected=faults,
            incoming_recommendations=incoming,
        )
        self._log.reps.append(rep)

        self._save_keypoints(rep_number, msg)
        self._write_incremental()

    def _build_incoming_recommendations(
        self, current_fault_types: set[str],
    ) -> list[IncomingRecommendation]:
        if not self._log.reps:
            return []
        prev = self._log.reps[-1]
        return [
            IncomingRecommendation(
                fault_type=rec.fault_type,
                recommendation=rec.recommendation,
                was_corrected=rec.fault_type not in current_fault_types,
            )
            for rec in prev.outgoing_recommendations
        ]

    # ------------------------------------------------------------------
    # Diagnosis attachment
    # ------------------------------------------------------------------

    def on_rep_diagnosis(
        self, rep_number: int, diagnosis: dict[str, Any], rep_score: dict[str, Any] | None = None,
    ) -> None:
        rep_number = self._pipeline_rep_to_seq.get(rep_number, rep_number)
        rep = self._find_rep(rep_number)
        if rep is None:
            return

        causes_by_symptom: dict[str, list[CauseRecord]] = {}
        for cause in diagnosis.get("immediate_causes", []):
            record = CauseRecord(
                cause_id=cause["cause_id"],
                confidence=cause["score"],
                explanation=cause["explanation"],
            )
            for symptom_id in cause.get("implicated_by", []):
                causes_by_symptom.setdefault(symptom_id, []).append(record)

        for fault in rep.faults_detected:
            matching = causes_by_symptom.get(fault.fault_type, [])
            fault.probable_causes = matching

        if rep_score is not None:
            composite = rep_score.get("composite_score", 0.0)
            self.update_best_rep(rep_number, composite)

        self._write_incremental()

    # ------------------------------------------------------------------
    # Teaching agent hooks
    # ------------------------------------------------------------------

    def set_outgoing_recommendations(
        self, rep_number: int, recommendations: list[RecommendationRecord],
    ) -> None:
        rep = self._find_rep(rep_number)
        if rep is not None:
            rep.outgoing_recommendations = recommendations
            self._write_incremental()

    def set_outgoing_recommendations_latest(
        self, recommendations: list[RecommendationRecord],
    ) -> None:
        if self._log.reps:
            self._log.reps[-1].outgoing_recommendations = recommendations
            self._write_incremental()

    def set_cues_played(self, rep_number: int, cues: list[str]) -> None:
        rep = self._find_rep(rep_number)
        if rep is not None:
            rep.cues_played = cues
            self._write_incremental()

    def set_cues_played_latest(self, cues: list[str]) -> None:
        if self._log.reps:
            self._log.reps[-1].cues_played = cues
            self._write_incremental()

    def update_best_rep(self, rep_number: int, composite_score: float) -> None:
        if composite_score > self._best_composite_score:
            self._best_composite_score = composite_score
            old_best = self._log.best_rep_number
            self._log.best_rep_number = rep_number

            if old_best is not None:
                self._remove_best_keypoints(old_best)
            self._copy_as_best_keypoints(rep_number)
            self._write_incremental()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self, passed: bool) -> None:
        self._log.passed = passed
        self._log.completed_at = datetime.now(timezone.utc).isoformat()
        self._log.total_reps_to_pass = len(self._log.reps) if passed else 0
        self._write_incremental()

    # ------------------------------------------------------------------
    # Keypoint persistence
    # ------------------------------------------------------------------

    def _save_keypoints(self, rep_number: int, msg: dict[str, Any]) -> None:
        bottom = msg.get("bottom_kpts")
        if bottom is not None:
            self._write_json(
                self._keypoints_dir / f"rep_{rep_number}_bottom.json", bottom,
            )
        standing = msg.get("standing_kpts")
        if standing is not None:
            self._write_json(
                self._keypoints_dir / f"rep_{rep_number}_standing.json", standing,
            )

    def _copy_as_best_keypoints(self, rep_number: int) -> None:
        for suffix in ("bottom", "standing"):
            src = self._keypoints_dir / f"rep_{rep_number}_{suffix}.json"
            dst = self._keypoints_dir / f"rep_{rep_number}_best_{suffix}.json"
            if src.exists():
                dst.write_text(src.read_text())

    def _remove_best_keypoints(self, rep_number: int) -> None:
        for suffix in ("bottom", "standing"):
            path = self._keypoints_dir / f"rep_{rep_number}_best_{suffix}.json"
            path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_rep(self, rep_number: int) -> RepRecord | None:
        for rep in self._log.reps:
            if rep.rep_number == rep_number:
                return rep
        return None

    def _write_incremental(self) -> None:
        self._write_json(self._log_path, self._log.model_dump())

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(nan_to_none(data), indent=2, default=str))
