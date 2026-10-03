"""
Session Tracker for Set Boundary Detection

Monitors the timing gap between reps to detect when a set ends
(pause > set_timeout_seconds). Accumulates per-set and per-session
statistics and triggers set-complete messages through the IPCBridge.
"""

import statistics
import time
from typing import Any, Dict, List, Optional

from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.config import CoachingConfig
from biomechanics.deadlift.rule_base import TIER_RANK
from biomechanics.diagnosis.bridge import (
    build_anthro_dict,
    build_frame_from_live_pipeline,
    build_rep_kinematic_summary,
    build_rep_trajectory,
    build_rom_dict,
    mediapipe_to_viewer_coords,
)
from biomechanics.diagnosis.demo_builder import DemoData, build_demo_data
from biomechanics.diagnosis.engine import HypothesisEngine
from biomechanics.diagnosis.rep_scoring import score_rep, score_set
from biomechanics.diagnosis.types import (
    DiagnosisResult,
    RepKinematicSummary,
    RepTrajectory,
    SetFeatures,
)
from biomechanics.faults.observability import capture_mode_from_env
from biomechanics.utils.types import RepData


ASSESSMENT_DIAGNOSIS_WINDOW = 3


class SessionTracker:
    """
    Tracks set and session boundaries using rep timing.

    A set boundary is detected when the pause between consecutive reps
    exceeds ``set_timeout_seconds``. When a set ends, a summary is
    sent via the IPCBridge.
    """

    def __init__(
        self,
        ipc_bridge: IPCBridge,
        config: Optional[CoachingConfig] = None,
    ):
        config = config or CoachingConfig()
        self.ipc_bridge = ipc_bridge
        self.set_timeout: float = config.set_timeout_seconds

        # Current set state
        self.current_set_number: int = 0
        self.current_set_reps: List[RepData] = []
        self.last_rep_time: float = 0.0
        self.set_active: bool = False

        # Session-level accumulators
        self.total_reps: int = 0
        self.total_sets: int = 0
        self.all_reps: List[RepData] = []

        # Last completed set summary (populated in _end_current_set)
        self.last_set_summary: Optional[Dict[str, Any]] = None

        # Diagnosis integration (populated via set_athlete_params)
        self._rep_kinematic_buffer: list[RepKinematicSummary] = []
        self._rep_trajectory_buffer: list[RepTrajectory | None] = []
        self._bottom_frame_buffer: list[tuple[int, list]] = []
        self._athlete_params: dict | None = None
        self._baseline: dict | None = None
        # The diagnosis engine models squats only; off for any other exercise.
        self.diagnosis_enabled: bool = True
        # An exercise's own set diagnosis (the deadlift's graph), installed by
        # the pipeline process with the profile; None on the squat path.
        self.set_diagnosis = None
        self._set_best_velocity_mps: float = -1.0

        # Assessment mode: per-rep rolling-window diagnosis
        self._assessment_mode: bool = False

        # Best composite rep score this set, for "best rep so far" highlights.
        self._set_best_score: float = -1.0
        self.capture_mode: str = capture_mode_from_env()

        # Last-rep snapshot for on-demand replay
        self._last_rep_bottom_kpts: list | None = None
        self._last_rep_standing_kpts: list | None = None
        self._last_rep_kinematic_summary: RepKinematicSummary | None = None
        self._last_rep_trajectory: RepTrajectory | None = None
        self._last_rep_data: RepData | None = None
        self._last_set_diagnosis: DiagnosisResult | None = None

    def set_athlete_params(self, athlete_params: dict, baseline: dict) -> None:
        self._athlete_params = athlete_params
        self._baseline = baseline

    def set_assessment_mode(self, enabled: bool) -> None:
        self._assessment_mode = enabled

    @property
    def _diagnosing(self) -> bool:
        return self.diagnosis_enabled and self._athlete_params is not None

    def on_exercise_changed(self, diagnosis_enabled: bool) -> None:
        """A new exercise starts: nothing from the previous one may be replayed
        or diagnosed, diagnosis runs only if the engine models the new one, and
        set numbers restart at 1 as they do in the workout plan."""
        if self.set_active:
            # Skipped mid-set: the partial set belongs to the old exercise.
            self._end_current_set()
        self.diagnosis_enabled = diagnosis_enabled
        self.reset_rep_buffers()
        self._last_set_diagnosis = None
        self.current_set_number = 0

    # ------------------------------------------------------------------
    # Rep handling
    # ------------------------------------------------------------------

    def on_rep_complete(
        self,
        rep: RepData,
        bottom_kpts: Optional[List] = None,
        bottom_angles: Optional[Dict[str, float]] = None,
        standing_kpts: Optional[List] = None,
        trajectory_samples: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """
        Process a completed rep. Detects set boundaries and forwards
        the rep to the IPCBridge.
        """
        now = rep.end_time

        # Check if the previous set timed out
        if self.set_active and self.last_rep_time > 0 and now - self.last_rep_time > self.set_timeout:
            self._end_current_set()

        # Start a new set if needed
        if not self.set_active:
            self.current_set_number += 1
            self.current_set_reps = []
            self.set_active = True
            self._set_best_score = -1.0
            self._set_best_velocity_mps = -1.0

        self.current_set_reps.append(rep)

        # Compute kinematics before IPC send so the summary can be included.
        # Guarded: a degenerate bottom frame must not block the rep_complete send.
        summary: RepKinematicSummary | None = None
        trajectory: RepTrajectory | None = None
        if self._diagnosing and bottom_kpts is not None and bottom_angles is not None:
            try:
                frame = build_frame_from_live_pipeline(
                    bottom_kpts, bottom_angles, standing_kpts=standing_kpts,
                )
                summary = build_rep_kinematic_summary(
                    frame,
                    self._athlete_params,
                    rep.rep_number,
                    descent_time_s=rep.descent_time,
                    ascent_time_s=rep.ascent_time,
                    features=rep.features,
                )
                trajectory = build_rep_trajectory(trajectory_samples)
                self._rep_kinematic_buffer.append(summary)
                self._rep_trajectory_buffer.append(trajectory)
                self._bottom_frame_buffer.append((rep.rep_number, frame["kpts"]))
            except Exception as e:
                summary = None
                trajectory = None
                print(f"[SESSION TRACKER] Rep kinematics failed for rep {rep.rep_number}: {e}")

        self.ipc_bridge.send_rep_complete(
            rep,
            bottom_kpts=bottom_kpts,
            bottom_angles=bottom_angles,
            standing_kpts=standing_kpts,
            rep_kinematic_summary=summary,
            set_number=self.current_set_number,
            highlights=self._rep_highlights(rep, summary, trajectory),
        )

        # Store last-rep snapshot for on-demand replay
        self._last_rep_bottom_kpts = bottom_kpts
        self._last_rep_standing_kpts = standing_kpts
        self._last_rep_kinematic_summary = summary
        self._last_rep_trajectory = trajectory
        self._last_rep_data = rep

        if self._assessment_mode:
            self._run_assessment_diagnosis(rep.rep_number)
        if self.set_diagnosis is not None and rep.features:
            rolling = self.set_diagnosis.on_rep(rep.features)
            if rolling is not None:
                self.ipc_bridge.send_rep_diagnosis(rep.rep_number, rolling)

        self.last_rep_time = now
        self.total_reps += 1
        self.all_reps.append(rep)

    def _rep_highlights(
        self,
        rep: RepData,
        summary: RepKinematicSummary | None,
        trajectory: RepTrajectory | None,
    ) -> list[str]:
        """What went right this rep, for positive reinforcement."""
        if rep.features.get("dl_schema") is not None:
            return self._deadlift_highlights(rep)
        highlights: list[str] = []
        if rep.depth_target_met:
            highlights.append("depth_target_met")
        if rep.is_clean:
            highlights.append("clean")
        if summary is not None and self._athlete_params is not None:
            anthro = build_anthro_dict(self._athlete_params)
            rom = build_rom_dict(self._athlete_params, self._baseline or {})
            score = score_rep(summary, anthro, rom, trajectory).composite_score
            if score > self._set_best_score:
                if len(self.current_set_reps) > 1:
                    highlights.append("best_rep_so_far")
                self._set_best_score = score
        return highlights

    def _deadlift_highlights(self, rep: RepData) -> list[str]:
        """No depth to praise: a clean rep, and the fastest rep of the set with
        nothing the coach would correct (no fault at or above its min tier)."""
        highlights: list[str] = []
        if rep.is_clean:
            highlights.append("clean")
        correctable = any(
            TIER_RANK.get(fault.severity.value, 0) >= TIER_RANK.get(fault.details.get("min_tier", "mild"), 1)
            for fault in rep.faults
        )
        velocity = rep.features.get("concentric_velocity_mps")
        if not correctable and isinstance(velocity, float) and velocity == velocity:
            if velocity > self._set_best_velocity_mps:
                if self._set_best_velocity_mps >= 0.0:
                    highlights.append("best_rep_so_far")
                self._set_best_velocity_mps = velocity
        return highlights

    def _run_assessment_diagnosis(self, rep_number: int) -> None:
        if not self._rep_kinematic_buffer or self._athlete_params is None:
            return
        window = self._rep_kinematic_buffer[-ASSESSMENT_DIAGNOSIS_WINDOW:]
        anthro = build_anthro_dict(self._athlete_params)
        rom = build_rom_dict(self._athlete_params, self._baseline or {})
        set_features = SetFeatures(
            user_id=0,
            set_id=f"assessment_rep_{rep_number}",
            rep_count=len(window),
            per_rep_kinematics=list(window),
            anthropometry=anthro,
            rom=rom,
            capture_mode=self.capture_mode,
        )
        diagnosis_result = HypothesisEngine().diagnose(set_features)
        latest_kin = self._rep_kinematic_buffer[-1]
        rep_score = score_rep(latest_kin, anthro, rom, self._last_rep_trajectory)
        self.ipc_bridge.send_rep_diagnosis(rep_number, diagnosis_result, rep_score=rep_score)

    # ------------------------------------------------------------------
    # Set timeout polling
    # ------------------------------------------------------------------

    def check_set_timeout(self, current_time: Optional[float] = None) -> bool:
        """
        Check if the current set has timed out. Call this periodically
        (e.g. every frame) to detect set boundaries between reps.

        Returns:
            True if a set was ended, False otherwise.
        """
        now = current_time if current_time is not None else time.time()

        if self.set_active and self.last_rep_time > 0 and now - self.last_rep_time > self.set_timeout:
            self._end_current_set()
            return True

        return False

    # ------------------------------------------------------------------
    # Set lifecycle
    # ------------------------------------------------------------------

    def _end_current_set(self) -> None:
        """Finalize the current set and send summary via IPC."""
        if self.current_set_reps:
            self.last_set_summary = self._compute_set_summary(
                self.current_set_number, self.current_set_reps
            )
            self.ipc_bridge.send_set_complete(
                self.current_set_number, self.current_set_reps
            )
            self.total_sets += 1

        if self.set_diagnosis is not None and self.current_set_reps:
            outcome = self.set_diagnosis.finish_set(f"set_{self.current_set_number}")
            if outcome is not None:
                diagnosis_result, score_summary = outcome
                self.ipc_bridge.send_diagnosis_complete(
                    self.current_set_number, diagnosis_result, score_summary,
                )
                self._last_set_diagnosis = diagnosis_result
            self.set_diagnosis.reset()

        if self._rep_kinematic_buffer and self._diagnosing:
            anthro = build_anthro_dict(self._athlete_params)
            rom = build_rom_dict(self._athlete_params, self._baseline or {})
            set_features = SetFeatures(
                user_id=0,
                set_id="live",
                rep_count=len(self._rep_kinematic_buffer),
                per_rep_kinematics=list(self._rep_kinematic_buffer),
                anthropometry=anthro,
                rom=rom,
                capture_mode=self.capture_mode,
            )
            diagnosis_result = HypothesisEngine().diagnose(set_features)
            score_summary = score_set(
                self._rep_kinematic_buffer, anthro, rom, self._rep_trajectory_buffer,
            )
            self.ipc_bridge.send_diagnosis_complete(
                self.current_set_number, diagnosis_result, score_summary,
            )
            # Outlives the buffer below: "show me that" almost always comes
            # during rest, after the recap, when the buffer is already gone.
            self._last_set_diagnosis = diagnosis_result

        self._rep_kinematic_buffer = []
        self._rep_trajectory_buffer = []
        self._bottom_frame_buffer = []
        self.current_set_reps = []
        self.set_active = False

    def reset_rep_buffers(self) -> None:
        """Clear per-rep diagnosis buffers and the last-rep snapshot.

        Called at phase boundaries (assessment rounds, assessment→calibration),
        where the previous rep must stop counting as "your last rep" — otherwise
        an assessment rep is replayed minutes later during the workout.
        """
        self._rep_kinematic_buffer = []
        self._rep_trajectory_buffer = []
        self._bottom_frame_buffer = []
        self._last_rep_bottom_kpts = None
        self._last_rep_standing_kpts = None
        self._last_rep_kinematic_summary = None
        self._last_rep_trajectory = None
        self._last_rep_data = None

    def bottom_frame_for_rep(self, rep_number: int) -> list | None:
        """Viewer-coords bottom-frame kpts for a rep, or the latest buffered frame."""
        for buffered_rep_number, kpts in self._bottom_frame_buffer:
            if buffered_rep_number == rep_number:
                return kpts
        if self._bottom_frame_buffer:
            return self._bottom_frame_buffer[-1][1]
        return None

    def get_last_rep_snapshot(self) -> dict | None:
        if self._last_rep_bottom_kpts is None:
            return None
        snapshot: dict = {
            "bottom_kpts": self._last_rep_bottom_kpts,
            "standing_kpts": self._last_rep_standing_kpts,
        }
        if self._last_rep_kinematic_summary is not None:
            snapshot["kinematic_summary"] = (
                self._last_rep_kinematic_summary.model_dump()
                if hasattr(self._last_rep_kinematic_summary, "model_dump")
                else self._last_rep_kinematic_summary
            )
            # RepKinematicSummary carries no score — the replay overlay needs
            # one, so score the rep here rather than reading a key that
            # never exists.
            if self._athlete_params is not None:
                anthro = build_anthro_dict(self._athlete_params)
                rom = build_rom_dict(self._athlete_params, self._baseline or {})
                score = score_rep(
                    self._last_rep_kinematic_summary, anthro, rom,
                    self._last_rep_trajectory,
                )
                snapshot["rep_score"] = round(score.composite_score * 100, 1)
        if self._last_rep_data is not None:
            snapshot["rep_data"] = (
                self._last_rep_data.model_dump()
                if hasattr(self._last_rep_data, "model_dump")
                else self._last_rep_data
            )
        return snapshot

    def build_on_demand_demo(self) -> DemoData | None:
        """Build a choreographed demo from the current kinematic buffer.

        Runs the diagnosis engine on accumulated rep data and builds
        a corrected pose stack from the last rep's bottom keypoints.
        Returns None if insufficient data.
        """
        if not self._diagnosing or self._last_rep_bottom_kpts is None:
            return None

        anthro = build_anthro_dict(self._athlete_params)
        rom = build_rom_dict(self._athlete_params, self._baseline or {})

        if self._rep_kinematic_buffer:
            set_features = SetFeatures(
                user_id=0,
                set_id="on_demand_demo",
                rep_count=len(self._rep_kinematic_buffer),
                per_rep_kinematics=list(self._rep_kinematic_buffer),
                anthropometry=anthro,
                rom=rom,
                capture_mode=self.capture_mode,
            )
            diagnosis = HypothesisEngine().diagnose(set_features)
        else:
            # Mid-rest: the buffer was cleared at set end, so reuse the
            # diagnosis the athlete was just given feedback on.
            diagnosis = self._last_set_diagnosis

        if diagnosis is None or not diagnosis.immediate_causes:
            return None

        observed = self._last_rep_bottom_kpts
        if hasattr(observed, "tolist"):
            observed = observed.tolist()
        # _last_rep_bottom_kpts is stored raw from the pipeline (MediaPipe
        # world, Y-down). build_demo_data validates shank tilt against a Y-up
        # frame, so an untransformed pose is rejected every time.
        observed = mediapipe_to_viewer_coords(observed)
        return build_demo_data(observed, diagnosis, anthro=anthro, rom=rom)

    def _compute_set_summary(
        self, set_number: int, reps: List[RepData]
    ) -> Dict[str, Any]:
        """Compute summary stats for a completed set."""
        depths = [r.max_depth_angle for r in reps]
        if len(depths) > 1 and not any(depth == depth for depth in depths):
            # No depth at all (a deadlift set), where statistics.stdev would fail;
            # one rep takes the unchanged path below.
            avg_depth = depth_consistency = float("nan")
        else:
            avg_depth = statistics.mean(depths) if depths else 0.0
            depth_consistency = statistics.stdev(depths) if len(depths) > 1 else 0.0

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

        return {
            "set_number": set_number,
            "total_reps": len(reps),
            "clean_reps": sum(1 for r in reps if r.is_clean),
            "avg_depth": round(avg_depth, 1),
            "depth_consistency": round(depth_consistency, 1),
            "fault_summary": fault_summary,
        }

    def force_end_set(self) -> None:
        """Manually end the current set (e.g. user stops exercising)."""
        self._end_current_set()

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    @property
    def session_stats(self) -> Dict:
        """Aggregate statistics for the entire session."""
        return {
            "total_reps": self.total_reps,
            "total_sets": self.total_sets,
            "avg_depth": (
                sum(r.max_depth_angle for r in self.all_reps) / len(self.all_reps)
                if self.all_reps
                else 0.0
            ),
            "clean_rep_percentage": (
                sum(1 for r in self.all_reps if r.is_clean) / len(self.all_reps) * 100
                if self.all_reps
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all state for a new session."""
        self.current_set_number = 0
        self.current_set_reps = []
        self.last_rep_time = 0.0
        self.set_active = False
        self.total_reps = 0
        self.total_sets = 0
        self.all_reps = []
        self._rep_kinematic_buffer = []
        self._rep_trajectory_buffer = []
        self._bottom_frame_buffer = []
        self._last_rep_bottom_kpts = None
        self._last_rep_standing_kpts = None
        self._last_rep_kinematic_summary = None
        self._last_rep_trajectory = None
        self._last_rep_data = None
        self._last_set_diagnosis = None
