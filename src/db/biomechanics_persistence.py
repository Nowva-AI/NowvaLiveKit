"""
Persistence layer for live biomechanics data: sessions, sets, reps, and
cue events with effectiveness outcomes. Writes run on a background worker
thread so cloud DB latency never blocks the coaching event loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session

from db.models import (
    BiomechanicsRep,
    BiomechanicsSession,
    BiomechanicsSet,
    CueEvent,
)

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_S = 5.0
CHRONIC_FAULT_SESSION_THRESHOLD = 0.5
CHRONIC_MIN_SESSIONS = 3

# Fault types the pipeline evaluates at or after rep completion. They never
# appear in a rep's faults_detailed and their rep_number tags the NEXT rep,
# so next-rep outcome evaluation and rep linking would produce false labels.
# The squat's once-per-rep verdicts (knee_valgus, hip_shoot, balance,
# hip_shift, bar-tilt bilateral_asymmetry, ...) are judged when the rep
# finishes, carry its own number and sit in its faults_detailed, so they are
# not listed. lockout stays listed for the deadlift's lockout rule.
END_OF_REP_FAULT_TYPES = frozenset({
    "depth",
    "lockout",
    "range_of_motion",
    "trunk_stability",
})

Op = tuple[str, dict]


# ------------------------------------------------------------------
# Pure helpers (no DB access — unit-testable)
# ------------------------------------------------------------------

def evaluate_cue_outcome(
    fault_type: str,
    cue_severity_score: float | None,
    next_rep_faults: list[dict],
) -> dict:
    """Compare a fired cue against the following rep's faults.

    Effective means the fault disappeared or its severity decreased.
    Judged against the worst occurrence of the fault in the next rep.
    """
    severities = [
        f["severity_score"]
        for f in next_rep_faults
        if f.get("fault_type") == fault_type and f.get("severity_score") is not None
    ]
    present = any(f.get("fault_type") == fault_type for f in next_rep_faults)
    if not present:
        return {
            "present_next_rep": False,
            "severity_next_rep": None,
            "effective": True,
        }
    severity_next = max(severities) if severities else None
    effective = (
        severity_next is not None
        and cue_severity_score is not None
        and severity_next < cue_severity_score
    )
    return {
        "present_next_rep": True,
        "severity_next_rep": severity_next,
        "effective": effective,
    }


def mean_fault_severity(fault_type: str, reps_faults: list[list[dict]]) -> float:
    """Mean severity of a fault type across a set's reps. 0.0 if absent."""
    severities = [
        f["severity_score"]
        for rep_faults in reps_faults
        for f in rep_faults
        if f.get("fault_type") == fault_type and f.get("severity_score") is not None
    ]
    if not severities:
        return 0.0
    return sum(severities) / len(severities)


def _build_rep_row(message: dict) -> dict:
    return {
        "rep_number": message.get("rep_number", 0),
        "set_number": message.get("set_number"),
        "is_clean": bool(message.get("is_clean", False)),
        "depth_class": message.get("depth_class_int"),
        "max_depth_angle": message.get("max_depth_angle"),
        "kinematics": message.get("rep_kinematic_summary"),
        "faults": message.get("faults_detailed") or [],
        "timing": {
            "rep_duration_ms": message.get("rep_duration_ms"),
            "descent_time_s": message.get("descent_time_s"),
            "ascent_time_s": message.get("ascent_time_s"),
        },
        "bottom_kpts": message.get("bottom_kpts"),
        "standing_kpts": message.get("standing_kpts"),
    }


def _rep_score_columns(score: dict) -> dict:
    return {
        "composite_score": score.get("composite_score"),
        "depth_score": score.get("depth_score"),
        "trunk_control_score": score.get("trunk_control_score"),
        "knee_tracking_score": score.get("knee_tracking_score"),
        "symmetry_score": score.get("symmetry_score"),
        "tempo_score": score.get("tempo_score"),
    }


def _aggregate_effectiveness(rows: list) -> list[dict]:
    """Group evaluated cue events by (fault_type, cue_key) with hit rates."""
    stats: dict[tuple, dict] = {}
    for row in rows:
        key = (row.fault_type, row.cue_key)
        if key not in stats:
            stats[key] = {
                "fault_type": row.fault_type,
                "cue_key": row.cue_key,
                "n_evaluated": 0,
                "n_effective": 0,
            }
        stats[key]["n_evaluated"] += 1
        if row.effective:
            stats[key]["n_effective"] += 1
    out = []
    for entry in stats.values():
        entry["effectiveness"] = round(
            entry["n_effective"] / entry["n_evaluated"], 3
        )
        out.append(entry)
    out.sort(key=lambda e: (e["fault_type"], e["cue_key"] or ""))
    return out


def build_baseline_summary(
    started_at: datetime,
    mean_session_score: float | None,
    total_reps: int,
    total_sets: int,
    set_rows: list,
    rep_rows: list,
) -> dict:
    """Aggregate one session's rows into a compact baseline for comparisons."""
    dimension_columns = (
        ("depth", "depth_score_avg"),
        ("trunk_control", "trunk_score_avg"),
        ("knee_tracking", "knee_score_avg"),
        ("symmetry", "symmetry_score_avg"),
        ("tempo", "tempo_score_avg"),
    )
    per_dimension = {}
    for key, attr in dimension_columns:
        values = [
            getattr(row, attr) for row in set_rows if getattr(row, attr) is not None
        ]
        if values:
            per_dimension[key] = round(sum(values) / len(values), 3)

    valgus_values: list[float] = []
    trunk_values: list[float] = []
    fault_counts: dict[str, dict] = {}
    for rep in rep_rows:
        kinematics = rep.kinematics or {}
        valgus_l = kinematics.get("knee_valgus_l")
        valgus_r = kinematics.get("knee_valgus_r")
        if valgus_l is not None and valgus_r is not None:
            valgus_values.append(max(valgus_l, valgus_r))
        trunk_pitch = kinematics.get("trunk_pitch_at_bottom")
        if trunk_pitch is not None:
            trunk_values.append(trunk_pitch)
        for fault in rep.faults or []:
            fault_type = fault.get("fault_type")
            if not fault_type:
                continue
            entry = fault_counts.setdefault(
                fault_type,
                {"fault_type": fault_type, "count": 0, "severity_sum": 0.0, "severity_n": 0},
            )
            entry["count"] += 1
            if fault.get("severity_score") is not None:
                entry["severity_sum"] += fault["severity_score"]
                entry["severity_n"] += 1

    top_faults = [
        {
            "fault_type": entry["fault_type"],
            "count": entry["count"],
            "avg_severity": (
                round(entry["severity_sum"] / entry["severity_n"], 2)
                if entry["severity_n"]
                else None
            ),
        }
        for entry in sorted(
            fault_counts.values(), key=lambda e: e["count"], reverse=True
        )
    ]

    # Per-set fault breakdown
    reps_by_set: dict[int, list] = {}
    for rep in rep_rows:
        sn = getattr(rep, "set_number", None)
        if sn is not None:
            reps_by_set.setdefault(sn, []).append(rep)

    per_set: list[dict] = []
    for set_row in sorted(set_rows, key=lambda s: s.set_number):
        set_faults: dict[str, int] = {}
        for rep in reps_by_set.get(set_row.set_number, []):
            for fault in rep.faults or []:
                ft = fault.get("fault_type")
                if ft:
                    set_faults[ft] = set_faults.get(ft, 0) + 1
        per_set.append({
            "set_number": set_row.set_number,
            "rep_count": set_row.rep_count or 0,
            "mean_score": set_row.mean_score,
            "faults": set_faults,
        })

    started_local_date = (
        started_at.replace(tzinfo=timezone.utc).astimezone().date()
    )
    return {
        "session_date": started_at.isoformat(),
        "days_ago": max(0, (datetime.now().date() - started_local_date).days),
        "mean_score": mean_session_score,
        "per_dimension": per_dimension,
        "total_reps": total_reps,
        "total_sets": total_sets,
        "top_faults": top_faults,
        "per_set": per_set,
        "avg_knee_valgus_deg": (
            round(sum(valgus_values) / len(valgus_values), 1) if valgus_values else None
        ),
        "avg_trunk_pitch_deg": (
            round(sum(trunk_values) / len(trunk_values), 1) if trunk_values else None
        ),
    }


def _extract_fault_series(reps: list, fault_type: str) -> list[dict]:
    """Chronological per-rep severity series for one fault type."""
    out = []
    for rep in reps:
        for fault in rep.faults or []:
            if fault.get("fault_type") == fault_type:
                out.append({
                    "timestamp": rep.created_at,
                    "rep_number": rep.rep_number,
                    "severity": fault.get("severity"),
                    "severity_score": fault.get("severity_score"),
                })
    return out


# ------------------------------------------------------------------
# Recorder
# ------------------------------------------------------------------

class BiomechanicsRecorder:
    """Serialized background writer for one workout session.

    ``record_*`` methods enqueue raw IPC messages from any thread; the
    worker thread turns them into ordered row operations and commits
    them one message at a time. A DB failure drops that message's writes
    and logs — it never propagates into the coaching path.
    """

    def __init__(
        self,
        user_id: Any,
        exercise: str = "squat",
        calibration_snapshot: dict | None = None,
        session_factory: Callable[[], Session] | None = None,
    ):
        self.session_id = uuid.uuid4()
        self._user_id = (
            user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
        )
        self._exercise = exercise
        self._calibration_snapshot = calibration_snapshot
        self._session_factory = session_factory
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._disabled = False

        # Worker-side state — touched only by the worker thread once started
        self._pending_cues: list[dict] = []
        self._current_set_cues: list[dict] = []
        self._prev_set_cues: list[dict] = []
        self._current_set_reps: list[dict] = []
        self._current_set_number: int | None = None
        self._set_mean_scores: list[float] = []
        self._session_causes: dict[str, dict] = {}
        self._total_reps = 0
        self._total_sets = 0

    # -------------------- public API (any thread) --------------------

    @property
    def exercise(self) -> str:
        return self._exercise

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._run, name="biomech-recorder", daemon=True
        )
        self._worker.start()
        self._queue.put(("session_start", {}))

    def record_fault(self, message: dict) -> None:
        self._queue.put(("fault", message))

    def record_rep(self, message: dict) -> None:
        self._queue.put(("rep", message))

    def record_rep_diagnosis(self, message: dict) -> None:
        self._queue.put(("rep_diagnosis", message))

    def record_set(self, message: dict) -> None:
        self._queue.put(("set", message))

    def record_cue_delivered(self, fault_type: str, cue_key: str | None = None) -> None:
        self._queue.put(("cue_delivered", {"fault_type": fault_type, "cue_key": cue_key}))

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> bool:
        """Flush remaining writes, finalize the session row, stop the worker.

        Returns True if the flush completed before the timeout.
        """
        self._queue.put(None)
        if self._worker is None:
            return True
        self._worker.join(timeout=timeout_s)
        flushed = not self._worker.is_alive()
        if not flushed:
            logger.warning(
                "[BIOMECH DB] Flush timed out after %.1fs — writes may be incomplete",
                timeout_s,
            )
        self._worker = None
        return flushed

    # -------------------- worker loop --------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._apply(self._ops_session_end())
                return
            kind, message = item
            try:
                ops = self._ops_for(kind, message)
            except Exception:
                logger.exception("[BIOMECH DB] Failed to build ops for %s", kind)
                continue
            self._apply(ops)

    def _ops_for(self, kind: str, message: dict) -> list[Op]:
        if kind == "session_start":
            return self._ops_session_start()
        if kind == "fault":
            return self._ops_fault(message)
        if kind == "rep":
            return self._ops_rep(message)
        if kind == "rep_diagnosis":
            return self._ops_rep_diagnosis(message)
        if kind == "set":
            return self._ops_set(message)
        if kind == "cue_delivered":
            return self._ops_cue_delivered(message)
        logger.warning("[BIOMECH DB] Unknown record kind: %s", kind)
        return []

    # -------------------- ops builders (pure state machine) --------------------

    def _ops_session_start(self) -> list[Op]:
        return [(
            "insert_session",
            {
                "id": self.session_id,
                "user_id": self._user_id,
                "exercise": self._exercise,
                "calibration_snapshot": self._calibration_snapshot,
            },
        )]

    def _ops_fault(self, message: dict) -> list[Op]:
        fault_type = message.get("fault_type", "unknown")
        rep_number = message.get("rep_number")
        # One cue event per (fault_type, rep) — slow reps can emit the same
        # fault twice past the pipeline's 3s cooldown
        for existing in self._pending_cues:
            if (
                existing["fault_type"] == fault_type
                and existing["rep_number"] == rep_number
            ):
                return []
        cue_id = uuid.uuid4()
        cue = {
            "id": cue_id,
            "fault_type": fault_type,
            "rep_number": rep_number,
            "severity_score": message.get("severity_score"),
            "evaluated": False,
            "linked": False,
            "end_of_rep": fault_type in END_OF_REP_FAULT_TYPES,
        }
        self._pending_cues.append(cue)
        self._current_set_cues.append(cue)
        return [(
            "insert_cue",
            {
                "id": cue_id,
                "session_id": self.session_id,
                "user_id": self._user_id,
                "rep_number": message.get("rep_number"),
                "fault_type": cue["fault_type"],
                "severity": message.get("severity"),
                "severity_score": message.get("severity_score"),
                "cue_key": message.get("cue"),
                "message": message.get("message"),
            },
        )]

    def _ops_rep(self, message: dict) -> list[Op]:
        rep_id = uuid.uuid4()
        rep_number = message.get("rep_number", 0)
        set_number = message.get("set_number")
        faults_detailed = message.get("faults_detailed") or []

        ops: list[Op] = []
        # A new set_number with reps still buffered means the previous set
        # never received diagnosis_complete — close it without scores so its
        # reps and totals don't leak into this set's row.
        if set_number is not None:
            if (
                self._current_set_number is not None
                and set_number != self._current_set_number
            ):
                ops.extend(self._ops_flush_unscored_set())
            self._current_set_number = set_number

        row = _build_rep_row(message)
        row.update({
            "id": rep_id,
            "session_id": self.session_id,
            "user_id": self._user_id,
        })
        ops.append(("insert_rep", row))

        # Link cues that fired during this rep to the new rep row.
        # End-of-rep fault types carry the next rep's number — skip them.
        link_ids = []
        for cue in self._pending_cues:
            if cue["linked"] or cue["end_of_rep"]:
                continue
            if cue["rep_number"] == rep_number:
                cue["linked"] = True
                link_ids.append(cue["id"])
        if link_ids:
            ops.append(("link_cues", {"cue_ids": link_ids, "rep_id": rep_id}))

        # Evaluate cues from earlier reps against this rep's faults.
        # End-of-rep fault types never appear in faults_detailed — leave
        # their outcome columns NULL rather than fabricate effective=True.
        for cue in self._pending_cues:
            if cue["evaluated"] or cue["end_of_rep"] or cue["rep_number"] is None:
                continue
            if cue["rep_number"] < rep_number:
                outcome = evaluate_cue_outcome(
                    cue["fault_type"], cue["severity_score"], faults_detailed
                )
                cue["evaluated"] = True
                ops.append(("cue_outcome", {"cue_id": cue["id"], **outcome}))

        self._current_set_reps.append({
            "id": rep_id,
            "rep_number": rep_number,
            "faults": faults_detailed,
        })
        return ops

    def _ops_rep_diagnosis(self, message: dict) -> list[Op]:
        rep_score = message.get("rep_score")
        if not rep_score:
            return []
        rep_number = message.get("rep_number", 0)
        rep = next(
            (r for r in reversed(self._current_set_reps) if r["rep_number"] == rep_number),
            None,
        )
        if rep is None:
            return []
        return [(
            "update_rep_scores",
            {"rep_id": rep["id"], **_rep_score_columns(rep_score)},
        )]

    def _ops_set(self, message: dict) -> list[Op]:
        set_id = uuid.uuid4()
        scoring = message.get("scoring") or {}
        diagnosis = message.get("diagnosis") or {}
        per_dimension = scoring.get("per_dimension") or {}

        ops: list[Op] = [(
            "insert_set",
            {
                "id": set_id,
                "session_id": self.session_id,
                "user_id": self._user_id,
                "set_number": message.get("set_number", 0),
                "rep_count": len(self._current_set_reps),
                "mean_score": scoring.get("mean_score"),
                "depth_score_avg": per_dimension.get("depth"),
                "trunk_score_avg": per_dimension.get("trunk_control"),
                "knee_score_avg": per_dimension.get("knee_tracking"),
                "symmetry_score_avg": per_dimension.get("symmetry"),
                "tempo_score_avg": per_dimension.get("tempo"),
                "trend_slope": scoring.get("trend_slope"),
                "best_rep_number": scoring.get("best_rep"),
                "worst_rep_number": scoring.get("worst_rep"),
                "diagnosis": diagnosis,
                "scoring": scoring,
            },
        )]
        ops.extend(self._ops_close_current_set(set_id, scoring, diagnosis))
        return ops

    def _ops_flush_unscored_set(self) -> list[Op]:
        """Close a set that never received diagnosis_complete (no scores)."""
        if not self._current_set_reps:
            return []
        set_id = uuid.uuid4()
        ops: list[Op] = [(
            "insert_set",
            {
                "id": set_id,
                "session_id": self.session_id,
                "user_id": self._user_id,
                "set_number": self._current_set_number or 0,
                "rep_count": len(self._current_set_reps),
            },
        )]
        ops.extend(self._ops_close_current_set(set_id, scoring=None, diagnosis=None))
        return ops

    def _ops_close_current_set(
        self, set_id: uuid.UUID, scoring: dict | None, diagnosis: dict | None
    ) -> list[Op]:
        ops: list[Op] = []
        rep_ids = [r["id"] for r in self._current_set_reps]
        if rep_ids:
            ops.append(("assign_set_to_reps", {"rep_ids": rep_ids, "set_id": set_id}))
        cue_ids = [c["id"] for c in self._current_set_cues]
        if cue_ids:
            ops.append(("assign_set_to_cues", {"cue_ids": cue_ids, "set_id": set_id}))

        # Backfill per-rep score columns from the set's scoring payload
        if scoring:
            reps_by_number = {r["rep_number"]: r for r in self._current_set_reps}
            for score in scoring.get("per_rep_scores") or []:
                rep = reps_by_number.get(score.get("rep_number"))
                if rep is not None:
                    ops.append((
                        "update_rep_scores",
                        {"rep_id": rep["id"], **_rep_score_columns(score)},
                    ))

        # Previous set's cues: did the fault persist into this set?
        # End-of-rep fault types can't appear in rep faults — leave NULL.
        reps_faults = [r["faults"] for r in self._current_set_reps]
        for cue in self._prev_set_cues:
            if cue["end_of_rep"]:
                continue
            ops.append((
                "cue_next_set",
                {
                    "cue_id": cue["id"],
                    "severity_next_set": mean_fault_severity(
                        cue["fault_type"], reps_faults
                    ),
                },
            ))

        # Session running totals
        self._total_reps += len(self._current_set_reps)
        self._total_sets += 1
        if scoring and scoring.get("mean_score") is not None:
            self._set_mean_scores.append(scoring["mean_score"])
        if diagnosis:
            for cause in diagnosis.get("session_causes") or []:
                cause_id = cause.get("cause_id")
                if cause_id:
                    self._session_causes[cause_id] = cause
        session_update = {
            "total_reps": self._total_reps,
            "total_sets": self._total_sets,
        }
        if self._set_mean_scores:
            session_update["mean_session_score"] = round(
                sum(self._set_mean_scores) / len(self._set_mean_scores), 3
            )
        if self._session_causes:
            session_update["session_causes"] = list(self._session_causes.values())
        ops.append(("update_session", session_update))

        # Rotate set state
        self._prev_set_cues = self._current_set_cues
        self._current_set_cues = []
        self._current_set_reps = []
        self._pending_cues = []
        self._current_set_number = None
        return ops

    def _ops_cue_delivered(self, message: dict) -> list[Op]:
        fault_type = message.get("fault_type")
        for cue in reversed(self._prev_set_cues + self._pending_cues):
            if cue["fault_type"] == fault_type:
                return [("mark_delivered", {"cue_id": cue["id"]})]
        return []

    def _ops_session_end(self) -> list[Op]:
        # Reps from a set that never completed still count toward totals
        ops = self._ops_flush_unscored_set()
        ops.append(("finalize_session", {}))
        return ops

    # -------------------- ORM apply layer --------------------

    def _resolve_session_factory(self) -> Callable[[], Session] | None:
        if self._session_factory is not None:
            return self._session_factory
        try:
            from db.database import SessionLocal
            self._session_factory = SessionLocal
            return self._session_factory
        except Exception:
            self._disabled = True
            logger.exception(
                "[BIOMECH DB] Database unavailable — persistence disabled for this session"
            )
            return None

    def _apply(self, ops: list[Op]) -> None:
        if self._disabled or not ops:
            return
        factory = self._resolve_session_factory()
        if factory is None:
            return
        db = None
        try:
            db = factory()
            for kind, payload in ops:
                self._apply_op(db, kind, dict(payload))
            db.commit()
        except Exception:
            if db is not None:
                db.rollback()
            logger.exception("[BIOMECH DB] Write failed — %d ops dropped", len(ops))
        finally:
            if db is not None:
                db.close()

    def _apply_op(self, db: Session, kind: str, payload: dict) -> None:
        if kind == "insert_session":
            db.add(BiomechanicsSession(**payload))
            db.flush()
        elif kind == "insert_cue":
            db.add(CueEvent(**payload))
            db.flush()
        elif kind == "insert_rep":
            db.add(BiomechanicsRep(**payload))
            db.flush()
        elif kind == "insert_set":
            db.add(BiomechanicsSet(**payload))
            db.flush()
        elif kind == "link_cues":
            db.query(CueEvent).filter(
                CueEvent.id.in_(payload["cue_ids"])
            ).update({"rep_id": payload["rep_id"]}, synchronize_session=False)
        elif kind == "cue_outcome":
            cue_id = payload.pop("cue_id")
            db.query(CueEvent).filter(CueEvent.id == cue_id).update(
                payload, synchronize_session=False
            )
        elif kind == "cue_next_set":
            cue_id = payload.pop("cue_id")
            db.query(CueEvent).filter(CueEvent.id == cue_id).update(
                payload, synchronize_session=False
            )
        elif kind == "mark_delivered":
            db.query(CueEvent).filter(CueEvent.id == payload["cue_id"]).update(
                {"delivered": True}, synchronize_session=False
            )
        elif kind == "update_rep_scores":
            rep_id = payload.pop("rep_id")
            db.query(BiomechanicsRep).filter(BiomechanicsRep.id == rep_id).update(
                payload, synchronize_session=False
            )
        elif kind == "assign_set_to_reps":
            db.query(BiomechanicsRep).filter(
                BiomechanicsRep.id.in_(payload["rep_ids"])
            ).update({"set_id": payload["set_id"]}, synchronize_session=False)
        elif kind == "assign_set_to_cues":
            db.query(CueEvent).filter(
                CueEvent.id.in_(payload["cue_ids"])
            ).update({"set_id": payload["set_id"]}, synchronize_session=False)
        elif kind == "update_session":
            db.query(BiomechanicsSession).filter(
                BiomechanicsSession.id == self.session_id
            ).update(payload, synchronize_session=False)
        elif kind == "finalize_session":
            db.query(BiomechanicsSession).filter(
                BiomechanicsSession.id == self.session_id
            ).update({"completed_at": datetime.utcnow()}, synchronize_session=False)


# ------------------------------------------------------------------
# Query API (voice agent progress context + flywheel metrics)
# ------------------------------------------------------------------

def get_last_completed_session(
    db: Session, user_id, exercise: str = "squat"
) -> dict | None:
    """Most recent finished session summary, or None for first-time users.

    Sessions with zero reps (started then aborted) don't count.
    """
    row = (
        db.query(BiomechanicsSession)
        .filter(
            BiomechanicsSession.user_id == user_id,
            BiomechanicsSession.exercise == exercise,
            BiomechanicsSession.completed_at.isnot(None),
            BiomechanicsSession.total_reps > 0,
        )
        .order_by(BiomechanicsSession.started_at.desc())
        .first()
    )
    if row is None:
        return None
    return {
        "session_id": str(row.id),
        "started_at": row.started_at,
        "completed_at": row.completed_at,
        "total_reps": row.total_reps,
        "total_sets": row.total_sets,
        "mean_session_score": row.mean_session_score,
        "session_causes": row.session_causes,
    }


def get_progress_baseline(
    db: Session, user_id, exercise: str = "squat"
) -> dict | None:
    """Compact summary of the user's last completed session for comparisons.

    Only sessions with at least one scored set qualify — an aborted shell
    session must not shadow the last real one.
    """
    session_row = (
        db.query(BiomechanicsSession)
        .filter(
            BiomechanicsSession.user_id == user_id,
            BiomechanicsSession.exercise == exercise,
            BiomechanicsSession.completed_at.isnot(None),
            BiomechanicsSession.mean_session_score.isnot(None),
        )
        .order_by(BiomechanicsSession.started_at.desc())
        .first()
    )
    if session_row is None:
        return None
    set_rows = (
        db.query(BiomechanicsSet)
        .filter(BiomechanicsSet.session_id == session_row.id)
        .order_by(BiomechanicsSet.set_number.asc())
        .all()
    )
    rep_rows = (
        db.query(BiomechanicsRep)
        .filter(BiomechanicsRep.session_id == session_row.id)
        .all()
    )
    return build_baseline_summary(
        started_at=session_row.started_at,
        mean_session_score=session_row.mean_session_score,
        total_reps=session_row.total_reps,
        total_sets=session_row.total_sets,
        set_rows=set_rows,
        rep_rows=rep_rows,
    )


def get_score_progress(
    db: Session, user_id, since: datetime | None = None, exercise: str = "squat"
) -> list[dict]:
    """Per-set scores over time, chronological."""
    query = (
        db.query(BiomechanicsSet)
        .join(
            BiomechanicsSession,
            BiomechanicsSet.session_id == BiomechanicsSession.id,
        )
        .filter(
            BiomechanicsSet.user_id == user_id,
            BiomechanicsSession.exercise == exercise,
        )
    )
    if since is not None:
        query = query.filter(BiomechanicsSet.created_at >= since)
    rows = query.order_by(BiomechanicsSet.created_at.asc()).all()
    return [
        {
            "timestamp": row.created_at,
            "set_number": row.set_number,
            "rep_count": row.rep_count,
            "mean_score": row.mean_score,
            "depth": row.depth_score_avg,
            "trunk_control": row.trunk_score_avg,
            "knee_tracking": row.knee_score_avg,
            "symmetry": row.symmetry_score_avg,
            "tempo": row.tempo_score_avg,
            "trend_slope": row.trend_slope,
        }
        for row in rows
    ]


def get_fault_progress(
    db: Session,
    user_id,
    fault_type: str,
    since: datetime | None = None,
    limit: int = 500,
) -> list[dict]:
    """Per-rep severity history for one fault type, chronological."""
    query = db.query(BiomechanicsRep).filter(BiomechanicsRep.user_id == user_id)
    if since is not None:
        query = query.filter(BiomechanicsRep.created_at >= since)
    rows = (
        query.order_by(BiomechanicsRep.created_at.desc()).limit(limit).all()
    )
    rows.reverse()
    return _extract_fault_series(rows, fault_type)


def get_rep_kinematics_history(
    db: Session, user_id, since: datetime | None = None, limit: int = 500
) -> list[dict]:
    """Per-rep kinematic summaries over time, chronological (ML/progress source)."""
    query = db.query(BiomechanicsRep).filter(BiomechanicsRep.user_id == user_id)
    if since is not None:
        query = query.filter(BiomechanicsRep.created_at >= since)
    rows = (
        query.order_by(BiomechanicsRep.created_at.desc()).limit(limit).all()
    )
    rows.reverse()
    return [
        {
            "timestamp": row.created_at,
            "rep_number": row.rep_number,
            "composite_score": row.composite_score,
            "is_clean": row.is_clean,
            "kinematics": row.kinematics,
        }
        for row in rows
    ]


def get_cue_effectiveness(
    db: Session, user_id=None, fault_type: str | None = None
) -> list[dict]:
    """Effectiveness rate per (fault_type, cue_key) for cues actually spoken.

    Cross-user when user_id is None.
    """
    query = db.query(CueEvent).filter(
        CueEvent.effective.isnot(None),
        CueEvent.delivered.is_(True),
    )
    if user_id is not None:
        query = query.filter(CueEvent.user_id == user_id)
    if fault_type is not None:
        query = query.filter(CueEvent.fault_type == fault_type)
    return _aggregate_effectiveness(query.all())


def _compute_fault_trend(per_session_counts: list[int]) -> str:
    """Compare older-half mean to newer-half mean of per-session occurrence counts."""
    if len(per_session_counts) < 2:
        return "stable"
    mid = len(per_session_counts) // 2
    older_mean = sum(per_session_counts[:mid]) / mid
    newer_mean = sum(per_session_counts[mid:]) / (len(per_session_counts) - mid)
    if older_mean == 0:
        return "worsening" if newer_mean > 0 else "stable"
    ratio = newer_mean / older_mean
    if ratio < 0.8:
        return "improving"
    if ratio > 1.2:
        return "worsening"
    return "stable"


def get_multi_session_fault_trends(
    db: Session,
    user_id,
    exercise: str = "squat",
    max_sessions: int = 10,
) -> dict:
    """Aggregate fault occurrence across the last N sessions for trend analysis."""
    session_rows = (
        db.query(BiomechanicsSession)
        .filter(
            BiomechanicsSession.user_id == user_id,
            BiomechanicsSession.exercise == exercise,
            BiomechanicsSession.completed_at.isnot(None),
            BiomechanicsSession.mean_session_score.isnot(None),
        )
        .order_by(BiomechanicsSession.started_at.desc())
        .limit(max_sessions)
        .all()
    )
    if not session_rows:
        return {
            "sessions_analyzed": 0,
            "total_reps": 0,
            "fault_profile": [],
            "chronic_faults": [],
        }

    session_ids = [s.id for s in session_rows]
    # Chronological order (oldest first) for trend computation
    session_ids.reverse()

    rep_rows = (
        db.query(BiomechanicsRep)
        .filter(BiomechanicsRep.session_id.in_(session_ids))
        .all()
    )

    total_reps = len(rep_rows)
    n_sessions = len(session_ids)
    session_id_to_idx = {sid: idx for idx, sid in enumerate(session_ids)}

    # fault_type -> {total, severity_sum, severity_n, per_session_counts[]}
    accum: dict[str, dict] = {}

    for rep in rep_rows:
        s_idx = session_id_to_idx.get(rep.session_id)
        if s_idx is None:
            continue
        for fault in rep.faults or []:
            ft = fault.get("fault_type")
            if not ft:
                continue
            if ft not in accum:
                accum[ft] = {
                    "total": 0,
                    "severity_sum": 0.0,
                    "severity_n": 0,
                    "session_set": set(),
                    "per_session": [0] * n_sessions,
                }
            entry = accum[ft]
            entry["total"] += 1
            entry["session_set"].add(s_idx)
            entry["per_session"][s_idx] += 1
            if fault.get("severity_score") is not None:
                entry["severity_sum"] += fault["severity_score"]
                entry["severity_n"] += 1

    fault_profile = []
    for ft, entry in accum.items():
        sessions_present = len(entry["session_set"])
        fault_profile.append({
            "fault_type": ft,
            "total_occurrences": entry["total"],
            "sessions_present": sessions_present,
            "session_frequency": round(sessions_present / n_sessions, 3),
            "rep_frequency": round(entry["total"] / total_reps, 3) if total_reps else 0,
            "avg_severity": (
                round(entry["severity_sum"] / entry["severity_n"], 2)
                if entry["severity_n"]
                else None
            ),
            "trend": _compute_fault_trend(entry["per_session"]),
        })

    fault_profile.sort(key=lambda e: e["total_occurrences"], reverse=True)

    chronic_faults = [
        e["fault_type"]
        for e in fault_profile
        if e["session_frequency"] >= CHRONIC_FAULT_SESSION_THRESHOLD
        and n_sessions >= CHRONIC_MIN_SESSIONS
    ]

    return {
        "sessions_analyzed": n_sessions,
        "total_reps": total_reps,
        "fault_profile": fault_profile,
        "chronic_faults": chronic_faults,
    }
