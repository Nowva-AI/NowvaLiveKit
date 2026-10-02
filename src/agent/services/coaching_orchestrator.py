"""
Coaching Orchestrator — Priority-based dispatch for real-time coaching cues.

Separates deterministic cached audio cues (faults, rep counts, positive
reinforcement) from LLM-generated speech (motivation, set recaps).
Cached cues always take priority over LLM-generated speech.
"""

import asyncio
import enum
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from affect.voice_style import set_coaching_moment
from agent.services.coaching_constants import (
    ADJUSTMENT_EXPLAIN_CUES,
    ADJUSTMENT_ON_TARGET_CUE,
    CUE_DISPLAY_LABELS,
    CUE_TEXT_MAP,
    TRACKING_LOST_CUE,
)
from biomechanics.coaching.cue_cache import (
    DEFAULT_FAULT_CUE_PRIORITY,
    FAULT_TO_CUE_MAP,
    PREEMPT_GAP_RATIO_ORCHESTRATOR,
    SIDE_CUE_SIDES,
    base_cue_key,
    can_cue_fault,
    fault_cue_priority,
)
from biomechanics.faults.observability import TRIANGULATED, capture_mode_from_env
from agent.services.progress_context import (
    build_progress_comparison_lines,
    build_session_comparison_line,
    fault_label,
)

logger = logging.getLogger(__name__)

# A cached cue played within this window is included in motivation context
# so the LLM does not contradict a correction the athlete just heard.
RECENT_CUE_CONTEXT_WINDOW_S = 10.0

# Below this diagnosis confidence, the recap prompt asks the LLM to hedge.
LOW_CONFIDENCE_THRESHOLD = 0.5

# The pipeline diagnoses a set only once rest_start reaches it, after the
# recap is already queued. The recap waits this long for that diagnosis.
DIAGNOSIS_WAIT_S = 2.5

# Faults the cameras cannot see. The recap must never claim them.
UNOBSERVABLE_FAULTS_LINE = (
    "Never say or imply anything about lower-back tucking (butt wink), upper-back rounding, "
    "bracing, or foot arch — the cameras can't see those."
)

# Diagnosis causes that point at the feet: when the latest set's top cause is
# one of these, the next fault cue arms the stance / toe-out monitor.
STANCE_CAUSE_IDS = frozenset({"narrow_stance", "stance_toe_mismatch", "narrow_foot_angle"})

# Praise: never within this many reps of the last praise. Generic praise is
# kept for a new best rep; everything else is praise for a fix that held.
POSITIVE_CUE_MIN_REP_GAP = 3
BEST_REP_HIGHLIGHT = "best_rep_so_far"
BEST_REP_CUE_KEY = "strong"

# One focus per set. A fault is worth a cue at moderate severity or worse, or
# when it was mild on MILD_REPEAT_MIN_REPS of the last MILD_REPEAT_WINDOW_REPS reps.
CUE_SEVERITIES = frozenset({"moderate", "severe"})
SEVERITY_RANK = {"mild": 1, "moderate": 2, "severe": 3}
MILD_REPEAT_WINDOW_REPS = 3
MILD_REPEAT_MIN_REPS = 2
# After this many cues for one fault in a set, the rest waits for the recap.
MAX_CUES_PER_FAULT_PER_SET = 2
# Knee cave this bad is cued even when it isn't the set's focus (safety).
SAFETY_FAULT_TYPE = "knee_valgus"
SAFETY_SEVERITY = "severe"
# A cued fault counts as fixed once it stays gone this many reps in a row.
FIX_HOLD_REPS = 2
FIXED_CUE_SUFFIX = "_fixed"

# Side-view faults are judged only on the 3-camera rig. On one camera Nova
# says nothing about them: no cue mid-set and no mention in the recap.
SIDE_VIEW_FAULTS = frozenset({"forward_lean", "hip_shoot", "velocity_loss", "balance"})
SIDE_VIEW_SYMPTOMS = frozenset(
    {"excessive_trunk_lean", "hip_shoot", "velocity_loss", "balance_forward"}
)

# Diagnosis symptom -> the fault rule whose cue addresses it (the set focus)
SYMPTOM_FAULT_TYPES = {
    "knee_not_tracking_toes": "knee_valgus",
    "hip_shoot": "hip_shoot",
    "heel_rise": "heel_rise",
    "balance_forward": "balance",
    "hip_shift": "hip_shift",
    "depth_limit": "depth",
    "uneven_setup": "foot_placement",
    "incomplete_lockout": "lockout",
    "fast_descent": "tempo",
    "velocity_loss": "velocity_loss",
}

# A set stopped short of its target (failed rep, bar racked early) ends this
# long after its last rep.
SET_IDLE_TIMEOUT_S = 15.0

# Effort from rep speed: concentric time vs the set's fastest rep. Needs no
# voice enrollment, only enough reps to compare.
MIN_REPS_FOR_TEMPO_EFFORT = 3
EFFORT_WORKING_RATIO = 1.15
EFFORT_NEAR_LIMIT_RATIO = 1.35

# A cue that fixed the next rep on fewer than this share of at least
# CUE_HISTORY_MIN_EVALUATED past attempts is flagged to the recap.
CUE_HISTORY_MIN_EVALUATED = 5
CUE_HISTORY_MAX_EFFECTIVENESS = 0.3

# Rep-to-rep depth spread (std dev, degrees) put into words for the recap
DEPTH_VERY_CONSISTENT_DEG = 4.0
DEPTH_FAIRLY_CONSISTENT_DEG = 8.0

# Intra-set stance/toe-out adjustment monitoring (Feature 2).
# Stance is in shoulder-width multiples; toe-out in degrees of external rotation.
STANCE_TOLERANCE = 0.1
TOE_OUT_TOLERANCE_DEG = 3.0
# Cap the coaching loop so a lifter who cannot reach the target (or whose
# feet are mistracked) is not corrected on a loop for the whole set.
MAX_ADJUSTMENT_UTTERANCES = 4


# =============================================================================
# PRIORITY LEVELS
# =============================================================================

class CuePriority(enum.IntEnum):
    """Lower number = higher priority."""
    LLM_MOTIVATION = 2
    LLM_SET_RECAP = 3
    LLM_EXERCISE_RECAP = 4
    POSITIVE_CUE = 6
    FAULT_CUE = 7  # intentional, validated live — fault cues yield to everything else; do not restore to top priority


# =============================================================================
# EVENT DATACLASS
# =============================================================================

@dataclass(order=True)
class CoachingEvent:
    """A coaching event to be dispatched by the orchestrator."""
    priority: int
    timestamp: float = field(compare=False)
    event_type: str = field(compare=False)  # "cached_cue" | "llm_motivation" | "llm_set_recap" | "llm_exercise_recap"
    cue_key: Optional[str] = field(compare=False, default=None)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class CueLogEntry:
    """Record of a cue that was actually dispatched (for set reports)."""
    wall_time: float       # time.time() for X-axis plotting
    label: str             # Human-readable ("Knees out!", "Rep 3", "Good rep!")
    category: str          # "fault" | "rep" | "positive" | "motivation"


@dataclass
class PendingCueOutcome:
    """A fault cue waiting to be judged on the reps after it."""
    fault_type: str
    cue_key: str
    after_rep: int
    clean_reps: int = 0


def ineffective_cue_faults(effectiveness: List[Dict[str, Any]]) -> Set[str]:
    """Faults whose cue has rarely fixed the next rep for this athlete.

    Rows come from get_cue_effectiveness; side variants count toward their fault.
    """
    totals: Dict[str, List[int]] = {}
    for row in effectiveness:
        if not row.get("cue_key"):
            continue
        evaluated_effective = totals.setdefault(row["fault_type"], [0, 0])
        evaluated_effective[0] += row.get("n_evaluated", 0)
        evaluated_effective[1] += row.get("n_effective", 0)
    return {
        fault_type
        for fault_type, (evaluated, effective) in totals.items()
        if evaluated >= CUE_HISTORY_MIN_EVALUATED
        and effective / evaluated < CUE_HISTORY_MAX_EFFECTIVENESS
    }


def load_phrase(load: Optional[tuple]) -> Optional[str]:
    if load is None:
        return None
    weight, unit = load
    if weight <= 0:
        return "bodyweight"
    return f"{weight:g} {unit}" if unit else f"{weight:g}"


def _depth_consistency_words(spread_deg: float) -> str:
    if spread_deg < DEPTH_VERY_CONSISTENT_DEG:
        return "very consistent from rep to rep"
    if spread_deg < DEPTH_FAIRLY_CONSISTENT_DEG:
        return "fairly consistent from rep to rep"
    return "varied from rep to rep"


def _rep_span(rep_numbers: List[int]) -> str:
    if len(rep_numbers) == 1:
        return f"rep {rep_numbers[0]}"
    return f"reps {rep_numbers[0]} to {rep_numbers[-1]}"


def _cue_outcome_line(outcome: Dict[str, Any]) -> Optional[str]:
    reps_after = outcome["reps_after"]
    if not reps_after:
        return None
    label = fault_label(outcome["fault_type"])
    faulted = len(outcome["faulted_after"])
    if faulted == 0:
        return (
            f"CUE OUTCOME: after your cue on {label} at rep {outcome['after_rep']}, "
            f"it was gone on {_rep_span(reps_after)} — name that improvement."
        )
    return (
        f"CUE OUTCOME: after your cue on {label} at rep {outcome['after_rep']}, "
        f"it still showed on {faulted} of the {len(reps_after)} reps after."
    )


def _spoken_cue_text(cue_key: Optional[str]) -> Optional[str]:
    text = CUE_TEXT_MAP.get(cue_key or "")
    return text.rstrip("!.") if text else None


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class CoachingOrchestrator:
    """
    Owns the priority queue and dispatches coaching cues and LLM calls.

    Cached cues (faults, rep counts, positive reinforcement) play immediately
    on a secondary audio track. LLM calls (motivation, set recaps) only fire
    when the queue is clear and are secondary to cached cues.
    """

    def __init__(
        self,
        play_cached_audio_fn: Callable,
        generate_llm_reply_fn: Callable,
        get_cue_audio_fn: Callable,
        advance_set_fn: Optional[Callable] = None,
        on_workout_complete_fn: Optional[Callable] = None,
        prune_context_fn: Optional[Callable] = None,
        affect_service: Any = None,
    ):
        self._play_cached = play_cached_audio_fn
        self._generate_llm = generate_llm_reply_fn
        self._get_cue_audio = get_cue_audio_fn
        self._advance_set = advance_set_fn
        self._on_workout_complete = on_workout_complete_fn
        self._prune_context = prune_context_fn
        # Speech affect / effort perception (optional). Read at dispatch time so it is fresh.
        self._affect_service = affect_service
        self._athlete_state: Optional[Dict[str, Any]] = None
        self._best_ascent_time_s: Optional[float] = None

        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._processing: bool = False
        self._processor_task: Optional[asyncio.Task] = None

        # Per-set tracking for motivation triggers
        self._set_rep_count: int = 0
        self._set_target_reps: Optional[int] = None
        self._set_number: int = 0  # Increments across sets (1-indexed in output)
        self._clean_streak: int = 0
        self._set_clean_count: int = 0
        self._set_shallow_count: int = 0
        self._set_shallow_depths: List[str] = []
        self._recent_faults: List[str] = []
        self._last_motivation_rep: int = 0
        self._motivation_interval: int = 3
        self._positive_cue_keys: List[str] = []
        self._last_positive_rep: Optional[int] = None
        # Pipeline set number of the latest rep — what diagnosis_complete carries
        self._pipeline_set_number: Optional[int] = None

        # Fault cue rate limiting (orchestrator-level, per cue key)
        self._last_fault_cue_time: float = 0.0
        self._last_fault_cue_priority: int = DEFAULT_FAULT_CUE_PRIORITY
        self._min_fault_cue_gap: float = 8.0  # seconds between fault cues

        # Rest state — suppress rep/fault processing during rest
        self._resting: bool = False
        self._rest_seconds: int = 120
        self._total_sets: Optional[int] = None

        # Set report data collection
        self._set_cue_log: List[CueLogEntry] = []
        self._set_angle_samples: List[Dict[str, Any]] = []
        self._set_rep_events: List[Dict[str, Any]] = []
        self._set_start_wall_time: float = time.time()

        # Deferred report generation — accumulate per-set snapshots,
        # generate all charts at session end (stop())
        self._pending_reports: List[Dict[str, Any]] = []

        # Accumulated per-set summaries for exercise recap
        self._all_set_summaries: List[Dict[str, Any]] = []

        # The exercise being coached, and the one the workout moves to after
        # this exercise's last set (set by CoachingService when the plan
        # advances). Only the squat has depth, diagnosis and stance coaching.
        self._exercise_name: str = ""
        self._is_squat: bool = True
        self._next_exercise: Optional[Dict[str, Any]] = None

        # Diagnosis data — populated by set_diagnosis_data(), consumed by recaps
        self._pending_diagnosis: Optional[Dict[str, Any]] = None
        self._pending_scoring: Optional[Dict[str, Any]] = None
        self._pending_diagnosis_set_number: Optional[int] = None
        self._diagnosis_arrived: Optional[asyncio.Event] = None
        self._diagnosis_wait_s: float = DIAGNOSIS_WAIT_S
        # Latest diagnosis, kept after the recap consumes it (stance follow-up)
        self._latest_diagnosis: Optional[Dict[str, Any]] = None
        # Session-level: long-term causes and anatomy notes are voiced once
        self._longterm_cause_voiced: bool = False
        self._contextual_note_voiced: bool = False
        # Fault types the pipeline flagged approximate; the recap hedges them
        self._approximate_fault_types: Set[str] = set()

        # Latest frame data snapshot — for real-time form queries (Feature 3)
        self._latest_angles: Dict[str, Any] = {}
        self._latest_angles_time: float = 0.0

        # Last fault cue context — what correction was last given (Feature 3)
        self._last_cue_context: Optional[Dict[str, Any]] = None

        # After-cue adjustment monitoring (Feature 2)
        self._adjustment_active: bool = False
        self._adjustment_param: str = ""
        self._adjustment_target: float = 0.0
        self._adjustment_tolerance: float = 0.05
        self._last_feedback_time: float = 0.0
        self._feedback_interval: float = 1.5
        self._adjustment_speaking: bool = False
        self._adjustment_utterances: int = 0
        self._speak_adjustment_fn: Optional[Callable] = None

        # The last fault cue, waiting to be judged on the reps after it
        self._pending_outcome: Optional[PendingCueOutcome] = None

        # One focus per set: faults seen this rep (decided at rep_complete),
        # the set's focus fault, and the fault cues actually spoken this set
        self._rep_fault_candidates: Dict[str, Dict[str, Any]] = {}
        self._set_focus_fault: Optional[str] = None
        self._set_cue_history: List[Dict[str, Any]] = []
        # No cues while the next-set announcement plays or tracking is lost
        self.cues_suppressed: bool = False
        self._tracking_lost: bool = False
        self._tracking_lost_cued: bool = False
        self._side_view_observable: bool = capture_mode_from_env() == TRIANGULATED
        self._idle_task: Optional[asyncio.Task] = None
        self._set_idle_timeout_s: float = SET_IDLE_TIMEOUT_S
        # The last set of the exercise is done; nothing counts until reset
        self._exercise_done: bool = False
        self._rest_started_at: Optional[float] = None
        self.last_set_data: Optional[Dict[str, Any]] = None
        # Returns (weight, unit) for the set in progress, or None when unknown
        self._get_set_load_fn: Optional[Callable[[], Optional[tuple]]] = None
        # Faults whose cue hasn't helped this athlete before (from the DB)
        self.ineffective_cue_faults: Set[str] = set()
        self._cue_history_voiced: Set[str] = set()

        # Last-session baseline — set by CoachingService once loaded from DB
        self.progress_baseline: Optional[Dict[str, Any]] = None

        # Multi-session fault trends — set by CoachingService once loaded from DB
        self.fault_trends: Optional[Dict[str, Any]] = None
        self._celebrated_faults: set = set()

        # Called with (fault_type, cue_key) after a fault cue actually plays —
        # lets the persistence layer distinguish delivered cues from detections
        self.on_fault_cue_delivered: Optional[Callable[[str, Optional[str]], None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background queue processor."""
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("[ORCHESTRATOR] Started")

    def stop(self):
        """Stop the queue processor and generate any pending set reports."""
        self._processing = False
        self._cancel_idle_timer()
        set_coaching_moment(None)
        if self._processor_task:
            # Avoid cancelling ourselves when stop() is called from within a
            # callback running inside the processor task.  Setting _processing
            # to False is enough — the loop will exit on its own.
            current = asyncio.current_task()
            if self._processor_task is not current:
                self._processor_task.cancel()
            self._processor_task = None
        self._generate_pending_reports()
        logger.info("[ORCHESTRATOR] Stopped")

    def reset_set(
        self,
        target_reps: Optional[int] = None,
        positive_cue_keys: Optional[List[str]] = None,
        total_sets: Optional[int] = None,
    ):
        """Reset per-set tracking state for a new set."""
        if total_sets is not None:
            self._total_sets = total_sets
        self._set_rep_count = 0
        self._set_target_reps = target_reps
        self._clean_streak = 0
        self._set_clean_count = 0
        self._set_shallow_count = 0
        self._set_shallow_depths = []
        self._recent_faults = []
        self._last_motivation_rep = 0
        self._last_positive_rep = None
        # The cue list arrives once, at pipeline start; a reset without one keeps it.
        if positive_cue_keys is not None:
            self._positive_cue_keys = list(positive_cue_keys)
        self._last_fault_cue_time = 0.0
        self._last_fault_cue_priority = DEFAULT_FAULT_CUE_PRIORITY
        self._resting = False
        self._adjustment_active = False
        self._adjustment_utterances = 0
        self._pending_outcome = None
        self._rep_fault_candidates = {}
        self._set_focus_fault = None
        self._set_cue_history = []
        self._tracking_lost = False
        self._tracking_lost_cued = False
        self._exercise_done = False
        self._cancel_idle_timer()
        set_coaching_moment(None)
        # Stale across sets this would keep framing every later form query
        # around a cue the lifter has long since moved past.
        self._last_cue_context = None
        # Reset report data
        self._set_cue_log = []
        self._set_angle_samples = []
        self._set_rep_events = []
        self._set_start_wall_time = time.time()
        # Effort tracking restarts with every set
        self._best_ascent_time_s = None
        if self._affect_service is not None:
            try:
                self._affect_service.on_set_reset()
            except Exception as e:
                logger.debug(f"[ORCHESTRATOR] affect on_set_reset failed: {e}")

    def on_rest_complete(self):
        """Called when rest timer expires — resume rep/fault processing."""
        self._resting = False
        self._rest_started_at = None
        logger.info("[ORCHESTRATOR] Rest complete — resuming")

    async def end_set_early(self, reps: Optional[int] = None) -> bool:
        """End the set now (the athlete said they're done) through the normal
        completion path: recap, then rest or the exercise recap. ``reps``
        overrides the counted reps. False when no set is in progress."""
        if self._resting or self._exercise_done:
            return False
        if reps is not None:
            self._set_rep_count = reps
        await self._complete_set("force_end")
        return True

    def set_exercise(self, exercise_name: str, is_squat: bool) -> None:
        """The exercise the pipeline is now analysing."""
        self._exercise_name = exercise_name
        self._is_squat = is_squat

    def set_next_exercise(self, exercise_name: str, total_sets: Optional[int]) -> None:
        """The workout plan moved on; the next exercise starts once this set is wrapped up."""
        self._next_exercise = {"exercise_name": exercise_name, "total_sets": total_sets}

    @property
    def next_exercise_pending(self) -> bool:
        return self._next_exercise is not None

    def begin_next_exercise(self, target_reps: Optional[int]) -> None:
        """Start the pending next exercise: set numbering and the recap history restart."""
        next_exercise = self._next_exercise or {}
        self._next_exercise = None
        self._set_number = 0
        self._all_set_summaries = []
        self.reset_set(
            target_reps=target_reps,
            positive_cue_keys=self._positive_cue_keys,
            total_sets=next_exercise.get("total_sets"),
        )
        logger.info(f"[ORCHESTRATOR] Next exercise: {next_exercise.get('exercise_name')}")

    # ------------------------------------------------------------------
    # Public state accessors
    # ------------------------------------------------------------------

    @property
    def resting(self) -> bool:
        """Whether rep/fault processing is suppressed: resting, or the
        exercise's last set is done."""
        return self._resting or self._exercise_done

    @property
    def exercise_done(self) -> bool:
        """The exercise's last set is done; its recap follows."""
        return self._exercise_done

    @property
    def set_focus_fault(self) -> Optional[str]:
        """The one fault this set is coached on, once known."""
        return self._set_focus_fault

    def rest_seconds_left(self) -> Optional[int]:
        if not self._resting or self._rest_started_at is None:
            return None
        elapsed = time.monotonic() - self._rest_started_at
        return max(0, round(self._rest_seconds - elapsed))

    @property
    def latest_angles(self) -> Dict[str, Any]:
        return self._latest_angles

    @property
    def latest_angles_time(self) -> float:
        return self._latest_angles_time

    @property
    def last_cue_context(self) -> Optional[Dict[str, Any]]:
        return self._last_cue_context

    @property
    def top_cause_id(self) -> Optional[str]:
        """Top immediate cause of the latest set diagnosis."""
        immediate = (self._latest_diagnosis or {}).get("immediate_causes") or []
        return immediate[0].get("cause_id") if immediate else None

    @resting.setter
    def resting(self, value: bool) -> None:
        self._resting = value

    @property
    def set_rep_count(self) -> int:
        """Reps completed in the current set."""
        return self._set_rep_count

    @set_rep_count.setter
    def set_rep_count(self, value: int) -> None:
        self._set_rep_count = value

    @property
    def positive_cue_keys(self) -> List[str]:
        """Cue keys eligible for positive reinforcement after clean reps."""
        return self._positive_cue_keys

    @positive_cue_keys.setter
    def positive_cue_keys(self, value: List[str]) -> None:
        self._positive_cue_keys = value

    @property
    def rest_seconds(self) -> int:
        """Rest duration used to time set recap speech."""
        return self._rest_seconds

    @rest_seconds.setter
    def rest_seconds(self, value: int) -> None:
        self._rest_seconds = value

    def set_diagnosis_data(
        self,
        diagnosis: Dict[str, Any],
        scoring: Dict[str, Any],
        set_number: Optional[int] = None,
    ) -> None:
        """Store diagnosis results from the pipeline and signal any waiting recap."""
        self._pending_diagnosis = diagnosis
        self._pending_scoring = scoring
        self._pending_diagnosis_set_number = set_number
        self._latest_diagnosis = diagnosis
        if self._diagnosis_arrived is not None:
            self._diagnosis_arrived.set()
        logger.info(
            f"[ORCHESTRATOR] Diagnosis data received: set={set_number} "
            f"confidence={diagnosis.get('confidence', 0):.2f} "
            f"score={scoring.get('mean_score', 0):.3f}"
        )
        # The diagnosis lands during the rest after its set: it picks the
        # next set's focus, unless that set is already under way.
        if self._set_rep_count == 0 and not self._exercise_done:
            self.carry_focus_from(diagnosis)

    def carry_focus_from(self, diagnosis: Dict[str, Any]) -> None:
        """Make a diagnosis's top fixable cause the coming set's focus."""
        focus = self._focus_from_diagnosis(diagnosis)
        if focus is not None:
            self._set_focus_fault = focus
            logger.info(f"[ORCHESTRATOR] Next set's focus from the diagnosis: {focus}")

    def _is_unseen_side_view(self, fault_type: str) -> bool:
        return not self._side_view_observable and fault_type in SIDE_VIEW_FAULTS

    def _visible_causes(self, causes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop causes only side-view symptoms point to, when there's one camera."""
        if self._side_view_observable:
            return list(causes)
        return [
            cause for cause in causes
            if not cause.get("implicated_by")
            or not set(cause["implicated_by"]) <= SIDE_VIEW_SYMPTOMS
        ]

    def _focus_cause(self, diagnosis: Dict[str, Any]) -> Optional[tuple]:
        """(cause, fault type) for the top fixable cause the camera sees well enough to cue."""
        for cause in self._visible_causes(diagnosis.get("immediate_causes") or []):
            if cause.get("observability") == "approximate":
                continue
            for symptom_id in cause.get("implicated_by") or []:
                fault_type = SYMPTOM_FAULT_TYPES.get(symptom_id)
                if fault_type and not self._is_unseen_side_view(fault_type):
                    return cause, fault_type
        return None

    def _focus_from_diagnosis(self, diagnosis: Dict[str, Any]) -> Optional[str]:
        focus = self._focus_cause(diagnosis)
        return focus[1] if focus else None

    def top_cause(self, diagnosis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The one cause a recap or handoff names: the set focus's cause, so the
        athlete hears one thing, else the top cause the camera can see."""
        focus = self._focus_cause(diagnosis or {})
        if focus:
            return focus[0]
        visible = self._visible_causes((diagnosis or {}).get("immediate_causes") or [])
        return visible[0] if visible else None

    def set_athlete_state(self, state: Dict[str, Any]) -> None:
        """Store an athlete state dict (effort, affect, confident) pushed by the affect service."""
        self._athlete_state = dict(state)

    def _current_athlete_state(self) -> Dict[str, Any]:
        """Freshest athlete state: the live affect service if attached, else the last pushed dict."""
        if self._affect_service is not None:
            try:
                state = self._affect_service.last_state
                return {"effort": state.effort, "affect": state.affect, "confident": bool(state.confident)}
            except Exception as e:
                logger.debug(f"[ORCHESTRATOR] affect state read failed: {e}")
        return dict(self._athlete_state or {})

    def _voice_strained(self) -> bool:
        """The voice confidently sounds strained or frustrated (needs enrollment)."""
        state = self._current_athlete_state()
        return bool(state.get("confident")) and state.get("affect") in ("strained", "frustrated")

    def _athlete_state_line(self, effort: Optional[str]) -> Optional[str]:
        """One context line for the LLM when the athlete sounds strained or
        frustrated, or rep speed says they're near their limit.

        Voice affect needs voice-enrollment confidence; effort comes from rep
        speed (``_tempo_effort``), which works from the first session.
        """
        notes = []
        if self._voice_strained():
            notes.append(f"sounds {self._current_athlete_state()['affect']}")
        if effort == "near_limit":
            notes.append("rep speed says they're near their limit")
        if not notes:
            return None
        return (
            f"ATHLETE STATE: {', '.join(notes)} — "
            "keep it calm and brief, lead with what went right, no humor."
        )

    def _tempo_effort(self) -> Optional[str]:
        """Effort from rep speed this set: None until enough reps to compare."""
        ratios = [r["ascent_ratio"] for r in self._set_rep_events if r.get("ascent_ratio")]
        if len(ratios) < MIN_REPS_FOR_TEMPO_EFFORT:
            return None
        last_faults = self._set_rep_events[-1].get("faults") or []
        if self._side_view_observable and "velocity_loss" in last_faults:
            return "near_limit"
        if ratios[-1] >= EFFORT_NEAR_LIMIT_RATIO:
            return "near_limit"
        if ratios[-1] >= EFFORT_WORKING_RATIO:
            return "working"
        return "fresh"

    def _consume_diagnosis(self) -> tuple:
        """Return diagnosis data if already available, otherwise skip."""
        diagnosis = self._pending_diagnosis
        scoring = self._pending_scoring
        self._pending_diagnosis = None
        self._pending_scoring = None
        self._pending_diagnosis_set_number = None
        return diagnosis, scoring

    async def _await_set_diagnosis(
        self, set_number: Optional[int], expects_diagnosis: Optional[bool] = None,
    ) -> tuple:
        """This set's diagnosis, waiting up to _diagnosis_wait_s for it.

        The pipeline diagnoses a set only after rest_start reaches it, which
        is after the recap is queued; reading straight away gave the recap
        the previous set's diagnosis, or none. Without a set number to match
        (older callers), whatever is pending is used as before.
        """
        if expects_diagnosis is None:
            expects_diagnosis = self._is_squat
        # Only the squat is diagnosed; nothing is coming for other exercises.
        if set_number is None or not expects_diagnosis:
            return self._consume_diagnosis()
        deadline = time.monotonic() + self._diagnosis_wait_s
        while True:
            if self._pending_diagnosis is not None:
                if self._pending_diagnosis_set_number in (None, set_number):
                    return self._consume_diagnosis()
                logger.info(
                    f"[ORCHESTRATOR] Dropping diagnosis for set "
                    f"{self._pending_diagnosis_set_number} — recapping set {set_number}"
                )
                self._consume_diagnosis()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._diagnosis_arrived = asyncio.Event()
            try:
                await asyncio.wait_for(self._diagnosis_arrived.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            finally:
                self._diagnosis_arrived = None
        logger.info(
            f"[ORCHESTRATOR] No diagnosis for set {set_number} within "
            f"{self._diagnosis_wait_s:.1f}s — recapping without it"
        )
        return None, None

    # ------------------------------------------------------------------
    # Adjustment monitoring (Feature 2: after-cue coaching loop)
    # ------------------------------------------------------------------

    def _maybe_start_adjustment_from_fault(self) -> Optional[str]:
        """Check stance width and toe-out against targets after a fault cue.

        Called intra-set after a fault cue when the latest set diagnosis
        blames the stance or foot angle — guide the user to fix that root
        cause between reps.

        Returns the cue key explaining what was armed, so the caller can play
        it — otherwise the lifter hears "knees out" then, with no stated
        reason, "a little wider".
        """
        if self._adjustment_utterances >= MAX_ADJUSTMENT_UTTERANCES:
            return None

        angles = self._latest_angles
        if not angles or "stance_width_ratio" not in angles:
            return None

        target_stance = angles.get("target_stance_ratio", 0.0)
        target_toe = angles.get("target_toe_out_deg", 0.0)
        if target_stance <= 0 and target_toe <= 0:
            return None

        current_stance = angles["stance_width_ratio"]
        current_toe = (
            angles.get("foot_direction_angle_l", 0.0)
            + angles.get("foot_direction_angle_r", 0.0)
        ) / 2.0

        stance_off = target_stance > 0 and (target_stance - current_stance) > STANCE_TOLERANCE
        toe_off = target_toe > 0 and (target_toe - current_toe) > TOE_OUT_TOLERANCE_DEG

        if stance_off:
            self.start_adjustment_monitor(
                "stance_width", target_stance, tolerance=STANCE_TOLERANCE
            )
            logger.info(
                f"[ORCHESTRATOR] Adjustment monitor: stance_width "
                f"current={current_stance:.2f} target={target_stance:.2f}"
            )
            return ADJUSTMENT_EXPLAIN_CUES["stance_width"]
        if toe_off:
            self.start_adjustment_monitor(
                "toe_out", target_toe, tolerance=TOE_OUT_TOLERANCE_DEG
            )
            logger.info(
                f"[ORCHESTRATOR] Adjustment monitor: toe_out "
                f"current={current_toe:.1f}° target={target_toe:.1f}°"
            )
            return ADJUSTMENT_EXPLAIN_CUES["toe_out"]
        return None

    def start_adjustment_monitor(self, param: str, target: float, tolerance: float = 0.05) -> None:
        self._adjustment_active = True
        self._adjustment_param = param
        self._adjustment_target = target
        self._adjustment_tolerance = tolerance
        # Deliberately not resetting _adjustment_utterances: the budget is
        # per set, so a repeat cue re-aiming the monitor cannot refill it.
        # Start the clock now so the first correction does not talk over the
        # fault cue that triggered it.
        self._last_feedback_time = time.monotonic()

    def stop_adjustment_monitor(self) -> None:
        self._adjustment_active = False
        self._adjustment_param = ""

    # ------------------------------------------------------------------
    # Angle sample recording (called from voice agent on frame_data)
    # ------------------------------------------------------------------

    def record_angle_sample(self, angles_dict: Dict[str, Any]) -> None:
        """Record a joint angle snapshot for set report timeseries and real-time queries."""
        if not angles_dict:
            return

        self._latest_angles = dict(angles_dict)
        self._latest_angles_time = time.time()

        if self._resting:
            return
        knee_l = angles_dict.get("knee_flexion_l")
        knee_r = angles_dict.get("knee_flexion_r")
        trunk_flexion = angles_dict.get("trunk_flexion", 0.0)
        # The pipeline sends None for angles whose keypoints were missing;
        # such frames add nothing to the set's angle series.
        if knee_l is not None and knee_r is not None and trunk_flexion is not None:
            self._set_angle_samples.append({
                "wall_time": time.time(),
                "avg_knee": (knee_l + knee_r) / 2.0,
                "trunk_flexion": trunk_flexion,
            })

        # After-cue adjustment monitoring — only while standing between reps
        if self._adjustment_active and angles_dict.get("rep_phase") == "idle":
            self._maybe_speak_adjustment(angles_dict)

    def _maybe_speak_adjustment(self, angles_dict: Dict[str, Any]) -> None:
        """Compare the tracked parameter to its target and cue the correction."""
        if self._adjustment_speaking or self._speak_adjustment_fn is None:
            return
        now = time.monotonic()
        if now - self._last_feedback_time < self._feedback_interval:
            return

        if self._adjustment_param == "stance_width":
            current = angles_dict.get("stance_width_ratio")
        elif self._adjustment_param == "toe_out":
            angle_l = angles_dict.get("foot_direction_angle_l")
            angle_r = angles_dict.get("foot_direction_angle_r")
            current = None if angle_l is None or angle_r is None else (angle_l + angle_r) / 2.0
        else:
            current = None

        # A frame without stance metrics (no 3D skeleton, uncalibrated
        # shoulder width) must not read as "you are at zero" — that would
        # cue a correction the lifter does not need.
        if current is None:
            return

        param = self._adjustment_param
        delta = self._adjustment_target - current
        hit_target = abs(delta) <= self._adjustment_tolerance
        self._last_feedback_time = now
        self._adjustment_utterances += 1

        # Read param first: stopping clears it, and the confirmation still
        # needs to name what the lifter just got right.
        if hit_target or self._adjustment_utterances >= MAX_ADJUSTMENT_UTTERANCES:
            self.stop_adjustment_monitor()

        self._adjustment_speaking = True
        asyncio.create_task(self._run_adjustment_speech(param, delta, hit_target))

    async def _run_adjustment_speech(self, param: str, delta: float, hit_target: bool) -> None:
        try:
            await self._speak_adjustment_fn(param, delta, hit_target)
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Adjustment feedback failed: {e}")
        finally:
            self._adjustment_speaking = False

    # ------------------------------------------------------------------
    # Event enqueueing (called from IPC handler)
    # ------------------------------------------------------------------

    async def on_fault(
        self,
        cue_key: Optional[str],
        fault_type: str,
        severity: str,
        message: str = "",
        observability: Optional[str] = None,
        side: Optional[str] = None,
    ):
        """Collect a fault for this rep. Whether it gets a cue is decided at
        rep_complete, once the rep's faults are all in (some arrive mid-rep)."""
        if self._resting or self._exercise_done:
            return
        if not self._add_fault_candidate(cue_key, fault_type, severity, observability, side):
            return
        self._recent_faults.append(fault_type)
        if len(self._recent_faults) > 10:
            self._recent_faults = self._recent_faults[-10:]

    def _add_fault_candidate(
        self,
        cue_key: Optional[str],
        fault_type: str,
        severity: str,
        observability: Optional[str],
        side: Optional[str],
    ) -> bool:
        """Keep this rep's worst severity per fault; False if it is never cued mid-set."""
        if self._is_unseen_side_view(fault_type):
            logger.info(f"[ORCHESTRATOR] Side-view fault on one camera — ignored (type={fault_type})")
            return False
        # Approximate faults are recorded upstream but never spoken mid-set:
        # the camera setup can't confirm them.
        if observability == "approximate":
            self._approximate_fault_types.add(fault_type)
            logger.info(f"[ORCHESTRATOR] Approximate fault — no cue (type={fault_type})")
            return False
        # The pipeline leaves cue empty when its own gap was closed; the
        # decision is made here, so fall back to the fault's own cue.
        if not cue_key and fault_type in FAULT_TO_CUE_MAP:
            cue_key = FAULT_TO_CUE_MAP[fault_type]
            if side in SIDE_CUE_SIDES:
                cue_key = f"{cue_key}_{side}"
        current = self._rep_fault_candidates.get(fault_type)
        if current is None or SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(current["severity"], 0):
            self._rep_fault_candidates[fault_type] = {"cue_key": cue_key, "severity": severity}
        return True

    async def on_shallow_rep(
        self,
        cue_key: Optional[str],
        depth_class_name: str = "",
    ):
        """Cue a rep that was rejected for depth.

        Fires on every shallow rep and bypasses the fault-cue gap: this cue
        stands in for the rep-count callout the lifter didn't get, so
        rate-limiting it would leave them with nothing to explain why the
        counter didn't move. Fire-and-forget on the rep audio track.
        """
        if self._resting or self._exercise_done:
            return
        self._set_shallow_count += 1
        self._set_shallow_depths.append(depth_class_name)
        # Faults from a descent that didn't count belong to no rep
        self._rep_fault_candidates = {}
        # Still squatting: the idle timeout counts from the last attempt
        if self._set_rep_count > 0:
            self._arm_idle_timer()

        logger.info(
            f"[ORCHESTRATOR] Shallow rep ({depth_class_name or 'unknown'}) — "
            f"{self._set_shallow_count} this set, not counted"
        )

        if self.cues_suppressed:
            return
        if not cue_key or not self._get_cue_audio(cue_key):
            logger.info(f"[ORCHESTRATOR] No cached audio for shallow cue: {cue_key}")
            return

        asyncio.create_task(self._play_cached(cue_key))
        self._set_cue_log.append(CueLogEntry(
            wall_time=time.time(),
            label=CUE_DISPLAY_LABELS.get(cue_key, cue_key),
            category="fault",
        ))
        logger.info(f"[ORCHESTRATOR] → Fire-and-forget SHALLOW REP cue: {cue_key}")

    async def on_rep_complete(
        self,
        rep_number: int,
        depth: str,
        is_clean: bool,
        faults: List[str],
        max_depth_angle: float = 0.0,
        rep_duration_ms: int = 0,
        ascent_time_s: float = 0.0,
        highlights: Optional[List[str]] = None,
        set_number: Optional[int] = None,
        faults_detailed: Optional[List[Dict[str, Any]]] = None,
    ):
        """Count the rep, then decide this rep's single cue and praise."""
        if self._resting or self._exercise_done:
            return
        # The pipeline rate-limits fault messages per type (3 s), so a fast
        # rep's fault may never arrive as one; its own verdicts always do.
        for fault in faults_detailed or []:
            details = fault.get("details") or {}
            self._add_fault_candidate(
                None, fault.get("fault_type", ""), fault.get("severity", ""),
                details.get("observability"), details.get("side"),
            )
        self._set_rep_count += 1  # Track relative reps per set (ignores absolute pipeline number)
        if set_number is not None:
            self._pipeline_set_number = set_number

        # Concentric slowdown vs the fastest rep of the set is the effort signal (velocity-loss proxy)
        ascent_ratio = None
        if ascent_time_s and ascent_time_s > 0:
            if self._best_ascent_time_s is None or ascent_time_s < self._best_ascent_time_s:
                self._best_ascent_time_s = ascent_time_s
            ascent_ratio = round(ascent_time_s / self._best_ascent_time_s, 3)
            if self._affect_service is not None:
                try:
                    self._affect_service.on_rep_effort(ascent_time_s)
                except Exception as e:
                    logger.debug(f"[ORCHESTRATOR] affect on_rep_effort failed: {e}")

        # Record rep event for set report
        self._set_rep_events.append({
            "wall_time": time.time(),
            "rep_number": self._set_rep_count,
            "is_clean": is_clean,
            "depth": depth,
            "max_depth_angle": max_depth_angle,
            "rep_duration_ms": rep_duration_ms,
            "faults": faults,
            "ascent_time_s": ascent_time_s,
            "ascent_ratio": ascent_ratio,
        })

        if is_clean:
            self._clean_streak += 1
            self._set_clean_count += 1
        else:
            self._clean_streak = 0

        completes_set = (
            self._set_target_reps is not None
            and self._set_rep_count >= self._set_target_reps
        )
        # Praise for a fix never plays over the set's end: the recap says it.
        fix_praise_key = self._resolve_cue_outcome(faults, can_praise=not completes_set)
        candidates, self._rep_fault_candidates = self._rep_fault_candidates, {}

        # Rep count cue — fire-and-forget on separate audio track
        # (bypasses queue since rep sounds play on a dedicated track)
        rep_cue_key = f"rep_{self._set_rep_count}"
        if self._get_cue_audio(rep_cue_key):
            asyncio.create_task(self._play_cached(rep_cue_key))
            # Log for set report
            self._set_cue_log.append(CueLogEntry(
                wall_time=time.time(),
                label=f"Rep {self._set_rep_count}",
                category="rep",
            ))
            logger.info(f"[ORCHESTRATOR] → Fire-and-forget REP COUNT cue: {rep_cue_key}")
        else:
            # Every counted rep is supposed to beep; silence here means the
            # validation sound never loaded, not a missing per-rep variant.
            logger.warning(
                f"[ORCHESTRATOR] No audio for rep cue {rep_cue_key} — rep went unheard"
            )

        logger.info(
            f"[ORCHESTRATOR] Rep {self._set_rep_count}/{self._set_target_reps or '?'} complete — "
            f"clean={is_clean}, streak={self._clean_streak}, set_clean={self._set_clean_count}"
        )

        if completes_set:
            await self._complete_set("rep_count")
            return  # Set complete — skip motivation/positive cues

        fault_event = self._choose_fault_cue(candidates)
        if fault_event is not None:
            await self._queue.put(fault_event)
            logger.info(
                f"[ORCHESTRATOR] ⬆ Enqueued FAULT cue: {fault_event.cue_key} "
                f"(type={fault_event.data['fault_type']}, severity={fault_event.data['severity']})"
            )
        # One line per rep: a correction takes the slot over any praise.
        praise_allowed = fault_event is None

        if fix_praise_key and praise_allowed:
            self._last_positive_rep = self._set_rep_count
            await self._queue.put(CoachingEvent(
                priority=CuePriority.POSITIVE_CUE,
                timestamp=time.monotonic(),
                event_type="cached_cue",
                cue_key=fix_praise_key,
            ))
            logger.info(f"[ORCHESTRATOR] ⬆ Enqueued FIX PRAISE cue: {fix_praise_key}")

        # Generic praise only for a new best rep
        positive_key = self._choose_positive_cue(is_clean, highlights) if praise_allowed else None
        if positive_key and self._get_cue_audio(positive_key):
            self._last_positive_rep = self._set_rep_count
            await self._queue.put(CoachingEvent(
                priority=CuePriority.POSITIVE_CUE,
                timestamp=time.monotonic(),
                event_type="cached_cue",
                cue_key=positive_key,
            ))
            logger.info(f"[ORCHESTRATOR] ⬆ Enqueued POSITIVE cue: {positive_key}")

        # Evaluate LLM motivation at "top of rep" (uses per-set rep number)
        if not self.cues_suppressed and self._should_trigger_motivation(self._set_rep_count):
            context = self._build_motivation_context(self._set_rep_count, depth, is_clean)
            await self._queue.put(CoachingEvent(
                priority=CuePriority.LLM_MOTIVATION,
                timestamp=time.monotonic(),
                event_type="llm_motivation",
                data=context,
            ))
            self._last_motivation_rep = self._set_rep_count
            logger.info(f"[ORCHESTRATOR] Enqueued motivation at rep {self._set_rep_count}")

        self._arm_idle_timer()

    def _passes_bandwidth(self, fault_type: str, severity: str) -> bool:
        if severity in CUE_SEVERITIES:
            return True
        recent = self._set_rep_events[-MILD_REPEAT_WINDOW_REPS:]
        return sum(fault_type in (r.get("faults") or []) for r in recent) >= MILD_REPEAT_MIN_REPS

    def _set_cue_count(self, fault_type: str) -> int:
        return sum(1 for cue in self._set_cue_history if cue["fault_type"] == fault_type)

    def _choose_fault_cue(self, candidates: Dict[str, Dict[str, Any]]) -> Optional[CoachingEvent]:
        """At most one cue per rep, for the set's focus fault.

        A fault qualifies when it is bad enough (moderate+, or mild on 2 of the
        last 3 reps) and hasn't used its cues for the set. Only the focus fault
        is cued, except severe knee cave (safety). With no focus yet, the
        priority order picks, and the fault picked becomes the focus.
        """
        if self.cues_suppressed or self._tracking_lost:
            return None
        eligible = []
        for fault_type, candidate in candidates.items():
            severity = candidate["severity"]
            if not self._passes_bandwidth(fault_type, severity):
                continue
            if self._set_cue_count(fault_type) >= MAX_CUES_PER_FAULT_PER_SET:
                continue
            is_safety = fault_type == SAFETY_FAULT_TYPE and severity == SAFETY_SEVERITY
            if self._set_focus_fault not in (None, fault_type) and not is_safety:
                continue
            cue_key = candidate["cue_key"]
            # A side variant without audio on disk falls back to the base cue.
            if cue_key and not self._get_cue_audio(cue_key):
                cue_key = base_cue_key(cue_key)
            if not cue_key or not self._get_cue_audio(cue_key):
                logger.info(f"[ORCHESTRATOR] No cached audio for {fault_type} cue '{cue_key}' — skipped")
                continue
            eligible.append((fault_cue_priority(fault_type), fault_type, cue_key, severity))
        if not eligible:
            return None
        priority, fault_type, cue_key, severity = min(eligible)

        # Rate limit fault cues to avoid overwhelming the lifter. A
        # higher-priority fault may jump the gap — otherwise a rarer but more
        # important fault is starved by the frequent ones.
        now = time.monotonic()
        if not can_cue_fault(
            now - self._last_fault_cue_time,
            self._min_fault_cue_gap,
            priority,
            self._last_fault_cue_priority,
            preempt_floor_ratio=PREEMPT_GAP_RATIO_ORCHESTRATOR,
        ):
            return None
        self._last_fault_cue_time = now
        self._last_fault_cue_priority = priority
        if self._set_focus_fault is None:
            self._set_focus_fault = fault_type
            logger.info(f"[ORCHESTRATOR] Set focus: {fault_type}")
        return CoachingEvent(
            # Offset by fault rank so that when several are already
            # queued, the most important one dispatches first.
            priority=CuePriority.FAULT_CUE + priority,
            timestamp=time.monotonic(),
            event_type="cached_cue",
            cue_key=cue_key,
            data={"fault_type": fault_type, "severity": severity, "after_rep": self._set_rep_count},
        )

    def _resolve_cue_outcome(self, faults: List[str], can_praise: bool) -> Optional[str]:
        """Judge the last fault cue on the reps after it; returns the praise
        line to play, if any.

        The cue plays between reps, so the next rep is the first one the
        lifter can act on. The fix counts once it holds FIX_HOLD_REPS reps in
        a row; then a praise line for that fault plays. If the fault comes
        back, nothing is said — the recap carries it.
        """
        outcome = self._pending_outcome
        if outcome is None or self._set_rep_count <= outcome.after_rep:
            return None
        if outcome.fault_type in faults:
            self._pending_outcome = None
            logger.info(f"[ORCHESTRATOR] Cue outcome: {outcome.cue_key} persists (rep {self._set_rep_count})")
            return None
        outcome.clean_reps += 1
        if outcome.clean_reps < FIX_HOLD_REPS:
            return None
        self._pending_outcome = None
        logger.info(f"[ORCHESTRATOR] Cue outcome: {outcome.cue_key} fixed (rep {self._set_rep_count})")
        if not can_praise or self.cues_suppressed or self._tracking_lost:
            return None
        praise_key = f"{base_cue_key(outcome.cue_key)}{FIXED_CUE_SUFFIX}"
        if not self._get_cue_audio(praise_key):
            praise_key = ADJUSTMENT_ON_TARGET_CUE
        if not self._get_cue_audio(praise_key):
            logger.info(f"[ORCHESTRATOR] No praise audio for fixed {outcome.fault_type} — skipped")
            return None
        return praise_key

    def _choose_positive_cue(
        self, is_clean: bool, highlights: Optional[List[str]],
    ) -> Optional[str]:
        """Generic praise only for a new best rep, never within
        POSITIVE_CUE_MIN_REP_GAP reps of the last praise."""
        if not is_clean or self.cues_suppressed or self._tracking_lost:
            return None
        if BEST_REP_HIGHLIGHT not in (highlights or []):
            return None
        if BEST_REP_CUE_KEY not in self._positive_cue_keys:
            return None
        if (
            self._last_positive_rep is not None
            and self._set_rep_count - self._last_positive_rep < POSITIVE_CUE_MIN_REP_GAP
        ):
            return None
        return BEST_REP_CUE_KEY

    async def on_tracking_quality(self, status: str, reason: str = "keypoints") -> None:
        """Pipeline tracking quality. While the athlete's legs are lost no fault
        cue or praise plays (missing joints read as fixed faults); the athlete
        hears the tracking_lost cue once per set. A dropped rig camera is only
        logged — the others still see the athlete."""
        if reason != "keypoints":
            logger.info(f"[ORCHESTRATOR] Tracking {status} ({reason}) — not voiced")
            return
        if status == "recovered":
            if self._tracking_lost:
                logger.info("[ORCHESTRATOR] Tracking recovered — cues resume")
            self._tracking_lost = False
            return
        if status != "lost":
            return
        self._tracking_lost = True
        self._pending_outcome = None
        logger.info("[ORCHESTRATOR] Tracking lost — fault cues paused")
        if self._resting or self._exercise_done or self._tracking_lost_cued:
            return
        self._tracking_lost_cued = True
        if not self._get_cue_audio(TRACKING_LOST_CUE):
            logger.info(f"[ORCHESTRATOR] No audio for {TRACKING_LOST_CUE} — skipped")
            return
        asyncio.create_task(self._play_cached(TRACKING_LOST_CUE))

    # ------------------------------------------------------------------
    # Set completion
    # ------------------------------------------------------------------

    def _set_token(self) -> tuple:
        return (self._set_number, self._set_start_wall_time)

    def _arm_idle_timer(self) -> None:
        self._cancel_idle_timer()
        self._idle_task = asyncio.create_task(self._end_set_when_idle(self._set_token()))

    def _cancel_idle_timer(self) -> None:
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    async def _end_set_when_idle(self, set_token: tuple) -> None:
        """A failed rep or a set racked short never reaches the target count;
        end it once the reps stop."""
        await asyncio.sleep(self._set_idle_timeout_s)
        if self._resting or self._exercise_done or self._set_token() != set_token:
            return
        self._idle_task = None
        logger.info(
            f"[ORCHESTRATOR] No rep for {self._set_idle_timeout_s:.0f}s after rep "
            f"{self._set_rep_count} — ending the set"
        )
        await self._complete_set("idle")

    async def _complete_set(self, trigger: str) -> None:
        """Close the set: queue its recap and rest, or the exercise recap."""
        self._cancel_idle_timer()
        self._set_number += 1
        set_data = self._build_set_summary()
        set_data["trigger"] = trigger
        self.last_set_data = set_data

        is_last_set = (
            self._total_sets is not None
            and self._set_number >= self._total_sets
        )
        self._exercise_done = is_last_set

        if is_last_set:
            # Last set — snapshot report data and advance session
            set_data["_report"] = {
                "angle_samples": list(self._set_angle_samples),
                "cue_log": list(self._set_cue_log),
                "rep_events": list(self._set_rep_events),
                "start_time": self._set_start_wall_time,
            }
            new_target_reps = None
            if self._advance_set:
                try:
                    new_target_reps = await self._advance_set()
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] advance_set_fn failed: {e}")
            # Queue exercise recap instead of set recap
            await self._queue_exercise_recap(set_data)
            if self._next_exercise is not None:
                # Another exercise follows: rest, then start it
                self.begin_next_exercise(new_target_reps)
                self._resting = True
                self._rest_started_at = time.monotonic()
        else:
            # Mid-workout set complete — normal flow
            await self.on_set_complete(set_data)
            if self._advance_set:
                try:
                    new_target_reps = await self._advance_set()
                    if self._next_exercise is not None:
                        # The plan moved to another exercise before the
                        # set count said so (unknown total, skipped sets)
                        self.begin_next_exercise(new_target_reps)
                    else:
                        self.reset_set(
                            target_reps=new_target_reps,
                            positive_cue_keys=self._positive_cue_keys,
                        )
                    # Enter rest mode — will be cleared by on_rest_complete()
                    self._resting = True
                    self._rest_started_at = time.monotonic()
                    logger.info("[ORCHESTRATOR] Entering rest mode")
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] advance_set_fn failed: {e}")

    async def on_set_complete(self, set_data: dict):
        """Enqueue LLM set recap.

        Snapshots report data into the event so it survives reset_set().
        Flushes stale in-set events (faults, motivation) before queuing
        so the recap dispatches immediately.
        """
        self._flush_queue()
        set_data["_report"] = {
            "angle_samples": list(self._set_angle_samples),
            "cue_log": list(self._set_cue_log),
            "rep_events": list(self._set_rep_events),
            "start_time": self._set_start_wall_time,
        }
        await self._queue.put(CoachingEvent(
            priority=CuePriority.LLM_SET_RECAP,
            timestamp=time.monotonic(),
            event_type="llm_set_recap",
            data=set_data,
        ))
        logger.info(f"[ORCHESTRATOR] Set {set_data.get('set_number', '?')} complete — queued recap")

    async def _queue_exercise_recap(self, final_set_data: dict):
        """Queue the exercise recap after the last set completes."""
        self._flush_queue()
        # Add the final set's summary to the accumulated list
        self._all_set_summaries.append({
            "set_number": final_set_data.get("set_number", 0),
            "total_reps": final_set_data.get("total_reps", 0),
            "clean_reps": final_set_data.get("clean_reps", 0),
            "shallow_reps": final_set_data.get("shallow_reps", 0),
            "avg_depth": final_set_data.get("avg_depth", 0),
            "depth_consistency": final_set_data.get("depth_consistency", 0),
            "avg_duration_ms": final_set_data.get("avg_duration_ms", 0),
            "fault_summary": final_set_data.get("fault_summary", {}),
        })

        # Stash report data for the final set
        report = final_set_data.get("_report")
        if report:
            self._pending_reports.append({
                "set_number": final_set_data.get("set_number", 0),
                "report": report,
            })

        await self._queue.put(CoachingEvent(
            priority=CuePriority.LLM_EXERCISE_RECAP,
            timestamp=time.monotonic(),
            event_type="llm_exercise_recap",
            data={
                "all_set_summaries": list(self._all_set_summaries),
                "total_sets": self._total_sets,
                "diagnosis_set_number": final_set_data.get("diagnosis_set_number"),
                # Snapshotted: the next exercise may start before this plays
                "exercise_name": self._exercise_name,
                "is_squat": self._is_squat,
                "next_exercise": (self._next_exercise or {}).get("exercise_name"),
            },
        ))
        logger.info("[ORCHESTRATOR] Last set complete — queued exercise recap")

    # ------------------------------------------------------------------
    # Set summary computation
    # ------------------------------------------------------------------

    def _build_set_summary(self) -> dict:
        """Compute set stats from accumulated per-rep data."""
        depths = [
            r["max_depth_angle"] for r in self._set_rep_events
            if r.get("max_depth_angle", 0) > 0
        ]
        avg_depth = round(statistics.mean(depths), 1) if depths else 0
        depth_consistency = round(statistics.stdev(depths), 1) if len(depths) > 1 else 0

        durations = [
            r["rep_duration_ms"] for r in self._set_rep_events
            if r.get("rep_duration_ms", 0) > 0
        ]
        avg_duration_ms = round(statistics.mean(durations)) if durations else 0

        # Aggregate faults across all reps
        fault_counts: Dict[str, int] = {}
        for rep in self._set_rep_events:
            for ft in rep.get("faults", []):
                fault_counts[ft] = fault_counts.get(ft, 0) + 1
        fault_summary = {
            ft: {"count": cnt, "pct": round(cnt / len(self._set_rep_events) * 100)}
            for ft, cnt in fault_counts.items()
        } if self._set_rep_events else {}

        return {
            "set_number": self._set_number,
            "exercise_name": self._exercise_name,
            "is_squat": self._is_squat,
            # The pipeline's number for this set, matched against diagnosis_complete
            "diagnosis_set_number": (
                self._pipeline_set_number
                if self._pipeline_set_number is not None
                else self._set_number
            ),
            "total_reps": self._set_rep_count,
            "target_reps": self._set_target_reps,
            "clean_reps": self._set_clean_count,
            "shallow_reps": self._set_shallow_count,
            "shallow_depths": list(self._set_shallow_depths),
            "avg_depth": avg_depth,
            "depth_consistency": depth_consistency,
            "avg_duration_ms": avg_duration_ms,
            "fault_summary": fault_summary,
            "load": load_phrase(self._set_load()),
            "effort": self._tempo_effort(),
            "focus_fault": self._set_focus_fault,
            "cue_outcomes": self._cue_outcomes(),
            "per_rep": [
                {
                    "rep": r["rep_number"],
                    "depth": r["depth"],
                    "depth_angle": round(r.get("max_depth_angle", 0), 1),
                    "clean": r["is_clean"],
                    "faults": r.get("faults", []),
                    "duration_ms": r.get("rep_duration_ms", 0),
                }
                for r in self._set_rep_events
            ],
        }

    def _cue_outcome(self, cue: Dict[str, Any]) -> Dict[str, Any]:
        """A spoken cue with the reps after it and which still showed the fault."""
        fault_type = cue["fault_type"]
        reps_after = [r for r in self._set_rep_events if r["rep_number"] > cue["after_rep"]]
        return {
            "fault_type": fault_type,
            "cue_key": cue["cue_key"],
            "after_rep": cue["after_rep"],
            "reps_after": [r["rep_number"] for r in reps_after],
            "faulted_after": [
                r["rep_number"] for r in reps_after if fault_type in (r.get("faults") or [])
            ],
        }

    def _cue_outcomes(self) -> List[Dict[str, Any]]:
        """The outcome of each fault's first cue this set."""
        outcomes: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for cue in self._set_cue_history:
            if cue["fault_type"] not in seen:
                seen.add(cue["fault_type"])
                outcomes.append(self._cue_outcome(cue))
        return outcomes

    def state_fragments(self) -> List[str]:
        """Short facts for the conversational agent's workout state line:
        last set's reps, the focus or top issue, the last cue and its outcome."""
        parts: List[str] = []
        last_set = self.last_set_data or {}
        if last_set:
            parts.append(
                f"last set {last_set.get('clean_reps', 0)}/{last_set.get('total_reps', 0)} clean"
            )
        if self._set_focus_fault:
            parts.append(f"focus {fault_label(self._set_focus_fault)}")
        else:
            spoken = self._spoken_fault_summary(last_set.get("fault_summary") or {})
            if spoken:
                top = max(spoken, key=lambda ft: spoken[ft].get("count", 0))
                parts.append(f"top issue {fault_label(top)}")
        if self._set_cue_history:
            outcome = self._cue_outcome(self._set_cue_history[-1])
        else:
            outcome = (last_set.get("cue_outcomes") or [None])[-1]
        if outcome:
            reps_after, faulted = outcome["reps_after"], outcome["faulted_after"]
            if not reps_after:
                result = "just given"
            elif faulted:
                result = f"still showing on {len(faulted)} of {len(reps_after)} reps"
            elif len(reps_after) >= FIX_HOLD_REPS:
                result = "fixed"
            else:
                result = "looking better"
            cue_text = _spoken_cue_text(outcome["cue_key"]) or fault_label(outcome["fault_type"])
            parts.append(f"last cue '{cue_text}' after rep {outcome['after_rep']}: {result}")
        return parts

    def _aggregate_current_session_faults(self) -> dict[str, int]:
        """Aggregate fault counts across all completed sets in this session."""
        totals: dict[str, int] = {}
        for summary in self._all_set_summaries:
            for fault_type, stats in summary.get("fault_summary", {}).items():
                totals[fault_type] = totals.get(fault_type, 0) + stats.get("count", 0)
        return totals

    # ------------------------------------------------------------------
    # Motivation trigger logic
    # ------------------------------------------------------------------

    def _should_trigger_motivation(self, rep_number: int) -> bool:
        """Fire LLM motivation at the midpoint of the set."""
        if not self._set_target_reps or self._set_target_reps <= 3:
            return False
        midpoint = self._set_target_reps // 2
        return rep_number == midpoint

    def _build_motivation_context(self, rep_number: int, depth: str, is_clean: bool) -> dict:
        """Build context dict for LLM motivation call."""
        reps_remaining = None
        if self._set_target_reps:
            reps_remaining = self._set_target_reps - rep_number

        last_cue_text = None
        if self._last_cue_context:
            cue_age = time.time() - self._last_cue_context.get("timestamp", 0)
            if cue_age < RECENT_CUE_CONTEXT_WINDOW_S:
                last_cue_text = CUE_TEXT_MAP.get(self._last_cue_context.get("cue_key", ""))

        return {
            "rep_number": rep_number,
            "reps_remaining": reps_remaining,
            "target_reps": self._set_target_reps,
            "clean_streak": self._clean_streak,
            "recent_faults": list(set(self._recent_faults[-5:])),
            "last_depth": depth,
            "last_clean": is_clean,
            "last_cue_text": last_cue_text,
            "effort": self._tempo_effort(),
        }

    # ------------------------------------------------------------------
    # Queue processing
    # ------------------------------------------------------------------

    async def _process_loop(self):
        """Main dispatch loop — runs continuously while active."""
        while self._processing:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._dispatch(event)
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Dispatch error: {e}")

    async def _dispatch(self, event: CoachingEvent):
        """Dispatch a single event based on its type and priority."""
        age_s = time.monotonic() - event.timestamp
        logger.info(
            f"[ORCHESTRATOR] ▶ Dispatching: type={event.event_type} priority={event.priority} "
            f"cue_key={event.cue_key} age={age_s:.1f}s"
        )

        from profiler.collector import SessionProfiler
        SessionProfiler.get_instance().record(
            "coaching", "cue_dispatched",
            cue_key=event.cue_key or "",
            cue_event_type=event.event_type,
            priority=int(event.priority),
            queue_age_ms=round(age_s * 1000, 1),
        )

        if event.event_type == "cached_cue":
            await self._dispatch_cached_cue(event)

        elif event.event_type == "llm_motivation":
            # Drop stale motivation events
            age = time.monotonic() - event.timestamp
            if age > 1.5:
                logger.info("[ORCHESTRATOR] Dropping stale motivation event (%.1fs old)", age)
                return
            # Drain any pending cached cues first, then speak motivation
            await self._drain_cached_cues()
            logger.info("[ORCHESTRATOR] → Firing LLM motivation")
            await self._speak_llm_motivation(event.data)
            logger.info("[ORCHESTRATOR] ✓ LLM motivation complete")

        elif event.event_type == "llm_set_recap":
            logger.info("[ORCHESTRATOR] → Firing LLM set recap")
            try:
                await self._speak_llm_set_recap(event.data)
            finally:
                set_coaching_moment(None)
            logger.info("[ORCHESTRATOR] ✓ LLM set recap complete")

        elif event.event_type == "llm_exercise_recap":
            logger.info("[ORCHESTRATOR] → Firing LLM exercise recap")
            try:
                await self._speak_llm_exercise_recap(event.data)
            finally:
                set_coaching_moment(None)
            logger.info("[ORCHESTRATOR] ✓ LLM exercise recap complete")

    async def _dispatch_cached_cue(self, event: CoachingEvent):
        """Play a cached cue."""
        # Drop stale cached cues — if a cue sat in the queue too long,
        # the coaching moment has passed
        age = time.monotonic() - event.timestamp
        if age > 1.0:
            logger.info("[ORCHESTRATOR] Dropping stale cached cue: %s (%.1fs old)", event.cue_key, age)
            return

        try:
            logger.info(f"[ORCHESTRATOR] → Playing cached cue: {event.cue_key}")
            set_token = self._set_token()
            await self._play_cached(event.cue_key)
            # Log successfully played cue for set report
            self._log_dispatched_cue(event)
            fault_type = (event.data or {}).get("fault_type")
            if fault_type:
                self._last_cue_context = {
                    "cue_key": event.cue_key,
                    "fault_type": fault_type,
                    "severity": (event.data or {}).get("severity"),
                    "timestamp": time.time(),
                }
            if fault_type and self.on_fault_cue_delivered:
                try:
                    self.on_fault_cue_delivered(fault_type, event.cue_key)
                except Exception:
                    logger.exception("[ORCHESTRATOR] cue-delivered callback failed")
            # Playback above is awaited, so the set may have ended while this
            # cue was still speaking. Arming anything now would leak follow-up
            # coaching into the next set with stale targets. Compare the set
            # identity rather than _resting: the final set and a verbally
            # force-ended set both land here with _resting already cleared.
            if self._resting or self._exercise_done or self._set_token() != set_token:
                logger.info("[ORCHESTRATOR] Set ended during cue — no follow-up armed")
                return
            if fault_type:
                after_rep = (event.data or {}).get("after_rep", self._set_rep_count)
                self._set_cue_history.append({
                    "fault_type": fault_type, "cue_key": event.cue_key, "after_rep": after_rep,
                })
                self._pending_outcome = PendingCueOutcome(
                    fault_type=fault_type, cue_key=event.cue_key, after_rep=after_rep,
                )
            # When the latest diagnosis blames the feet, follow the cue with
            # the stance / toe-out check. Only without a load: nobody should
            # move their feet under a loaded bar — that waits for the recap.
            if (
                fault_type
                and not self._adjustment_active
                and self.top_cause_id in STANCE_CAUSE_IDS
                and self._is_bodyweight_set()
            ):
                explain_key = self._maybe_start_adjustment_from_fault()
                if explain_key and self._get_cue_audio(explain_key):
                    await self._play_cached(explain_key)
            logger.info(f"[ORCHESTRATOR] ✓ Cached cue played: {event.cue_key}")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Cached cue playback failed ({event.cue_key}): {e}")

    def _set_load(self) -> Optional[tuple]:
        return self._get_set_load_fn() if self._get_set_load_fn is not None else None

    def _is_bodyweight_set(self) -> bool:
        load = self._set_load()
        return load is not None and load[0] <= 0

    def _flush_queue(self):
        """Drop all pending events — called at set boundaries to clear stale cues."""
        dropped = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if dropped:
            logger.info(f"[ORCHESTRATOR] Flushed {dropped} stale events at set boundary")

    async def _drain_cached_cues(self):
        """Drain and play any remaining cached cues before an LLM event."""
        while not self._queue.empty():
            try:
                next_event = self._queue.get_nowait()
                if next_event.event_type == "cached_cue":
                    await self._dispatch_cached_cue(next_event)
                else:
                    # Re-enqueue non-cached events
                    await self._queue.put(next_event)
                    break
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Cue logging for set reports
    # ------------------------------------------------------------------

    def _log_dispatched_cue(self, event: CoachingEvent) -> None:
        """Log a successfully dispatched cached cue for the set report."""
        cue_key = event.cue_key or ""
        data = event.data or {}

        # Determine category and label
        if data.get("fault_type"):
            category = "fault"
            label = CUE_DISPLAY_LABELS.get(cue_key, cue_key)
        elif cue_key.startswith("rep_"):
            category = "rep"
            label = f"Rep {cue_key.split('_', 1)[1]}"
        else:
            category = "positive"
            label = CUE_DISPLAY_LABELS.get(cue_key, cue_key)

        self._set_cue_log.append(CueLogEntry(
            wall_time=time.time(),
            label=label,
            category=category,
        ))

    # ------------------------------------------------------------------
    # Set report generation
    # ------------------------------------------------------------------

    def _generate_set_report_from_snapshot(self, set_number: int, report: dict) -> None:
        """Generate a set report PNG from snapshotted data."""
        try:
            from agent.services.set_report import generate_set_report
            generate_set_report(
                set_number=set_number,
                angle_samples=report.get("angle_samples", []),
                cue_log=report.get("cue_log", []),
                rep_events=report.get("rep_events", []),
                set_start_time=report.get("start_time", 0.0),
            )
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Set report generation failed: {e}")

    def _generate_pending_reports(self) -> None:
        """Generate all deferred set reports (called at session end)."""
        if not self._pending_reports:
            return
        logger.info(f"[ORCHESTRATOR] Generating {len(self._pending_reports)} set report(s)...")
        for entry in self._pending_reports:
            self._generate_set_report_from_snapshot(entry["set_number"], entry["report"])
        self._pending_reports.clear()

    # ------------------------------------------------------------------
    # LLM speech helpers
    # ------------------------------------------------------------------

    async def _speak_llm_motivation(self, context: dict):
        """Generate and speak LLM motivation."""
        set_coaching_moment("last_rep" if context.get("effort") == "near_limit" else None)
        reps_remaining = context.get("reps_remaining")
        clean_streak = context.get("clean_streak", 0)
        recent_faults = context.get("recent_faults", [])
        rep_number = context.get("rep_number", 0)

        parts = [f"Rep {rep_number} just finished."]
        if reps_remaining is not None and reps_remaining > 0:
            parts.append(f"{reps_remaining} reps to go.")
        if clean_streak >= 2:
            parts.append(f"{clean_streak} clean reps in a row.")
        if recent_faults:
            labels = ", ".join(fault_label(ft) for ft in recent_faults[-2:])
            parts.append(f"Recent issues: {labels}.")
        last_cue_text = context.get("last_cue_text")
        if last_cue_text:
            parts.append(
                f"You just told them: '{last_cue_text}' — don't contradict it; reinforcing it is fine."
            )

        context_str = " ".join(parts)

        if context.get("effort") == "near_limit":
            instructions = (
                "Mid-set and their rep speed says they're near their limit. One calm, steady push — "
                "2 to 4 words, no shouting, no humor. React to the data below. "
                f"Here is the data: {context_str}"
            )
        else:
            instructions = (
                "Mid-set. One short motivational push — 2 to 5 words, calm and direct; intensity "
                "only if the data below earns it. No humor mid-set. "
                f"Here is the data: {context_str}"
            )

        logger.info(f"[ORCHESTRATOR] LLM motivation instructions: {instructions[:100]}...")
        try:
            handle = await self._generate_llm(instructions)
            # Log motivation for set report
            self._set_cue_log.append(CueLogEntry(
                wall_time=time.time(),
                label="Motivation",
                category="motivation",
            ))
            logger.info("[ORCHESTRATOR] ✓ LLM motivation spoken")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] LLM motivation failed: {e}", exc_info=True)

    def _spoken_fault_summary(self, fault_summary: Dict[str, Any]) -> Dict[str, Any]:
        """The faults Nova may talk about: side-view faults drop out on one camera."""
        return {
            fault_type: stats for fault_type, stats in fault_summary.items()
            if not self._is_unseen_side_view(fault_type)
        }

    def _cue_history_lines(self, fault_types) -> List[str]:
        """Once per session per fault: its usual cue hasn't helped before."""
        lines = []
        for fault_type in fault_types:
            if fault_type in self.ineffective_cue_faults and fault_type not in self._cue_history_voiced:
                self._cue_history_voiced.add(fault_type)
                lines.append(
                    f"CUE HISTORY: the usual cue for {fault_label(fault_type)} hasn't helped this "
                    "athlete in past sessions — explain that fix from a different angle."
                )
        return lines

    def _top_adjustment(self, diagnosis: Optional[dict]) -> Optional[str]:
        top = self.top_cause(diagnosis)
        if not top or not top.get("parameter_delta"):
            return None
        from biomechanics.diagnosis.demo_builder import summarize_cue_magnitude
        return summarize_cue_magnitude(top.get("cause_id", ""), top["parameter_delta"])

    def _last_set_note(self, data: dict, diagnosis: Optional[dict]) -> str:
        """Compact facts behind the recap, kept in the chat so follow-up
        questions ("what do you mean?") aren't answered blind."""
        parts = [
            f"Set {data.get('set_number', 0)}: {data.get('clean_reps', 0)} of "
            f"{data.get('total_reps', 0)} reps clean"
        ]
        if data.get("load"):
            parts.append(f"load {data['load']}")
        if data.get("focus_fault"):
            parts.append(f"focus {fault_label(data['focus_fault'])}")
        for outcome in (data.get("cue_outcomes") or [])[:1]:
            cue_text = _spoken_cue_text(outcome["cue_key"]) or fault_label(outcome["fault_type"])
            if outcome["reps_after"]:
                result = (
                    "gone after" if not outcome["faulted_after"]
                    else f"still on {len(outcome['faulted_after'])} of {len(outcome['reps_after'])} reps after"
                )
                parts.append(f"cue '{cue_text}' at rep {outcome['after_rep']}, {result}")
        adjustment = self._top_adjustment(diagnosis)
        if adjustment:
            parts.append(f"adjustment: {adjustment}")
        return "; ".join(parts) + "."

    async def _speak_llm_set_recap(self, data: dict):
        """Speak the set recap: one takeaway, one cue for next set, one positive."""
        set_coaching_moment("recap")
        diagnosis, scoring = await self._await_set_diagnosis(data.get("diagnosis_set_number"))

        set_num = data.get("set_number", 0)
        total_reps = data.get("total_reps", 0)
        clean_reps = data.get("clean_reps", 0)
        shallow_reps = data.get("shallow_reps", 0)
        avg_depth = data.get("avg_depth", 0)
        depth_consistency = data.get("depth_consistency", 0)
        avg_duration_ms = data.get("avg_duration_ms", 0)
        fault_summary = data.get("fault_summary", {})
        per_rep = data.get("per_rep", [])
        effort = data.get("effort")
        spoken_faults = self._spoken_fault_summary(fault_summary)

        next_set = set_num + 1
        total_sets_str = f" of {self._total_sets}" if self._total_sets else ""
        is_squat = data.get("is_squat", True)
        exercise_prefix = "" if is_squat or not data.get("exercise_name") else f"{data['exercise_name']} "

        # Build structured context for LLM
        parts = [
            f"{exercise_prefix}Set {set_num}{total_sets_str} just finished: {clean_reps} of {total_reps} reps clean.",
            f"Next up is set {next_set}{total_sets_str}.",
        ]
        target_reps = data.get("target_reps")
        if target_reps and total_reps < target_reps:
            parts.append(
                f"They stopped at {total_reps} of {target_reps} target reps — treat that as "
                "information about the load or the day, never as failure."
            )
        if shallow_reps and is_squat:
            attempted = total_reps + shallow_reps
            parts.append(
                f"{shallow_reps} of {attempted} attempts were too shallow to count "
                f"(never reached their depth target) — those did not go toward the rep total."
            )
        if data.get("load"):
            parts.append(f"Load: {data['load']}.")
        if avg_depth and is_squat:
            parts.append(f"Depth was {_depth_consistency_words(depth_consistency)}.")
        if effort == "near_limit":
            parts.append("Rep speed: the last reps slowed down a lot — close to their limit.")
        elif effort == "working":
            parts.append("Rep speed: reps slowed a little toward the end.")
        if spoken_faults:
            fault_lines = []
            for fault_type, stats in spoken_faults.items():
                count = stats.get("count", 0)
                approximate = " (approximate)" if fault_type in self._approximate_fault_types else ""
                fault_lines.append(f"{fault_label(fault_type)} on {count} of {total_reps} reps{approximate}")
            parts.append(f"Faults: {'; '.join(fault_lines)}.")
            if any(fault_type in self._approximate_fault_types for fault_type in spoken_faults):
                parts.append(
                    "Faults marked approximate are hard to judge from this camera setup — "
                    "if you mention one, say it looked like it, never state it as fact."
                )
        if data.get("focus_fault"):
            parts.append(f"This set's focus was {fault_label(data['focus_fault'])}.")
        for outcome in data.get("cue_outcomes") or []:
            line = _cue_outcome_line(outcome)
            if line:
                parts.append(line)
        parts.extend(self._cue_history_lines(spoken_faults))
        parts.extend(self._build_rep_highlights(per_rep))

        # Enrich with diagnosis data if available
        if diagnosis and scoring:
            parts.extend(self._build_diagnosis_context(diagnosis, scoring))
            parts.extend(
                build_progress_comparison_lines(self.progress_baseline, scoring)
            )

        # Multi-session trend context and chronic fault celebration
        if self.fault_trends:
            from agent.services.progress_context import (
                build_trend_comparison_lines,
                build_chronic_fault_celebration,
            )
            fault_counts = {
                ft: stats.get("count", 0) for ft, stats in spoken_faults.items()
            }
            parts.extend(
                build_trend_comparison_lines(self.fault_trends, fault_counts, total_reps)
            )
            completed_sets = len(self._all_set_summaries)
            if completed_sets >= 2:
                current_faults = self._aggregate_current_session_faults()
                celebration = build_chronic_fault_celebration(
                    self.fault_trends, current_faults, completed_sets
                )
                if celebration:
                    new_celebrations = []
                    for ft in self.fault_trends.get("chronic_faults", []):
                        if ft not in self._celebrated_faults and current_faults.get(ft, 0) == 0:
                            self._celebrated_faults.add(ft)
                            new_celebrations.append(ft)
                    if new_celebrations:
                        parts.append(celebration)

        athlete_line = self._athlete_state_line(effort)
        if athlete_line:
            parts.append(athlete_line)

        context_str = " ".join(parts)

        humor_line = self._humor_line(
            total_reps, clean_reps, shallow_reps,
            affect=self._current_athlete_state().get("affect"),
            near_limit=effort == "near_limit",
        )
        speech_rules = (
            "Plain qualitative words: whole numbers only where they help (a rep number, a cm "
            "adjustment, a score change), never decimals, degrees or seconds. "
        )

        if diagnosis and scoring:
            instructions = (
                f"Your athlete just finished a set. Here's their biomechanics analysis and rep data. "
                f"Give three beats in 2-3 short sentences, in whatever order feels natural: one "
                f"takeaway about the set, the ONE adjustment for the next set (in body units when "
                f"the ADJUSTMENT gives them), and one specific thing that went well — a CUE OUTCOME "
                f"where the fix held is the best one. "
                f"{speech_rules}"
                f"Open differently than you did after the last set. "
                f"{humor_line} "
                f"Here is the data: {context_str}"
            )
        else:
            instructions = (
                f"Your athlete just finished a set. Give three beats in 2-3 short sentences, in "
                f"whatever order feels natural: one takeaway about the set, ONE specific thing to "
                f"do next set if a fault kept repeating, and one specific thing that went well — a "
                f"CUE OUTCOME where the fix held is the best one. "
                f"{speech_rules}"
                f"Open differently than you did after the last set. "
                f"{humor_line} "
                f"Here is the data: {context_str}"
            )

        logger.info(f"[ORCHESTRATOR] LLM set recap instructions: {instructions[:120]}...")
        try:
            handle = await self._generate_llm(
                instructions, last_set_note=self._last_set_note(data, diagnosis),
            )
            logger.info("[ORCHESTRATOR] ✓ LLM set recap spoken")
            # Prune old conversation items to prevent progressive latency
            if self._prune_context:
                try:
                    await self._prune_context(max_items=6)
                except Exception as e:
                    logger.warning(f"[ORCHESTRATOR] Post-recap context prune failed: {e}")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] LLM set recap failed: {e}", exc_info=True)
        finally:
            # Report stashing runs always (data integrity)
            report = data.get("_report")
            if report:
                self._pending_reports.append({"set_number": set_num, "report": report})
            # Accumulate set summary for exercise recap
            summary: Dict[str, Any] = {
                "set_number": set_num,
                "total_reps": total_reps,
                "clean_reps": clean_reps,
                "shallow_reps": shallow_reps,
                "avg_depth": avg_depth,
                "depth_consistency": depth_consistency,
                "avg_duration_ms": avg_duration_ms,
                "fault_summary": fault_summary,
            }
            if diagnosis and scoring:
                summary["diagnosis"] = diagnosis
                summary["scoring"] = scoring
            self._all_set_summaries.append(summary)
            # Sets closed by _complete_set were already reset there
            if not data.get("trigger"):
                self.reset_set(self._set_target_reps, self._positive_cue_keys)

    @staticmethod
    def _humor_line(
        total_reps: int, clean_reps: int, shallow_reps: int,
        affect: Optional[str] = None, near_limit: bool = False,
    ) -> str:
        """Machine-checked humor gate: only allow humor after a genuinely good set and a settled athlete."""
        good_set = (
            total_reps > 0
            and shallow_reps == 0
            and clean_reps >= max(1, round(total_reps * 0.6))
            and affect not in ("frustrated", "strained")
            and not near_limit
        )
        if good_set:
            return "A light touch of dry humor is welcome if it comes naturally — never forced."
        return "No humor in this reply — keep it straight, supportive, and forward-looking."

    def _build_rep_highlights(self, per_rep: list) -> List[str]:
        """Compress the per-rep dump into best/roughest highlight lines."""
        highlights: List[str] = []
        if not per_rep:
            return highlights
        clean = [r for r in per_rep if r.get("clean")]
        if clean:
            best = max(clean, key=lambda r: r.get("depth_angle", 0))
            highlights.append(f"Best rep: rep {best['rep']}, their deepest clean rep.")
        faulted = [
            (r, [ft for ft in r.get("faults", []) if not self._is_unseen_side_view(ft)])
            for r in per_rep
        ]
        faulted = [(r, faults) for r, faults in faulted if faults]
        if faulted:
            worst, worst_faults = max(faulted, key=lambda pair: len(pair[1]))
            labels = ", ".join(fault_label(ft) for ft in worst_faults)
            highlights.append(f"Roughest rep: rep {worst['rep']} — {labels}.")
        return highlights

    def _build_diagnosis_context(self, diagnosis: dict, scoring: dict) -> List[str]:
        """Build diagnosis-enriched context lines for the LLM prompt."""
        parts: List[str] = []

        mean_pct = round(scoring.get("mean_score", 0) * 100)
        dims = scoring.get("per_dimension", {})
        dim_labels = [
            ("depth", "depth"), ("trunk", "trunk_control"),
            ("knee", "knee_tracking"), ("symmetry", "symmetry"),
            ("tempo", "tempo"),
        ]
        if not self._side_view_observable:
            # Trunk control is a side-view measure
            dim_labels = [pair for pair in dim_labels if pair[1] != "trunk_control"]
        dim_str = ", ".join(
            f"{label}: {round(dims[key] * 100)}"
            for label, key in dim_labels if key in dims
        )
        parts.append(f"\nFORM SCORE: {mean_pct} out of 100 ({dim_str})")

        slope = scoring.get("trend_slope", 0)
        if slope > 0.01:
            trend = "improving"
        elif slope < -0.01:
            trend = "declining"
        else:
            trend = "stable"
        parts.append(f"TREND: {trend} over the set")

        top = self.top_cause(diagnosis)
        if top:
            parts.append(f"TOP ISSUE: {top.get('explanation', 'N/A')}")
            adjustment = self._top_adjustment(diagnosis)
            if adjustment:
                parts.append(f"ADJUSTMENT: {adjustment}")

        session_causes = self._visible_causes(diagnosis.get("session_causes", []))
        if session_causes:
            parts.append(f"SESSION PATTERN: {session_causes[0].get('explanation', '')}")

        # One long-term cause per session, in the first recap that has one
        longterm = self._visible_causes(diagnosis.get("longterm_causes") or [])
        if longterm and not self._longterm_cause_voiced:
            self._longterm_cause_voiced = True
            parts.append(
                f"LONG-TERM (mention once, briefly, as something to build over the coming weeks — "
                f"not a fix for the next set): {longterm[0].get('explanation', '')}"
            )

        notes = self._visible_causes(diagnosis.get("contextual_notes") or [])
        if notes and not self._contextual_note_voiced:
            self._contextual_note_voiced = True
            parts.append(
                f"ANATOMY NOTE (say once, reassuringly — this is how they're built, not a fault): "
                f"{notes[0].get('explanation', '')}"
            )

        confidence = diagnosis.get("confidence", 0)
        top_is_approximate = bool(top) and top.get("observability") == "approximate"
        if confidence < LOW_CONFIDENCE_THRESHOLD or top_is_approximate:
            parts.append(
                "This analysis is low-confidence — hedge the adjustment (say it 'looked like'), "
                "and never mention confidence or percentages."
            )

        parts.append(UNOBSERVABLE_FAULTS_LINE)
        return parts

    async def _speak_llm_exercise_recap(self, data: dict):
        """Generate and speak comprehensive exercise recap after all sets."""
        set_coaching_moment("recap")
        # The final set's diagnosis follows workout_complete — wait for it
        diagnosis, scoring = await self._await_set_diagnosis(
            data.get("diagnosis_set_number"), data.get("is_squat"),
        )

        all_summaries = data.get("all_set_summaries", [])
        total_sets = data.get("total_sets", len(all_summaries))

        # Attach final set's diagnosis if available
        if diagnosis and scoring and all_summaries:
            all_summaries[-1]["diagnosis"] = diagnosis
            all_summaries[-1]["scoring"] = scoring

        # Compute aggregate stats
        total_reps = sum(s.get("total_reps", 0) for s in all_summaries)
        total_clean = sum(s.get("clean_reps", 0) for s in all_summaries)

        # Aggregate the faults Nova may talk about across all sets
        all_faults: Dict[str, int] = {}
        for s in all_summaries:
            for fault_type, stats in self._spoken_fault_summary(s.get("fault_summary", {})).items():
                all_faults[fault_type] = all_faults.get(fault_type, 0) + stats.get("count", 0)

        # Per-set breakdown
        per_set_lines = []
        for s in all_summaries:
            sn = s.get("set_number", 0)
            tr = s.get("total_reps", 0)
            cr = s.get("clean_reps", 0)
            faults_in_set = self._spoken_fault_summary(s.get("fault_summary", {}))
            fault_str = ""
            if faults_in_set:
                fault_parts = [
                    f"{fault_label(ft)} {st['count']} times"
                    for ft, st in faults_in_set.items()
                ]
                fault_str = f", faults: {'; '.join(fault_parts)}"
            per_set_lines.append(f"Set {sn}: {cr} of {tr} clean{fault_str}")

        total_shallow = sum(s.get("shallow_reps", 0) for s in all_summaries)
        last_set = self.last_set_data or {}
        exercise_name = data.get("exercise_name") or "Exercise"
        next_exercise = data.get("next_exercise")
        is_squat = data.get("is_squat", True)

        parts = [
            f"{exercise_name} complete! {total_sets} sets finished.",
            f"Total: {total_clean} of {total_reps} reps clean.",
        ]
        if next_exercise:
            parts.append(f"The workout continues: {next_exercise} is next, after the rest.")
        if total_shallow and is_squat:
            parts.append(
                f"{total_shallow} attempts across the session were too shallow to count."
            )
        if last_set.get("load"):
            parts.append(f"Load: {last_set['load']}.")
        if per_set_lines:
            parts.append(f"Per-set breakdown: {'; '.join(per_set_lines)}.")
        if all_faults:
            top_faults = sorted(all_faults.items(), key=lambda x: x[1], reverse=True)[:3]
            fault_str = ", ".join(
                f"{fault_label(ft)} {cnt} times across all sets" for ft, cnt in top_faults
            )
            parts.append(f"Most common faults: {fault_str}.")
        for outcome in last_set.get("cue_outcomes") or []:
            line = _cue_outcome_line(outcome)
            if line:
                parts.append(f"Last set — {line}")

        # Diagnosis progression across sets
        diagnosed_sets = [s for s in all_summaries if "scoring" in s]
        has_diagnosis = len(diagnosed_sets) > 0
        if diagnosed_sets:
            score_lines = []
            for s in diagnosed_sets:
                sn = s["set_number"]
                mean_pct = round(s["scoring"].get("mean_score", 0) * 100)
                score_lines.append(f"Set {sn}: {mean_pct} out of 100")
            parts.append(f"Form score progression: {'; '.join(score_lines)}.")

            if len(diagnosed_sets) >= 2:
                first_score = diagnosed_sets[0]["scoring"].get("mean_score", 0)
                last_score = diagnosed_sets[-1]["scoring"].get("mean_score", 0)
                delta = round((last_score - first_score) * 100)
                if delta > 0:
                    parts.append(f"Overall: improved by {delta} points from first to last set.")
                elif delta < 0:
                    parts.append(f"Overall: declined by {abs(delta)} points from first to last set.")

            all_issues = []
            for s in diagnosed_sets:
                immediate = self._visible_causes(s.get("diagnosis", {}).get("immediate_causes", []))
                if immediate:
                    all_issues.append(f"Set {s['set_number']}: {immediate[0].get('explanation', '')}")
            if all_issues:
                parts.append(f"Key issues by set: {'; '.join(all_issues)}.")

            session_mean = statistics.mean(
                s["scoring"].get("mean_score", 0) for s in diagnosed_sets
            )
            comparison = build_session_comparison_line(
                self.progress_baseline, session_mean
            )
            if comparison:
                parts.append(comparison)

        # Multi-session trend context and chronic fault celebration
        if self.fault_trends:
            from agent.services.progress_context import (
                build_trend_comparison_lines,
                build_chronic_fault_celebration,
            )
            parts.extend(
                build_trend_comparison_lines(self.fault_trends, all_faults, total_reps)
            )
            completed_sets = len(all_summaries)
            current_faults = {}
            for s in all_summaries:
                for ft, stats in s.get("fault_summary", {}).items():
                    current_faults[ft] = current_faults.get(ft, 0) + stats.get("count", 0)
            celebration = build_chronic_fault_celebration(
                self.fault_trends, current_faults, completed_sets
            )
            if celebration:
                uncelebrated = [
                    ft for ft in self.fault_trends.get("chronic_faults", [])
                    if ft not in self._celebrated_faults and current_faults.get(ft, 0) == 0
                ]
                if uncelebrated:
                    for ft in uncelebrated:
                        self._celebrated_faults.add(ft)
                    parts.append(celebration)

        context_str = " ".join(parts)

        humor_line = self._humor_line(total_reps, total_clean, total_shallow)
        closing = (
            f"End by telling them {next_exercise} is up next after the rest — the workout isn't over."
            if next_exercise
            else "End warm and steady — they just finished."
        )
        speech_rules = (
            "Plain qualitative words: whole numbers only where they help (a rep number, a score "
            "change), never decimals, degrees or seconds. "
        )

        if has_diagnosis:
            instructions = (
                f"Your athlete just finished their entire exercise. Here's their comprehensive biomechanics analysis. "
                f"{context_str} "
                f"Give honest feedback in 2-3 short sentences, in whatever order feels natural: how "
                f"their form moved across the sets, one takeaway for next session, and one specific "
                f"thing that went well. "
                f"{speech_rules}"
                f"{humor_line} "
                f"{closing}"
            )
        else:
            instructions = (
                f"Your athlete just finished their entire exercise. Give them a recap. "
                f"{context_str} "
                f"Give honest feedback in 2-3 short sentences, in whatever order feels natural: how "
                f"the sets went, one takeaway for next session, and one specific thing that went well. "
                f"{speech_rules}"
                f"{humor_line} "
                f"{closing}"
            )

        exercise_note = f"Exercise done: {total_sets} sets, {total_clean} of {total_reps} reps clean"
        if all_faults:
            exercise_note += f"; most common fault {fault_label(max(all_faults, key=all_faults.get))}"

        logger.info(f"[ORCHESTRATOR] LLM exercise recap instructions: {instructions[:120]}...")
        try:
            handle = await self._generate_llm(instructions, last_set_note=exercise_note + ".")
            logger.info("[ORCHESTRATOR] ✓ LLM exercise recap spoken")
            # Only fire workout_complete AFTER speech has played out, and only
            # after the workout's last exercise
            if self._on_workout_complete and not data.get("next_exercise"):
                try:
                    await self._on_workout_complete()
                except Exception as e:
                    logger.error(f"[ORCHESTRATOR] on_workout_complete callback failed: {e}")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] LLM exercise recap failed: {e}", exc_info=True)
