"""
Exercise Profile Base Class

Defines the ExerciseProfile interface that bundles exercise-specific concerns:
fault rules, rep counting signal, coaching cues, calibration config, and
depth categorization. The pipeline delegates to the active profile for all
exercise-specific behavior.

Subclasses override only the methods that differ from the default (squat)
behavior. This is a concrete base class, not an ABC — every method has a
sensible default so new exercises can be added incrementally. Every hook
for non-squat exercises (docs/exercises/SHARED-PLUMBING.md §1, §4, §9)
defaults to today's squat path, except reports_depth, which only the squat sets.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.utils.types import (
    LEG_KEYPOINTS,
    FaultEvent,
    FaultSeverity,
    JointAngles,
    Skeleton3D,
    depth_category,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from biomechanics.faults.session_reference import SessionReference
    from biomechanics.utils.frame_context import FrameContext, RepAnalyzer

# The orchestrator ends a set after this long without a rep (squat value).
DEFAULT_SET_IDLE_TIMEOUT_S = 15.0


class ExerciseProfile:
    """
    Base exercise profile with squat defaults.

    Subclasses override methods to customize behavior for their exercise.
    The pipeline calls these methods during initialization and per-frame
    processing — profiles are factories, not owners: a hook that needs a
    threshold receives the pipeline's config, and a profile keeps no config
    or session state of its own.
    """

    name: str = "default"
    movement_pattern: Optional[str] = "squat"
    # The assessment, calibration and causal-diagnosis engine model this
    # exercise. Squats only today; everything else skips those phases.
    uses_diagnosis_engine: bool = False
    # The BiLSTM rep counter was trained on this exercise's movement.
    uses_bilstm_counter: bool = False
    # Nova offers this exercise for camera coaching. Off until the profile's
    # faults are validated on real lifts.
    coaching_ready: bool = False
    # How Nova names the exercise when listing what it coaches ("" = from name).
    display_name: str = ""

    # --- Shared exercise plumbing (squat defaults) ---
    # Set boundaries may refine the rig's camera calibration on this
    # exercise's frames (hinged poses near plates would degrade it).
    allows_camera_refine: bool = True
    # The exercise needs the 3D bar tracker.
    needs_bar_3d: bool = False
    # This exercise's standing frames may feed the session's body measurement.
    feeds_body_calibration: bool = True
    # Until coaching_ready, get_profile serves an untracked stand-in instead.
    gate_until_ready: bool = False
    # Set summaries and recaps speak about depth (the squat only).
    reports_depth: bool = False
    # Readiness HUD hint when the profile's own readiness check fails ("" = generic).
    readiness_hint: str = ""
    # Static start tier per fault type: verdicts below it are recorded, never
    # cued. A missing fault type starts at mild.
    min_cue_tiers: Mapping[str, FaultSeverity] = MappingProxyType({})
    # Keypoints whose loss means tracking is lost and frames are not trusted.
    tracking_keypoints: tuple[int, ...] = LEG_KEYPOINTS
    # The orchestrator ends a set after this long without a rep.
    set_idle_timeout_s: float = DEFAULT_SET_IDLE_TIMEOUT_S

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        """Create the fault rules for this exercise.

        Returns a list of FaultRule instances configured with appropriate
        thresholds. The pipeline passes these to the RuleEngine.

        Default: returns None to let RuleEngine use its built-in _create_rules().
        """
        return None

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        """Compute the rep counting signal from the current frame.

        The returned float drives the HipPositionRepCounter FSM. Different
        exercises use different signals:
        - Squats: hip Y-position relative to ankle (cm)
        - Deadlifts: trunk flexion angle
        - Bench press: elbow flexion angle

        Args:
            skeleton_3d: Current frame's 3D skeleton.
            angles: Current frame's joint angles (may be None early in pipeline).

        Returns:
            Signal value for the rep counter.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_rep_signal()"
        )

    def create_rep_counter_config(
        self, config: BiomechanicsConfig
    ) -> HipPositionCounterConfig:
        """Return rep counter config tuned for this exercise's signal range.

        Default: uses the pipeline's hip_counter config as-is.
        """
        return config.hip_counter

    def get_fault_to_cue_map(self) -> Dict[str, str]:
        """Fault type -> cue key for this exercise's corrections.

        The spoken text for each key lives in the voice agent's CUE_TEXT_MAP.
        A fault with no entry here is never cued.
        """
        return {}

    def get_cue_dict(self) -> Dict[str, str]:
        """Every cue key the voice agent should have ready for this exercise:
        its corrections, the generic positives and the rep counts."""
        from biomechanics.coaching.cue_cache import GENERIC_POSITIVE_CUE_KEYS, build_cue_dict

        return build_cue_dict(*self.get_fault_to_cue_map().values(), *GENERIC_POSITIVE_CUE_KEYS)

    def get_depth_metric(self, angles: JointAngles) -> float:
        """Return the per-frame depth metric tracked across the rep.

        Stored as max_depth_angle / min_depth_angle in RepData.
        Default: avg_knee_flexion (squat).
        """
        return angles.avg_knee_flexion

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        """Return named asymmetry values averaged across the rep.

        Default: knee + hip asymmetry (squat).
        """
        return {"knee": angles.knee_asymmetry, "hip": angles.hip_asymmetry}

    def categorize_depth(self, angle: float) -> str:
        """Categorize rep depth from the max angle achieved.

        Default: delegates to the standard squat depth_category().
        """
        return depth_category(angle)

    def get_readiness_check(self) -> Optional[Callable[[FrameContext], bool]]:
        """A setup-position predicate for a profile without a rep analyser,
        ANDed into the standing gate over its consecutive-frame window. It reads
        only the context (raw hip-centred skeleton, no angles); a profile never
        holds analyser state to build it. None keeps the standing gate as is.
        """
        return None

    # ------------------------------------------------------------------
    # Rep analyser and per-rep hooks (SHARED-PLUMBING.md §1-§2)
    # ------------------------------------------------------------------

    def create_rep_analyzer(self, config: BiomechanicsConfig) -> Optional[RepAnalyzer]:
        """An analyser owning every per-frame and per-rep state of a non-squat
        exercise. None (the squat) keeps the pipeline's squat path."""
        return None

    def observe_frame(self, ctx: FrameContext) -> None:
        """Sees every analysed frame after the readiness latch. Default: nothing."""

    def create_session_reference(self, config: BiomechanicsConfig) -> Optional[SessionReference]:
        """The exercise's session reference; None lets the engine build the squat's."""
        return None

    def create_set_diagnosis(self) -> Any:
        """The exercise's set-level diagnosis, or None (only the squat engine today)."""
        return None

    def on_rep_rejected(
        self, reason: str, features: BaseModel, angles: JointAngles, config: BiomechanicsConfig,
    ) -> List[FaultEvent]:
        """Faults explaining a rep the analyser rejected (not counted)."""
        return []

    def shallow_rep_cue(
        self, fault: FaultEvent, streak: int, cued_this_set: int, config: BiomechanicsConfig,
    ) -> Optional[str]:
        """Cue for a rejected rep's shallow_rep message. streak counts consecutive
        rejections with this reason this set (from 1); cued_this_set counts the
        set's shallow_rep messages that carried a cue. None = cue-less."""
        return None

    def get_hedged_recap_faults(self, config: BiomechanicsConfig) -> frozenset[str]:
        """Fault types whose recorded-only verdicts the recap may mention, hedged
        (the exercise's faults.<exercise>.hedged_recap_faults). Default: none."""
        return frozenset()

    def cue_start_tier(self, fault_type: str, ctx: FrameContext, config: BiomechanicsConfig) -> FaultSeverity:
        """The lowest severity at which a verdict may be cued, given this frame.

        Override when a start tier depends on session state (a certified mount,
        a revoked one); the pipeline stamps verdicts below it recorded-only.
        """
        return self.min_cue_tiers.get(fault_type, FaultSeverity.MILD)

    # ------------------------------------------------------------------
    # Rep counter factory
    # ------------------------------------------------------------------

    def create_rep_counter(self, config: BiomechanicsConfig):
        """Return a SignalRepCounter wired to this profile's signal and metrics.

        BiLSTM rep counters are exercise-specific and trained separately.
        When a trained BiLSTM model exists for an exercise, the pipeline
        constructs it directly (as it does today for squat). This method
        only handles the signal-based fallback that every exercise gets
        out of the box.
        """
        from biomechanics.faults.hip_position_counter import SignalRepCounter
        return SignalRepCounter(
            config=self.create_rep_counter_config(config),
            depth_metric_fn=self.get_depth_metric,
            asymmetry_fn=self.get_asymmetry_metrics,
        )

    # ------------------------------------------------------------------
    # ML / BiLSTM hooks (for future per-exercise models)
    # ------------------------------------------------------------------

    def get_feature_extractor(self):
        """Return the feature extractor for this exercise's movement family.

        Lower-body exercises use LandmarkFeatureExtractor (14d).
        Upper-body exercises override to return UpperBodyFeatureExtractor (14d).
        Used when training and running per-exercise BiLSTM models.
        """
        from biomechanics.ml.feature_extractor import LandmarkFeatureExtractor
        return LandmarkFeatureExtractor()
