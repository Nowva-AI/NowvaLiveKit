"""
Exercise Profile Base Class

Defines the ExerciseProfile interface that bundles exercise-specific concerns:
fault rules, rep counting signal, coaching cues, calibration config, and
depth categorization. The pipeline delegates to the active profile for all
exercise-specific behavior.

Subclasses override only the methods that differ from the default (squat)
behavior. This is a concrete base class, not an ABC — every method has a
sensible default so new exercises can be added incrementally.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.utils.types import JointAngles, Skeleton3D, depth_category

if TYPE_CHECKING:
    from biomechanics.deadlift.analyzer import DeadliftRepAnalyzer
    from biomechanics.deadlift.diagnosis import DeadliftSetDiagnosis
    from biomechanics.faults.session_reference import SessionReference


class ExerciseProfile:
    """
    Base exercise profile with squat defaults.

    Subclasses override methods to customize behavior for their exercise.
    The pipeline calls these methods during initialization and per-frame
    processing — profiles are factories, not owners.
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

    # Hooks for exercises that bring their own per-frame analysis (the deadlift).
    # Every default is today's squat path, so profiles that leave them alone
    # behave exactly as before.
    # While not coaching_ready, get_profile() hands out the profile's gated
    # stand-in (gated_profile) instead, so unvalidated rules never run.
    gate_until_ready: bool = False
    # The pipeline tracks the bar in 3D on every view for this exercise.
    needs_bar_3d: bool = False
    # Camera refines at set boundaries may use this exercise's frames, and its
    # frames are buffered for them.
    allows_camera_refine: bool = True
    # This exercise's frames feed the session's body measurement.
    feeds_body_calibration: bool = True
    # The voice agent's set idle timeout for this exercise; None keeps its default.
    set_idle_timeout_s: Optional[float] = None
    # The agent waits for this exercise's set diagnosis before the recap; None
    # leaves it to the agent (it waits on the squat).
    waits_for_diagnosis: Optional[bool] = None
    # The closed-loop setup guidance the agent runs for this exercise ("" = none).
    closed_loop_cue: str = ""
    # Keypoints whose loss mutes cues mid-set; None = the legs (hips, knees, ankles).
    tracking_keypoints: Optional[tuple] = None

    def create_rep_analyzer(self, config: BiomechanicsConfig) -> Optional["DeadliftRepAnalyzer"]:
        """An object owning this exercise's per-frame state, or None for the
        squat path (rep features from the pipeline's own trajectory)."""
        return None

    def create_session_reference(self) -> Optional["SessionReference"]:
        """The rule engine's session reference; None builds the squat's."""
        return None

    def create_set_diagnosis(self, capture_mode: str) -> Optional["DeadliftSetDiagnosis"]:
        """A set diagnosis for exercises outside the squat engine; None = none."""
        return None

    def min_cue_tiers(self, config: BiomechanicsConfig) -> Dict[str, str]:
        """fault_type -> lowest severity the agent may speak; empty = every tier."""
        return {}

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

    def get_readiness_check(self) -> Optional[Callable[[Skeleton3D, JointAngles], bool]]:
        """Return a predicate for setup-position validation.

        None means use the default standing-pose gate.
        """
        return None

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
