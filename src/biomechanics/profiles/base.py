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

from typing import Callable, Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.utils.types import JointAngles, Skeleton3D, depth_category


class ExerciseProfile:
    """
    Base exercise profile with squat defaults.

    Subclasses override methods to customize behavior for their exercise.
    The pipeline calls these methods during initialization and per-frame
    processing — profiles are factories, not owners.
    """

    name: str = "default"
    movement_pattern: str = "squat"

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

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        """Return exercise-specific cue dictionary, or None to use cue_cache defaults."""
        return None

    def get_fault_to_cue_map(self) -> Optional[Dict[str, str]]:
        """Return fault-type to cue-key mapping, or None to use cue_cache defaults."""
        return None

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
