"""Hypothesis engine for causal set diagnosis, on a DiagnosisGraph (squat by default).

Operates at set-level: takes the per-rep kinematics of a set, detects
symptoms the camera setup can actually see, maps them to candidate causes via
the knowledge graph, scores each cause on the set's median rep, and returns a
structured diagnosis whose confidence reflects how well it was measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from pydantic import BaseModel

from biomechanics.faults.observability import (
    APPROXIMATE,
    NOT_OBSERVABLE,
    OBSERVABLE,
    measurement_observability,
)

from .graph.evidence_tests import ANKLE_DF_UNRESTRICTED_DEG, CONTROLLED_DESCENT_S
from .graph.loader import SYMPTOM_GRAPH, CAUSE_GRAPH
from .graph.parameter_deltas import (
    delta_widen_stance,
    foot_angle_target_deg,
)
from .rep_scoring import score_set
from .types import (
    DetectedSymptom,
    DiagnosisResult,
    HypothesizedCause,
    RepKinematicSummary,
    SetFeatures,
    SetScoreSummary,
)

# Pseudo-posterior mass for "none of the candidate causes explains this
# symptom" — keeps weak evidence from being normalized into confidence.
UNEXPLAINED_LEAK_SCORE = 0.10

HYPOTHESIS_SCORE_THRESHOLD = 0.15
LOAD_REDUCTION_PCT = 10.0

# Measurement confidence: fewer reps than this cannot confirm a pattern, and a
# symptom read from monocular depth regression counts for half.
MIN_REPS_FOR_FULL_CONFIDENCE = 3
OBSERVABILITY_WEIGHT = {OBSERVABLE: 1.0, APPROXIMATE: 0.5}
# Features summarised by the set's last reps rather than the median: fatigue
# shows up at the end of a set, not in its middle.
_TAIL_FIELDS = ("velocity_loss_pct",)
TAIL_REPS = 2
# The core measurements of a squat diagnosis; confidence scales with how many
# of them the set actually measured.
_CORE_MEASURES = ("knee_valgus_max", "depth_ratio", "hip_shift_ratio", "trunk_pitch_at_bottom")
# Toe-out difference that counts as fully asymmetric setup, for scaling it
# against stagger in the setup-asymmetry feature.
SETUP_FLARE_SCALE_DEG = 60.0
# Side-view measures a single camera cannot resolve. They are blanked before
# causes are scored, so a frontal symptom can't voice them through a shared
# cause (e.g. incomplete lockout -> weight too heavy, evidenced by speed loss).
_SIDE_VIEW_FIELDS = (
    "trunk_pitch_at_bottom",
    "hip_shoot_deg",
    "balance_ratio",
    "concentric_velocity_mps",
    "velocity_loss_pct",
)


@dataclass(frozen=True)
class DiagnosisGraph:
    """Everything exercise-specific the engine reads.

    symptoms / causes: the loaded YAML graphs (graph.loader.load_graph).
    rep_model: the per-rep summary model; its numeric fields are aggregated
    into the set's median rep. extract_feature reads a symptom's feature off a
    rep; fill_values gives the explanation templates their values; score_set
    scores a set of reps. core_measures set measurement confidence,
    side_view_fields are blanked when the side view is not observable, and
    tail_fields are summarised by the set's last reps instead of the median.
    """
    symptoms: Mapping[str, Mapping[str, Any]]
    causes: Mapping[str, Mapping[str, Any]]
    rep_model: type[BaseModel]
    extract_feature: Callable[[Any, str], float]
    fill_values: Callable[[Any, dict, dict, SetScoreSummary | None], dict[str, Any]]
    score_set: Callable[[list[Any], dict, dict], SetScoreSummary]
    core_measures: tuple[str, ...]
    side_view_fields: tuple[str, ...]
    tail_fields: tuple[str, ...]


def _squat_extract_feature(rep: RepKinematicSummary, feature_name: str) -> float:
    if feature_name == "knee_valgus_max":
        sides = [v for v in (rep.knee_valgus_l, rep.knee_valgus_r) if not math.isnan(v)]
        return max(sides) if sides else math.nan
    if feature_name == "excess_trunk_lean":
        return rep.trunk_pitch_at_bottom - rep.expected_pitch_reference
    if feature_name == "hip_shift_abs":
        return abs(rep.hip_shift_ratio)
    if feature_name == "depth_deficit":
        return max(0.0, rep.depth_ratio) if math.isfinite(rep.depth_ratio) else math.nan
    if feature_name == "balance_forward":
        return max(0.0, rep.balance_ratio) if math.isfinite(rep.balance_ratio) else math.nan
    if feature_name == "fast_descent_s":
        if rep.descent_time_s <= 0.0:
            return math.nan
        return max(0.0, CONTROLLED_DESCENT_S - rep.descent_time_s)
    if feature_name == "neck_extension_deg":
        return max(0.0, -rep.neck_flexion_deg) if math.isfinite(rep.neck_flexion_deg) else math.nan
    if feature_name == "setup_asymmetry":
        stagger = abs(rep.stagger_ratio)
        flare = abs(rep.foot_direction_angle_l - rep.foot_direction_angle_r) / SETUP_FLARE_SCALE_DEG
        finite = [v for v in (stagger, flare) if math.isfinite(v)]
        return max(finite) if finite else math.nan
    return getattr(rep, feature_name)


def _squat_fill_values(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> dict[str, Any]:
    stagger = rep.stagger_ratio if math.isfinite(rep.stagger_ratio) else 0.0
    flare = rep.foot_direction_angle_l - rep.foot_direction_angle_r
    flare = flare if math.isfinite(flare) else 0.0
    if abs(stagger) * SETUP_FLARE_SCALE_DEG >= abs(flare):
        setup_side = "left" if stagger > 0 else "right"
        setup_issue = "ahead of the other"
    else:
        setup_side = "left" if flare > 0 else "right"
        setup_issue = "turned out more than the other"
    width_increase_m = delta_widen_stance(rep, anthro, rom)["__width_increase_per_side_m"]

    return {
        "ratio": anthro.get("femur_torso_ratio", 1.0),
        "athlete_lean": _finite_or(rep.expected_pitch_athlete, 0.0),
        "current_ratio": _finite_or(rep.stance_width_ratio, 0.0),
        "width_cm": width_increase_m * 100.0,
        "current_angle": _finite_or(
            (rep.foot_direction_angle_l + rep.foot_direction_angle_r) / 2.0, 0.0
        ),
        "recommended_angle": foot_angle_target_deg(anthro, rom),
        "current_df": _finite_or(_nanmax(rep.ankle_df_l_max, rep.ankle_df_r_max), 0.0),
        "expected_df": ANKLE_DF_UNRESTRICTED_DEG,
        "shift_side": "right" if rep.hip_shift_ratio > 0 else "left",
        "setup_side": setup_side,
        "setup_issue": setup_issue,
        "stiffer_ankle_side": "left" if rep.ankle_df_l_max < rep.ankle_df_r_max else "right",
        "ankle_gap": _finite_or(abs(rep.ankle_df_l_max - rep.ankle_df_r_max), 0.0),
        "tighter_side": "left" if rep.hip_flexion_l_max < rep.hip_flexion_r_max else "right",
        "first_bad_rep": _first_degraded_rep(set_summary)
        if set_summary is not None
        else rep.rep_number,
        "reduction_pct": LOAD_REDUCTION_PCT,
        # There is no load to take off a bodyweight squat.
        "load_advice": (
            f"Take about {LOAD_REDUCTION_PCT:.0f}% off, or end sets before the reps slow down that much."
            if rep.bar_detected
            else "Stop a rep or two before the reps slow down that much, or rest a little longer between sets."
        ),
    }


def _first_degraded_rep(set_summary: SetScoreSummary) -> int:
    for rep_score in set_summary.per_rep_scores:
        if rep_score.composite_score < set_summary.mean_score:
            return rep_score.rep_number
    return set_summary.worst_rep_number


SQUAT_GRAPH = DiagnosisGraph(
    symptoms=SYMPTOM_GRAPH,
    causes=CAUSE_GRAPH,
    rep_model=RepKinematicSummary,
    extract_feature=_squat_extract_feature,
    fill_values=_squat_fill_values,
    score_set=score_set,
    core_measures=_CORE_MEASURES,
    side_view_fields=_SIDE_VIEW_FIELDS,
    tail_fields=_TAIL_FIELDS,
)


class HypothesisEngine:
    def __init__(self, graph: DiagnosisGraph | None = None) -> None:
        self._graph = graph if graph is not None else SQUAT_GRAPH

    def diagnose(self, set_features: SetFeatures) -> DiagnosisResult:
        return self.diagnose_reps(
            set_features.set_id,
            set_features.per_rep_kinematics,
            set_features.anthropometry,
            set_features.rom,
            set_features.capture_mode,
        )

    def diagnose_reps(
        self,
        set_id: str,
        reps: list[BaseModel],
        anthro: dict,
        rom: dict,
        capture_mode: str,
    ) -> DiagnosisResult:
        """diagnose() for reps of the graph's own rep model, which SetFeatures
        (squat summaries only) cannot carry."""
        set_summary = self._graph.score_set(reps, anthro, rom) if len(reps) >= 2 else None
        # Causes are judged on the set's median rep: a single worst rep may
        # not even show the symptom that implicated them.
        aggregate_rep = self._without_unseen_side_view(
            self._aggregate_rep(reps, set_summary), capture_mode
        )

        detected_symptoms = self._detect_symptoms(reps, anthro, capture_mode)
        cause_scores = self._score_causes(
            detected_symptoms, aggregate_rep, anthro, rom, set_summary
        )
        filtered_causes = {
            cause_id: info
            for cause_id, info in cause_scores.items()
            if info["aggregate_score"] > HYPOTHESIS_SCORE_THRESHOLD
        }

        hypotheses = self._build_hypotheses(
            filtered_causes, aggregate_rep, anthro, rom, set_summary
        )
        combined_perturbation = self._merge_perturbations(hypotheses)

        immediate = [h for h in hypotheses if h.tier == 1]
        session = [h for h in hypotheses if h.tier == 2]
        longterm = [h for h in hypotheses if h.tier == 3]
        contextual = [h for h in hypotheses if h.tier == 0]

        confidence = self._compute_confidence(reps, detected_symptoms)

        return DiagnosisResult(
            set_id=set_id,
            detected_symptoms=detected_symptoms,
            immediate_causes=immediate,
            session_causes=session,
            longterm_causes=longterm,
            contextual_notes=contextual,
            combined_perturbation=combined_perturbation,
            confidence=confidence,
        )


    def _detect_symptoms(
        self, reps: list[BaseModel], anthro: dict, capture_mode: str
    ) -> list[DetectedSymptom]:
        detected = []

        for symptom_id, symptom_def in self._graph.symptoms.items():
            observability = measurement_observability(symptom_def["measurement"], capture_mode)
            if observability == NOT_OBSERVABLE:
                continue

            detection = symptom_def["detection"]
            feature_name = detection["feature"]
            aggregation = detection["aggregation"]

            values_per_rep = [
                self._extract_feature(rep, feature_name) for rep in reps
            ]

            aggregated_value = self._aggregate(values_per_rep, aggregation)
            expected_value = symptom_def["expected_value_fn"](anthro)
            threshold = symptom_def["threshold"]
            scoring = symptom_def["severity_scoring"]

            severity = self._compute_severity(
                aggregated_value, expected_value, threshold, scoring
            )

            if severity > 0.0:
                contributing_reps = [
                    rep.rep_number
                    for rep, value in zip(reps, values_per_rep)
                    if self._compute_severity(
                        value, expected_value, threshold, scoring
                    )
                    > 0.0
                ]
                detected.append(
                    DetectedSymptom(
                        symptom_id=symptom_id,
                        severity=severity,
                        contributing_reps=contributing_reps,
                        observability=observability,
                    )
                )

        return detected

    def _score_causes(
        self,
        detected_symptoms: list[DetectedSymptom],
        representative_rep: BaseModel,
        anthro: dict,
        rom: dict,
        set_summary: SetScoreSummary | None,
    ) -> dict[str, dict[str, Any]]:
        cause_posteriors: dict[str, list[float]] = {}
        cause_implicated_by: dict[str, list[str]] = {}
        cause_approximate: set[str] = set()

        for symptom in detected_symptoms:
            symptom_def = self._graph.symptoms[symptom.symptom_id]
            candidates = symptom_def["candidate_causes"]

            raw_scores: list[tuple[str, float]] = []
            for candidate in candidates:
                cause_id = candidate["cause_id"]
                prior = candidate["prior"]
                cause_def = self._graph.causes[cause_id]
                evidence_fn = cause_def["evidence_test_fn"]
                evidence_score = evidence_fn(
                    representative_rep, anthro, rom, set_summary
                )
                posterior = prior * evidence_score
                raw_scores.append((cause_id, posterior))

            total = sum(score for _, score in raw_scores) + UNEXPLAINED_LEAK_SCORE
            for cause_id, posterior in raw_scores:
                normalized = (posterior / total) * symptom.severity
                cause_posteriors.setdefault(cause_id, []).append(normalized)
                cause_implicated_by.setdefault(cause_id, []).append(
                    symptom.symptom_id
                )
                if symptom.observability == APPROXIMATE:
                    cause_approximate.add(cause_id)

        result: dict[str, dict[str, Any]] = {}
        for cause_id, posteriors in cause_posteriors.items():
            aggregate_score = 1.0 - math.prod(1.0 - p for p in posteriors)
            result[cause_id] = {
                "aggregate_score": aggregate_score,
                "posteriors": posteriors,
                "implicated_by": cause_implicated_by[cause_id],
                "observability": APPROXIMATE if cause_id in cause_approximate else OBSERVABLE,
            }

        return result

    def _build_hypotheses(
        self,
        filtered_causes: dict[str, dict[str, Any]],
        representative_rep: BaseModel,
        anthro: dict,
        rom: dict,
        set_summary: SetScoreSummary | None,
    ) -> list[HypothesizedCause]:
        hypotheses = []

        for cause_id, info in filtered_causes.items():
            cause_def = self._graph.causes[cause_id]
            tier = cause_def["tier"]
            evidence_fn = cause_def["evidence_test_fn"]
            evidence_score = evidence_fn(
                representative_rep, anthro, rom, set_summary
            )

            parameter_delta = None
            if tier == 1 and cause_def["parameter_delta_fn"] is not None:
                delta_fn = cause_def["parameter_delta_fn"]
                parameter_delta = delta_fn(representative_rep, anthro, rom)

            explanation = self._fill_template(
                cause_def["explanation_template"],
                representative_rep,
                anthro,
                rom,
                set_summary,
            )

            avg_prior = (
                sum(info["posteriors"]) / len(info["posteriors"])
                if info["posteriors"]
                else 0.0
            )

            hypotheses.append(
                HypothesizedCause(
                    cause_id=cause_id,
                    tier=tier,
                    score=info["aggregate_score"],
                    evidence_score=evidence_score,
                    prior=avg_prior,
                    implicated_by=info["implicated_by"],
                    parameter_delta=parameter_delta,
                    explanation=explanation,
                    observability=info["observability"],
                )
            )

        hypotheses.sort(key=lambda h: (-h.tier, -h.score))
        return hypotheses

    def _merge_perturbations(
        self, hypotheses: list[HypothesizedCause]
    ) -> dict:
        combined: dict[str, float] = {}
        foot_target_deltas: list[list[float]] = []

        for hypothesis in hypotheses:
            if hypothesis.tier != 1 or hypothesis.parameter_delta is None:
                continue

            for key, value in hypothesis.parameter_delta.items():
                if key == "__foot_target_delta":
                    foot_target_deltas.append(value)
                elif isinstance(value, str):
                    combined[key] = value
                else:
                    combined[key] = combined.get(key, 0.0) + value

        if foot_target_deltas:
            merged_foot = [0.0] * 6
            for delta in foot_target_deltas:
                for i in range(6):
                    merged_foot[i] += delta[i]
            combined["__foot_target_delta"] = merged_foot

        return combined

    def _compute_confidence(
        self,
        reps: list[BaseModel],
        detected_symptoms: list[DetectedSymptom],
    ) -> float:
        """How well the set was measured, 0-1 — not how bad it was.

        Below 0.5 the coach should hedge: too few reps to confirm a pattern,
        core measurements missing (feet or knees never seen), or findings
        resting on monocular depth regression. A clean verdict on a set whose
        knees were never measured is not a confident one.
        """
        rep_factor = min(1.0, len(reps) / MIN_REPS_FOR_FULL_CONFIDENCE)
        measured = [
            any(math.isfinite(self._extract_feature(rep, measure)) for rep in reps)
            for measure in self._graph.core_measures
        ]
        coverage = sum(measured) / len(measured)
        if not detected_symptoms:
            return round(rep_factor * coverage, 3)
        weights = [OBSERVABILITY_WEIGHT.get(s.observability, 0.5) for s in detected_symptoms]
        return round(rep_factor * coverage * sum(weights) / len(weights), 3)

    def _without_unseen_side_view(
        self, rep: BaseModel, capture_mode: str
    ) -> BaseModel:
        if measurement_observability("side_view", capture_mode) != NOT_OBSERVABLE:
            return rep
        return rep.model_copy(update={name: math.nan for name in self._graph.side_view_fields})

    def _aggregate_rep(
        self,
        reps: list[BaseModel],
        set_summary: SetScoreSummary | None,
    ) -> BaseModel:
        """The set's median rep: every numeric measure is its median over the
        reps (fatigue measures take the worst rep). Labelled with the worst rep's
        number so explanations still point at a real rep."""
        if len(reps) == 1:
            return reps[0]
        rep_model = self._graph.rep_model
        values: dict[str, Any] = {}
        for name, field in rep_model.model_fields.items():
            column = [getattr(rep, name) for rep in reps]
            if field.annotation is bool:
                values[name] = any(column)
            elif field.annotation is int:
                values[name] = int(round(float(np.median(column))))
            else:
                finite = [v for v in column if math.isfinite(v)]
                if not finite:
                    values[name] = math.nan
                elif name in self._graph.tail_fields:
                    values[name] = _tail_mean(finite)
                else:
                    values[name] = float(np.median(finite))
        worst = set_summary.worst_rep_number if set_summary is not None else reps[-1].rep_number
        values["rep_number"] = worst
        return rep_model(**values)

    def _extract_feature(
        self, rep: BaseModel, feature_name: str
    ) -> float:
        return self._graph.extract_feature(rep, feature_name)

    def _aggregate(self, values: list[float], method: str) -> float:
        values = [v for v in values if not math.isnan(v)]
        if not values:
            return 0.0
        if method == "median":
            return float(np.median(values))
        if method == "tail_mean":
            return _tail_mean(values)
        if method == "max":
            return max(values)
        elif method == "mean":
            return sum(values) / len(values)
        elif method == "last":
            return values[-1]
        return sum(values) / len(values)

    def _compute_severity(
        self,
        value: float,
        expected: float,
        threshold: float,
        scoring: str,
    ) -> float:
        excess = value - expected
        if scoring == "relative_excess":
            if excess <= threshold:
                return 0.0
            scale = threshold * 2.0 if threshold > 0 else 10.0
            return min(1.0, (excess - threshold) / scale)
        elif scoring == "zscore":
            stdev_estimate = max(threshold, 1.0)
            z = excess / stdev_estimate
            if z <= 1.0:
                return 0.0
            return min(1.0, (z - 1.0) / 2.0)
        return 0.0

    def _fill_template(
        self,
        template: str,
        rep: BaseModel,
        anthro: dict,
        rom: dict,
        set_summary: SetScoreSummary | None,
    ) -> str:
        fill_values = self._graph.fill_values(rep, anthro, rom, set_summary)
        try:
            return template.format(**fill_values)
        except (KeyError, IndexError):
            return template


def _finite_or(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


def _nanmax(first: float, second: float) -> float:
    finite = [value for value in (first, second) if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _tail_mean(values: list[float]) -> float:
    tail = values[-TAIL_REPS:]
    return sum(tail) / len(tail)
