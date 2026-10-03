"""Faults for an RDL attempt stopped above the count floor (rdl.md §3.6, §5.7).

Not a rule: profile.on_rep_rejected calls build_rejected_faults for every
"partial" excursion, and it reads the thresholds the analyser stamped into the
features. Every partial yields exactly one shallow_rep fault: rdl_hip_sink when
the hips sank into a cueable squat pattern, rdl_partial otherwise.
"""

from __future__ import annotations

from biomechanics.analysis.hinge_features import HingeRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.observability import APPROXIMATE
from biomechanics.faults.rules.rdl_hip_sink import RdlHipSinkRule
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles

_SEVERITY_RANK = {
    FaultSeverity.NONE: 0,
    FaultSeverity.MILD: 1,
    FaultSeverity.MODERATE: 2,
    FaultSeverity.SEVERE: 3,
}


class _PartialVerdict(RepFaultRule):
    """Hand travel short of the count floor; any shortfall is mild."""

    @property
    def fault_type(self) -> FaultType:
        return FaultType.RDL_PARTIAL

    def judge_rep(self, features: HingeRepFeatures, reference: None, angles: JointAngles) -> FaultEvent | None:
        moderate, severe = features.partial_bands
        return self._rep_fault(
            features.count_floor_travel - features.bar_travel_ratio,
            {"mild": 0.0, "moderate": moderate, "severe": severe},
            angles,
            features.rep_number,
            unit="ratio",
            phase="bottom",
            bar_travel_ratio=features.bar_travel_ratio,
        )


def _cueable_hip_sink(features: HingeRepFeatures, angles: JointAngles) -> FaultEvent | None:
    mild, moderate, severe = features.hip_sink_bands
    fault = RdlHipSinkRule(SeverityTiers(mild=mild, moderate=moderate, severe=severe)).judge_rep(
        features, None, angles,
    )
    if fault is None or fault.details.get("observability") == APPROXIMATE:
        return None
    # At or above moderate, and never below the session's start tier: the
    # shallow cue is chosen before the pipeline stamps recorded-only verdicts.
    floor = max(_SEVERITY_RANK[FaultSeverity.MODERATE], _SEVERITY_RANK[FaultSeverity(features.hip_sink_start_tier)])
    return fault if _SEVERITY_RANK[fault.severity] >= floor else None


def build_rejected_faults(features: HingeRepFeatures, angles: JointAngles) -> list[FaultEvent]:
    """The one shallow_rep fault of a partial attempt (never counted)."""
    fault = _cueable_hip_sink(features, angles) or _PartialVerdict().judge_rep(features, None, angles)
    if fault is None:
        return []
    fault.details.update(shallow_rep=True, counted=False, range_label=features.range_label)
    return [fault]
