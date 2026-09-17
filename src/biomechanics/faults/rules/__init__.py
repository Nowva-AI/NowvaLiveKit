"""
Fault Detection Rules

Individual rule implementations for detecting specific form faults.
"""

from biomechanics.faults.rules.depth import DepthRule, DepthCategory
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.faults.rules.forward_lean import ForwardLeanRule
from biomechanics.faults.rules.knee_valgus import KneeValgusRule
from biomechanics.faults.rules.tempo import EccentricTempoRule, StallingRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.back_rounding import BackRoundingRule
from biomechanics.faults.rules.lockout import LockoutRule
from biomechanics.faults.rules.elbow_flare import ElbowFlareRule
from biomechanics.faults.rules.bar_path import BarPathRule
from biomechanics.faults.rules.shoulder_stability import ShoulderStabilityRule
from biomechanics.faults.rules.trunk_stability import TrunkStabilityRule
from biomechanics.faults.rules.heel_rise import HeelRiseRule

__all__ = [
    "DepthRule",
    "DepthCategory",
    "SymmetryRule",
    "ForwardLeanRule",
    "KneeValgusRule",
    "EccentricTempoRule",
    "StallingRule",
    "RangeOfMotionRule",
    "BackRoundingRule",
    "LockoutRule",
    "ElbowFlareRule",
    "BarPathRule",
    "ShoulderStabilityRule",
    "TrunkStabilityRule",
    "HeelRiseRule",
]
