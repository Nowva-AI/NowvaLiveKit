"""Import before `harness`: the pre-IK overhaul deletes the old filters that smoothing/harness.py imports at module
level (JointAngleFilter, DerivativeTracker, PredictiveStateEstimator, VelocityClamp). WS4 only uses harness.make_data,
fast_eval and summarize, which do not touch them, so missing names are replaced by placeholder classes."""
from __future__ import annotations

import importlib
import sys
import types

REQUIRED_NAMES = {
    "biomechanics.utils.filters": ("JointAngleFilter",),
    "biomechanics.utils.derivatives": ("DerivativeTracker",),
    "biomechanics.utils.predictive_state": ("PredictiveStateEstimator",),
    "biomechanics.utils.velocity_clamp": ("VelocityClamp",),
}

for module_name, names in REQUIRED_NAMES.items():
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    for name in names:
        if not hasattr(module, name):
            setattr(module, name, type(name, (), {}))
