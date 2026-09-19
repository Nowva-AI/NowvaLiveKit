"""Deterministic ground-truth evaluation harness for the pre-IK filter chain on triangulated 3-camera input.

Usage (from the harness folder, repo src on sys.path):
    from preik_harness import evaluate, evaluate_chains, baseline_factories, production_chain, SCENARIOS
    result = evaluate(production_chain(["blend", "vclamp"]), scenarios=["clean", "valgus"], calibration="tpose")
"""

from __future__ import annotations

import logging
import sys

REPO_SRC = "/Users/naiahoard/NowvaLiveKit/src"
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

# the production gates/filters log every 30 failing frames; silence them for batch evaluation
logging.getLogger("biomechanics").setLevel(logging.ERROR)

from .api import CALIBRATION_MODES, compare, evaluate, evaluate_chains, get_prepared  # noqa: E402
from .delivery import DeliveryConfig  # noqa: E402
from .cameras import PERSON_BA_MODES, PersonBAConfig, RigConfig  # noqa: E402
from .detector import PROFILES, NoiseProfile  # noqa: E402
from .body import SCENARIOS  # noqa: E402
from .chains import (BASELINE_CHAINS, DiagnosticConfidenceFloor, ProductionChain, baseline_factories,  # noqa: E402
                     production_chain)
from .runner import FrameContext  # noqa: E402
from .proposed import ProposedChain, proposed_chain  # noqa: E402

__all__ = ["CALIBRATION_MODES", "DeliveryConfig", "RigConfig", "PERSON_BA_MODES", "PersonBAConfig", "PROFILES", "NoiseProfile", "evaluate", "evaluate_chains", "compare", "get_prepared", "SCENARIOS", "BASELINE_CHAINS",
           "ProductionChain", "DiagnosticConfidenceFloor", "baseline_factories", "production_chain", "FrameContext", "ProposedChain", "proposed_chain"]
