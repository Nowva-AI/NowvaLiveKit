"""Speech affect perception runtime: encoder inference, per-user baselines, athlete state, voice style.

Pure numpy, onnxruntime and soxr. No livekit imports; the agent glue lives in
src/agent/services/affect_service.py and src/agent/agents/shared/affect_mixin.py.
"""

from __future__ import annotations
