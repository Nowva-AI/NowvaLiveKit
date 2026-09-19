"""
NaN/inf sanitization for payloads that are serialized to JSON (IPC, files).

Missing joint angles are NaN inside the pipeline (C9); JSON has no NaN, so
every serialization boundary converts non-finite floats to None.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def nan_to_none(value: Any) -> Any:
    """Recursively replace NaN/inf floats with None in nested dicts, lists and tuples."""
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [nan_to_none(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: nan_to_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [nan_to_none(item) for item in value]
    return value
