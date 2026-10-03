"""The one filter every agent-side reader of rep verdicts goes through.

A recorded-only verdict sits below its fault's cue start tier: it is stored for
the record but never cued, counted or stated. An approximate verdict is hard to
judge from the current camera setup, so spoken readers drop it unless they hedge it.
Verdicts come as faults_detailed entries (flags under "details") or as fault
messages (flags at the top level); both shapes are read.
"""

from __future__ import annotations

from typing import Any

from biomechanics.faults.observability import APPROXIMATE


def _verdict_flag(fault: dict[str, Any], key: str) -> Any:
    details = fault.get("details") or {}
    return details[key] if key in details else fault.get(key)


def is_recorded_only(fault: dict[str, Any]) -> bool:
    return bool(_verdict_flag(fault, "recorded_only"))


def is_approximate(fault: dict[str, Any]) -> bool:
    return _verdict_flag(fault, "observability") == APPROXIMATE


def reportable_faults(faults: list[dict[str, Any]], spoken: bool) -> list[dict[str, Any]]:
    """The verdicts a reader may count or state. Recorded-only verdicts always drop
    out. With spoken=True approximate ones drop out too; a reader that hedges them
    passes spoken=False and hedges each one it keeps."""
    return [
        fault for fault in faults
        if not is_recorded_only(fault) and not (spoken and is_approximate(fault))
    ]
