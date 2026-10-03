"""Squat delivery golden master: a scripted two-set squat workout through the
coaching service and orchestrator must produce exactly the recorded cues,
prompts and set summaries. Pins the squat while the delivery side grows
other exercises.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.coaching_orchestrator import CoachingOrchestrator  # noqa: E402
from agent.services.coaching_service import CoachingService  # noqa: E402
from biomechanics.coaching.cue_cache import FAULT_TO_CUE_MAP, SQUAT_CUES  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "squat_delivery_golden.json"


class _StubState:
    def __init__(self, values: dict) -> None:
        self.values = values

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value

    def save_state(self) -> None:
        pass


class _SilentCueAudio:
    session = object()
    rep_track_ready = False

    async def cache_cues(self, cues: dict) -> None:
        pass


def _fault(fault_type: str, severity: str, side: str | None, observability: str = "observable") -> dict:
    return {
        "fault_type": fault_type, "severity": severity,
        "details": {"side": side, "observability": observability},
    }


def _rep(rep_number: int, depth_angle: float, ascent_s: float, detailed: list[dict]) -> dict:
    return {
        "type": "rep_complete", "rep_number": rep_number, "set_number": 1,
        "is_clean": not detailed, "depth_category": "parallel",
        "faults_in_rep": list(dict.fromkeys(f["fault_type"] for f in detailed)),
        "faults_detailed": detailed, "max_depth_angle": depth_angle,
        "rep_duration_ms": 2000, "ascent_time_s": ascent_s,
    }


def _build_service(prompts: list[dict]) -> tuple[CoachingService, CoachingOrchestrator]:
    from agent.core.workout_session import WorkoutSession

    session = WorkoutSession.create_quick_session(
        "u1", "Barbell Back Squat", sets=2, reps=4, weight=60.0, rest_seconds=90, weight_unit="kg",
    )
    state = _StubState({
        "user.id": "u1",
        "workout.current_session": session.to_dict(),
        "workout.exercise_name": "Barbell Back Squat",
    })
    service = CoachingService(session=None, state=state)

    async def _reply(instructions: str, last_set_note: str | None = None):
        prompts.append({"instructions": instructions, "last_set_note": last_set_note})

    service._coaching_llm_reply = _reply
    with patch.object(CoachingOrchestrator, "start"):
        service._init_orchestrator()
    orchestrator = service._coaching_orchestrator
    orchestrator._play_cached = AsyncMock()
    orchestrator._get_cue_audio = lambda key: bool(key)
    orchestrator._side_view_observable = False
    orchestrator._min_fault_cue_gap = 0.0
    orchestrator._diagnosis_wait_s = 0.0
    service._audio_cue_service = _SilentCueAudio()
    service._workout_active_flag = True
    return service, orchestrator


async def _play_queue(orchestrator: CoachingOrchestrator, heard: list[str]) -> list[dict]:
    """Play every queued cue and recap the way the dispatch loop would."""
    set_data: list[dict] = []
    while not orchestrator._queue.empty():
        event = orchestrator._queue.get_nowait()
        if event.event_type == "cached_cue":
            heard.append(event.cue_key)
            await orchestrator._dispatch_cached_cue(event)
        elif event.event_type == "llm_set_recap":
            set_data.append({k: v for k, v in event.data.items() if k != "_report"})
            await orchestrator._speak_llm_set_recap(event.data)
        elif event.event_type == "llm_exercise_recap":
            await orchestrator._speak_llm_exercise_recap(event.data)
    return set_data


def run_squat_scenario() -> dict:
    prompts: list[dict] = []
    heard: list[str] = []

    async def _run() -> dict:
        service, orchestrator = _build_service(prompts)
        handle = service._handle_message
        await handle({
            "type": "cache_cues", "exercise_name": "Barbell Back Squat", "profile": "squat",
            "cues": dict(SQUAT_CUES), "fault_to_cue": dict(FAULT_TO_CUE_MAP), "reports_depth": True,
        })
        set_data: list[dict] = []

        # Set 1: knee cave cued at rep 1, fixed over reps 2-3, then a
        # rate-limited approximate hip shift and a mild heel rise.
        await handle({
            "type": "fault", "fault_type": "knee_valgus", "severity": "moderate",
            "cue": "knees_out_left", "rep_number": 1, "side": "left", "observability": "observable",
        })
        await handle(_rep(1, 95.2, 0.90, [_fault("knee_valgus", "moderate", "left")]))
        set_data += await _play_queue(orchestrator, heard)
        await handle(_rep(2, 96.0, 0.95, []))
        set_data += await _play_queue(orchestrator, heard)
        await handle(_rep(3, 94.1, 1.05, []))
        set_data += await _play_queue(orchestrator, heard)
        await handle({
            "type": "fault", "fault_type": "heel_rise", "severity": "mild",
            "cue": None, "rep_number": 4, "side": "right", "observability": "observable",
        })
        await handle(_rep(4, 92.7, 1.30, [
            _fault("hip_shift", "moderate", "left", observability="approximate"),
            _fault("heel_rise", "mild", "right"),
        ]))
        verdict = service.last_rep_verdict()
        set_data += await _play_queue(orchestrator, heard)
        fragments = orchestrator.state_fragments()
        rest_prompt = service._build_rest_complete_prompt()
        orchestrator._set_focus_fault = "knee_valgus"
        focus_prompt = service._build_rest_complete_prompt()
        orchestrator._set_focus_fault = None
        await handle({"type": "rest_complete"})

        # Set 2: a side-view fault on one camera, knee cave only in the
        # rep's own verdicts (its message was rate-limited), then it persists.
        await handle({
            "type": "fault", "fault_type": "velocity_loss", "severity": "severe",
            "cue": "drive", "rep_number": 1, "side": None, "observability": "observable",
        })
        await handle(_rep(1, 93.0, 0.85, [
            _fault("velocity_loss", "severe", None), _fault("knee_valgus", "severe", "right"),
        ]))
        set_data += await _play_queue(orchestrator, heard)
        await handle(_rep(2, 91.5, 1.00, [_fault("knee_valgus", "moderate", "right")]))
        set_data += await _play_queue(orchestrator, heard)
        await handle(_rep(3, 94.0, 1.10, []))
        set_data += await _play_queue(orchestrator, heard)
        await handle(_rep(4, 95.5, 1.40, []))
        set_data += await _play_queue(orchestrator, heard)
        orchestrator.stop()
        return {
            "heard": heard,
            "prompts": prompts,
            "set_data": set_data,
            "last_set_data": {k: v for k, v in orchestrator.last_set_data.items() if k != "_report"},
            "verdict": verdict,
            "fragments": fragments,
            "rest_prompt": rest_prompt,
            "focus_prompt": focus_prompt,
        }

    return json.loads(json.dumps(asyncio.run(_run()), default=str))


class TestSquatDeliveryGolden:
    def test_squat_workout_matches_the_golden_master(self) -> None:
        assert run_squat_scenario() == json.loads(GOLDEN_PATH.read_text())
