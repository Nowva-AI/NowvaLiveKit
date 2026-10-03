"""
Squat delivery golden master (deadlift PLAN §5, milestone J0): the IPC streams
the squat golden sets recorded (tests/test_biomechanics/fixtures/squat_golden)
replayed through the real CoachingService and CoachingOrchestrator, with fake
TTS, LLM and database on the pipeline's frame clock. Snapshots the cues played,
the LLM prompts (recap text), the recorder calls and the messages sent back to
the pipeline. Regenerate with NOWVA_UPDATE_GOLDENS=1 and review the diff.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.workout_session import WorkoutSession
from agent.services import coaching_orchestrator as orchestrator_module
from agent.services import set_report
from agent.services.coaching_constants import COACHING_PERSONA, CUE_TEXT_MAP
from agent.services.coaching_orchestrator import SET_IDLE_TIMEOUT_S
from agent.services.coaching_service import CoachingService

STREAM_DIR = Path(__file__).parent / "test_biomechanics" / "fixtures" / "squat_golden"
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "squat_delivery_golden"
UPDATE_GOLDENS_ENV = "NOWVA_UPDATE_GOLDENS"
SQUAT_EXERCISE = "Barbell Back Squat"
SCENARIOS = (
    "clean", "knee_valgus", "hip_shoot", "shallow_descent", "shallow_descent_no_depth_target", "dropout",
)
VARIANTS = ("stored_params", "first_time")

# The pipeline golden's capture clock: 30 fps from this instant.
FRAME_DT_S = 1.0 / 30.0
CLOCK_START_S = 1.7e9
# A quick bodyweight squat workout whose first set is the recorded one.
USER_ID = "golden-athlete"
WORKOUT_SETS = 2
TARGET_REPS = 3
REST_SECONDS = 90
LLM_REPLY_TEXT = "Solid set, keep it up."
# Event-loop turns after each message: the queue processor dispatches and the
# fire-and-forget cue tasks run before the next message arrives.
SETTLE_YIELDS = 50
RECORDED_MESSAGE_FIELDS = (
    "type", "fault_type", "severity", "rep_number", "set_number", "cue", "is_clean", "faults_in_rep",
)

FLOAT_DECIMALS = 6
FLOAT_TOLERANCE = 1.5e-6
MAX_REPORTED_DIFFERENCES = 40


class _FakeClock:
    """time.time() and time.monotonic() for the orchestrator, on the stream's frame clock."""

    def __init__(self) -> None:
        self.now_s = CLOCK_START_S

    def at_frame(self, frame: int) -> None:
        self.now_s = CLOCK_START_S + frame * FRAME_DT_S

    def advance(self, seconds: float) -> None:
        self.now_s += seconds

    def time(self) -> float:
        return self.now_s

    def monotonic(self) -> float:
        return self.now_s

    def elapsed_s(self) -> float:
        return _normalise(self.now_s - CLOCK_START_S)


class _FakeAudioCueService:
    """Fake TTS: a clip for every cue key with text, as shipped; records what plays."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self.session = object()
        self.rep_track_ready = False
        self.cached: list[str] = []
        self.played: list[dict] = []

    async def cache_cues(self, cues: dict) -> None:
        self.cached = list(cues)

    def has_cue(self, cue_key: str) -> bool:
        return cue_key in CUE_TEXT_MAP

    async def play_cue(self, cue_key: str) -> None:
        self.played.append({"t_s": self._clock.elapsed_s(), "cue": cue_key})


class _FakeSpeechHandle:
    async def wait_for_playout(self) -> None:
        return None


class _FakeAgent:
    def __init__(self) -> None:
        from livekit.agents import llm

        self.chat_ctx = llm.ChatContext.empty()
        self.chat_ctx.items.append(llm.ChatMessage(role="system", content=["You are Nova."]))
        self.userdata = SimpleNamespace(compaction_service=None)

    async def update_chat_ctx(self, chat_ctx) -> None:
        self.chat_ctx = chat_ctx


class _FakeLLMSession:
    """Fake LLM: every generate_reply speaks one fixed line; the prompts are recorded."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self.current_agent = _FakeAgent()
        self.userdata = SimpleNamespace(affect_service=None)
        self.prompts: list[dict] = []

    def generate_reply(self, *, instructions=None, user_input=None, tool_choice=None, allow_interruptions=None):
        from livekit.agents import llm

        self.prompts.append({
            "t_s": self._clock.elapsed_s(),
            "persona": instructions == COACHING_PERSONA,
            "user_input": user_input,
        })
        items = self.current_agent.chat_ctx.items
        items.append(llm.ChatMessage(role="user", content=[user_input]))
        items.append(llm.ChatMessage(role="assistant", content=[LLM_REPLY_TEXT]))
        return _FakeSpeechHandle()


class _RecordingRecorder:
    """Fake database: the BiomechanicsRecorder calls, with each message's identifying fields."""

    exercise = "squat"

    def __init__(self) -> None:
        self.ops: list[dict] = []

    def _record(self, op: str, message: dict) -> None:
        self.ops.append({"op": op, **{key: message[key] for key in RECORDED_MESSAGE_FIELDS if key in message}})

    def record_fault(self, message: dict) -> None:
        self._record("record_fault", message)

    def record_rep(self, message: dict) -> None:
        self._record("record_rep", message)

    def record_rep_diagnosis(self, message: dict) -> None:
        self._record("record_rep_diagnosis", message)

    def record_set(self, message: dict) -> None:
        self._record("record_set", message)

    def record_cue_delivered(self, fault_type: str, cue_key: str | None = None) -> None:
        self.ops.append({"op": "record_cue_delivered", "fault_type": fault_type, "cue_key": cue_key})


class _PipelineIPC:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(copy.deepcopy(message))


class _StubState:
    def __init__(self, values: dict) -> None:
        self.values = values

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value

    def save_state(self) -> None:
        pass


def _normalise(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = round(value, FLOAT_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _workout_state() -> _StubState:
    session = WorkoutSession.create_quick_session(
        USER_ID, SQUAT_EXERCISE, sets=WORKOUT_SETS, reps=TARGET_REPS, weight=0.0,
        rest_seconds=REST_SECONDS, weight_unit="kg",
    )
    return _StubState({
        "user.id": USER_ID,
        "workout.exercise_name": SQUAT_EXERCISE,
        "workout.current_session": session.to_dict(),
    })


async def _settle() -> None:
    for _ in range(SETTLE_YIELDS):
        await asyncio.sleep(0)


def _chat_texts(agent: _FakeAgent) -> list[str]:
    return [item.text_content or "" for item in agent.chat_ctx.items]


# A workout's first squat set, as the voice agent hears the pipeline.
async def _replay(stream: list[dict], clock: _FakeClock) -> dict:
    llm_session = _FakeLLMSession(clock)
    audio = _FakeAudioCueService(clock)
    service = CoachingService(session=llm_session, state=_workout_state(), audio_cue_service=audio)
    pipeline_ipc = _PipelineIPC()
    service._coaching_ipc = pipeline_ipc
    service._init_orchestrator()
    orchestrator = service._coaching_orchestrator
    # What _start_biomech_recording wires once the workout is active, minus the database.
    recorder = _RecordingRecorder()
    service._biomech_recorder = recorder
    orchestrator.on_fault_cue_delivered = recorder.record_cue_delivered
    service._workout_active_flag = True

    dropped: list[dict] = []
    for entry in stream:
        clock.at_frame(entry["frame"])
        try:
            await service._handle_message(copy.deepcopy(entry["message"]))
        except Exception as error:
            # The listener hands messages over with run_coroutine_threadsafe and
            # never awaits the future: in production an exception drops the message.
            dropped.append({
                "t_s": clock.elapsed_s(), "type": entry["message"]["type"],
                "error": f"{type(error).__name__}: {error}",
            })
        await _settle()

    # A set short of its target ends on the orchestrator's idle timer.
    clock.advance(SET_IDLE_TIMEOUT_S)
    orchestrator._set_idle_timeout_s = 0.0
    orchestrator._arm_idle_timer()
    await _settle()

    snapshot = {
        "cues_played": audio.played,
        "llm_prompts": llm_session.prompts,
        "chat_context": _chat_texts(llm_session.current_agent),
        "recorder_ops": recorder.ops,
        "to_pipeline": pipeline_ipc.messages,
        "dropped_messages": dropped,
        "last_set":{key: value for key, value in (orchestrator.last_set_data or {}).items() if key != "_report"},
        "workout_state_line": service.workout_state_line(),
    }
    orchestrator.stop()
    return _normalise(snapshot)


# Pretty JSON with sorted keys; lists of scalars stay on one line.
def _format_json(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{pad} {json.dumps(key)}: {_format_json(value[key], indent + 1)}" for key in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [f"{pad} {_format_json(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value)


def _differences(expected: Any, actual: Any, path: str = "$") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        found: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                found.append(f"{path}.{key}: missing")
            elif key not in expected:
                found.append(f"{path}.{key}: unexpected {json.dumps(actual[key])[:160]}")
            else:
                found.extend(_differences(expected[key], actual[key], f"{path}.{key}"))
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        found = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            found.extend(_differences(expected_item, actual_item, f"{path}[{index}]"))
        if len(expected) != len(actual):
            found.append(f"{path}: expected {len(expected)} items, got {len(actual)}")
        return found
    both_numbers = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in (expected, actual)
    )
    if both_numbers:
        return [] if abs(expected - actual) <= FLOAT_TOLERANCE else [f"{path}: expected {expected}, got {actual}"]
    if expected == actual:
        return []
    return [f"{path}:\n    expected {json.dumps(expected)[:400]}\n    got      {json.dumps(actual)[:400]}"]


def _stream(scenario: str, variant: str) -> list[dict]:
    path = STREAM_DIR / f"{scenario}__{variant}.json"
    if not path.exists():
        pytest.fail(f"Missing squat golden {path.name}: regenerate tests/test_biomechanics/test_squat_golden.py first")
    return json.loads(path.read_text())["ipc"]


def _golden(scenario: str, variant: str) -> dict:
    path = GOLDEN_DIR / f"{scenario}__{variant}.json"
    if not path.exists():
        pytest.fail(f"Missing golden {path.name}: run with {UPDATE_GOLDENS_ENV}=1 to create it")
    return json.loads(path.read_text())


@pytest.fixture
def delivery_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """The orchestrator's clock, a 3-camera rig, and no set-report PNGs."""
    clock = _FakeClock()
    monkeypatch.setattr(orchestrator_module, "time", clock)
    monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
    monkeypatch.delenv("NOWVA_PROFILE", raising=False)
    monkeypatch.setattr(set_report, "generate_set_report", lambda **kwargs: None)
    return clock


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scenario", SCENARIOS)
class TestSquatDeliveryGolden:
    def test_squat_delivery_matches_golden(self, delivery_clock: _FakeClock, scenario: str, variant: str):
        snapshot = asyncio.run(_replay(_stream(scenario, variant), delivery_clock))
        path = GOLDEN_DIR / f"{scenario}__{variant}.json"
        if os.environ.get(UPDATE_GOLDENS_ENV) == "1":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_format_json(snapshot) + "\n")
            return
        differences = _differences(_golden(scenario, variant), snapshot)
        if differences:
            shown = differences[:MAX_REPORTED_DIFFERENCES]
            pytest.fail("\n".join([
                f"Squat delivery differs from {path.name} ({len(differences)} differences). If intended "
                f"(also after regenerating the pipeline golden), regenerate with {UPDATE_GOLDENS_ENV}=1.",
                *shown,
            ]))


class TestSquatDeliveryScenarios:
    """A regenerated delivery golden must still say what each set calls for."""

    def test_clean_set_counts_three_reps_and_cues_no_fault(self):
        golden = _golden("clean", "stored_params")
        assert [cue["cue"] for cue in golden["cues_played"]] == ["rep_1", "rep_2", "rep_3"]
        assert not [op for op in golden["recorder_ops"] if op["op"] == "record_cue_delivered"]

    def test_knee_cave_is_cued_with_knees_out(self):
        golden = _golden("knee_valgus", "stored_params")
        assert "knees_out" in [cue["cue"] for cue in golden["cues_played"]]
        assert {"op": "record_cue_delivered", "fault_type": "knee_valgus", "cue_key": "knees_out"} in golden["recorder_ops"]

    def test_hip_shoot_is_cued_with_chest_up_on_the_rig(self):
        golden = _golden("hip_shoot", "stored_params")
        assert "chest_up" in [cue["cue"] for cue in golden["cues_played"]]

    def test_shallow_rep_is_dropped_before_dispatch_today(self):
        """Known bug, pinned rather than fixed in J0: shallow_rep carries the depth
        fault's "category", which collides with SessionProfiler.record(category, ...)
        in CoachingService._handle_message, outside its try. The athlete never hears
        "deeper" and no row is written. The fix changes this golden on purpose."""
        golden = _golden("shallow_descent", "stored_params")
        assert [message["type"] for message in golden["dropped_messages"]] == ["shallow_rep"]
        assert "deeper" not in [cue["cue"] for cue in golden["cues_played"]]

    def test_every_set_gets_one_recap_and_a_rest(self):
        for scenario in SCENARIOS:
            for variant in VARIANTS:
                golden = _golden(scenario, variant)
                assert len(golden["llm_prompts"]) == 1, (scenario, variant)
                assert golden["to_pipeline"] == [{"type": "rest_start", "rest_seconds": REST_SECONDS}], (scenario, variant)
