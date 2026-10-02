"""Tests for CoachingService shutdown behavior."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import AsyncMock, patch

from agent.services.assessment_logger import AssessmentLogger
from agent.services.coaching_orchestrator import CoachingOrchestrator
from agent.services.coaching_service import CoachingService


class TestStopFinalizesAssessment:
    def test_stop_finalizes_active_assessment_log(self, tmp_path):
        service = CoachingService(session=None, state=None)
        service._started = True
        service._assessment_logger = AssessmentLogger(
            session_dir=tmp_path,
            session_id="test_session",
            user_height_cm=180.0,
        )

        asyncio.run(service.stop())

        log = json.loads((tmp_path / "assessment" / "assessment_log.json").read_text())
        assert log["completed_at"] is not None
        assert log["passed"] is False
        assert service._assessment_logger is None

    def test_stop_without_assessment_is_clean(self, tmp_path):
        service = CoachingService(session=None, state=None)
        service._started = True

        asyncio.run(service.stop())

        assert service._assessment_logger is None
        assert not (tmp_path / "assessment").exists()


def _wait_for(condition_fn, timeout_s: float = 5.0) -> bool:
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if condition_fn():
            return True
        time.sleep(0.02)
    return False


def _serve_one_client(server) -> threading.Thread:
    def _serve():
        try:
            server.accept_client()
            server.listen()
        except OSError:
            pass
    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return thread


class TestListenerReconnect:
    def test_listener_reconnects_after_server_restart(self, monkeypatch):
        """main.py stops the coaching socket between workout passes — the
        listener must reconnect when the server rebinds, not die silently."""
        import tempfile
        import agent.services.coaching_service as cs
        from agent.core.ipc_communication import IPCServer

        socket_path = tempfile.mkdtemp() + "/coaching.sock"
        monkeypatch.setattr(cs, "COACHING_SOCKET_PATH", socket_path)
        monkeypatch.setattr(cs, "LISTENER_RECONNECT_POLL_S", 0.05)

        service = CoachingService(session=None, state=None)

        server = IPCServer(socket_path=socket_path)
        server.bind()
        _serve_one_client(server)

        asyncio.run(service._start_ipc_listener())
        assert _wait_for(lambda: service._listener_running)

        # Workout ends: main.py stops the coaching server
        server.stop()
        assert _wait_for(lambda: not service._listener_running)

        # Next workout: main.py rebinds — the listener must come back
        server = IPCServer(socket_path=socket_path)
        server.bind()
        _serve_one_client(server)
        assert _wait_for(lambda: service._listener_running)

        service._stop_ipc_listener()
        server.stop()

    def test_start_assessment_resets_demo_latch(self, tmp_path):
        service = CoachingService(session=None, state=None)
        service._assessment_demo_played = True

        service.start_assessment(
            session_dir=tmp_path, session_id="s", user_height_cm=180.0,
        )

        assert service._assessment_demo_played is False


class _StubState:
    def __init__(self, user_id: str) -> None:
        self.values: dict = {"user.id": user_id}

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value

    def save_state(self) -> None:
        pass


class TestCalibrationCompletePersistence:
    def test_calibration_without_athlete_params_keeps_stored_params(self, monkeypatch):
        """A calibration_complete that lost its body measurements must not
        erase the stored ones — returning users would lose diagnosis."""
        import types
        import uuid
        from db.models import UserCalibration

        stored_params = {
            "shoulder_width_m": 0.39, "femur_avg_m": 0.46, "torso_avg_m": 0.55,
            "hip_width_m": 0.27, "tibia_avg_m": 0.44, "foot_avg_m": 0.21,
        }
        user_id = uuid.uuid4()
        row = UserCalibration(
            user_id=user_id, movement_pattern="squat", peaks={}, thresholds={},
            calibration_reps=5, athlete_params=dict(stored_params),
            baseline={"peakDorsi": 33.0, "peakKneeFlex": 118.0},
        )

        class _Query:
            def filter(self, *conditions):
                return self

            def first(self):
                return row

        class _Session:
            def query(self, model):
                return _Query()

            def add(self, new_row):
                raise AssertionError("existing row must be updated, not re-added")

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(
            sys.modules, "db.database", types.SimpleNamespace(SessionLocal=_Session),
        )
        service = CoachingService(session=None, state=_StubState(str(user_id)))

        async def _no_reply(instructions):
            return None

        monkeypatch.setattr(service, "_coaching_llm_reply", _no_reply)

        asyncio.run(service._on_calibration_complete({
            "type": "calibration_complete",
            "movement_pattern": "squat",
            "peaks": {"trunk_flexion": 41.0},
            "thresholds": {"knee_valgus": {"mild": 12.5}},
        }))

        assert row.athlete_params == stored_params
        assert row.baseline == {"peakDorsi": 33.0, "peakKneeFlex": 118.0}
        assert row.thresholds == {"knee_valgus": {"mild": 12.5}}


class _FakeRecorder:
    def __init__(self) -> None:
        self.faults: list[dict] = []
        self.reps: list[dict] = []
        self.sets: list[dict] = []

    def record_fault(self, message: dict) -> None:
        self.faults.append(message)

    def record_rep(self, message: dict) -> None:
        self.reps.append(message)

    def record_set(self, message: dict) -> None:
        self.sets.append(message)


def _wired_service() -> tuple[CoachingService, CoachingOrchestrator, _FakeRecorder]:
    service = CoachingService(session=None, state=None)
    orchestrator = CoachingOrchestrator(
        play_cached_audio_fn=AsyncMock(),
        generate_llm_reply_fn=AsyncMock(),
        get_cue_audio_fn=lambda key: bool(key),
    )
    orchestrator.reset_set(target_reps=10)
    recorder = _FakeRecorder()
    service._coaching_orchestrator = orchestrator
    service._biomech_recorder = recorder
    service._workout_active_flag = True
    return service, orchestrator, recorder


class TestPipelineMessageWiring:
    def test_approximate_fault_is_recorded_but_not_cued(self):
        service, orchestrator, recorder = _wired_service()
        message = {
            "type": "fault", "fault_type": "hip_shoot", "severity": "moderate",
            "cue": None, "rep_number": 2, "side": None, "observability": "approximate",
        }
        asyncio.run(service._handle_message(message))
        assert recorder.faults == [message]
        assert orchestrator._queue.empty()

    def test_side_cue_reaches_the_orchestrator_as_given(self):
        service, orchestrator, _ = _wired_service()

        async def _run():
            await service._handle_message({
                "type": "fault", "fault_type": "knee_valgus", "severity": "moderate",
                "cue": "knees_out_left", "rep_number": 2, "side": "left",
                "observability": "observable",
            })
            # The cue is decided once the rep is complete
            assert orchestrator._queue.empty()
            await service._handle_message({
                "type": "rep_complete", "rep_number": 2, "is_clean": False,
                "faults_in_rep": ["knee_valgus"],
            })
            orchestrator.stop()

        asyncio.run(_run())
        assert orchestrator._queue.get_nowait().cue_key == "knees_out_left"

    def test_fault_without_a_pipeline_cue_gets_its_side_cue(self):
        """The pipeline's own cue gap leaves cue empty; the agent decides."""
        service, orchestrator, _ = _wired_service()

        async def _run():
            await service._handle_message({
                "type": "fault", "fault_type": "knee_valgus", "severity": "moderate",
                "cue": None, "rep_number": 2, "side": "right",
                "observability": "observable",
            })
            await service._handle_message({
                "type": "rep_complete", "rep_number": 2, "is_clean": False,
                "faults_in_rep": ["knee_valgus"],
            })
            orchestrator.stop()

        asyncio.run(_run())
        assert orchestrator._queue.get_nowait().cue_key == "knees_out_right"

    def test_rep_complete_passes_highlights_and_set_number(self):
        service, orchestrator, _ = _wired_service()
        orchestrator.positive_cue_keys = ["strong"]
        asyncio.run(service._handle_message({
            "type": "rep_complete", "rep_number": 1, "is_clean": True,
            "faults_in_rep": [], "set_number": 3, "highlights": ["best_rep_so_far"],
        }))
        assert orchestrator._build_set_summary()["diagnosis_set_number"] == 3
        assert orchestrator._queue.get_nowait().cue_key == "strong"

    def test_diagnosis_complete_passes_its_set_number(self):
        service, orchestrator, _ = _wired_service()
        asyncio.run(service._handle_message({
            "type": "diagnosis_complete", "set_number": 3,
            "diagnosis": {
                "confidence": 0.8,
                "immediate_causes": [{"cause_id": "narrow_stance", "explanation": "x"}],
            },
            "scoring": {"mean_score": 0.7},
        }))
        assert orchestrator._pending_diagnosis_set_number == 3
        assert service.get_top_cause_id() == "narrow_stance"

    def test_diagnosis_before_the_workout_starts_is_dropped(self):
        """A diagnosis from before the workout (assessment, calibration) must
        not reach the first set's recap or pick its focus."""
        service, orchestrator, _ = _wired_service()
        service._workout_active_flag = False
        asyncio.run(service._handle_message({
            "type": "diagnosis_complete", "set_number": 1,
            "diagnosis": {"confidence": 0.8, "immediate_causes": []},
            "scoring": {"mean_score": 0.7},
        }))
        assert orchestrator._pending_diagnosis is None
        assert service.get_top_cause_id() is None


# =============================================================================
# Helpers for workout-state tests
# =============================================================================


def _quick_session_dict(sets: int = 3, reps: int = 5, weight: float = 0.0, unit: str = "kg") -> dict:
    from agent.core.workout_session import WorkoutSession

    session = WorkoutSession.create_quick_session(
        "u1", "Barbell Back Squat", sets=sets, reps=reps, weight=weight,
        rest_seconds=90, weight_unit=unit,
    )
    return session.to_dict()


def _workout_service(sets: int = 3, reps: int = 5, weight: float = 0.0, unit: str = "kg"):
    """A service with a live orchestrator wired to a WorkoutSession in state."""
    state = _StubState("u1")
    state.values["workout.current_session"] = _quick_session_dict(sets, reps, weight, unit)
    service = CoachingService(session=None, state=state)
    replies: list[str] = []

    async def _reply(instructions: str, last_set_note: str | None = None):
        replies.append(instructions)

    service._coaching_llm_reply = _reply
    # Real wiring, without the queue processor: tests drain the queue
    with patch.object(CoachingOrchestrator, "start"):
        service._init_orchestrator()
    orchestrator = service._coaching_orchestrator
    orchestrator._play_cached = AsyncMock()
    orchestrator._get_cue_audio = lambda key: bool(key)
    service._workout_active_flag = True
    return service, orchestrator, replies


def _rep_message(rep_number: int, faults: list[str] | None = None) -> dict:
    return {
        "type": "rep_complete", "rep_number": rep_number, "is_clean": not faults,
        "faults_in_rep": faults or [], "set_number": 1,
    }


def _queued(orchestrator) -> list:
    events = []
    while not orchestrator._queue.empty():
        events.append(orchestrator._queue.get_nowait())
    return events


class TestUnmeasuredRepDepth:
    def test_set_still_completes_when_a_rep_had_no_depth(self):
        """A rep with no frame showing both knees arrives with max_depth_angle
        None (NaN on the wire); the set must still close and get its recap."""
        async def _run():
            service, orchestrator, _ = _workout_service(sets=2, reps=2)
            await service._handle_message({**_rep_message(1), "max_depth_angle": None})
            await service._handle_message({**_rep_message(2), "max_depth_angle": 95.0})
            events = _queued(orchestrator)
            orchestrator.stop()
            return events

        assert [e.event_type for e in asyncio.run(_run())] == ["llm_set_recap"]


class TestRestCompleteAnnouncement:
    def test_reps_during_the_announcement_count(self):
        """The pipeline counts reps as soon as rest ends; the announcement
        used to swallow them."""
        async def _run():
            service, orchestrator, _ = _workout_service()
            orchestrator._resting = True

            async def _announce(instructions: str, last_set_note: str | None = None):
                await service._handle_message(_rep_message(1))

            service._coaching_llm_reply = _announce
            await service._handle_message({"type": "rest_complete"})
            return orchestrator

        orchestrator = asyncio.run(_run())
        assert orchestrator.set_rep_count == 1
        assert orchestrator.cues_suppressed is False
        orchestrator.stop()

    def test_no_cue_while_the_announcement_plays(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            orchestrator._resting = True
            queued = []

            async def _announce(instructions: str, last_set_note: str | None = None):
                await service._handle_message({
                    "type": "fault", "fault_type": "knee_valgus", "severity": "severe",
                    "cue": "knees_out", "observability": "observable",
                })
                await service._handle_message(_rep_message(1, ["knee_valgus"]))
                queued.extend(_queued(orchestrator))

            service._coaching_llm_reply = _announce
            await service._handle_message({"type": "rest_complete"})
            orchestrator.stop()
            return queued

        assert asyncio.run(_run()) == []


class TestRestCompletePrompt:
    def test_is_calm_and_names_the_set_focus(self):
        service, orchestrator, _ = _workout_service()
        orchestrator._set_focus_fault = "knee_valgus"
        prompt = service._build_rest_complete_prompt()
        assert "knees caving in" in prompt
        assert "energetic" not in prompt.lower()
        assert "do not ask" not in prompt.lower()

    def test_open_pain_report_reaches_the_next_set_line_without_a_question(self):
        """The SAFETY line rides on every coaching prompt; mid-workout the
        athlete can't answer a question without the wake word."""
        from agent.services.athlete_facts import FACTS_KEY

        session = _FakeSession()
        state = _StubState("u1")
        state.values[FACTS_KEY] = {
            "pain_flags": [{"body_part": "left knee", "note": "", "reported_at": "2026-10-01T10:00:00"}],
        }
        service = CoachingService(session=session, state=state)
        service._workout_active_flag = True
        asyncio.run(service._coaching_llm_reply(service._build_rest_complete_prompt()))
        assert "left knee" in session.prompts[0]
        assert "don't ask them anything" in session.prompts[0]


class TestNoQuestionsMidWorkout:
    def test_workout_coaching_lines_never_ask(self):
        session = _FakeSession()
        service = CoachingService(session=session, state=_StubState("u1"))
        service._workout_active_flag = True
        asyncio.run(service._coaching_llm_reply("recap set 1"))
        assert "don't ask them anything" in session.prompts[0]

    def test_assessment_lines_may_ask(self):
        session = _FakeSession()
        service = CoachingService(session=session, state=_StubState("u1"))
        asyncio.run(service._coaching_llm_reply("assessment feedback"))
        assert "don't ask them anything" not in session.prompts[0]


class TestForceEndSet:
    def test_force_end_gets_a_recap_and_rest(self):
        async def _run():
            service, orchestrator, _ = _workout_service(sets=3, reps=8)
            for rep in range(1, 4):
                await service._handle_message(_rep_message(rep))
            _queued(orchestrator)
            result = await service.force_end_current_set(reps=3)
            events = _queued(orchestrator)
            orchestrator.stop()
            return service, orchestrator, result, events

        service, orchestrator, result, events = asyncio.run(_run())
        assert result["status"] == "advanced"
        assert [e.event_type for e in events] == ["llm_set_recap"]
        assert events[0].data["total_reps"] == 3
        assert orchestrator.resting is True
        session = service._get_workout_session()
        assert session.get_current_set().set_number == 2
        assert session.last_completed_set().performed_reps == 3

    def test_second_done_during_rest_does_not_end_the_next_set(self):
        async def _run():
            service, orchestrator, _ = _workout_service(sets=3, reps=8)
            await service._handle_message(_rep_message(1))
            await service.force_end_current_set(reps=1)
            result = await service.force_end_current_set(reps=0)
            orchestrator.stop()
            return service, result

        service, result = asyncio.run(_run())
        assert result["status"] == "already_resting"
        assert service._get_workout_session().get_current_set().set_number == 2

    def test_force_ending_the_last_set_gets_the_exercise_recap(self):
        async def _run():
            service, orchestrator, _ = _workout_service(sets=1, reps=8)
            await service._handle_message(_rep_message(1))
            _queued(orchestrator)
            result = await service.force_end_current_set(reps=1)
            events = _queued(orchestrator)
            orchestrator.stop()
            return result, events

        result, events = asyncio.run(_run())
        assert result["status"] == "workout_complete"
        assert [e.event_type for e in events] == ["llm_exercise_recap"]


class TestSetLoadAndRpe:
    def test_completed_set_records_the_load(self):
        async def _run():
            service, orchestrator, _ = _workout_service(sets=2, reps=1, weight=60.0, unit="kg")
            await service._handle_message(_rep_message(1))
            orchestrator.stop()
            return service

        service = asyncio.run(_run())
        last = service._get_workout_session().last_completed_set()
        assert last.performed_weight == 60.0

    def test_record_set_rpe_lands_on_the_set_just_finished(self):
        async def _run():
            service, orchestrator, _ = _workout_service(sets=2, reps=1)
            await service._handle_message(_rep_message(1))
            orchestrator.stop()
            return service

        service = asyncio.run(_run())
        service.record_set_rpe(8.0)
        session = service._get_workout_session()
        assert session.last_completed_set().actual_rpe == 8.0
        assert session.get_current_set().actual_rpe is None

    def test_record_set_rpe_before_any_set_is_a_no_op(self):
        service, orchestrator, _ = _workout_service()
        service.record_set_rpe(7.0)
        assert service._get_workout_session().last_completed_set() is None


class TestWorkoutStateLine:
    def test_none_without_a_workout(self):
        service = CoachingService(session=None, state=_StubState("u1"))
        assert service.workout_state_line() is None

    def test_in_set(self):
        async def _run():
            service, orchestrator, _ = _workout_service(sets=3, reps=5, weight=60.0)
            await service._handle_message(_rep_message(1))
            await service._handle_message(_rep_message(2))
            line = service.workout_state_line()
            orchestrator.stop()
            return line

        line = asyncio.run(_run())
        assert "set 1/3" in line
        assert "2 reps" in line
        assert "60 kg" in line
        assert "\n" not in line

    def test_resting_after_a_set_with_its_cue_and_outcome(self):
        async def _run():
            service, orchestrator, _ = _workout_service(sets=3, reps=4)
            await service._handle_message({
                "type": "fault", "fault_type": "knee_valgus", "severity": "moderate",
                "cue": "knees_out", "observability": "observable",
            })
            await service._handle_message(_rep_message(1, ["knee_valgus"]))
            event = orchestrator._queue.get_nowait()
            await orchestrator._dispatch_cached_cue(event)
            for rep in range(2, 5):
                await service._handle_message(_rep_message(rep))
            line = service.workout_state_line()
            orchestrator.stop()
            return line

        line = asyncio.run(_run())
        assert "resting before set 2/3" in line
        assert "last set 3/4 clean" in line
        assert "knees out" in line.lower()
        assert "fixed" in line
        assert len(line) < 300


# =============================================================================
# Chat context: summary and the last-set item
# =============================================================================


class _FakeHandle:
    async def wait_for_playout(self) -> None:
        return None


class _FakeAgent:
    def __init__(self) -> None:
        from livekit.agents import llm

        self.chat_ctx = llm.ChatContext.empty()
        self.chat_ctx.items.append(llm.ChatMessage(role="system", content=["You are Nova."]))

    async def update_chat_ctx(self, chat_ctx) -> None:
        self.chat_ctx = chat_ctx


class _FakeCompaction:
    def get_summary(self) -> str:
        return "They squatted earlier."


class _FakeSession:
    """Appends the turn to the agent's context, like generate_reply does."""

    def __init__(self, reply_text: str = "Knees tracked better that set.") -> None:
        import types

        self.current_agent = _FakeAgent()
        self.current_agent.userdata = types.SimpleNamespace(compaction_service=_FakeCompaction())
        self.reply_text = reply_text
        self.prompts: list[str] = []

    def generate_reply(self, *, instructions=None, user_input=None, tool_choice=None,
                       allow_interruptions=None):
        from livekit.agents import llm

        self.prompts.append(user_input)
        items = self.current_agent.chat_ctx.items
        items.append(llm.ChatMessage(role="user", content=[user_input]))
        items.append(llm.ChatMessage(role="assistant", content=[self.reply_text]))
        return _FakeHandle()


def _texts(agent) -> list[str]:
    return [item.text_content or "" for item in agent.chat_ctx.items]


class TestCoachingContext:
    def test_five_prunes_leave_one_summary(self):
        from livekit.agents import llm

        session = _FakeSession()
        service = CoachingService(session=session, state=_StubState("u1"))

        async def _run():
            for turn in range(5):
                for i in range(10):
                    session.current_agent.chat_ctx.items.append(
                        llm.ChatMessage(role="user", content=[f"turn {turn} msg {i}"])
                    )
                await service._prune_conversation_context(max_items=6)

        asyncio.run(_run())
        summaries = [t for t in _texts(session.current_agent) if t.startswith("[CONVERSATION SUMMARY]")]
        assert len(summaries) == 1

    def test_recap_leaves_one_last_set_item(self):
        session = _FakeSession()
        service = CoachingService(session=session, state=_StubState("u1"))

        async def _run():
            await service._coaching_llm_reply("recap set 1", last_set_note="Set 1: 4 of 5 reps clean.")
            await service._coaching_llm_reply("recap set 2", last_set_note="Set 2: 5 of 5 reps clean.")

        asyncio.run(_run())
        texts = _texts(session.current_agent)
        last_set = [t for t in texts if t.startswith("[LAST SET]")]
        assert len(last_set) == 1
        assert "Set 2: 5 of 5 reps clean." in last_set[0]
        assert session.reply_text in last_set[0]
        # The ephemeral turn itself is still stripped
        assert "recap set 2" not in texts

    def test_plain_coaching_reply_adds_no_last_set_item(self):
        session = _FakeSession()
        service = CoachingService(session=session, state=_StubState("u1"))
        asyncio.run(service._coaching_llm_reply("motivate"))
        assert not any(t.startswith("[LAST SET]") for t in _texts(session.current_agent))

    def test_open_pain_report_reaches_every_coaching_prompt(self):
        from agent.services.athlete_facts import FACTS_KEY

        session = _FakeSession()
        state = _StubState("u1")
        state.values[FACTS_KEY] = {
            "pain_flags": [{"body_part": "lower back", "note": "", "reported_at": "2026-10-01T10:00:00"}],
        }
        service = CoachingService(session=session, state=state)
        asyncio.run(service._coaching_llm_reply("recap set 1"))
        assert "SAFETY" in session.prompts[0]
        assert "lower back" in session.prompts[0]

    def test_no_safety_line_without_a_pain_report(self):
        session = _FakeSession()
        service = CoachingService(session=session, state=_StubState("u1"))
        asyncio.run(service._coaching_llm_reply("recap set 1"))
        assert "SAFETY" not in session.prompts[0]


def _proceed_anyway_result(body_measurement: str = "complete") -> dict:
    return {
        "type": "assessment_result", "passed": False, "round": 2,
        "final_round": True, "proceed_anyway": True, "body_measurement": body_measurement,
        "diagnosis": {
            "immediate_causes": [{
                "cause_id": "knee_track_cue", "tier": 1, "score": 0.7,
                "explanation": "Your knees drift inside your toes on the way up.",
                "implicated_by": ["knee_not_tracking_toes"], "observability": "observable",
            }],
            "session_causes": [], "contextual_notes": [],
        },
        "scoring": {"mean_score": 0.74},
        "demo": {"available": False, "cues": []},
    }


class TestAssessmentProceedsAnyway:
    def test_unresolved_check_starts_the_workout_instead_of_another_retry(self):
        service, orchestrator, replies = _workout_service()
        asyncio.run(service._on_assessment_result(_proceed_anyway_result()))
        orchestrator.stop()
        assert len(replies) == 1
        assert "knees drift inside your toes" in replies[0]
        assert "try it again" not in replies[0].lower()

    def test_provisional_body_measurement_is_hedged(self):
        service, orchestrator, replies = _workout_service()
        asyncio.run(service._on_assessment_result(_proceed_anyway_result("provisional")))
        orchestrator.stop()
        assert "first read" in replies[0]

    def test_carried_cause_is_the_first_workout_sets_focus(self):
        service, orchestrator, _ = _workout_service()
        asyncio.run(service._on_assessment_result(_proceed_anyway_result()))
        orchestrator.reset_set(target_reps=5)
        service._apply_carried_focus()
        orchestrator.stop()
        assert orchestrator._set_focus_fault == "knee_valgus"


def test_proceed_prompt_names_the_same_cause_the_first_set_focuses_on():
    service, orchestrator, replies = _workout_service()
    message = _proceed_anyway_result()
    message["diagnosis"]["immediate_causes"].insert(0, {
        "cause_id": "foot_placement_asymmetry", "tier": 1, "score": 0.9,
        "explanation": "Your feet are set up unevenly.",
        "implicated_by": ["uneven_setup"], "observability": "approximate",
    })
    asyncio.run(service._on_assessment_result(message))
    orchestrator.stop()
    assert "knees drift inside your toes" in replies[0]
    assert "set up unevenly" not in replies[0]


class TestTrackingQuality:
    def test_lost_tracking_plays_its_cue_once_and_pauses_fault_cues(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            lost = {"type": "tracking_quality", "status": "lost", "missing": ["left_knee"]}
            await service._handle_message(lost)
            await service._handle_message(lost)
            await asyncio.sleep(0)
            await service._handle_message({
                "type": "fault", "fault_type": "knee_valgus", "severity": "severe",
                "cue": "knees_out", "observability": "observable",
            })
            await service._handle_message(_rep_message(1, ["knee_valgus"]))
            paused = [e.cue_key for e in _queued(orchestrator)]
            await service._handle_message({"type": "tracking_quality", "status": "recovered", "missing": []})
            await service._handle_message({
                "type": "fault", "fault_type": "knee_valgus", "severity": "severe",
                "cue": "knees_out", "observability": "observable",
            })
            await service._handle_message(_rep_message(2, ["knee_valgus"]))
            resumed = [e.cue_key for e in _queued(orchestrator)]
            plays = [call.args[0] for call in orchestrator._play_cached.await_args_list]
            orchestrator.stop()
            return paused, resumed, plays

        paused, resumed, plays = asyncio.run(_run())
        assert plays.count("tracking_lost") == 1
        assert "knees_out" not in paused
        assert "knees_out" in resumed

    def test_camera_dropout_is_not_voiced_and_keeps_cues(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            await service._handle_message({
                "type": "tracking_quality", "status": "lost", "reason": "camera",
                "missing": [], "cameras": ["1"],
            })
            await asyncio.sleep(0)
            await service._handle_message({
                "type": "fault", "fault_type": "knee_valgus", "severity": "severe",
                "cue": "knees_out", "observability": "observable",
            })
            await service._handle_message(_rep_message(1, ["knee_valgus"]))
            queued = [e.cue_key for e in _queued(orchestrator)]
            plays = [call.args[0] for call in orchestrator._play_cached.await_args_list]
            orchestrator.stop()
            return queued, plays

        queued, plays = asyncio.run(_run())
        assert "tracking_lost" not in plays
        assert "knees_out" in queued

    def test_tracking_loss_does_not_outlive_its_set(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            await service._handle_message({"type": "tracking_quality", "status": "lost", "missing": ["left_knee"]})
            orchestrator.reset_set(target_reps=5)
            lost_after_reset = orchestrator._tracking_lost
            orchestrator.stop()
            return lost_after_reset

        assert asyncio.run(_run()) is False

    def test_missing_clip_is_skipped(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            orchestrator._get_cue_audio = lambda key: key != "tracking_lost"
            await service._handle_message({"type": "tracking_quality", "status": "lost", "missing": []})
            await asyncio.sleep(0)
            plays = [call.args[0] for call in orchestrator._play_cached.await_args_list]
            orchestrator.stop()
            return plays

        assert "tracking_lost" not in asyncio.run(_run())


class TestLastRepVerdict:
    def test_none_before_the_first_rep(self):
        service, orchestrator, _ = _workout_service()
        assert service.last_rep_verdict() is None

    def test_latest_rep_with_faults_deduped(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            await service._handle_message(_rep_message(1))
            await service._handle_message({
                **_rep_message(2, ["knee_valgus", "hip_shift"]),
                "faults_detailed": [
                    {"fault_type": "knee_valgus", "severity": "moderate",
                     "details": {"side": "left", "observability": "observable"}},
                    {"fault_type": "knee_valgus", "severity": "mild",
                     "details": {"side": "right", "observability": "observable"}},
                    {"fault_type": "hip_shift", "severity": "mild",
                     "details": {"side": "right", "observability": "approximate"}},
                ],
            })
            orchestrator.stop()
            return service.last_rep_verdict()

        assert asyncio.run(_run()) == {
            "rep_number": 2,
            "faults": [
                {"fault_type": "knee_valgus", "side": "left", "observability": "observable"},
                {"fault_type": "hip_shift", "side": "right", "observability": "approximate"},
            ],
        }

    def test_reps_before_the_workout_starts_are_ignored(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            service._workout_active_flag = False
            await service._handle_message(_rep_message(1))
            orchestrator.stop()
            return service.last_rep_verdict()

        assert asyncio.run(_run()) is None

    def test_clean_rep_has_no_faults(self):
        async def _run():
            service, orchestrator, _ = _workout_service()
            await service._handle_message(_rep_message(1))
            orchestrator.stop()
            return service.last_rep_verdict()

        assert asyncio.run(_run()) == {"rep_number": 1, "faults": []}
