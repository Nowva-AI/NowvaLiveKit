"""Tests for the intra-set stance/toe-out adjustment monitor.

The monitor starts when a fault cue fires mid-set while the latest set
diagnosis blames the stance or foot angle, and coaches the lifter back to
their target stance while they stand between reps.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.coaching_constants import PREEMPTIVE_TEXT
from agent.services.coaching_orchestrator import (
    STANCE_TOLERANCE,
    TOE_OUT_TOLERANCE_DEG,
    CoachingEvent,
    CoachingOrchestrator,
    CuePriority,
)

TARGET_STANCE = 1.5
TARGET_TOE_OUT = 25.0


def _make_orchestrator() -> CoachingOrchestrator:
    orch = CoachingOrchestrator(
        play_cached_audio_fn=AsyncMock(),
        generate_llm_reply_fn=AsyncMock(),
        get_cue_audio_fn=lambda key: bool(key),
    )
    orch._speak_adjustment_fn = AsyncMock()
    return orch


def _armed_orchestrator(param: str = "stance_width") -> CoachingOrchestrator:
    """An orchestrator already monitoring, with the feedback interval elapsed."""
    orch = _make_orchestrator()
    target = TARGET_STANCE if param == "stance_width" else TARGET_TOE_OUT
    tolerance = STANCE_TOLERANCE if param == "stance_width" else TOE_OUT_TOLERANCE_DEG
    orch.start_adjustment_monitor(param, target, tolerance)
    orch._last_feedback_time = 0.0
    return orch


def _frame(
    rep_phase: str = "idle",
    stance: float | None = 1.1,
    toe_out: float | None = 10.0,
    with_targets: bool = True,
) -> dict:
    sample: dict = {"knee_flexion_l": 5.0, "knee_flexion_r": 5.0, "rep_phase": rep_phase}
    if stance is not None:
        sample["stance_width_ratio"] = stance
    if toe_out is not None:
        sample["foot_direction_angle_l"] = toe_out
        sample["foot_direction_angle_r"] = toe_out
    if with_targets:
        sample["target_stance_ratio"] = TARGET_STANCE
        sample["target_toe_out_deg"] = TARGET_TOE_OUT
    return sample


def _spoken(call: dict) -> str:
    """The text handed to generate_reply, which passes a ChatMessage."""
    user_input = call["user_input"]
    content = getattr(user_input, "content", None)
    return str(content[0]) if content else str(user_input)


def _diagnosis_blaming(cause_id: str) -> dict:
    return {
        "confidence": 0.8,
        "immediate_causes": [{"cause_id": cause_id, "score": 0.7, "explanation": "x"}],
    }


def _fault_cue_event(
    cue_key: str = "chest_up", fault_type: str = "hip_shoot"
) -> CoachingEvent:
    return CoachingEvent(
        priority=CuePriority.FAULT_CUE,
        timestamp=time.monotonic(),
        event_type="cached_cue",
        cue_key=cue_key,
        data={"fault_type": fault_type, "severity": "moderate"},
    )


class TestMonitorStart:
    def test_narrow_stance_starts_stance_monitor(self):
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(stance=1.1))
        orch._maybe_start_adjustment_from_fault()
        assert orch._adjustment_active
        assert orch._adjustment_param == "stance_width"
        assert orch._adjustment_target == pytest.approx(TARGET_STANCE)

    def test_correct_stance_falls_through_to_toe_out(self):
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(stance=TARGET_STANCE, toe_out=10.0))
        orch._maybe_start_adjustment_from_fault()
        assert orch._adjustment_param == "toe_out"
        assert orch._adjustment_target == pytest.approx(TARGET_TOE_OUT)

    def test_no_monitor_when_both_already_on_target(self):
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(stance=TARGET_STANCE, toe_out=TARGET_TOE_OUT))
        orch._maybe_start_adjustment_from_fault()
        assert not orch._adjustment_active

    def test_no_monitor_without_targets(self):
        """Before calibration the pipeline sends no targets — stay silent."""
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(with_targets=False))
        orch._maybe_start_adjustment_from_fault()
        assert not orch._adjustment_active

    def test_no_monitor_without_stance_metrics(self):
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(stance=None))
        orch._maybe_start_adjustment_from_fault()
        assert not orch._adjustment_active


class TestMonitorTriggering:
    def test_stance_cause_in_latest_diagnosis_starts_the_monitor(self):
        async def _run():
            orch = _make_orchestrator()
            orch.record_angle_sample(_frame(stance=1.1))

            # No diagnosis yet: a fault cue alone does not arm it
            await orch._dispatch_cached_cue(_fault_cue_event("knees_out", "knee_valgus"))
            assert not orch._adjustment_active

            orch.set_diagnosis_data(_diagnosis_blaming("narrow_stance"), {"mean_score": 0.7})
            await orch._dispatch_cached_cue(_fault_cue_event("knees_out", "knee_valgus"))
            assert orch._adjustment_active

        asyncio.run(_run())

    def test_each_stance_cause_arms_the_monitor(self):
        async def _run():
            for cause_id in ("narrow_stance", "stance_toe_mismatch", "narrow_foot_angle"):
                orch = _make_orchestrator()
                orch.record_angle_sample(_frame(stance=1.1))
                orch.set_diagnosis_data(_diagnosis_blaming(cause_id), {"mean_score": 0.7})
                await orch._dispatch_cached_cue(_fault_cue_event())
                assert orch._adjustment_active, cause_id

        asyncio.run(_run())

    def test_other_top_cause_does_not_arm_the_monitor(self):
        async def _run():
            orch = _make_orchestrator()
            orch.record_angle_sample(_frame(stance=1.1))
            orch.set_diagnosis_data(_diagnosis_blaming("ankle_mobility"), {"mean_score": 0.7})
            await orch._dispatch_cached_cue(_fault_cue_event())
            assert not orch._adjustment_active

        asyncio.run(_run())

    def test_follow_up_survives_the_recap_consuming_the_diagnosis(self):
        """The recap consumes the pending diagnosis; the next set's cues
        still need to know what it blamed."""
        async def _run():
            orch = _make_orchestrator()
            orch.record_angle_sample(_frame(stance=1.1))
            orch.set_diagnosis_data(_diagnosis_blaming("narrow_stance"), {"mean_score": 0.7})
            orch._consume_diagnosis()
            await orch._dispatch_cached_cue(_fault_cue_event())
            assert orch._adjustment_active

        asyncio.run(_run())

    def test_running_monitor_is_not_explained_again(self):
        async def _run():
            orch = _make_orchestrator()
            spoken: list[str] = []

            async def _record(key):
                spoken.append(key)

            orch._play_cached = _record
            orch.record_angle_sample(_frame(stance=1.1))
            orch.set_diagnosis_data(_diagnosis_blaming("narrow_stance"), {"mean_score": 0.7})
            await orch._dispatch_cached_cue(_fault_cue_event())
            await orch._dispatch_cached_cue(_fault_cue_event("knees_out", "knee_valgus"))
            assert spoken == ["chest_up", "stance_explain", "knees_out"]

        asyncio.run(_run())


class TestFeedbackLoop:
    def test_speaks_while_standing(self):
        async def _run():
            orch = _armed_orchestrator()
            orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))
            await asyncio.sleep(0)
            orch._speak_adjustment_fn.assert_awaited_once()
            param, delta, hit = orch._speak_adjustment_fn.await_args.args
            assert param == "stance_width"
            assert delta == pytest.approx(TARGET_STANCE - 1.1)
            assert hit is False

        asyncio.run(_run())

    def test_silent_during_the_rep(self):
        async def _run():
            orch = _armed_orchestrator()
            for phase in ("descending", "bottom", "ascending"):
                orch.record_angle_sample(_frame(rep_phase=phase, stance=1.1))
            await asyncio.sleep(0)
            orch._speak_adjustment_fn.assert_not_awaited()

        asyncio.run(_run())

    def test_hitting_target_confirms_then_stops(self):
        async def _run():
            orch = _armed_orchestrator()
            orch.record_angle_sample(_frame(rep_phase="idle", stance=TARGET_STANCE))
            await asyncio.sleep(0)
            param, _, hit = orch._speak_adjustment_fn.await_args.args
            assert hit is True
            # The confirmation must still name the parameter even though the
            # monitor has already been torn down.
            assert param == "stance_width"
            assert not orch._adjustment_active

        asyncio.run(_run())

    def test_missing_metrics_do_not_cue_a_phantom_correction(self):
        """A frame with no 3D skeleton must not read as 'stance is zero'."""

        async def _run():
            orch = _armed_orchestrator()
            orch.record_angle_sample(_frame(rep_phase="idle", stance=None))
            await asyncio.sleep(0)
            orch._speak_adjustment_fn.assert_not_awaited()
            assert orch._adjustment_active

        asyncio.run(_run())

    def test_throttled_to_the_feedback_interval(self):
        async def _run():
            orch = _armed_orchestrator()
            for _ in range(5):
                orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))
            await asyncio.sleep(0)
            assert orch._speak_adjustment_fn.await_count == 1

        asyncio.run(_run())

    def test_does_not_talk_over_itself(self):
        """A slow generate_reply must not accumulate overlapping speech."""

        async def _run():
            orch = _armed_orchestrator()
            released = asyncio.Event()

            async def _slow(param, delta, hit):
                await released.wait()

            orch._speak_adjustment_fn = AsyncMock(side_effect=_slow)

            orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))
            await asyncio.sleep(0)

            orch._last_feedback_time = 0.0  # interval elapsed again
            orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))
            await asyncio.sleep(0)
            assert orch._speak_adjustment_fn.await_count == 1

            released.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            orch._last_feedback_time = 0.0
            orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))
            await asyncio.sleep(0)
            assert orch._speak_adjustment_fn.await_count == 2

        asyncio.run(_run())

    def test_toe_out_averages_both_feet(self):
        async def _run():
            orch = _armed_orchestrator("toe_out")
            sample = _frame(rep_phase="idle")
            sample["foot_direction_angle_l"] = 20.0
            sample["foot_direction_angle_r"] = 10.0
            orch.record_angle_sample(sample)
            await asyncio.sleep(0)
            _, delta, _ = orch._speak_adjustment_fn.await_args.args
            assert delta == pytest.approx(TARGET_TOE_OUT - 15.0)

        asyncio.run(_run())


class TestSetBoundary:
    def test_new_set_clears_the_monitor(self):
        orch = _armed_orchestrator()
        orch.reset_set(target_reps=5)
        assert not orch._adjustment_active

    def test_silent_during_rest(self):
        async def _run():
            orch = _armed_orchestrator()
            orch.resting = True
            orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))
            await asyncio.sleep(0)
            orch._speak_adjustment_fn.assert_not_awaited()

        asyncio.run(_run())


class TestPreemptiveOutcomeTiming:
    """A cue fires during a rep that is already underway, so that rep's
    faults were fixed before the lifter heard anything."""

    def _armed(self, fired_at_rep: int):
        from agent.services.coaching_orchestrator import PendingCueOutcome

        orch = _make_orchestrator()
        orch._play_raw_frames_fn = AsyncMock()
        outcome = PendingCueOutcome(
            fault_type="forward_lean",
            cue_key="chest_up",
            fired_at_rep=fired_at_rep,
            positive_audio=["pos"],
            negative_audio=["neg"],
        )
        done: asyncio.Future = asyncio.Future()
        done.set_result(None)
        outcome.generation_task = done
        orch._pending_outcome = outcome
        return orch

    def test_not_judged_on_the_rep_the_cue_fired_in(self):
        async def _run():
            orch = self._armed(fired_at_rep=3)
            orch._set_rep_count = 3
            orch._resolve_pending_outcome(["forward_lean"])
            await asyncio.sleep(0)
            orch._play_raw_frames_fn.assert_not_awaited()
            assert orch._pending_outcome is not None

        asyncio.run(_run())

    def test_negative_plays_when_fault_persists_on_the_next_rep(self):
        async def _run():
            orch = self._armed(fired_at_rep=3)
            orch._set_rep_count = 4
            orch._resolve_pending_outcome(["forward_lean"])
            await asyncio.sleep(0)
            orch._play_raw_frames_fn.assert_awaited_once_with(["neg"])
            assert orch._pending_outcome is None

        asyncio.run(_run())

    def test_positive_plays_when_fault_is_gone_on_the_next_rep(self):
        async def _run():
            orch = self._armed(fired_at_rep=3)
            orch._set_rep_count = 4
            orch._resolve_pending_outcome(["knee_valgus"])
            await asyncio.sleep(0)
            orch._play_raw_frames_fn.assert_awaited_once_with(["pos"])

        asyncio.run(_run())

    def test_resolving_does_not_block_the_rep_count_cue(self):
        """Playout must not be awaited inline — it delays the rep sound."""

        async def _run():
            orch = self._armed(fired_at_rep=3)
            orch._set_rep_count = 4
            blocked = asyncio.Event()

            async def _never_finishes(audio):
                await blocked.wait()

            orch._play_raw_frames_fn = AsyncMock(side_effect=_never_finishes)

            # Synchronous call: it returns before playback even starts.
            orch._resolve_pending_outcome(["forward_lean"])
            assert orch._pending_outcome is None

            await asyncio.sleep(0)
            orch._play_raw_frames_fn.assert_awaited_once()
            blocked.set()
            await asyncio.sleep(0)

        asyncio.run(_run())


class TestAdjustmentSpeech:
    """Exercises the real CoachingService method against a session double
    with livekit's actual generate_reply signature — the orchestrator tests
    mock the callback, so a bad kwarg would otherwise be swallowed."""

    def _service(self):
        from agent.services.coaching_service import CoachingService

        calls: list[dict] = []

        class _Handle:
            async def wait_for_playout(self):
                return None

        class _Session:
            current_agent = None

            def generate_reply(
                self, *, user_input=None, instructions=None, tool_choice=None,
                tools=None, allow_interruptions=None, chat_ctx=None,
                input_modality=None,
            ):
                calls.append({"user_input": user_input, "instructions": instructions})
                return _Handle()

        service = CoachingService.__new__(CoachingService)
        service._session = _Session()
        service.is_coaching_speaking = False
        # No cue audio on disk — exercises the LLM fallback path.
        service._audio_cue_service = None
        return service, calls

    def test_speaks_through_the_real_generate_reply_signature(self):
        service, calls = self._service()
        asyncio.run(service._speak_adjustment_feedback("stance_width", 0.3, False))
        assert len(calls) == 1, "generate_reply was never reached"
        assert "narrow" in _spoken(calls[0]).lower()

    def test_confirms_when_the_target_is_hit(self):
        service, calls = self._service()
        asyncio.run(service._speak_adjustment_feedback("stance_width", 0.0, True))
        assert "on target" in _spoken(calls[0]).lower()

    def test_toe_out_wording_names_the_direction(self):
        service, calls = self._service()
        asyncio.run(service._speak_adjustment_feedback("toe_out", 8.0, False))
        assert "more turn-out" in _spoken(calls[0])

        service, calls = self._service()
        asyncio.run(service._speak_adjustment_feedback("toe_out", -8.0, False))
        assert "too far" in _spoken(calls[0])


class TestMonitorBounds:
    def test_stops_nagging_after_the_utterance_cap(self):
        from agent.services.coaching_orchestrator import MAX_ADJUSTMENT_UTTERANCES

        async def _run():
            orch = _armed_orchestrator()
            for _ in range(MAX_ADJUSTMENT_UTTERANCES + 3):
                orch._last_feedback_time = 0.0
                orch._adjustment_speaking = False
                orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))
                await asyncio.sleep(0)
            assert orch._speak_adjustment_fn.await_count == MAX_ADJUSTMENT_UTTERANCES
            assert not orch._adjustment_active

        asyncio.run(_run())

    def test_cue_finishing_after_the_set_ends_arms_nothing(self):
        """_dispatch_cached_cue awaits playback, so the set can end mid-cue."""

        async def _run():
            orch = _make_orchestrator()
            orch.record_angle_sample(_frame(stance=1.1))

            async def _play_then_end_set(cue_key):
                orch.reset_set(target_reps=5)
                orch.resting = True

            orch._play_cached = _play_then_end_set
            await orch._dispatch_cached_cue(_fault_cue_event())
            assert not orch._adjustment_active
            assert orch._pending_outcome is None

        asyncio.run(_run())


class TestPreemptiveGenerationHandoff:
    """The TTS task must still populate the outcome it was created for,
    even when the next rep detaches it from the orchestrator first."""

    def _orchestrator(self, tts_delay: float):
        played: list = []
        orch = _make_orchestrator()

        async def _tts(text):
            await asyncio.sleep(tts_delay)
            return [f"audio:{text}"]

        async def _play(frames):
            played.append(frames)

        orch._generate_tts_fn = _tts
        orch._play_raw_frames_fn = _play
        return orch, played

    def _run_cue_then_reps(self, tts_delay: float):
        async def _run():
            orch, played = self._orchestrator(tts_delay)
            await orch._dispatch_cached_cue(_fault_cue_event())

            orch._set_rep_count = 1  # the rep the cue played during
            orch._resolve_pending_outcome(["hip_shoot"])
            await asyncio.sleep(0)
            judged_own_rep = orch._pending_outcome is None

            orch._set_rep_count = 2  # first rep they could act on
            orch._resolve_pending_outcome([])
            await asyncio.sleep(tts_delay + 0.3)
            return judged_own_rep, played

        return asyncio.run(_run())

    def test_audio_arrives_when_tts_beats_the_rep(self):
        judged_own_rep, played = self._run_cue_then_reps(0.0)
        assert judged_own_rep is False
        assert played == [[f"audio:{PREEMPTIVE_TEXT['chest_up'][0]}"]]

    def test_audio_still_arrives_when_tts_finishes_after_the_rep(self):
        judged_own_rep, played = self._run_cue_then_reps(0.2)
        assert judged_own_rep is False
        assert played == [[f"audio:{PREEMPTIVE_TEXT['chest_up'][0]}"]]

    def test_side_variant_cue_uses_its_base_outcome_lines(self):
        async def _run():
            orch, played = self._orchestrator(0.0)
            await orch._dispatch_cached_cue(_fault_cue_event("knees_out_left", "knee_valgus"))
            assert orch._pending_outcome is not None
            orch._set_rep_count = 2
            orch._resolve_pending_outcome([])
            await asyncio.sleep(0.3)
            return played

        played = asyncio.run(_run())
        assert played == [[f"audio:{PREEMPTIVE_TEXT['knees_out'][0]}"]]


class TestUtteranceBudgetIsPerSet:
    """A repeat fault cue re-aims the monitor but must not refill the
    budget, or a lifter who never reaches target gets cued all set."""

    def _exhaust(self, orch):
        from agent.services.coaching_orchestrator import MAX_ADJUSTMENT_UTTERANCES

        for _ in range(MAX_ADJUSTMENT_UTTERANCES):
            orch._last_feedback_time = 0.0
            orch._adjustment_speaking = False
            orch.record_angle_sample(_frame(rep_phase="idle", stance=1.1))

    def test_repeat_cue_cannot_refill_the_budget(self):
        from agent.services.coaching_orchestrator import MAX_ADJUSTMENT_UTTERANCES

        async def _run():
            orch = _armed_orchestrator()
            orch.record_angle_sample(_frame(stance=1.1))
            self._exhaust(orch)
            await asyncio.sleep(0)
            assert not orch._adjustment_active

            # Same fault fires again later in the same set.
            orch._maybe_start_adjustment_from_fault()
            assert not orch._adjustment_active
            assert orch._speak_adjustment_fn.await_count == MAX_ADJUSTMENT_UTTERANCES

        asyncio.run(_run())

    def test_the_next_set_gets_a_fresh_budget(self):
        async def _run():
            orch = _armed_orchestrator()
            orch.record_angle_sample(_frame(stance=1.1))
            self._exhaust(orch)
            await asyncio.sleep(0)

            orch.reset_set(target_reps=5)
            orch.record_angle_sample(_frame(stance=1.1))
            orch._maybe_start_adjustment_from_fault()
            assert orch._adjustment_active

        asyncio.run(_run())


class TestAdjustmentCuesArePreCached:
    """The polling corrections fire every 1.5s between reps, so they play
    from the cue cache rather than paying an LLM round-trip each time."""

    def _service(self):
        from agent.services.coaching_service import CoachingService

        played: list[str] = []

        class _Cues:
            def has_cue(self, key):
                return True

            async def play_cue(self, key):
                played.append(key)

        service = CoachingService.__new__(CoachingService)
        service.is_coaching_speaking = False
        service._audio_cue_service = _Cues()
        service._session = None  # any LLM fallback would blow up here
        service._visual_bridge = None
        return service, played

    def test_direction_cues_play_from_cache(self):
        service, played = self._service()
        asyncio.run(service._speak_adjustment_feedback("stance_width", 0.3, False))
        asyncio.run(service._speak_adjustment_feedback("stance_width", -0.3, False))
        asyncio.run(service._speak_adjustment_feedback("toe_out", 8.0, False))
        asyncio.run(service._speak_adjustment_feedback("toe_out", -8.0, False))
        assert played == [
            "stance_wider", "stance_narrower", "toe_out_more", "toe_out_less",
        ]

    def test_on_target_plays_the_shared_confirmation(self):
        service, played = self._service()
        asyncio.run(service._speak_adjustment_feedback("toe_out", 0.0, True))
        assert played == ["adjust_good"]


class TestArmingExplanation:
    """The lifter hears "chest up" then gets asked to move their feet — the
    monitor must say why before it starts polling."""

    def test_arming_returns_the_stance_explanation_cue(self):
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(stance=1.1))
        assert orch._maybe_start_adjustment_from_fault() == "stance_explain"

    def test_arming_returns_the_toe_out_explanation_cue(self):
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(stance=TARGET_STANCE, toe_out=10.0))
        assert orch._maybe_start_adjustment_from_fault() == "toe_out_explain"

    def test_nothing_to_fix_returns_no_explanation(self):
        orch = _make_orchestrator()
        orch.record_angle_sample(_frame(stance=TARGET_STANCE, toe_out=TARGET_TOE_OUT))
        assert orch._maybe_start_adjustment_from_fault() is None

    def test_explanation_is_spoken_right_after_the_fault_cue(self):
        async def _run():
            orch = _make_orchestrator()
            spoken: list[str] = []
            orch._play_cached = lambda key: _record(spoken, key)
            orch.record_angle_sample(_frame(stance=1.1))
            orch.set_diagnosis_data(_diagnosis_blaming("narrow_stance"), {"mean_score": 0.7})
            await orch._dispatch_cached_cue(_fault_cue_event())
            assert spoken == ["chest_up", "stance_explain"]

        async def _record(sink, key):
            sink.append(key)

        asyncio.run(_run())
