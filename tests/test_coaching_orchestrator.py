"""
Tests for CoachingOrchestrator

Verifies priority ordering, motivation trigger logic,
set recap dispatch, and stale event handling.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.coaching_orchestrator import (
    CoachingEvent,
    CoachingOrchestrator,
    CuePriority,
)


# =============================================================================
# FIXTURES
# =============================================================================


def _make_callbacks():
    """Create mock async callbacks for the orchestrator."""
    return {
        "play_cached_audio_fn": AsyncMock(),
        "generate_llm_reply_fn": AsyncMock(),
        "get_cue_audio_fn": lambda key: True if key else False,
    }


@pytest.fixture
def mock_callbacks():
    return _make_callbacks()


@pytest.fixture
def orchestrator(mock_callbacks):
    return CoachingOrchestrator(**mock_callbacks)


# =============================================================================
# TEST PRIORITY ORDERING
# =============================================================================


class TestCuePriority:
    """Test that priority values are correctly ordered."""

    def test_llm_motivation_highest_priority(self):
        assert CuePriority.LLM_MOTIVATION < CuePriority.POSITIVE_CUE
        assert CuePriority.LLM_MOTIVATION < CuePriority.FAULT_CUE

    def test_cached_cues_before_fault(self):
        assert CuePriority.POSITIVE_CUE < CuePriority.FAULT_CUE

    def test_motivation_before_recap(self):
        assert CuePriority.LLM_MOTIVATION < CuePriority.LLM_SET_RECAP

    def test_event_ordering(self):
        """CoachingEvents should sort by priority."""
        events = [
            CoachingEvent(CuePriority.LLM_SET_RECAP, time.monotonic(), "llm_set_recap"),
            CoachingEvent(CuePriority.FAULT_CUE, time.monotonic(), "cached_cue", "knees_out"),
            CoachingEvent(CuePriority.LLM_MOTIVATION, time.monotonic(), "llm_motivation"),
            CoachingEvent(CuePriority.POSITIVE_CUE, time.monotonic(), "cached_cue", "good_depth"),
        ]
        sorted_events = sorted(events)
        assert sorted_events[0].priority == CuePriority.LLM_MOTIVATION
        assert sorted_events[1].priority == CuePriority.LLM_SET_RECAP
        assert sorted_events[2].priority == CuePriority.POSITIVE_CUE
        assert sorted_events[3].priority == CuePriority.FAULT_CUE


# =============================================================================
# TEST MOTIVATION TRIGGERS
# =============================================================================


class TestMotivationTriggers:
    """Test midpoint-based motivation trigger logic."""

    def test_no_trigger_without_target(self, orchestrator):
        orchestrator._set_target_reps = 0
        assert orchestrator._should_trigger_motivation(3) is False

    def test_no_trigger_short_set(self, orchestrator):
        orchestrator._set_target_reps = 3
        assert orchestrator._should_trigger_motivation(1) is False

    def test_trigger_at_midpoint(self, orchestrator):
        orchestrator._set_target_reps = 10
        assert orchestrator._should_trigger_motivation(5) is True

    def test_no_trigger_before_midpoint(self, orchestrator):
        orchestrator._set_target_reps = 10
        assert orchestrator._should_trigger_motivation(4) is False

    def test_no_trigger_after_midpoint(self, orchestrator):
        orchestrator._set_target_reps = 10
        assert orchestrator._should_trigger_motivation(6) is False

    def test_midpoint_odd_target(self, orchestrator):
        orchestrator._set_target_reps = 7
        assert orchestrator._should_trigger_motivation(3) is True

    def test_build_motivation_context(self, orchestrator):
        orchestrator._set_target_reps = 10
        orchestrator._clean_streak = 4
        orchestrator._recent_faults = ["knee_valgus", "forward_lean"]
        context = orchestrator._build_motivation_context(6, "parallel", True)
        assert context["rep_number"] == 6
        assert context["reps_remaining"] == 4
        assert context["clean_streak"] == 4
        assert "knee_valgus" in context["recent_faults"]


# =============================================================================
# TEST EVENT ENQUEUEING
# =============================================================================


class TestEventEnqueueing:
    """Test that events are correctly enqueued."""

    def test_on_fault_enqueues(self, orchestrator):
        async def _run():
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            assert not orchestrator._queue.empty()
            event = orchestrator._queue.get_nowait()
            # Fault events sit at FAULT_CUE offset by their rank, so the
            # more important fault dispatches first when several are queued.
            assert event.priority >= CuePriority.FAULT_CUE
            assert event.cue_key == "knees_out"
        asyncio.run(_run())

    def test_on_fault_no_audio_skips(self):
        cbs = _make_callbacks()
        cbs["get_cue_audio_fn"] = lambda key: False
        orch = CoachingOrchestrator(**cbs)

        async def _run():
            await orch.on_fault("unknown_cue", "unknown", "mild")
            assert orch._queue.empty()
        asyncio.run(_run())

    def test_on_fault_rate_limited(self, orchestrator):
        async def _run():
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            assert not orchestrator._queue.empty()
            orchestrator._queue.get_nowait()  # drain first
            # Same-or-lower priority fault within gap should be skipped
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            assert orchestrator._queue.empty()
        asyncio.run(_run())

    def test_on_fault_tracks_recent_faults(self, orchestrator):
        async def _run():
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            assert "knee_valgus" in orchestrator._recent_faults
        asyncio.run(_run())

    def test_on_shallow_rep_plays_cue(self, orchestrator, mock_callbacks):
        async def _run():
            await orchestrator.on_shallow_rep("deeper", "Quarter")
            await asyncio.sleep(0)  # let the fire-and-forget task run
            mock_callbacks["play_cached_audio_fn"].assert_awaited_once_with("deeper")
            assert orchestrator._set_shallow_count == 1
        asyncio.run(_run())

    def test_on_shallow_rep_bypasses_fault_rate_limit(self, orchestrator, mock_callbacks):
        """Consecutive shallow reps each get a cue — unlike fault cues."""
        async def _run():
            for _ in range(3):
                await orchestrator.on_shallow_rep("deeper", "Quarter")
            await asyncio.sleep(0)
            assert mock_callbacks["play_cached_audio_fn"].await_count == 3
            assert orchestrator._set_shallow_count == 3
        asyncio.run(_run())

    def test_on_shallow_rep_does_not_count_a_rep(self, orchestrator):
        async def _run():
            await orchestrator.on_shallow_rep("deeper", "Half")
            assert orchestrator.set_rep_count == 0
        asyncio.run(_run())

    def test_on_shallow_rep_suppressed_while_resting(self, orchestrator, mock_callbacks):
        async def _run():
            orchestrator.resting = True
            await orchestrator.on_shallow_rep("deeper", "Quarter")
            await asyncio.sleep(0)
            mock_callbacks["play_cached_audio_fn"].assert_not_awaited()
            assert orchestrator._set_shallow_count == 0
        asyncio.run(_run())

    def test_shallow_reps_reach_set_summary(self, orchestrator):
        async def _run():
            await orchestrator.on_shallow_rep("deeper", "Quarter")
            await orchestrator.on_shallow_rep("deeper", "Half")
            summary = orchestrator._build_set_summary()
            assert summary["shallow_reps"] == 2
            assert summary["shallow_depths"] == ["Quarter", "Half"]
        asyncio.run(_run())

    def test_reset_set_clears_shallow_count(self, orchestrator):
        async def _run():
            await orchestrator.on_shallow_rep("deeper", "Quarter")
            orchestrator.reset_set()
            assert orchestrator._set_shallow_count == 0
            assert orchestrator._set_shallow_depths == []
        asyncio.run(_run())

    def test_on_rep_complete_fires_rep_cue(self, orchestrator, mock_callbacks):
        async def _run():
            await orchestrator.on_rep_complete(1, "parallel", True, [])
            await asyncio.sleep(0)  # let fire-and-forget task run
            mock_callbacks["play_cached_audio_fn"].assert_awaited_once_with("rep_1")
        asyncio.run(_run())

    def test_on_rep_complete_clean_enqueues_positive(self, orchestrator, monkeypatch):
        orchestrator._positive_cue_keys = ["good_rep", "strong"]
        # Production fires the positive cue only 30% of the time — pin the roll
        monkeypatch.setattr("agent.services.coaching_orchestrator.random.random", lambda: 0.0)

        async def _run():
            await orchestrator.on_rep_complete(1, "parallel", True, [])
            events = []
            while not orchestrator._queue.empty():
                events.append(orchestrator._queue.get_nowait())
            positive = [e for e in events if e.priority == CuePriority.POSITIVE_CUE]
            assert len(positive) == 1
            assert positive[0].cue_key in ("good_rep", "strong")
        asyncio.run(_run())

    def test_on_rep_complete_faulted_no_positive(self, orchestrator):
        orchestrator._positive_cue_keys = ["good_rep"]

        async def _run():
            await orchestrator.on_rep_complete(1, "parallel", False, ["knee_valgus"])
            events = []
            while not orchestrator._queue.empty():
                events.append(orchestrator._queue.get_nowait())
            positive = [e for e in events if e.priority == CuePriority.POSITIVE_CUE]
            assert len(positive) == 0
        asyncio.run(_run())

    def test_on_rep_complete_tracks_clean_streak(self, orchestrator):
        async def _run():
            await orchestrator.on_rep_complete(1, "parallel", True, [])
            assert orchestrator._clean_streak == 1
            await orchestrator.on_rep_complete(2, "parallel", True, [])
            assert orchestrator._clean_streak == 2
            await orchestrator.on_rep_complete(3, "parallel", False, ["knee_valgus"])
            assert orchestrator._clean_streak == 0
        asyncio.run(_run())

    def test_on_set_complete_enqueues(self, orchestrator):
        async def _run():
            set_data = {"set_number": 1, "total_reps": 5, "clean_reps": 3}
            await orchestrator.on_set_complete(set_data)
            event = orchestrator._queue.get_nowait()
            assert event.priority == CuePriority.LLM_SET_RECAP
            assert event.data["total_reps"] == 5
        asyncio.run(_run())


# =============================================================================
# TEST DISPATCH
# =============================================================================


class TestDispatch:
    """Test event dispatch behavior."""

    def test_cached_cue_plays(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)

        async def _run():
            event = CoachingEvent(
                CuePriority.FAULT_CUE, time.monotonic(), "cached_cue", "knees_out"
            )
            await orch._dispatch(event)
            mock_callbacks["play_cached_audio_fn"].assert_awaited_once_with("knees_out")
        asyncio.run(_run())

    def test_stale_motivation_dropped(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_MOTIVATION,
                time.monotonic() - 10.0,
                "llm_motivation",
                data={"rep_number": 3, "reps_remaining": 5},
            )
            await orch._dispatch(event)
            mock_callbacks["generate_llm_reply_fn"].assert_not_awaited()
        asyncio.run(_run())

    def test_set_recap_calls_llm(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_SET_RECAP,
                time.monotonic(),
                "llm_set_recap",
                data={
                    "set_number": 1, "total_reps": 5, "clean_reps": 4,
                    "avg_depth": 95.0, "depth_consistency": 3.5, "fault_summary": {},
                },
            )
            await orch._dispatch(event)
            mock_callbacks["generate_llm_reply_fn"].assert_awaited_once()
            call_args = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
            assert "just finished a set" in call_args
            assert "5 reps" in call_args
        asyncio.run(_run())

    def test_set_recap_with_fault_trends_and_faults(self, mock_callbacks):
        """Recap must survive dict-shaped fault_summary stats when trends exist."""
        orch = CoachingOrchestrator(**mock_callbacks)
        orch.fault_trends = {
            "sessions_analyzed": 3,
            "total_reps": 45,
            "fault_profile": [{"fault_type": "forward_lean", "total_occurrences": 9}],
            "chronic_faults": ["forward_lean"],
        }

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_SET_RECAP,
                time.monotonic(),
                "llm_set_recap",
                data={
                    "set_number": 1, "total_reps": 5, "clean_reps": 4,
                    "avg_depth": 95.0, "depth_consistency": 3.5,
                    "fault_summary": {"forward_lean": {"count": 1, "pct": 20}},
                },
            )
            await orch._dispatch(event)
            mock_callbacks["generate_llm_reply_fn"].assert_awaited_once()
            call_args = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
            assert "CROSS-SESSION TREND" in call_args
        asyncio.run(_run())

    def test_motivation_calls_llm(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_MOTIVATION,
                time.monotonic(),
                "llm_motivation",
                data={"rep_number": 3, "reps_remaining": 5, "clean_streak": 2, "recent_faults": []},
            )
            await orch._dispatch(event)
            mock_callbacks["generate_llm_reply_fn"].assert_awaited_once()
            call_args = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
            assert "motivational push" in call_args
        asyncio.run(_run())


# =============================================================================
# TEST RESET
# =============================================================================


class TestReset:
    """Test reset_set clears per-set state."""

    def test_reset_clears_state(self, orchestrator):
        orchestrator._set_rep_count = 5
        orchestrator._clean_streak = 3
        orchestrator._recent_faults = ["knee_valgus"]
        orchestrator._last_motivation_rep = 3
        orchestrator._last_fault_cue_time = 99.0
        orchestrator.reset_set(target_reps=8, positive_cue_keys=["strong"])
        assert orchestrator._set_rep_count == 0
        assert orchestrator._clean_streak == 0
        assert orchestrator._recent_faults == []
        assert orchestrator._last_motivation_rep == 0
        assert orchestrator._set_target_reps == 8
        assert orchestrator._positive_cue_keys == ["strong"]
        assert orchestrator._last_fault_cue_time == 0.0
        assert orchestrator._set_clean_count == 0


# =============================================================================
# TEST SET COMPLETION (rep-count boundary)
# =============================================================================


class TestSetCompletion:
    """Test rep-count-based set boundary detection."""

    def test_set_complete_at_target_reps(self):
        """on_rep_complete should trigger on_set_complete when reps hit target."""
        cbs = _make_callbacks()
        advance_fn = AsyncMock(return_value=5)
        orch = CoachingOrchestrator(**cbs, advance_set_fn=advance_fn)
        orch.reset_set(target_reps=3)

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])
            await orch.on_rep_complete(2, "parallel", True, [])
            await orch.on_rep_complete(3, "parallel", True, [])

            events = []
            while not orch._queue.empty():
                events.append(orch._queue.get_nowait())
            recap_events = [e for e in events if e.event_type == "llm_set_recap"]
            assert len(recap_events) == 1
            assert recap_events[0].data["trigger"] == "rep_count"
            assert recap_events[0].data["total_reps"] == 3
            assert recap_events[0].data["clean_reps"] == 3

            advance_fn.assert_awaited_once()
            # State should be reset for next set
            assert orch._set_rep_count == 0
            assert orch._set_target_reps == 5

        asyncio.run(_run())

    def test_no_set_complete_before_target(self):
        """on_rep_complete should NOT trigger set_complete before target."""
        cbs = _make_callbacks()
        advance_fn = AsyncMock(return_value=3)
        orch = CoachingOrchestrator(**cbs, advance_set_fn=advance_fn)
        orch.reset_set(target_reps=5)

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])
            await orch.on_rep_complete(2, "parallel", True, [])

            events = []
            while not orch._queue.empty():
                events.append(orch._queue.get_nowait())
            recap_events = [e for e in events if e.event_type == "llm_set_recap"]
            assert len(recap_events) == 0
            advance_fn.assert_not_awaited()

        asyncio.run(_run())

    def test_no_set_complete_without_target(self):
        """No target_reps set -> never trigger set_complete from rep count."""
        cbs = _make_callbacks()
        advance_fn = AsyncMock()
        orch = CoachingOrchestrator(**cbs, advance_set_fn=advance_fn)

        async def _run():
            for i in range(1, 11):
                await orch.on_rep_complete(i, "parallel", True, [])
            advance_fn.assert_not_awaited()

        asyncio.run(_run())

    def test_set_complete_without_advance_fn(self):
        """Set complete should still trigger recap even without advance_set_fn."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch.reset_set(target_reps=2)

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])
            await orch.on_rep_complete(2, "parallel", True, [])

            events = []
            while not orch._queue.empty():
                events.append(orch._queue.get_nowait())
            recap_events = [e for e in events if e.event_type == "llm_set_recap"]
            assert len(recap_events) == 1

        asyncio.run(_run())

    def test_clean_count_tracks_correctly(self):
        """_set_clean_count should track total clean reps, not just streak."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch.reset_set(target_reps=5)

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])   # clean
            await orch.on_rep_complete(2, "parallel", False, [])  # not clean
            await orch.on_rep_complete(3, "parallel", True, [])   # clean
            assert orch._set_clean_count == 2
            assert orch._clean_streak == 1

        asyncio.run(_run())


# =============================================================================
# TEST FULL FLOW (integration-style)
# =============================================================================


class TestFullFlow:
    """Integration-style tests with the processor loop running."""

    def test_fault_plays_before_motivation(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)

        async def _run():
            orch.start()
            try:
                # Enqueue motivation first, then fault
                await orch._queue.put(CoachingEvent(
                    CuePriority.LLM_MOTIVATION, time.monotonic(), "llm_motivation",
                    data={"rep_number": 3, "reps_remaining": 5, "clean_streak": 0, "recent_faults": []},
                ))
                await orch._queue.put(CoachingEvent(
                    CuePriority.FAULT_CUE, time.monotonic(), "cached_cue", "knees_out",
                ))
                await asyncio.sleep(0.3)
                cbs["play_cached_audio_fn"].assert_awaited_once_with("knees_out")
            finally:
                orch.stop()
        asyncio.run(_run())

    def test_rep_complete_full_flow(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch._positive_cue_keys = ["strong"]

        async def _run():
            orch.start()
            try:
                await orch.on_rep_complete(1, "parallel", True, [])
                await asyncio.sleep(0.3)
                calls = cbs["play_cached_audio_fn"].call_args_list
                cue_keys = [c[0][0] for c in calls]
                assert "rep_1" in cue_keys
                # No LLM motivation on rep 1
                cbs["generate_llm_reply_fn"].assert_not_awaited()
            finally:
                orch.stop()
        asyncio.run(_run())


# =============================================================================
# TEST REST MODE
# =============================================================================


class TestRestMode:
    """Test that resting flag suppresses rep/fault processing."""

    def test_resting_blocks_on_fault(self, orchestrator):
        """Faults are ignored while resting."""
        orchestrator._resting = True

        async def _run():
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            assert orchestrator._queue.empty()

        asyncio.run(_run())

    def test_resting_blocks_on_rep_complete(self, orchestrator):
        """Reps are ignored while resting."""
        orchestrator._resting = True

        async def _run():
            await orchestrator.on_rep_complete(1, "parallel", True, [])
            assert orchestrator._queue.empty()
            assert orchestrator._set_rep_count == 0  # Not incremented

        asyncio.run(_run())

    def test_on_rest_complete_clears_flag(self, orchestrator):
        """on_rest_complete() resumes normal processing."""
        orchestrator._resting = True
        orchestrator.on_rest_complete()
        assert orchestrator._resting is False

    def test_rest_set_after_advance(self):
        """After set completion with advance, orchestrator enters rest mode."""
        cbs = _make_callbacks()
        advance_mock = AsyncMock(return_value=5)
        orch = CoachingOrchestrator(**cbs, advance_set_fn=advance_mock)
        orch.reset_set(target_reps=3)

        async def _run():
            orch.start()
            try:
                for i in range(1, 4):
                    await orch.on_rep_complete(i, "parallel", True, [])
                await asyncio.sleep(0.3)
                assert orch._resting is True
                advance_mock.assert_awaited_once()
            finally:
                orch.stop()

        asyncio.run(_run())

    def test_reset_set_clears_resting(self, orchestrator):
        """reset_set() clears resting as safety."""
        orchestrator._resting = True
        orchestrator.reset_set(target_reps=5)
        assert orchestrator._resting is False


# =============================================================================
# TEST RELATIVE REP COUNTING
# =============================================================================


class TestRelativeRepCounting:
    """Test that orchestrator tracks per-set relative reps, not absolute pipeline numbers."""

    def test_rep_count_increments_by_one(self):
        """_set_rep_count should increment by 1 regardless of absolute rep_number."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch.reset_set(target_reps=10)

        async def _run():
            # Pipeline sends absolute rep numbers 5, 6, 7
            await orch.on_rep_complete(5, "parallel", True, [])
            assert orch._set_rep_count == 1
            await orch.on_rep_complete(6, "parallel", True, [])
            assert orch._set_rep_count == 2
            await orch.on_rep_complete(7, "parallel", True, [])
            assert orch._set_rep_count == 3

        asyncio.run(_run())

    def test_no_immediate_set_complete_after_reset(self):
        """After reset, high absolute rep numbers should NOT trigger immediate set completion."""
        cbs = _make_callbacks()
        advance_fn = AsyncMock(return_value=5)
        orch = CoachingOrchestrator(**cbs, advance_set_fn=advance_fn)
        orch.reset_set(target_reps=5)

        async def _run():
            # Simulate set 2: pipeline sends absolute rep 6 (first rep of new set)
            await orch.on_rep_complete(6, "parallel", True, [])
            # Should be counted as relative rep 1, NOT trigger set complete
            assert orch._set_rep_count == 1
            advance_fn.assert_not_awaited()

        asyncio.run(_run())

    def test_rep_cue_uses_relative_number(self):
        """Rep cue key should use per-set number, not absolute pipeline number."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch.reset_set(target_reps=10)

        async def _run():
            # Pipeline sends absolute rep 8, but it's the 1st rep of this set
            await orch.on_rep_complete(8, "parallel", True, [])
            await asyncio.sleep(0)  # let fire-and-forget task run
            cbs["play_cached_audio_fn"].assert_awaited_once_with("rep_1")

        asyncio.run(_run())

    def test_set_number_increments_across_sets(self):
        """_set_number should increment with each completed set."""
        cbs = _make_callbacks()
        advance_fn = AsyncMock(return_value=3)
        orch = CoachingOrchestrator(**cbs, advance_set_fn=advance_fn)
        orch.reset_set(target_reps=3)

        async def _run():
            # Complete set 1
            for i in range(1, 4):
                await orch.on_rep_complete(i, "parallel", True, [])

            # Check set_number in recap event
            events = []
            while not orch._queue.empty():
                events.append(orch._queue.get_nowait())
            recap = [e for e in events if e.event_type == "llm_set_recap"]
            assert len(recap) == 1
            assert recap[0].data["set_number"] == 1

            # Clear rest mode before set 2
            orch.on_rest_complete()

            # Now complete set 2 (absolute reps 4, 5, 6)
            for i in range(4, 7):
                await orch.on_rep_complete(i, "parallel", True, [])

            events = []
            while not orch._queue.empty():
                events.append(orch._queue.get_nowait())
            recap = [e for e in events if e.event_type == "llm_set_recap"]
            assert len(recap) == 1
            assert recap[0].data["set_number"] == 2

        asyncio.run(_run())

    def test_motivation_uses_relative_rep(self):
        """Motivation should fire based on per-set rep count, not absolute."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch.reset_set(target_reps=10)

        async def _run():
            # Simulate reps through the midpoint (relative rep 5 = 10 // 2)
            for i in range(5):
                await orch.on_rep_complete(i + 10, "parallel", False, [])

            events = []
            while not orch._queue.empty():
                events.append(orch._queue.get_nowait())
            motivation = [e for e in events if e.event_type == "llm_motivation"]
            assert len(motivation) == 1

        asyncio.run(_run())


# =============================================================================
# TEST DIAGNOSIS INTEGRATION
# =============================================================================


def _mock_diagnosis() -> dict:
    """Diagnosis dict matching the IPC message format."""
    return {
        "confidence": 0.82,
        "detected_symptoms": [
            {"symptom_id": "knee_valgus", "severity": 0.6, "contributing_reps": [1, 3]},
        ],
        "immediate_causes": [
            {
                "cause_id": "stance_narrow",
                "score": 0.75,
                "explanation": "Stance width is ~15% narrower than optimal for your proportions",
                "parameter_delta": {"stance_width_ratio": 0.15},
            },
        ],
        "session_causes": [
            {
                "cause_id": "ankle_mobility",
                "score": 0.6,
                "explanation": "Limited ankle dorsiflexion is forcing compensatory knee cave",
            },
        ],
        "combined_perturbation": {"stance_width_ratio": 0.15},
    }


def _mock_scoring() -> dict:
    """Scoring dict matching the IPC message format."""
    return {
        "mean_score": 0.72,
        "per_dimension": {
            "depth": 0.85,
            "trunk_control": 0.60,
            "knee_tracking": 0.75,
            "symmetry": 0.90,
            "ankle": 0.50,
        },
        "best_rep": 2,
        "worst_rep": 1,
        "trend_slope": 0.02,
    }


class TestDiagnosisData:
    """Test diagnosis data storage and consumption."""

    def test_set_diagnosis_data_stores(self, orchestrator):
        diagnosis = _mock_diagnosis()
        scoring = _mock_scoring()
        orchestrator.set_diagnosis_data(diagnosis, scoring)
        assert orchestrator._pending_diagnosis is diagnosis
        assert orchestrator._pending_scoring is scoring

    def test_consume_diagnosis_returns_and_clears(self, orchestrator):
        diagnosis = _mock_diagnosis()
        scoring = _mock_scoring()
        orchestrator.set_diagnosis_data(diagnosis, scoring)
        got_diag, got_score = orchestrator._consume_diagnosis()
        assert got_diag is diagnosis
        assert got_score is scoring
        assert orchestrator._pending_diagnosis is None
        assert orchestrator._pending_scoring is None

    def test_consume_diagnosis_returns_none_when_empty(self, orchestrator):
        got_diag, got_score = orchestrator._consume_diagnosis()
        assert got_diag is None
        assert got_score is None

    def test_set_recap_with_diagnosis_enriches_prompt(self):
        """Set recap should include diagnosis data in the LLM prompt."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch.set_diagnosis_data(_mock_diagnosis(), _mock_scoring())

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_SET_RECAP,
                time.monotonic(),
                "llm_set_recap",
                data={
                    "set_number": 1, "total_reps": 5, "clean_reps": 4,
                    "avg_depth": 95.0, "depth_consistency": 3.5,
                    "avg_duration_ms": 2500, "fault_summary": {},
                },
            )
            await orch._dispatch(event)
            call_args = cbs["generate_llm_reply_fn"].call_args[0][0]
            assert "FORM SCORE: 72 out of 100" in call_args
            assert "TREND: improving" in call_args
            assert "stance" in call_args.lower()
            assert "biomechanics analysis" in call_args

        asyncio.run(_run())

    def test_set_recap_without_diagnosis_uses_original_prompt(self):
        """Without diagnosis data, the set recap should use the original prompt."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_SET_RECAP,
                time.monotonic(),
                "llm_set_recap",
                data={
                    "set_number": 1, "total_reps": 5, "clean_reps": 4,
                    "avg_depth": 95.0, "depth_consistency": 3.5,
                    "avg_duration_ms": 2500, "fault_summary": {},
                },
            )
            await orch._dispatch(event)
            call_args = cbs["generate_llm_reply_fn"].call_args[0][0]
            assert "just finished a set" in call_args
            assert "FORM SCORE" not in call_args

        asyncio.run(_run())

    def test_set_recap_stores_diagnosis_in_set_summary(self):
        """Diagnosis data should be included in _all_set_summaries for exercise recap."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        diagnosis = _mock_diagnosis()
        scoring = _mock_scoring()
        orch.set_diagnosis_data(diagnosis, scoring)

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_SET_RECAP,
                time.monotonic(),
                "llm_set_recap",
                data={
                    "set_number": 1, "total_reps": 5, "clean_reps": 4,
                    "avg_depth": 95.0, "depth_consistency": 3.5,
                    "avg_duration_ms": 2500, "fault_summary": {},
                },
            )
            await orch._dispatch(event)
            assert len(orch._all_set_summaries) == 1
            summary = orch._all_set_summaries[0]
            assert "diagnosis" in summary
            assert "scoring" in summary
            assert summary["scoring"]["mean_score"] == 0.72

        asyncio.run(_run())

    def test_exercise_recap_with_diagnosis_progression(self):
        """Exercise recap should include form score progression from diagnosed sets."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_EXERCISE_RECAP,
                time.monotonic(),
                "llm_exercise_recap",
                data={
                    "all_set_summaries": [
                        {
                            "set_number": 1, "total_reps": 5, "clean_reps": 3,
                            "avg_depth": 90.0, "fault_summary": {},
                            "scoring": {"mean_score": 0.65},
                            "diagnosis": {"immediate_causes": [{"explanation": "Stance narrow"}]},
                        },
                        {
                            "set_number": 2, "total_reps": 5, "clean_reps": 4,
                            "avg_depth": 95.0, "fault_summary": {},
                            "scoring": {"mean_score": 0.78},
                            "diagnosis": {"immediate_causes": [{"explanation": "Minor knee cave"}]},
                        },
                    ],
                    "total_sets": 2,
                },
            )
            await orch._dispatch(event)
            call_args = cbs["generate_llm_reply_fn"].call_args[0][0]
            assert "Form score progression" in call_args
            assert "Set 1: 65 out of 100" in call_args
            assert "Set 2: 78 out of 100" in call_args
            assert "improved by 13 points" in call_args
            assert "biomechanics analysis" in call_args

        asyncio.run(_run())


# =============================================================================
# TEST ATHLETE STATE (speech affect / effort perception)
# =============================================================================


class _FakeAthleteState:
    def __init__(self, effort="fresh", affect="flat", confident=True):
        self.effort = effort
        self.affect = affect
        self.confident = confident


class _FakeAffectService:
    def __init__(self, effort="fresh", affect="flat", confident=True):
        self.last_state = _FakeAthleteState(effort, affect, confident)
        self.rep_efforts: list[float] = []
        self.set_resets = 0

    def on_rep_effort(self, ascent_time_s: float):
        self.rep_efforts.append(ascent_time_s)
        return self.last_state

    def on_set_reset(self):
        self.set_resets += 1


class TestAthleteState:
    """Affect gates: humor, recap tone line, motivation tone, and ascent-time effort tracking."""

    def test_humor_blocked_when_frustrated(self):
        good = CoachingOrchestrator._humor_line(8, 8, 0)
        assert "welcome" in good
        blocked = CoachingOrchestrator._humor_line(8, 8, 0, affect="frustrated")
        assert "No humor" in blocked
        assert "No humor" in CoachingOrchestrator._humor_line(8, 8, 0, affect="strained")

    def test_state_line_only_when_confident_and_notable(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)
        assert orch._athlete_state_line() is None
        orch.set_athlete_state({"effort": "fresh", "affect": "flat", "confident": True})
        assert orch._athlete_state_line() is None
        orch.set_athlete_state({"effort": "working", "affect": "strained", "confident": False})
        assert orch._athlete_state_line() is None
        orch.set_athlete_state({"effort": "working", "affect": "strained", "confident": True})
        line = orch._athlete_state_line()
        assert line is not None and "strained" in line and "no humor" in line

    def test_live_service_wins_over_pushed_dict(self, mock_callbacks):
        service = _FakeAffectService(effort="near_limit", affect="flat")
        orch = CoachingOrchestrator(**mock_callbacks, affect_service=service)
        orch.set_athlete_state({"effort": "fresh", "affect": "flat", "confident": True})
        assert orch._current_athlete_state()["effort"] == "near_limit"
        assert "near limit" in orch._athlete_state_line()

    def test_ascent_time_tracks_best_and_notifies_service(self, mock_callbacks):
        service = _FakeAffectService()
        orch = CoachingOrchestrator(**mock_callbacks, affect_service=service)
        orch.reset_set(target_reps=10)
        assert service.set_resets == 1

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [], ascent_time_s=1.0)
            await orch.on_rep_complete(2, "parallel", True, [], ascent_time_s=1.5)
            await orch.on_rep_complete(3, "parallel", True, [])

        asyncio.run(_run())
        assert service.rep_efforts == [1.0, 1.5]
        assert orch._best_ascent_time_s == 1.0
        events = orch._set_rep_events
        assert events[1]["ascent_ratio"] == 1.5
        assert events[2]["ascent_ratio"] is None

    def test_motivation_tone_near_limit(self, mock_callbacks):
        service = _FakeAffectService(effort="near_limit", affect="flat")
        orch = CoachingOrchestrator(**mock_callbacks, affect_service=service)

        asyncio.run(orch._speak_llm_motivation({"rep_number": 4, "reps_remaining": 4}))
        instructions = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        assert "near their limit" in instructions
        assert "Shout" not in instructions

    def test_recap_includes_state_line_when_strained(self, mock_callbacks):
        service = _FakeAffectService(effort="working", affect="strained")
        orch = CoachingOrchestrator(**mock_callbacks, affect_service=service)
        data = {
            "set_number": 1, "total_reps": 5, "clean_reps": 5, "shallow_reps": 0,
            "avg_depth": 95.0, "depth_consistency": 2.0, "avg_duration_ms": 2400,
            "fault_summary": {}, "per_rep": [],
        }
        asyncio.run(orch._speak_llm_set_recap(data))
        instructions = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        assert "ATHLETE STATE" in instructions
        assert "No humor" in instructions
