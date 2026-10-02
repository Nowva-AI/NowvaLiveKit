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
    UNOBSERVABLE_FAULTS_LINE,
    CoachingEvent,
    CoachingOrchestrator,
    CuePriority,
)


# =============================================================================
# FIXTURES
# =============================================================================


def _drain_events(orch) -> list:
    events = []
    while not orch._queue.empty():
        events.append(orch._queue.get_nowait())
    return events


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

    def test_on_fault_enqueues_at_rep_end(self, orchestrator):
        async def _run():
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            # Mid-rep: nothing yet — the cue is decided when the rep completes
            assert orchestrator._queue.empty()
            await orchestrator.on_rep_complete(1, "parallel", False, ["knee_valgus"])
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
            await orch.on_fault("unknown_cue", "unknown", "moderate")
            await orch.on_rep_complete(1, "parallel", False, ["unknown"])
            assert orch._queue.empty()
        asyncio.run(_run())

    def test_on_fault_rate_limited(self, orchestrator):
        async def _run():
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            await orchestrator.on_rep_complete(1, "parallel", False, ["knee_valgus"])
            assert not orchestrator._queue.empty()
            _drain_events(orchestrator)
            # Same-or-lower priority fault within gap should be skipped
            await orchestrator.on_fault("knees_out", "knee_valgus", "moderate")
            await orchestrator.on_rep_complete(2, "parallel", False, ["knee_valgus"])
            assert [e for e in _drain_events(orchestrator) if e.cue_key == "knees_out"] == []
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

    def test_on_rep_complete_best_rep_enqueues_positive(self, orchestrator):
        orchestrator._positive_cue_keys = ["good_rep", "strong"]

        async def _run():
            await orchestrator.on_rep_complete(1, "parallel", True, [], highlights=["best_rep_so_far"])
            events = []
            while not orchestrator._queue.empty():
                events.append(orchestrator._queue.get_nowait())
            positive = [e for e in events if e.priority == CuePriority.POSITIVE_CUE]
            assert len(positive) == 1
            assert positive[0].cue_key == "strong"
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
            "fault_profile": [{"fault_type": "knee_valgus", "total_occurrences": 9}],
            "chronic_faults": ["knee_valgus"],
        }

        async def _run():
            event = CoachingEvent(
                CuePriority.LLM_SET_RECAP,
                time.monotonic(),
                "llm_set_recap",
                data={
                    "set_number": 1, "total_reps": 5, "clean_reps": 4,
                    "avg_depth": 95.0, "depth_consistency": 3.5,
                    "fault_summary": {"knee_valgus": {"count": 4, "pct": 80}},
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
            "tempo": 0.50,
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
        assert orch._athlete_state_line(None) is None
        orch.set_athlete_state({"effort": "fresh", "affect": "flat", "confident": True})
        assert orch._athlete_state_line(None) is None
        orch.set_athlete_state({"effort": "working", "affect": "strained", "confident": False})
        assert orch._athlete_state_line(None) is None
        orch.set_athlete_state({"effort": "working", "affect": "strained", "confident": True})
        line = orch._athlete_state_line(None)
        assert line is not None and "strained" in line and "no humor" in line

    def test_live_service_wins_over_pushed_dict(self, mock_callbacks):
        service = _FakeAffectService(effort="fresh", affect="frustrated")
        orch = CoachingOrchestrator(**mock_callbacks, affect_service=service)
        orch.set_athlete_state({"effort": "fresh", "affect": "flat", "confident": True})
        assert orch._current_athlete_state()["affect"] == "frustrated"
        assert "frustrated" in orch._athlete_state_line(None)

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

        asyncio.run(orch._speak_llm_motivation(
            {"rep_number": 4, "reps_remaining": 4, "effort": "near_limit"}))
        instructions = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        assert "near their limit" in instructions
        assert "Shout" not in instructions

    def test_motivation_is_never_a_shout(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)
        asyncio.run(orch._speak_llm_motivation({"rep_number": 4, "reps_remaining": 4}))
        instructions = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        assert "Shout" not in instructions
        assert "calm" in instructions

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


# =============================================================================
# TEST SET RECAP WAITS FOR ITS OWN DIAGNOSIS
# =============================================================================


def _recap_prompts(callbacks: dict) -> list[str]:
    return [call[0][0] for call in callbacks["generate_llm_reply_fn"].call_args_list]


class TestRecapWaitsForItsDiagnosis:
    """The pipeline diagnoses a set only after rest_start reaches it, which is
    after the recap is queued. The recap must wait for that set's diagnosis."""

    def test_diagnosis_arriving_after_the_recap_is_queued_reaches_the_recap(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs, advance_set_fn=AsyncMock(return_value=2))
        orch.reset_set(target_reps=2, total_sets=3)

        async def _run():
            orch.start()
            try:
                await orch.on_rep_complete(1, "parallel", True, [], set_number=1)
                await orch.on_rep_complete(2, "parallel", True, [], set_number=1)
                # Recap is queued; the diagnosis lands a moment later
                await asyncio.sleep(0.2)
                orch.set_diagnosis_data(_mock_diagnosis(), _mock_scoring(), set_number=1)
                await asyncio.sleep(0.3)
            finally:
                orch.stop()

        asyncio.run(_run())
        prompts = _recap_prompts(cbs)
        assert len(prompts) == 1
        assert "FORM SCORE: 72 out of 100" in prompts[0]

    def test_previous_sets_late_diagnosis_is_not_used(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch._diagnosis_wait_s = 0.1
        orch.set_diagnosis_data(_mock_diagnosis(), _mock_scoring(), set_number=1)
        data = {
            "set_number": 2, "diagnosis_set_number": 2, "total_reps": 5, "clean_reps": 4,
            "fault_summary": {}, "per_rep": [],
        }
        asyncio.run(orch._speak_llm_set_recap(data))
        prompt = _recap_prompts(cbs)[0]
        assert "FORM SCORE" not in prompt
        assert orch._pending_diagnosis is None

    def test_recap_goes_ahead_without_diagnosis_after_the_wait(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs)
        orch._diagnosis_wait_s = 0.05
        data = {
            "set_number": 1, "diagnosis_set_number": 1, "total_reps": 5, "clean_reps": 4,
            "fault_summary": {}, "per_rep": [],
        }
        started = time.monotonic()
        asyncio.run(orch._speak_llm_set_recap(data))
        assert time.monotonic() - started < 1.0
        prompt = _recap_prompts(cbs)[0]
        assert "just finished a set" in prompt
        assert "FORM SCORE" not in prompt

    def test_exercise_recap_waits_for_the_final_sets_diagnosis(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs, advance_set_fn=AsyncMock(return_value=None))
        orch.reset_set(target_reps=1, total_sets=1)

        async def _run():
            orch.start()
            try:
                await orch.on_rep_complete(1, "parallel", True, [], set_number=1)
                await asyncio.sleep(0.2)
                orch.set_diagnosis_data(_mock_diagnosis(), _mock_scoring(), set_number=1)
                await asyncio.sleep(0.3)
            finally:
                orch.stop()

        asyncio.run(_run())
        prompts = _recap_prompts(cbs)
        assert len(prompts) == 1
        assert "Form score progression: Set 1: 72 out of 100" in prompts[0]

    def test_set_summary_carries_the_pipeline_set_number(self, orchestrator):
        asyncio.run(orchestrator.on_rep_complete(1, "parallel", True, [], set_number=4))
        assert orchestrator._build_set_summary()["diagnosis_set_number"] == 4

    def test_set_summary_falls_back_to_the_agent_set_count(self, orchestrator):
        orchestrator._set_number = 2
        asyncio.run(orchestrator.on_rep_complete(1, "parallel", True, []))
        assert orchestrator._build_set_summary()["diagnosis_set_number"] == 2


class TestSetEndedEarly:
    def test_verbally_ended_set_gets_a_recap_and_rest(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs, advance_set_fn=AsyncMock(return_value=5))
        orch.reset_set(target_reps=8, total_sets=3)

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])
            _drain_events(orch)
            ended = await orch.end_set_early(reps=4)
            return ended, _drain_events(orch)

        ended, events = asyncio.run(_run())
        assert ended is True
        assert [e.event_type for e in events] == ["llm_set_recap"]
        assert events[0].data["total_reps"] == 4
        assert events[0].data["trigger"] == "force_end"
        assert orch.resting is True

    def test_second_done_during_rest_is_ignored(self):
        cbs = _make_callbacks()
        advance = AsyncMock(return_value=5)
        orch = CoachingOrchestrator(**cbs, advance_set_fn=advance)
        orch.reset_set(target_reps=8, total_sets=3)

        async def _run():
            await orch.end_set_early(reps=3)
            return await orch.end_set_early(reps=0)

        assert asyncio.run(_run()) is False
        advance.assert_awaited_once()

    def test_verbally_ended_set_counts_toward_the_last_set(self):
        """Without counting it, the real last set ran a set recap instead of
        the exercise recap and the workout never completed."""
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs, advance_set_fn=AsyncMock(return_value=1))
        orch.reset_set(target_reps=1, total_sets=2)

        async def _run():
            await orch.end_set_early(reps=0)
            orch.on_rest_complete()
            _drain_events(orch)
            await orch.on_rep_complete(1, "parallel", True, [])
            return _drain_events(orch)

        events = asyncio.run(_run())
        assert [e.event_type for e in events] == ["llm_exercise_recap"]

    def test_nothing_counts_after_the_last_set(self):
        cbs = _make_callbacks()
        orch = CoachingOrchestrator(**cbs, advance_set_fn=AsyncMock(return_value=None))
        orch.reset_set(target_reps=1, total_sets=1)

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])
            _drain_events(orch)
            await orch.on_rep_complete(2, "parallel", True, [])
            ended = await orch.end_set_early(reps=1)
            return ended, _drain_events(orch)

        ended, events = asyncio.run(_run())
        assert ended is False
        assert events == []
        assert orch.resting is True


# =============================================================================
# TEST DIAGNOSIS CONTEXT FOR THE RECAP
# =============================================================================


def _diagnosis_with(**overrides) -> dict:
    diagnosis = _mock_diagnosis()
    diagnosis.update(overrides)
    return diagnosis


class TestDiagnosisContext:
    def test_dimensions_include_tempo_and_not_ankle(self, orchestrator):
        scoring = _mock_scoring()
        scoring["per_dimension"]["ankle"] = 0.4
        score_line = orchestrator._build_diagnosis_context(_mock_diagnosis(), scoring)[0]
        assert "tempo: 50" in score_line
        assert "ankle" not in score_line

    def test_longterm_cause_is_voiced_once_per_session(self, orchestrator):
        longterm = [{
            "cause_id": "ankle_mobility", "tier": 3, "score": 0.6,
            "explanation": "Ankle mobility work will open up your depth",
            "observability": "observable",
        }]
        first = " ".join(orchestrator._build_diagnosis_context(
            _diagnosis_with(longterm_causes=longterm), _mock_scoring()))
        second = " ".join(orchestrator._build_diagnosis_context(
            _diagnosis_with(longterm_causes=longterm), _mock_scoring()))
        assert "LONG-TERM" in first and "Ankle mobility work" in first
        assert "LONG-TERM" not in second

    def test_contextual_note_is_voiced_once_per_session(self, orchestrator):
        notes = [{
            "cause_id": "long_femurs", "tier": 0, "score": 0.5,
            "explanation": "Longer thighs mean more forward lean is normal for you",
            "observability": "observable",
        }]
        first = " ".join(orchestrator._build_diagnosis_context(
            _diagnosis_with(contextual_notes=notes), _mock_scoring()))
        second = " ".join(orchestrator._build_diagnosis_context(
            _diagnosis_with(contextual_notes=notes), _mock_scoring()))
        assert "ANATOMY NOTE" in first and "Longer thighs" in first
        assert "ANATOMY NOTE" not in second

    def test_hedges_when_the_top_cause_is_approximate(self, orchestrator):
        diagnosis = _mock_diagnosis()
        diagnosis["immediate_causes"][0]["observability"] = "approximate"
        text = " ".join(orchestrator._build_diagnosis_context(diagnosis, _mock_scoring()))
        assert "hedge" in text

    def test_hedges_when_measurement_confidence_is_low(self, orchestrator):
        text = " ".join(orchestrator._build_diagnosis_context(
            _diagnosis_with(confidence=0.3), _mock_scoring()))
        assert "hedge" in text

    def test_no_hedge_when_confident_and_observable(self, orchestrator):
        text = " ".join(orchestrator._build_diagnosis_context(_mock_diagnosis(), _mock_scoring()))
        assert "hedge" not in text

    def test_never_claims_faults_the_cameras_cannot_see(self, orchestrator):
        parts = orchestrator._build_diagnosis_context(_mock_diagnosis(), _mock_scoring())
        assert UNOBSERVABLE_FAULTS_LINE in parts
        assert "butt wink" in UNOBSERVABLE_FAULTS_LINE

    def test_top_cause_id_outlives_the_recap(self, orchestrator):
        orchestrator.set_diagnosis_data(_mock_diagnosis(), _mock_scoring())
        orchestrator._consume_diagnosis()
        assert orchestrator.top_cause_id == "stance_narrow"


class TestApproximateFaultsInRecap:
    def test_recap_marks_approximate_faults_for_hedging(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)
        asyncio.run(orch.on_fault(None, "hip_shift", "moderate", observability="approximate"))
        data = {
            "set_number": 1, "total_reps": 5, "clean_reps": 3, "per_rep": [],
            "fault_summary": {
                "hip_shift": {"count": 2, "pct": 40},
                "knee_valgus": {"count": 1, "pct": 20},
            },
        }
        asyncio.run(orch._speak_llm_set_recap(data))
        prompt = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        assert "hips sliding to one side on 2 of 5 reps (approximate)" in prompt
        assert "knees caving in on 1 of 5 reps;" in prompt or "knees caving in on 1 of 5 reps." in prompt
        assert "looked like" in prompt


# =============================================================================
# TEST POSITIVE REINFORCEMENT FROM HIGHLIGHTS
# =============================================================================


def _positive_cues(orch) -> list[str]:
    cues = []
    while not orch._queue.empty():
        event = orch._queue.get_nowait()
        if event.priority == CuePriority.POSITIVE_CUE:
            cues.append(event.cue_key)
    return cues


class TestPositiveHighlights:
    def _orchestrator(self, mock_callbacks) -> CoachingOrchestrator:
        orch = CoachingOrchestrator(**mock_callbacks)
        orch.reset_set(target_reps=12, positive_cue_keys=["strong", "clean", "great_depth"])
        return orch

    def test_new_best_rep_is_always_praised(self, mock_callbacks):
        orch = self._orchestrator(mock_callbacks)
        asyncio.run(orch.on_rep_complete(1, "parallel", True, [], highlights=["best_rep_so_far", "clean"]))
        assert _positive_cues(orch) == ["strong"]

    def test_ordinary_clean_rep_gets_no_generic_praise(self, mock_callbacks):
        orch = self._orchestrator(mock_callbacks)

        async def _run():
            for rep in range(1, 6):
                await orch.on_rep_complete(rep, "parallel", True, [], highlights=["clean", "depth_target_met"])

        asyncio.run(_run())
        assert _positive_cues(orch) == []

    def test_praise_waits_out_the_rep_gap(self, mock_callbacks):
        orch = self._orchestrator(mock_callbacks)

        async def _run():
            praised = []
            for rep in range(1, 5):
                await orch.on_rep_complete(rep, "parallel", True, [], highlights=["best_rep_so_far"])
                praised.append(_positive_cues(orch))
            return praised

        assert asyncio.run(_run()) == [["strong"], [], [], ["strong"]]

    def test_faulted_rep_is_never_praised(self, mock_callbacks):
        orch = self._orchestrator(mock_callbacks)
        asyncio.run(orch.on_rep_complete(
            1, "parallel", False, ["knee_valgus"], highlights=["best_rep_so_far"]))
        assert _positive_cues(orch) == []

    def test_highlight_without_cached_cue_is_skipped(self, mock_callbacks):
        orch = self._orchestrator(mock_callbacks)
        orch.positive_cue_keys = ["good_rep"]
        asyncio.run(orch.on_rep_complete(1, "parallel", True, [], highlights=["best_rep_so_far"]))
        assert _positive_cues(orch) == []


# =============================================================================
# ONE FOCUS PER SET (cue bandwidth, cap, safety)
# =============================================================================


async def _rep(orch, rep_number: int, *faults: tuple, ascent_time_s: float = 0.0) -> list:
    """One rep: its faults arrive, rep_complete decides; returns queued cue keys."""
    for cue_key, fault_type, severity in faults:
        await orch.on_fault(cue_key, fault_type, severity, observability="observable")
    await orch.on_rep_complete(
        rep_number, "parallel", not faults, [f[1] for f in faults], ascent_time_s=ascent_time_s,
    )
    return [e.cue_key for e in _drain_events(orch) if e.event_type == "cached_cue"]


KNEE = ("knees_out", "knee_valgus", "moderate")
KNEE_MILD = ("knees_out", "knee_valgus", "mild")
KNEE_SEVERE = ("knees_out", "knee_valgus", "severe")
LOCKOUT = ("lockout", "lockout", "moderate")


def _set_orchestrator(mock_callbacks, target_reps: int = 10) -> CoachingOrchestrator:
    orch = CoachingOrchestrator(**mock_callbacks)
    orch.reset_set(target_reps=target_reps, total_sets=3)
    orch._min_fault_cue_gap = 0.0  # judge the cue rules, not the time gap
    return orch


def _diagnosis_implicating(*symptom_ids: str, observability: str = "observable") -> dict:
    return {
        "confidence": 0.8,
        "immediate_causes": [{
            "cause_id": "some_cause", "tier": 1, "score": 0.7, "explanation": "x",
            "implicated_by": list(symptom_ids), "observability": observability,
        }],
    }


class TestCoachingMomentVoice:
    def test_recap_speaks_in_the_recap_style_and_a_new_set_clears_it(self, mock_callbacks):
        from affect import voice_style

        orch = CoachingOrchestrator(**mock_callbacks)
        seen_during_recap = []

        async def _capture(*args, **kwargs):
            seen_during_recap.append(voice_style._coaching_moment)

        mock_callbacks["generate_llm_reply_fn"].side_effect = _capture

        async def _run():
            await orch._dispatch(CoachingEvent(
                CuePriority.LLM_SET_RECAP, time.monotonic(), "llm_set_recap",
                data={
                    "set_number": 1, "total_reps": 5, "clean_reps": 5, "fault_summary": {},
                    "trigger": "rep_count",
                },
            ))
        asyncio.run(_run())

        assert seen_during_recap == ["recap"]
        assert voice_style._coaching_moment is None

    def test_near_limit_motivation_uses_the_last_rep_style(self, mock_callbacks):
        from affect import voice_style

        orch = CoachingOrchestrator(**mock_callbacks)
        seen = []

        async def _capture(*args, **kwargs):
            seen.append(voice_style._coaching_moment)

        mock_callbacks["generate_llm_reply_fn"].side_effect = _capture
        asyncio.run(orch._speak_llm_motivation({"rep_number": 4, "effort": "near_limit"}))
        voice_style.set_coaching_moment(None)
        assert seen == ["last_rep"]


def _mixed_diagnosis() -> dict:
    """An approximate foot-setup cause ranked above an observable knee cause."""
    return {
        "confidence": 0.8,
        "immediate_causes": [
            {
                "cause_id": "foot_placement_asymmetry", "tier": 1, "score": 0.8,
                "explanation": "Your feet are set up unevenly.",
                "implicated_by": ["uneven_setup"], "observability": "approximate",
            },
            {
                "cause_id": "knee_track_cue", "tier": 1, "score": 0.6,
                "explanation": "Your knees drift inside your toes.",
                "implicated_by": ["knee_not_tracking_toes"], "observability": "observable",
            },
        ],
    }


class TestOneThing:
    def test_recap_top_issue_is_the_next_sets_focus(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch.carry_focus_from(_mixed_diagnosis())
        context = " ".join(orch._build_diagnosis_context(_mixed_diagnosis(), _mock_scoring()))
        assert orch._set_focus_fault == "knee_valgus"
        assert "TOP ISSUE: Your knees drift inside your toes." in context

    def test_top_cause_falls_back_when_nothing_is_cueable(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        diagnosis = {"immediate_causes": _mixed_diagnosis()["immediate_causes"][:1]}
        assert orch.top_cause(diagnosis)["cause_id"] == "foot_placement_asymmetry"


class TestCarriedFocus:
    def test_unresolved_assessment_cause_becomes_the_first_sets_focus(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch.carry_focus_from(_diagnosis_implicating("knee_not_tracking_toes"))
        assert orch._set_focus_fault == "knee_valgus"

    def test_approximate_cause_is_not_carried(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch.carry_focus_from(_diagnosis_implicating("hip_shift", observability="approximate"))
        assert orch._set_focus_fault is None


class TestCueBandwidth:
    def test_single_mild_rep_is_not_cued(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        assert asyncio.run(_rep(orch, 1, KNEE_MILD)) == []

    def test_mild_on_two_of_the_last_three_reps_is_cued(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            first = await _rep(orch, 1, KNEE_MILD)
            second = await _rep(orch, 2)
            third = await _rep(orch, 3, KNEE_MILD)
            return first, second, third

        assert asyncio.run(_run()) == ([], [], ["knees_out"])

    def test_moderate_is_cued_straight_away(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        assert asyncio.run(_rep(orch, 1, KNEE)) == ["knees_out"]


class TestIdleTimer:
    def test_shallow_attempt_keeps_the_set_open(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await _rep(orch, 1)
            armed_after_rep = orch._idle_task
            await orch.on_shallow_rep("deeper", "Quarter")
            rearmed = orch._idle_task is not armed_after_rep and orch._idle_task is not None
            orch._cancel_idle_timer()
            return rearmed

        assert asyncio.run(_run()) is True


class TestPraiseKeysSurviveReset:
    def test_reset_without_keys_keeps_the_known_praise_cues(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)
        orch.reset_set(target_reps=5, positive_cue_keys=["strong", "clean"])
        orch.reset_set(target_reps=5)
        assert orch._positive_cue_keys == ["strong", "clean"]


class TestRepVerdictCandidates:
    def test_rate_limited_fault_still_cues_from_the_reps_own_verdict(self, mock_callbacks):
        """The pipeline sends one fault message per type per 3 s; the rep's
        faults_detailed still carries every verdict."""
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await orch.on_rep_complete(
                1, "parallel", False, ["knee_valgus"],
                faults_detailed=[{
                    "fault_type": "knee_valgus", "severity": "severe",
                    "details": {"side": "left", "observability": "observable"},
                }],
            )
            return [e.cue_key for e in _drain_events(orch) if e.event_type == "cached_cue"]

        assert asyncio.run(_run()) == ["knees_out_left"]

    def test_approximate_verdict_is_not_cued(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await orch.on_rep_complete(
                1, "parallel", False, ["hip_shift"],
                faults_detailed=[{
                    "fault_type": "hip_shift", "severity": "severe",
                    "details": {"side": "left", "observability": "approximate"},
                }],
            )
            return [e.cue_key for e in _drain_events(orch) if e.event_type == "cached_cue"]

        assert asyncio.run(_run()) == []


class TestCueCap:
    def test_at_most_two_cues_per_fault_per_set(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            heard = []
            for rep in range(1, 6):
                cues = await _rep(orch, rep, KNEE)
                for cue_key in cues:
                    await orch._dispatch_cached_cue(CoachingEvent(
                        CuePriority.FAULT_CUE, time.monotonic(), "cached_cue", cue_key,
                        data={"fault_type": "knee_valgus", "severity": "moderate", "after_rep": rep},
                    ))
                heard.extend(cues)
            return heard

        assert asyncio.run(_run()) == ["knees_out", "knees_out"]

    def test_next_set_gets_its_cues_back(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch._set_cue_history = [{"fault_type": "knee_valgus", "cue_key": "knees_out", "after_rep": 1}] * 2
        orch.reset_set(target_reps=10)
        orch._min_fault_cue_gap = 0.0
        assert asyncio.run(_rep(orch, 1, KNEE)) == ["knees_out"]


class TestSetFocus:
    def test_first_fault_cued_becomes_the_focus(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            return await _rep(orch, 1, LOCKOUT), await _rep(orch, 2, KNEE)

        first, second = asyncio.run(_run())
        assert first == ["lockout"]
        assert second == []
        assert orch.set_focus_fault == "lockout"

    def test_previous_sets_diagnosis_picks_the_focus(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch.set_diagnosis_data(_diagnosis_implicating("knee_not_tracking_toes"), _mock_scoring())
        assert orch.set_focus_fault == "knee_valgus"

        async def _run():
            return await _rep(orch, 1, LOCKOUT), await _rep(orch, 2, KNEE)

        assert asyncio.run(_run()) == ([], ["knees_out"])

    def test_severe_knee_cave_is_cued_outside_the_focus(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            return await _rep(orch, 1, LOCKOUT), await _rep(orch, 2, KNEE_SEVERE)

        assert asyncio.run(_run()) == (["lockout"], ["knees_out"])

    def test_diagnosis_after_the_set_started_keeps_the_focus(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        asyncio.run(_rep(orch, 1, LOCKOUT))
        orch.set_diagnosis_data(_diagnosis_implicating("knee_not_tracking_toes"), _mock_scoring())
        assert orch.set_focus_fault == "lockout"

    def test_approximate_cause_is_not_a_focus(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch.set_diagnosis_data(
            _diagnosis_implicating("hip_shift", observability="approximate"), _mock_scoring())
        assert orch.set_focus_fault is None

    def test_side_view_cause_is_not_a_focus_on_one_camera(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch._side_view_observable = False
        orch.set_diagnosis_data(_diagnosis_implicating("velocity_loss"), _mock_scoring())
        assert orch.set_focus_fault is None

    def test_side_view_cause_is_a_focus_on_the_rig(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch._side_view_observable = True
        orch.set_diagnosis_data(_diagnosis_implicating("hip_shoot"), _mock_scoring())
        assert orch.set_focus_fault == "hip_shoot"


class TestSideViewFaults:
    def test_no_cue_on_one_camera(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch._side_view_observable = False
        assert asyncio.run(_rep(orch, 1, ("drive", "velocity_loss", "severe"))) == []

    def test_cued_on_the_rig(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch._side_view_observable = True
        assert asyncio.run(_rep(orch, 1, ("drive", "velocity_loss", "severe"))) == ["drive"]

    def test_not_in_the_recap_on_one_camera(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)
        orch._side_view_observable = False
        data = {
            "set_number": 1, "total_reps": 5, "clean_reps": 2,
            "per_rep": [{"rep": 2, "clean": False, "faults": ["hip_shoot", "velocity_loss"]}],
            "fault_summary": {
                "hip_shoot": {"count": 2, "pct": 40},
                "velocity_loss": {"count": 1, "pct": 20},
                "balance": {"count": 1, "pct": 20},
            },
        }
        asyncio.run(orch._speak_llm_set_recap(data))
        prompt = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        for label in ("hips rising", "reps slowing", "weight drifting"):
            assert label not in prompt


class TestCueTiming:
    def test_mid_rep_fault_waits_for_rep_complete(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await orch.on_fault("heels_down", "heel_rise", "moderate", observability="observable")
            mid_rep = _drain_events(orch)
            await orch.on_rep_complete(1, "parallel", False, ["heel_rise"])
            return mid_rep, [e.cue_key for e in _drain_events(orch)]

        assert asyncio.run(_run()) == ([], ["heels_down"])

    def test_shallow_attempt_drops_its_faults(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await orch.on_fault("heels_down", "heel_rise", "moderate", observability="observable")
            await orch.on_shallow_rep("deeper", "Quarter")
            await orch.on_rep_complete(1, "parallel", True, [])
            return [e.cue_key for e in _drain_events(orch)]

        assert asyncio.run(_run()) == []


# =============================================================================
# FIX PRAISE (cue outcome)
# =============================================================================


async def _cued_knees(orch) -> None:
    """Rep 1 shows knee cave and its cue is heard."""
    await _rep(orch, 1, KNEE)
    await orch._dispatch_cached_cue(CoachingEvent(
        CuePriority.FAULT_CUE, time.monotonic(), "cached_cue", "knees_out_left",
        data={"fault_type": "knee_valgus", "severity": "moderate", "after_rep": 1},
    ))


class TestFixPraise:
    def test_fix_that_holds_two_reps_gets_its_own_praise(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await _cued_knees(orch)
            after_one = await _rep(orch, 2)
            after_two = await _rep(orch, 3)
            return after_one, after_two

        assert asyncio.run(_run()) == ([], ["knees_out_fixed"])

    def test_a_corrective_cue_takes_the_reps_slot_over_praise(self, mock_callbacks):
        """One utterance per rep: praise queued first would play first and leave
        the safety cue to go stale in the queue."""
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await _rep(orch, 1, LOCKOUT)
            await orch._dispatch_cached_cue(CoachingEvent(
                CuePriority.FAULT_CUE, time.monotonic(), "cached_cue", "lockout",
                data={"fault_type": "lockout", "severity": "moderate", "after_rep": 1},
            ))
            await _rep(orch, 2)
            return await _rep(orch, 3, KNEE_SEVERE)

        assert asyncio.run(_run()) == ["knees_out"]

    def test_falls_back_to_the_shared_confirmation(self, mock_callbacks):
        mock_callbacks["get_cue_audio_fn"] = lambda key: bool(key) and not key.endswith("_fixed")
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await _cued_knees(orch)
            await _rep(orch, 2)
            return await _rep(orch, 3)

        assert asyncio.run(_run()) == ["adjust_good"]

    def test_persisting_fault_gets_silence(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch._min_fault_cue_gap = 60.0  # no second cue either

        async def _run():
            await _cued_knees(orch)
            heard = await _rep(orch, 2, KNEE)
            heard += await _rep(orch, 3)
            heard += await _rep(orch, 4)
            return heard

        assert asyncio.run(_run()) == []
        assert orch._pending_outcome is None

    def test_no_praise_after_the_set_completing_rep(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks, advance_set_fn=AsyncMock(return_value=3))
        orch.reset_set(target_reps=3, total_sets=3)
        orch._min_fault_cue_gap = 0.0

        async def _run():
            await _cued_knees(orch)
            await _rep(orch, 2)
            return await _rep(orch, 3)

        heard = asyncio.run(_run())
        assert "knees_out_fixed" not in heard
        assert "adjust_good" not in heard

    def test_recap_states_the_cue_outcome(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks, advance_set_fn=AsyncMock(return_value=5))
        orch.reset_set(target_reps=5, total_sets=3)
        orch._min_fault_cue_gap = 0.0

        async def _run():
            await _cued_knees(orch)
            for rep in range(2, 6):
                await _rep(orch, rep)

        asyncio.run(_run())
        orch._diagnosis_wait_s = 0.0
        asyncio.run(orch._speak_llm_set_recap(orch.last_set_data))
        prompt = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        assert "CUE OUTCOME" in prompt
        assert "rep 1" in prompt and "reps 2 to 5" in prompt


# =============================================================================
# SET ENDS WITHOUT HITTING THE TARGET
# =============================================================================


class TestIdleSetEnd:
    def test_set_ends_when_the_reps_stop(self, mock_callbacks):
        advance = AsyncMock(return_value=8)
        orch = CoachingOrchestrator(**mock_callbacks, advance_set_fn=advance)
        orch.reset_set(target_reps=8, total_sets=3)
        orch._set_idle_timeout_s = 0.05

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])
            await orch.on_rep_complete(2, "parallel", True, [])
            await asyncio.sleep(0.15)
            return _drain_events(orch)

        events = asyncio.run(_run())
        recaps = [e for e in events if e.event_type == "llm_set_recap"]
        assert len(recaps) == 1
        assert recaps[0].data["trigger"] == "idle"
        assert recaps[0].data["total_reps"] == 2
        advance.assert_awaited_once()
        assert orch.resting is True

    def test_each_rep_restarts_the_clock(self, mock_callbacks):
        advance = AsyncMock(return_value=8)
        orch = CoachingOrchestrator(**mock_callbacks, advance_set_fn=advance)
        orch.reset_set(target_reps=8, total_sets=3)
        orch._set_idle_timeout_s = 0.1

        async def _run():
            for rep in range(1, 5):
                await orch.on_rep_complete(rep, "parallel", True, [])
                await asyncio.sleep(0.06)

        asyncio.run(_run())
        advance.assert_not_awaited()

    def test_no_clock_before_the_first_rep(self, mock_callbacks):
        advance = AsyncMock(return_value=8)
        orch = CoachingOrchestrator(**mock_callbacks, advance_set_fn=advance)
        orch.reset_set(target_reps=8, total_sets=3)
        orch._set_idle_timeout_s = 0.01

        async def _run():
            await asyncio.sleep(0.05)

        asyncio.run(_run())
        advance.assert_not_awaited()

    def test_last_set_stopped_short_gets_the_exercise_recap(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks, advance_set_fn=AsyncMock(return_value=None))
        orch.reset_set(target_reps=8, total_sets=1)
        orch._set_idle_timeout_s = 0.05

        async def _run():
            await orch.on_rep_complete(1, "parallel", True, [])
            await asyncio.sleep(0.15)
            return _drain_events(orch)

        assert [e.event_type for e in asyncio.run(_run())] == ["llm_exercise_recap"]


# =============================================================================
# EFFORT FROM REP SPEED
# =============================================================================


class TestTempoEffort:
    def test_needs_three_reps(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)

        async def _run():
            await _rep(orch, 1, ascent_time_s=1.0)
            await _rep(orch, 2, ascent_time_s=1.5)

        asyncio.run(_run())
        assert orch._tempo_effort() is None

    def test_slow_last_rep_is_near_the_limit_without_voice_confidence(self, mock_callbacks):
        service = _FakeAffectService(effort="fresh", affect="flat", confident=False)
        orch = CoachingOrchestrator(**mock_callbacks, affect_service=service)
        orch.reset_set(target_reps=10)

        async def _run():
            for rep, ascent in enumerate((1.0, 1.1, 1.5), start=1):
                await _rep(orch, rep, ascent_time_s=ascent)

        asyncio.run(_run())
        assert orch._tempo_effort() == "near_limit"
        assert "near their limit" in orch._athlete_state_line(orch._tempo_effort())

    def test_velocity_loss_means_near_limit_on_the_rig(self, mock_callbacks):
        orch = _set_orchestrator(mock_callbacks)
        orch._side_view_observable = True

        async def _run():
            await _rep(orch, 1, ascent_time_s=1.0)
            await _rep(orch, 2, ascent_time_s=1.0)
            await _rep(orch, 3, ("drive", "velocity_loss", "mild"), ascent_time_s=1.05)

        asyncio.run(_run())
        assert orch._tempo_effort() == "near_limit"


# =============================================================================
# RECAP CONTENT
# =============================================================================


class TestRecapContent:
    def _data(self, **overrides) -> dict:
        data = {
            "set_number": 1, "total_reps": 8, "clean_reps": 6, "shallow_reps": 0,
            "avg_depth": 92.7, "depth_consistency": 3.2, "avg_duration_ms": 1300,
            "fault_summary": {"knee_valgus": {"count": 2, "pct": 25}},
            "per_rep": [
                {"rep": 3, "depth_angle": 95.4, "clean": True, "faults": []},
                {"rep": 4, "depth_angle": 90.1, "clean": False, "faults": ["knee_valgus"]},
            ],
            "load": "60 kg", "effort": "fresh", "focus_fault": "knee_valgus",
            "cue_outcomes": [], "target_reps": 8,
        }
        data.update(overrides)
        return data

    def _prompt(self, mock_callbacks, orch=None, **overrides) -> str:
        orch = orch or CoachingOrchestrator(**mock_callbacks)
        asyncio.run(orch._speak_llm_set_recap(self._data(**overrides)))
        return mock_callbacks["generate_llm_reply_fn"].call_args[0][0]

    def test_no_decimal_dump(self, mock_callbacks):
        prompt = self._prompt(mock_callbacks)
        assert "92.7" not in prompt and "95.4" not in prompt and "1.3" not in prompt
        assert "degrees on average" not in prompt
        assert "very consistent" in prompt

    def test_three_beats_and_the_load(self, mock_callbacks):
        prompt = self._prompt(mock_callbacks)
        assert "three beats" in prompt
        assert "Load: 60 kg." in prompt
        assert "focus was knees caving in" in prompt

    def test_stopping_short_is_named_without_blame(self, mock_callbacks):
        prompt = self._prompt(mock_callbacks, total_reps=5, clean_reps=5)
        assert "5 of 8 target reps" in prompt

    def test_never_asks_a_question_the_athlete_cannot_answer(self, mock_callbacks):
        """The mic is off during rest unless the athlete says the wake word."""
        assert "how hard" not in self._prompt(mock_callbacks, effort="working")

    def test_no_humor_when_rep_speed_says_near_limit(self, mock_callbacks):
        prompt = self._prompt(mock_callbacks, effort="near_limit", clean_reps=8, fault_summary={})
        assert "No humor" in prompt
        assert "humor is welcome" not in prompt

    def test_no_effort_question_on_a_steady_set(self, mock_callbacks):
        assert "how hard" not in self._prompt(mock_callbacks, effort="fresh")

    def test_last_set_note_goes_with_the_recap(self, mock_callbacks):
        self._prompt(mock_callbacks)
        note = mock_callbacks["generate_llm_reply_fn"].call_args.kwargs["last_set_note"]
        assert note.startswith("Set 1: 6 of 8 reps clean")
        assert "load 60 kg" in note

    def test_unhelpful_cue_history_is_flagged_once(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)
        orch.ineffective_cue_faults = {"knee_valgus"}
        first = self._prompt(mock_callbacks, orch=orch)
        second = self._prompt(mock_callbacks, orch=orch)
        assert "CUE HISTORY" in first
        assert "CUE HISTORY" not in second

    def test_exercise_recap_is_short_and_plain(self, mock_callbacks):
        orch = CoachingOrchestrator(**mock_callbacks)
        event_data = {
            "all_set_summaries": [{
                "set_number": 1, "total_reps": 5, "clean_reps": 4, "avg_depth": 92.7,
                "avg_duration_ms": 1300, "fault_summary": {"velocity_loss": {"count": 2}},
            }],
            "total_sets": 1,
        }
        asyncio.run(orch._speak_llm_exercise_recap(event_data))
        prompt = mock_callbacks["generate_llm_reply_fn"].call_args[0][0]
        assert "2-3 short sentences" in prompt
        assert "92.7" not in prompt and "1.3 seconds" not in prompt
        assert "high note" not in prompt


class TestIneffectiveCueFaults:
    def test_flags_a_cue_that_rarely_helped(self):
        from agent.services.coaching_orchestrator import ineffective_cue_faults

        rows = [
            {"fault_type": "knee_valgus", "cue_key": "knees_out", "n_evaluated": 4, "n_effective": 0},
            {"fault_type": "knee_valgus", "cue_key": "knees_out_left", "n_evaluated": 3, "n_effective": 1},
            {"fault_type": "lockout", "cue_key": "lockout", "n_evaluated": 6, "n_effective": 4},
        ]
        assert ineffective_cue_faults(rows) == {"knee_valgus"}

    def test_needs_enough_attempts(self):
        from agent.services.coaching_orchestrator import ineffective_cue_faults

        rows = [{"fault_type": "knee_valgus", "cue_key": "knees_out", "n_evaluated": 3, "n_effective": 0}]
        assert ineffective_cue_faults(rows) == set()

    def test_ignores_faults_that_got_no_cue(self):
        from agent.services.coaching_orchestrator import ineffective_cue_faults

        rows = [{"fault_type": "hip_shift", "cue_key": None, "n_evaluated": 9, "n_effective": 0}]
        assert ineffective_cue_faults(rows) == set()
