"""Tests for fault cue priority and side-aware cue selection.

With a flat rate limit the most frequent faults claimed every cue slot and
rarer, more important faults were never heard. Priorities follow the squat
contract: knee cave, hip shoot, heel rise, balance, hip shift, bar tilt,
depth, foot setup, lockout, tempo, depth drift, velocity loss.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.coaching_orchestrator import CoachingOrchestrator
from biomechanics.coaching.cue_cache import (
    DEFAULT_FAULT_CUE_PRIORITY,
    FAULT_TO_CUE_MAP,
    SQUAT_CUES,
    CueCache,
    base_cue_key,
    fault_cue_priority,
)

CONTRACT_PRIORITY_ORDER = [
    "knee_valgus", "hip_shoot", "heel_rise", "balance", "hip_shift",
    "bilateral_asymmetry", "depth", "foot_placement", "lockout", "tempo",
    "depth_drift", "velocity_loss",
]

CONTRACT_CUE_KEYS = {
    "knee_valgus": "knees_out",
    "hip_shoot": "chest_up",
    "heel_rise": "heels_down",
    "balance": "whole_foot",
    "hip_shift": "even_it_out",
    "bilateral_asymmetry": "level_bar",
    "depth": "deeper",
    "foot_placement": "square_feet",
    "lockout": "lockout",
    "tempo": "slow_down",
    "depth_drift": "same_depth",
    "velocity_loss": "drive",
}

SIDED_CUE_KEYS = ["knees_out", "heels_down", "even_it_out", "square_feet"]


def _orchestrator(get_cue_audio_fn=lambda key: bool(key)) -> CoachingOrchestrator:
    return CoachingOrchestrator(
        play_cached_audio_fn=AsyncMock(),
        generate_llm_reply_fn=AsyncMock(),
        get_cue_audio_fn=get_cue_audio_fn,
    )


def _drain(orch) -> list[str]:
    out = []
    while not orch._queue.empty():
        out.append(orch._queue.get_nowait().cue_key)
    return out


def _squat_cache() -> CueCache:
    cache = CueCache()
    cache.prepare_for_exercise("squat")
    return cache


class TestPriorityOrder:
    def test_priorities_follow_the_contract_order(self):
        priorities = [fault_cue_priority(fault_type) for fault_type in CONTRACT_PRIORITY_ORDER]
        assert priorities == list(range(len(CONTRACT_PRIORITY_ORDER)))

    def test_unlisted_faults_rank_after_every_squat_fault(self):
        assert fault_cue_priority("forward_lean") == DEFAULT_FAULT_CUE_PRIORITY
        assert DEFAULT_FAULT_CUE_PRIORITY > fault_cue_priority("velocity_loss")


class TestFaultToCueMap:
    def test_every_squat_fault_maps_to_its_contract_cue(self):
        for fault_type, cue_key in CONTRACT_CUE_KEYS.items():
            assert FAULT_TO_CUE_MAP[fault_type] == cue_key
            assert cue_key in SQUAT_CUES

    def test_every_side_variant_is_a_squat_cue(self):
        for cue_key in SIDED_CUE_KEYS:
            assert f"{cue_key}_left" in SQUAT_CUES
            assert f"{cue_key}_right" in SQUAT_CUES

    def test_dead_cue_keys_are_dropped(self):
        for cue_key in ("hips_through", "flat_back"):
            assert cue_key not in SQUAT_CUES
            assert cue_key not in FAULT_TO_CUE_MAP.values()

    def test_forward_lean_stays_mapped_for_other_profiles(self):
        assert FAULT_TO_CUE_MAP["forward_lean"] == "chest_up"


class TestSideAwareCue:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_side_picks_the_side_variant(self, side):
        assert _squat_cache().get_cue_for_fault("knee_valgus", 100.0, side=side) == f"knees_out_{side}"

    @pytest.mark.parametrize("side", [None, "both", ""])
    def test_no_single_side_gets_the_base_cue(self, side):
        assert _squat_cache().get_cue_for_fault("knee_valgus", 100.0, side=side) == "knees_out"

    def test_fault_without_side_variants_gets_the_base_cue(self):
        assert _squat_cache().get_cue_for_fault("hip_shoot", 100.0, side="left") == "chest_up"

    def test_exercise_without_the_side_variant_falls_back_to_base(self):
        cache = _squat_cache()
        del cache.cues["heels_down_left"]
        assert cache.get_cue_for_fault("heel_rise", 100.0, side="left") == "heels_down"

    def test_base_cue_key_strips_the_side(self):
        assert base_cue_key("knees_out_left") == "knees_out"
        assert base_cue_key("square_feet_right") == "square_feet"
        assert base_cue_key("chest_up") == "chest_up"


class TestCueCachePreemption:
    def test_knee_valgus_preempts_same_frame(self):
        """Two faults on the same detection frame: knee cave (priority 0) wins."""
        cache = _squat_cache()
        assert cache.get_cue_for_fault("hip_shift", 100.0) == "even_it_out"
        assert cache.get_cue_for_fault("knee_valgus", 100.0) == "knees_out"

    def test_lower_priority_cannot_preempt_knee_valgus(self):
        cache = _squat_cache()
        assert cache.get_cue_for_fault("knee_valgus", 100.0) == "knees_out"
        assert cache.get_cue_for_fault("hip_shoot", 100.0) is None

    def test_same_fault_still_respects_the_gap(self):
        cache = _squat_cache()
        assert cache.get_cue_for_fault("hip_shoot", 100.0) == "chest_up"
        assert cache.get_cue_for_fault("hip_shoot", 100.0 + 0.5) is None

    def test_everything_flows_again_once_the_gap_expires(self):
        cache = _squat_cache()
        assert cache.get_cue_for_fault("knee_valgus", 100.0) == "knees_out"
        later = 100.0 + cache.min_cue_gap + 0.1
        assert cache.get_cue_for_fault("velocity_loss", later) == "drive"


class TestOrchestratorPreemption:
    def test_knee_valgus_jumps_the_fault_gap(self):
        async def _run():
            orch = _orchestrator()
            await orch.on_fault("chest_up", "hip_shoot", "mild")
            orch._last_fault_cue_time -= 4.7  # pretend 4.7s elapsed
            await orch.on_fault("knees_out", "knee_valgus", "moderate")
            assert "knees_out" in _drain(orch)

        asyncio.run(_run())

    def test_lower_priority_does_not_jump_the_gap(self):
        async def _run():
            orch = _orchestrator()
            await orch.on_fault("knees_out", "knee_valgus", "moderate")
            orch._last_fault_cue_time -= 4.7
            await orch.on_fault("level_bar", "bilateral_asymmetry", "mild")
            assert _drain(orch) == ["knees_out"]

        asyncio.run(_run())

    def test_queued_faults_dispatch_most_important_first(self):
        async def _run():
            orch = _orchestrator()
            # All three enqueue because each has strictly higher priority.
            await orch.on_fault("drive", "velocity_loss", "mild")
            await orch.on_fault("even_it_out", "hip_shift", "mild")
            await orch.on_fault("knees_out_left", "knee_valgus", "mild")
            assert _drain(orch) == ["knees_out_left", "even_it_out", "drive"]

        asyncio.run(_run())

    def test_priority_resets_between_sets(self):
        async def _run():
            orch = _orchestrator()
            await orch.on_fault("knees_out", "knee_valgus", "moderate")
            orch.reset_set(target_reps=5)
            # A new set must not be blocked by the previous set's top cue.
            await orch.on_fault("drive", "velocity_loss", "mild")
            assert "drive" in _drain(orch)

        asyncio.run(_run())


class TestOrchestratorCueSelection:
    def test_side_cue_is_played_as_given(self):
        async def _run():
            orch = _orchestrator()
            await orch.on_fault("knees_out_right", "knee_valgus", "moderate")
            assert _drain(orch) == ["knees_out_right"]

        asyncio.run(_run())

    def test_side_cue_without_audio_falls_back_to_base(self):
        async def _run():
            orch = _orchestrator(get_cue_audio_fn=lambda key: key == "knees_out")
            await orch.on_fault("knees_out_left", "knee_valgus", "moderate")
            assert _drain(orch) == ["knees_out"]

        asyncio.run(_run())

    def test_approximate_fault_is_never_cued(self):
        async def _run():
            orch = _orchestrator()
            await orch.on_fault("chest_up", "hip_shoot", "severe", observability="approximate")
            assert _drain(orch) == []
            assert orch._recent_faults == []

        asyncio.run(_run())

    def test_approximate_fault_does_not_take_the_cue_slot(self):
        async def _run():
            orch = _orchestrator()
            await orch.on_fault(None, "knee_valgus", "severe", observability="approximate")
            await orch.on_fault("drive", "velocity_loss", "mild", observability="observable")
            assert _drain(orch) == ["drive"]

        asyncio.run(_run())
