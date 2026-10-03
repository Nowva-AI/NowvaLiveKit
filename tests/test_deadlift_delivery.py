"""Deadlift delivery slice (.claude/deadlift/CONTRACT.md §1, §3-§5): cue cache,
cue text, the voice agent's deadlift behaviour, and the squat left unchanged.

Every IPC message here is synthetic and shaped as the contract freezes it.
Nothing calls a network API.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import agent.services.coaching_orchestrator as orchestrator_module  # noqa: E402
from agent.services.coaching_constants import (  # noqa: E402
    CUE_DISPLAY_LABELS,
    CUE_TEXT_MAP,
    FIXED_CUE_TEXT,
)
from agent.services.coaching_orchestrator import (  # noqa: E402
    MAX_ADJUSTMENT_UTTERANCES,
    SET_IDLE_TIMEOUT_S,
    CoachingEvent,
    CoachingOrchestrator,
    CuePriority,
    ExerciseCueConfig,
    bar_midfoot_cue_key,
)
from agent.services.coaching_service import POSITIVE_CUE_KEYS, CoachingService  # noqa: E402
from agent.services.progress_context import (  # noqa: E402
    FAULT_LABELS,
    build_greeting_progress_line,
    build_progress_report,
)
from biomechanics.coaching.cue_cache import (  # noqa: E402
    DEFAULT_FAULT_CUE_PRIORITY,
    FAULT_TO_CUE_MAP,
    CueCache,
    build_cue_dict,
    fault_cue_priority,
)

# CONTRACT §1, in priority order (ranks 20-30)
DEADLIFT_PRIORITY_ORDER = [
    "deadlift_bar_position", "deadlift_shoulders_behind", "deadlift_setup_hips",
    "deadlift_hips_shoot", "deadlift_bar_drift", "deadlift_lockout", "deadlift_lean_back",
    "deadlift_hip_shift", "deadlift_bar_tilt", "deadlift_bent_arms", "deadlift_velocity_loss",
]
FIRST_DEADLIFT_RANK = 20
SQUAT_PRIORITY_ORDER = [
    "knee_valgus", "hip_shoot", "heel_rise", "balance", "hip_shift",
    "bilateral_asymmetry", "depth", "foot_placement", "lockout", "tempo",
    "depth_drift", "velocity_loss",
]
DEADLIFT_FAULT_TO_CUE = {
    "deadlift_bar_position": "deadlift_bar_midfoot",
    "deadlift_shoulders_behind": "deadlift_shoulders_over",
    "deadlift_setup_hips": "deadlift_hips",
    "deadlift_hips_shoot": "deadlift_chest_with_hips",
    "deadlift_bar_drift": "deadlift_bar_close",
    "deadlift_lockout": "deadlift_lockout",
    "deadlift_lean_back": "deadlift_finish_neutral",
    "deadlift_hip_shift": "deadlift_even_feet",
    "deadlift_bar_tilt": "deadlift_level_bar",
    "deadlift_bent_arms": "deadlift_long_arms",
}
DEADLIFT_MIN_CUE_TIERS = {
    "deadlift_bar_position": "mild",
    "deadlift_shoulders_behind": "moderate",
    "deadlift_setup_hips": "severe",
    "deadlift_hips_shoot": "moderate",
    "deadlift_bar_drift": "moderate",
    "deadlift_lockout": "moderate",
    "deadlift_lean_back": "moderate",
    "deadlift_hip_shift": "moderate",
    "deadlift_bar_tilt": "moderate",
    "deadlift_bent_arms": "mild",
    "deadlift_velocity_loss": "recap",
}
CORRECTION_CUE_KEYS = [
    "deadlift_bar_midfoot", "deadlift_hips", "deadlift_hips_up", "deadlift_hips_down",
    "deadlift_shoulders_over", "deadlift_chest_with_hips", "deadlift_bar_close",
    "deadlift_lockout", "deadlift_finish_neutral", "deadlift_even_feet", "deadlift_level_bar",
    "deadlift_long_arms",
]
SIDE_CUE_KEYS = ["deadlift_even_feet_left", "deadlift_even_feet_right"]
FOOT_GUIDANCE_CUE_KEYS = ["deadlift_step_closer", "deadlift_closer", "deadlift_back"]
CONTRACT_CUE_KEYS = CORRECTION_CUE_KEYS + SIDE_CUE_KEYS + FOOT_GUIDANCE_CUE_KEYS
DEADLIFT_CUES = build_cue_dict(*CONTRACT_CUE_KEYS, "adjust_good", "good_rep", "strong", "clean", "perfect")

DEADLIFT_CACHE_CUES = {
    "type": "cache_cues",
    "exercise_name": "Barbell Deadlift",
    "profile": "deadlift",
    "cues": DEADLIFT_CUES,
    "fault_to_cue": DEADLIFT_FAULT_TO_CUE,
    "min_cue_tiers": DEADLIFT_MIN_CUE_TIERS,
    "set_idle_timeout_s": 30.0,
    "waits_for_diagnosis": True,
    "closed_loop": "deadlift_bar_midfoot",
}
SQUAT_CACHE_CUES = {
    "type": "cache_cues",
    "exercise_name": "Barbell Back Squat",
    "profile": "squat",
    "cues": build_cue_dict("knees_out", "knees_out_left", "knees_out_right", "strong"),
}

MAX_CUE_WORDS = 4
BANNED_CUE_WORDS = [
    "round", "flat back", "spine", "lumbar", "hinge", "lats", "injur", "pain",
    "eccentric", "concentric", "valgus",
]
BAR_SPEED_TOLERANCE = 1e-9


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


draft = _load_script("draft_cue_text")
generate = _load_script("generate_cue_audio")


def _orchestrator(**overrides) -> CoachingOrchestrator:
    callbacks = {
        "play_cached_audio_fn": AsyncMock(),
        "generate_llm_reply_fn": AsyncMock(),
        "get_cue_audio_fn": lambda key: bool(key),
    }
    callbacks.update(overrides)
    return CoachingOrchestrator(**callbacks)


def _deadlift_orchestrator(target_reps: int = 10, **overrides) -> CoachingOrchestrator:
    orch = _orchestrator(**overrides)
    # The deadlift is coached on the triangulated rig: its side-view faults are
    # never even emitted on one camera.
    orch._side_view_observable = True
    orch.set_exercise("Barbell Deadlift", is_squat=False)
    orch.apply_cue_config(ExerciseCueConfig.from_cache_cues(DEADLIFT_CACHE_CUES, "deadlift"))
    orch.reset_set(target_reps=target_reps, total_sets=3)
    return orch


def _fault(
    fault_type: str, severity: str, min_tier: str | None = None, **extra_details,
) -> dict:
    """One faults_detailed entry of rep_complete, as the contract shapes it."""
    details = {
        "side": None, "phase": "pull", "observability": "observable", "is_drift": False,
        "value": 6.0, "unit": "cm", "bar_source": "bar", "gravity_source": "measured",
        "min_tier": min_tier or DEADLIFT_MIN_CUE_TIERS.get(fault_type),
        **extra_details,
    }
    return {"fault_type": fault_type, "severity": severity, "details": details}


def _features(bar_speed_mps: float, velocity_loss_pct: float = 0.0) -> dict:
    return {
        "dl_schema": 1, "concentric_velocity_mps": bar_speed_mps,
        "velocity_loss_pct": velocity_loss_pct, "bar_source": "bar",
    }


async def _rep(
    orch: CoachingOrchestrator, rep_number: int, *faults: dict,
    highlights: list[str] | None = None, features: dict | None = None,
) -> None:
    await orch.on_rep_complete(
        rep_number, "n/a", not faults, [fault["fault_type"] for fault in faults],
        highlights=highlights or [], set_number=1, faults_detailed=list(faults),
        features=features,
    )


def _frame(phase: str, bar_midfoot_cm: float | None = None) -> dict:
    """frame_data as the service hands it to the orchestrator."""
    return {
        "knee_flexion_l": 20.0, "knee_flexion_r": 20.0, "trunk_flexion": 30.0,
        "rep_phase": phase, "deadlift_phase": phase,
        "bar_midfoot_live_cm": bar_midfoot_cm, "bar_source": "bar",
    }


def _drain(orch: CoachingOrchestrator) -> list[CoachingEvent]:
    events = []
    while not orch._queue.empty():
        events.append(orch._queue.get_nowait())
    return events


def _cue_keys(events: list[CoachingEvent]) -> list[str]:
    return [event.cue_key for event in events if event.event_type == "cached_cue"]


class _FakeAudioCueService:
    session = object()
    rep_track_ready = False

    async def cache_cues(self, cues: dict) -> None:
        pass


class _FakeRecorder:
    def __init__(self) -> None:
        self.exercise = "deadlift"
        self.faults: list[dict] = []
        self.reps: list[dict] = []

    def record_fault(self, message: dict) -> None:
        self.faults.append(message)

    def record_rep(self, message: dict) -> None:
        self.reps.append(message)


class _StubState:
    def __init__(self, exercise_name: str) -> None:
        self.values: dict = {"user.id": "u1", "workout.exercise_name": exercise_name}

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value


def _service(cache_cues: dict) -> tuple[CoachingService, CoachingOrchestrator, _FakeRecorder]:
    """A live service and orchestrator after the pipeline's cache_cues."""
    service = CoachingService(session=None, state=_StubState(cache_cues["exercise_name"]))
    orch = _orchestrator()
    orch.reset_set(target_reps=10, total_sets=3)
    service._coaching_orchestrator = orch
    service._audio_cue_service = _FakeAudioCueService()
    recorder = _FakeRecorder()
    recorder.exercise = cache_cues["profile"]
    service._biomech_recorder = recorder
    service._workout_active_flag = True
    asyncio.run(service._on_cache_cues(cache_cues))
    return service, orch, recorder


# =============================================================================
# CUE CACHE
# =============================================================================


class TestFaultCuePriority:
    def test_deadlift_ranks_follow_the_contract(self):
        ranks = [fault_cue_priority(fault_type) for fault_type in DEADLIFT_PRIORITY_ORDER]
        assert ranks == list(range(FIRST_DEADLIFT_RANK, FIRST_DEADLIFT_RANK + len(DEADLIFT_PRIORITY_ORDER)))

    def test_shoulders_outrank_hips_at_setup(self):
        assert fault_cue_priority("deadlift_shoulders_behind") < fault_cue_priority("deadlift_setup_hips")

    def test_squat_ranks_and_default_are_unchanged(self):
        assert [fault_cue_priority(ft) for ft in SQUAT_PRIORITY_ORDER] == list(range(len(SQUAT_PRIORITY_ORDER)))
        assert DEFAULT_FAULT_CUE_PRIORITY == len(SQUAT_PRIORITY_ORDER)


def _deadlift_cue_cache() -> CueCache:
    cache = CueCache()
    cache.cues = dict(DEADLIFT_CUES)
    cache.fault_to_cue = dict(DEADLIFT_FAULT_TO_CUE)
    return cache


class TestCueCacheVariant:
    @pytest.mark.parametrize("direction", ["up", "down"])
    def test_direction_picks_the_hip_variant(self, direction):
        cue_key = _deadlift_cue_cache().get_cue_for_fault("deadlift_setup_hips", 100.0, variant=direction)
        assert cue_key == f"deadlift_hips_{direction}"

    def test_variant_without_a_cue_gets_the_base(self):
        cue_key = _deadlift_cue_cache().get_cue_for_fault("deadlift_bar_position", 100.0, variant="forward")
        assert cue_key == "deadlift_bar_midfoot"

    def test_side_variant_works_as_for_the_squat(self):
        cue_key = _deadlift_cue_cache().get_cue_for_fault("deadlift_hip_shift", 100.0, side="left")
        assert cue_key == "deadlift_even_feet_left"

    def test_squat_cue_is_unchanged_by_a_variant_it_does_not_have(self):
        cache = CueCache()
        cache.prepare_for_exercise("squat")
        assert cache.get_cue_for_fault("knee_valgus", 100.0, side="left", variant="up") == "knees_out_left"

    def test_prepare_for_exercise_keeps_the_active_profile(self):
        cache = CueCache()
        assert cache.profile is None
        cache.prepare_for_exercise("Barbell Back Squat")
        assert cache.profile is not None
        assert cache.profile.name == cache.profile_name == "squat"


# =============================================================================
# CUE TEXT, LABELS, DRAFT AND AUDIO SCRIPTS
# =============================================================================


def _cue_lines() -> dict[str, list[str]]:
    return json.loads(draft.CUES_JSON_PATH.read_text())


class TestDeadliftCueText:
    def test_every_contract_key_has_text_a_label_and_three_short_lines(self):
        cue_lines = _cue_lines()
        for cue_key in CONTRACT_CUE_KEYS:
            assert cue_key in CUE_TEXT_MAP, cue_key
            assert cue_key in CUE_DISPLAY_LABELS, cue_key
            assert len(cue_lines[cue_key]) == draft.VARIANTS_PER_CUE, cue_key
            assert draft.over_length_lines(cue_key, cue_lines[cue_key]) == [], cue_key

    def test_every_correction_has_fix_praise(self):
        cue_lines = _cue_lines()
        for cue_key in CORRECTION_CUE_KEYS:
            praise_key = f"{cue_key}_fixed"
            assert praise_key in FIXED_CUE_TEXT, praise_key
            assert praise_key in POSITIVE_CUE_KEYS, praise_key
            assert len(cue_lines[praise_key]) == draft.VARIANTS_PER_CUE, praise_key
            assert draft.over_length_lines(praise_key, cue_lines[praise_key]) == [], praise_key

    def test_hip_base_cue_has_its_own_neutral_line(self):
        lines = _cue_lines()["deadlift_hips"]
        assert not any("higher" in line or "lower" in line for line in lines)

    def test_lines_avoid_jargon_and_never_claim_the_spine(self):
        cue_lines = _cue_lines()
        for cue_key, lines in cue_lines.items():
            if not cue_key.startswith("deadlift_"):
                continue
            for line in lines + [CUE_TEXT_MAP.get(cue_key, "")]:
                assert not any(word in line.lower() for word in BANNED_CUE_WORDS), (cue_key, line)

    def test_placeholder_cues_are_gone(self):
        for cue_key in ("deadlift_flat_back", "deadlift_even"):
            assert cue_key not in CUE_TEXT_MAP
            assert cue_key not in CUE_DISPLAY_LABELS

    def test_every_deadlift_fault_has_a_plain_label(self):
        for fault_type in DEADLIFT_PRIORITY_ORDER:
            assert fault_type in FAULT_LABELS, fault_type
            assert "deadlift" not in FAULT_LABELS[fault_type]


class TestDraftDeadliftMode:
    def test_deadlift_mode_drafts_every_contract_key_and_its_praise(self):
        expected = set(CONTRACT_CUE_KEYS) | {f"{key}_fixed" for key in CORRECTION_CUE_KEYS}
        assert set(draft.cue_keys_to_draft("deadlift")) == expected

    def test_every_deadlift_key_has_a_scenario(self):
        for cue_key in draft.cue_keys_to_draft("deadlift"):
            assert draft.scenario_for(cue_key), cue_key

    def test_deadlift_keys_get_the_deadlift_prompt(self):
        assert draft.exercise_for("deadlift_hips_up_fixed") == "deadlift"
        assert draft.exercise_for("deadlift_even_feet_left") == "deadlift"
        assert draft.exercise_for("knees_out_left") == "squat"
        assert "deadlift" in draft.SYSTEM_PROMPTS["deadlift"]
        assert draft.SYSTEM_PROMPTS["squat"] == draft.SYSTEM_PROMPT

    def test_side_scenario_names_the_side(self):
        scenario = draft.scenario_for("deadlift_even_feet_right")
        assert scenario.startswith(draft.DEADLIFT_CUE_SCENARIOS["deadlift_even_feet"])
        assert "drifting right" in scenario

    def test_scenarios_never_mention_the_spine(self):
        for cue_key, scenario in draft.DEADLIFT_CUE_SCENARIOS.items():
            assert not any(word in scenario.lower() for word in ("round", "spine", "lumbar")), cue_key

    def test_squat_mode_is_the_squat_list(self):
        squat_keys = draft.cue_keys_to_draft("squat")
        assert not any(key.startswith("deadlift_") for key in squat_keys)
        assert draft.cue_keys_to_draft()[: len(squat_keys)] == squat_keys

    def test_audio_script_covers_every_deadlift_key(self):
        cue_lines = generate.load_cue_lines(generate.CUES_JSON_PATH)
        for cue_key in draft.cue_keys_to_draft("deadlift"):
            assert cue_key in cue_lines, cue_key


# =============================================================================
# CACHE_CUES CONFIG AND FRAME_DATA WIRING
# =============================================================================


class TestCacheCuesConfig:
    def test_deadlift_fields_configure_the_orchestrator(self):
        _, orch, _ = _service(DEADLIFT_CACHE_CUES)
        assert orch.fault_to_cue == DEADLIFT_FAULT_TO_CUE
        assert orch._cue_config.min_cue_tiers == DEADLIFT_MIN_CUE_TIERS
        assert orch._set_idle_timeout_s == pytest.approx(30.0)
        assert orch._cue_config.closed_loop == "deadlift_bar_midfoot"
        assert orch._is_squat is False
        assert orch._expects_diagnosis() is True

    def test_squat_cache_cues_without_new_fields_keeps_todays_defaults(self):
        _, orch, _ = _service(SQUAT_CACHE_CUES)
        assert orch._cue_config == ExerciseCueConfig()
        assert orch.fault_to_cue == FAULT_TO_CUE_MAP
        assert orch._set_idle_timeout_s == pytest.approx(SET_IDLE_TIMEOUT_S)
        assert orch._is_squat is True
        assert orch._expects_diagnosis() is True

    def test_frame_data_carries_the_deadlift_fields(self):
        service, orch, _ = _service(DEADLIFT_CACHE_CUES)
        asyncio.run(service._handle_message({
            "type": "frame_data", "joint_angles": {"knee_flexion_l": 20.0, "knee_flexion_r": 20.0},
            "fps": 30.0, "frame_index": 7, "rep_phase": "setup",
            "deadlift_phase": "setup", "bar_midfoot_live_cm": None, "bar_source": "wrist_proxy",
        }))
        assert orch.latest_angles["deadlift_phase"] == "setup"
        assert orch.latest_angles["bar_source"] == "wrist_proxy"
        assert "bar_midfoot_live_cm" in orch.latest_angles


# =============================================================================
# §5.1 CUE FALLBACK
# =============================================================================


class TestCueFallback:
    def test_fault_without_a_pipeline_cue_uses_the_profile_map(self):
        async def _run():
            orch = _deadlift_orchestrator()
            await orch.on_fault(None, "deadlift_bar_drift", "moderate",
                                observability="observable", details=_fault("deadlift_bar_drift", "moderate")["details"])
            await _rep(orch, 1, _fault("deadlift_bar_drift", "moderate"))
            orch.record_angle_sample(_frame("floor"))
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run()) == ["deadlift_bar_close"]

    def test_hip_direction_picks_its_variant(self):
        async def _run():
            orch = _deadlift_orchestrator()
            await _rep(orch, 1, _fault("deadlift_setup_hips", "severe", direction="down", phase="setup"))
            orch.record_angle_sample(_frame("floor"))
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run()) == ["deadlift_hips_down"]

    def test_hip_variant_without_audio_falls_back_to_the_base(self):
        async def _run():
            orch = _deadlift_orchestrator(get_cue_audio_fn=lambda key: key == "deadlift_hips")
            await _rep(orch, 1, _fault("deadlift_setup_hips", "severe", direction="up", phase="setup"))
            orch.record_angle_sample(_frame("floor"))
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run()) == ["deadlift_hips"]

    def test_squat_fallback_is_unchanged_after_squat_cache_cues(self):
        service, orch, _ = _service(SQUAT_CACHE_CUES)

        async def _run():
            await service._handle_message({
                "type": "fault", "fault_type": "knee_valgus", "severity": "moderate",
                "cue": None, "rep_number": 2, "side": "right", "observability": "observable",
                "details": {"side": "right", "phase": "ascent", "observability": "observable"},
            })
            await service._handle_message({
                "type": "rep_complete", "rep_number": 2, "is_clean": False,
                "faults_in_rep": ["knee_valgus"],
            })
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run()) == ["knees_out_right"]

    def test_rest_complete_line_names_the_deadlift_cue(self):
        service, orch, _ = _service(DEADLIFT_CACHE_CUES)
        orch._set_focus_fault = "deadlift_bar_drift"
        prompt = service._build_rest_complete_prompt()
        assert "bar drifting away from the legs" in prompt
        assert CUE_TEXT_MAP["deadlift_bar_close"] in prompt


# =============================================================================
# §5.2 MINIMUM CUE TIER
# =============================================================================


class TestMinimumTier:
    def test_fault_under_its_static_tier_is_recorded_but_not_cued(self):
        async def _run():
            orch = _deadlift_orchestrator()
            for rep_number in (1, 2, 3):
                await _rep(orch, rep_number, _fault("deadlift_bar_drift", "mild"))
                orch.record_angle_sample(_frame("floor"))
            return orch, _cue_keys(_drain(orch))

        orch, cues = asyncio.run(_run())
        assert cues == []
        assert orch._build_set_summary()["fault_summary"]["deadlift_bar_drift"]["count"] == 3

    def test_rep_tier_raises_the_static_tier(self):
        """D1 is mild by default; a rep measured from the wrist proxy says moderate."""
        async def _run(min_tier: str):
            orch = _deadlift_orchestrator()
            for rep_number in (1, 2):
                await _rep(orch, rep_number, _fault("deadlift_bar_position", "mild", min_tier=min_tier))
                orch.record_angle_sample(_frame("floor"))
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run("mild")) == ["deadlift_bar_midfoot"]
        assert asyncio.run(_run("moderate")) == []

    def test_static_tier_holds_when_the_rep_tier_is_lower(self):
        async def _run():
            orch = _deadlift_orchestrator()
            await _rep(orch, 1, _fault("deadlift_setup_hips", "moderate", min_tier="mild"))
            orch.record_angle_sample(_frame("floor"))
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run()) == []

    def test_recap_only_fault_is_never_cued(self):
        async def _run():
            orch = _deadlift_orchestrator()
            await _rep(orch, 1, _fault("deadlift_velocity_loss", "severe", unit="pct"))
            orch.record_angle_sample(_frame("floor"))
            return orch, _cue_keys(_drain(orch))

        orch, cues = asyncio.run(_run())
        assert cues == []
        assert "deadlift_velocity_loss" in orch._build_set_summary()["fault_summary"]

    def test_fault_under_its_tier_still_reaches_the_db(self):
        service, orch, recorder = _service(DEADLIFT_CACHE_CUES)
        message = {
            "type": "fault", "fault_type": "deadlift_bar_drift", "severity": "mild",
            "cue": None, "rep_number": 1, "side": None, "observability": "observable",
            "details": _fault("deadlift_bar_drift", "mild")["details"],
        }
        asyncio.run(service._handle_message(message))
        assert recorder.faults == [message]
        assert orch._rep_fault_candidates == {}

    def test_squat_faults_have_no_minimum_tier(self):
        _, orch, _ = _service(SQUAT_CACHE_CUES)
        assert not orch._below_min_tier("knee_valgus", "mild", None)
        assert orch._reached_min_tier("knee_valgus")


def _diagnosis_implicating(symptom_id: str) -> dict:
    return {
        "confidence": 0.8,
        "immediate_causes": [{
            "cause_id": "bar_too_far", "score": 0.7, "tier": 1, "observability": "observable",
            "explanation": "Start with the bar about 4 cm closer to your shins.",
            "parameter_delta": {"bar_midfoot_cm": -4.0}, "implicated_by": [symptom_id],
        }],
    }


class TestDiagnosisFocusTier:
    @pytest.mark.parametrize("severity,adopted", [("mild", False), ("moderate", True)])
    def test_focus_is_adopted_only_once_its_fault_reached_its_tier(self, monkeypatch, severity, adopted):
        monkeypatch.setitem(orchestrator_module.SYMPTOM_FAULT_TYPES, "bar_drift", "deadlift_bar_drift")

        async def _run():
            orch = _deadlift_orchestrator(advance_set_fn=AsyncMock(return_value=1))
            orch.reset_set(target_reps=1, total_sets=3)
            await _rep(orch, 1, _fault("deadlift_bar_drift", severity))
            orch.set_diagnosis_data(_diagnosis_implicating("bar_drift"), {"mean_score": 0.7}, set_number=1)
            return orch.set_focus_fault

        assert asyncio.run(_run()) == ("deadlift_bar_drift" if adopted else None)


# =============================================================================
# §5.3 TOUCH-AND-GO: NO CUE BETWEEN REPS
# =============================================================================


class TestTouchAndGo:
    def test_cue_waits_for_the_dead_stop(self):
        async def _run():
            orch = _deadlift_orchestrator()
            await _rep(orch, 1, _fault("deadlift_bar_drift", "moderate"))
            spoken_in_rep = []
            for phase in ("lower", "pull", "top"):
                orch.record_angle_sample(_frame(phase))
                spoken_in_rep += _drain(orch)
            orch.record_angle_sample(_frame("floor"))
            return spoken_in_rep, _drain(orch)

        spoken_in_rep, at_floor = asyncio.run(_run())
        assert spoken_in_rep == []
        assert _cue_keys(at_floor) == ["deadlift_bar_close"]
        assert time.monotonic() - at_floor[0].timestamp < 1.0

    def test_cue_from_a_touch_and_go_rep_waits_through_the_next_rep(self):
        async def _run():
            orch = _deadlift_orchestrator()
            orch._play_cached = AsyncMock()
            await _rep(orch, 1, _fault("deadlift_bar_drift", "moderate"))
            orch._held_rep_events["cue"].timestamp -= 5.0
            orch.record_angle_sample(_frame("pull"))  # touch-and-go: straight into rep 2
            await _rep(orch, 2, highlights=["clean", "best_rep_so_far"], features=_features(0.6))
            orch.record_angle_sample(_frame("floor"))
            return _drain(orch)

        released = asyncio.run(_run())
        assert _cue_keys(released) == ["deadlift_bar_close"]
        # Fresh, so the dispatcher does not drop it, and judged on the reps after it is heard
        assert time.monotonic() - released[0].timestamp < 1.0
        assert released[0].data["after_rep"] == 2

    @pytest.mark.parametrize("phase", ["floor", "stance", "setup", "approach"])
    def test_every_dead_stop_phase_releases_the_cue(self, phase):
        async def _run():
            orch = _deadlift_orchestrator()
            await _rep(orch, 1, _fault("deadlift_bar_drift", "moderate"))
            orch.record_angle_sample(_frame(phase))
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run()) == ["deadlift_bar_close"]

    def test_cue_pending_at_the_last_rep_goes_into_the_recap(self):
        async def _run():
            orch = _deadlift_orchestrator(advance_set_fn=AsyncMock(return_value=2))
            orch.reset_set(target_reps=2, total_sets=3)
            await _rep(orch, 1, _fault("deadlift_bar_drift", "moderate"))
            orch.record_angle_sample(_frame("pull"))
            await _rep(orch, 2, _fault("deadlift_bar_drift", "moderate"))
            return orch, _drain(orch)

        orch, events = asyncio.run(_run())
        assert [event.event_type for event in events] == ["llm_set_recap"]
        recap = events[0].data
        assert recap["focus_fault"] == "deadlift_bar_drift"
        assert recap["fault_summary"]["deadlift_bar_drift"]["count"] == 2
        assert orch._held_rep_events == {}

    def test_rep_count_still_plays_at_once(self):
        async def _run():
            orch = _deadlift_orchestrator()
            await _rep(orch, 1, _fault("deadlift_bar_drift", "moderate"))
            await asyncio.sleep(0)
            return orch

        orch = asyncio.run(_run())
        orch._play_cached.assert_awaited_once_with("rep_1")

    def test_squat_cue_is_not_held(self):
        _, orch, _ = _service(SQUAT_CACHE_CUES)

        async def _run():
            await orch.on_fault("knees_out", "knee_valgus", "moderate", observability="observable")
            await orch.on_rep_complete(1, "parallel", False, ["knee_valgus"])
            return _cue_keys(_drain(orch))

        assert asyncio.run(_run()) == ["knees_out"]


# =============================================================================
# §5.4 SET IDLE TIMEOUT
# =============================================================================


class TestSetIdleTimeout:
    def test_default_is_fifteen_seconds(self):
        assert _orchestrator()._set_idle_timeout_s == pytest.approx(15.0)

    def test_set_ends_after_the_profiles_timeout(self):
        config = ExerciseCueConfig.from_cache_cues({**DEADLIFT_CACHE_CUES, "set_idle_timeout_s": 0.05}, "deadlift")

        async def _run():
            orch = _orchestrator(advance_set_fn=AsyncMock(return_value=8))
            orch.apply_cue_config(config)
            orch.reset_set(target_reps=8, total_sets=3)
            await _rep(orch, 1)
            await asyncio.sleep(0.15)
            return _drain(orch)

        events = asyncio.run(_run())
        assert [event.data["trigger"] for event in events if event.event_type == "llm_set_recap"] == ["idle"]


# =============================================================================
# §5.5 CLOSED-LOOP D1 FOOT GUIDANCE
# =============================================================================


def _guided_orchestrator() -> CoachingOrchestrator:
    orch = _deadlift_orchestrator()
    orch._feedback_interval = 0.0
    return orch


async def _stand_at_bar(orch: CoachingOrchestrator, *offsets_cm: float | None) -> list[str]:
    """Stance frames, one per offset; returns the cue keys played."""
    for offset_cm in offsets_cm:
        orch.record_angle_sample(_frame("stance", offset_cm))
        await asyncio.sleep(0)
    return [call.args[0] for call in orch._play_cached.await_args_list]


class TestClosedLoopGuidance:
    @pytest.mark.parametrize("offset_cm,cue_key", [
        (25.0, "deadlift_step_closer"),
        (15.5, "deadlift_step_closer"),
        (15.0, "deadlift_closer"),
        (2.5, "deadlift_closer"),
        (2.0, "adjust_good"),
        (-1.5, "adjust_good"),
        (-4.0, "deadlift_back"),
    ])
    def test_offset_picks_the_cue(self, offset_cm, cue_key):
        assert bar_midfoot_cue_key(offset_cm) == cue_key

    def test_guidance_walks_the_feet_in_then_confirms_once(self):
        orch = _guided_orchestrator()
        played = asyncio.run(_stand_at_bar(orch, 25.0, 9.0, 1.0, 1.0))
        assert played == ["deadlift_step_closer", "deadlift_closer", "adjust_good"]

    def test_bar_behind_midfoot_says_back(self):
        orch = _guided_orchestrator()
        assert asyncio.run(_stand_at_bar(orch, -5.0, 0.5)) == ["deadlift_back", "adjust_good"]

    def test_nothing_to_confirm_when_already_over_midfoot(self):
        orch = _guided_orchestrator()
        assert asyncio.run(_stand_at_bar(orch, 1.0, -1.0)) == []

    def test_arms_only_past_the_mild_threshold(self):
        orch = _guided_orchestrator()
        assert asyncio.run(_stand_at_bar(orch, 2.5, -2.8, 3.0)) == []
        assert not orch._bar_guidance_armed

    def test_once_armed_it_guides_down_to_the_tolerance(self):
        orch = _guided_orchestrator()
        played = asyncio.run(_stand_at_bar(orch, 4.0, 2.5, 1.5))
        assert played == ["deadlift_closer", "deadlift_closer", "adjust_good"]

    def test_unmeasured_offset_is_ignored(self):
        orch = _guided_orchestrator()
        assert asyncio.run(_stand_at_bar(orch, None)) == []

    def test_wrist_proxy_never_guides_the_feet(self):
        async def _run():
            orch = _guided_orchestrator()
            for offset_cm in (25.0, 10.0, -5.0):
                frame = _frame("stance", offset_cm)
                frame["bar_source"] = "wrist_proxy"
                orch.record_angle_sample(frame)
                await asyncio.sleep(0)
            return orch

        orch = asyncio.run(_run())
        orch._play_cached.assert_not_awaited()
        assert not orch._bar_guidance_armed

    def test_only_speaks_while_standing_at_the_bar(self):
        async def _run():
            orch = _guided_orchestrator()
            for phase in ("approach", "floor", "top"):
                orch.record_angle_sample(_frame(phase, 30.0))
                await asyncio.sleep(0)
            return orch

        asyncio.run(_run())._play_cached.assert_not_awaited()

    @pytest.mark.parametrize("phase", ["setup", "pull"])
    def test_setup_and_pull_disarm_it(self, phase):
        async def _run():
            orch = _guided_orchestrator()
            await _stand_at_bar(orch, 10.0)
            assert orch._bar_guidance_armed
            orch.record_angle_sample(_frame(phase))
            return orch

        assert not asyncio.run(_run())._bar_guidance_armed

    def test_shares_the_stance_monitors_utterance_budget(self):
        orch = _guided_orchestrator()
        orch._adjustment_utterances = MAX_ADJUSTMENT_UTTERANCES - 1
        played = asyncio.run(_stand_at_bar(orch, 20.0, 10.0, 1.0))
        assert played == ["deadlift_step_closer"]
        assert orch._adjustment_utterances == MAX_ADJUSTMENT_UTTERANCES

    def test_waits_while_another_adjustment_speaks(self):
        orch = _guided_orchestrator()
        orch._adjustment_speaking = True
        orch.record_angle_sample(_frame("stance", 20.0))
        orch._play_cached.assert_not_called()

    def test_waits_out_the_feedback_interval(self):
        orch = _deadlift_orchestrator()
        played = asyncio.run(_stand_at_bar(orch, 20.0, 10.0))
        assert played == ["deadlift_step_closer"]

    def test_squat_profile_has_no_foot_guidance(self):
        _, orch, _ = _service(SQUAT_CACHE_CUES)
        orch._feedback_interval = 0.0
        assert asyncio.run(_stand_at_bar(orch, 20.0)) == []


# =============================================================================
# §5.6 SQUAT STANCE MONITOR GATED ON THE SQUAT PROFILE
# =============================================================================


def _stance_frame() -> dict:
    return {
        "knee_flexion_l": 5.0, "knee_flexion_r": 5.0, "rep_phase": "idle",
        "stance_width_ratio": 1.1, "foot_direction_angle_l": 10.0, "foot_direction_angle_r": 10.0,
        "target_stance_ratio": 1.5, "target_toe_out_deg": 25.0,
    }


def _fault_cue_event(cue_key: str, fault_type: str) -> CoachingEvent:
    return CoachingEvent(
        priority=CuePriority.FAULT_CUE, timestamp=time.monotonic(), event_type="cached_cue",
        cue_key=cue_key, data={"fault_type": fault_type, "severity": "moderate"},
    )


def _orchestrator_with_stale_stance_diagnosis(is_squat: bool) -> CoachingOrchestrator:
    orch = _orchestrator()
    orch._speak_adjustment_fn = AsyncMock()
    orch._get_set_load_fn = lambda: (0.0, None)
    orch.record_angle_sample(_stance_frame())
    # A squat set's diagnosis, left over when the workout moves on (O6)
    orch.set_diagnosis_data(
        {"confidence": 0.8, "immediate_causes": [{"cause_id": "narrow_stance", "explanation": "x"}]},
        {"mean_score": 0.7},
    )
    orch.set_exercise("Barbell Deadlift" if not is_squat else "Barbell Back Squat", is_squat=is_squat)
    return orch


class TestStanceMonitorGate:
    def test_stale_squat_diagnosis_never_arms_it_during_deadlifts(self):
        orch = _orchestrator_with_stale_stance_diagnosis(is_squat=False)
        orch.apply_cue_config(ExerciseCueConfig.from_cache_cues(DEADLIFT_CACHE_CUES, "deadlift"))
        asyncio.run(orch._dispatch_cached_cue(_fault_cue_event("deadlift_bar_close", "deadlift_bar_drift")))
        assert not orch._adjustment_active

    def test_it_still_arms_for_the_squat(self):
        orch = _orchestrator_with_stale_stance_diagnosis(is_squat=True)
        asyncio.run(orch._dispatch_cached_cue(_fault_cue_event("knees_out", "knee_valgus")))
        assert orch._adjustment_active


# =============================================================================
# WAITS_FOR_DIAGNOSIS AND §5.7 RECAP
# =============================================================================


def _deadlift_scoring() -> dict:
    return {"mean_score": 0.75, "per_dimension": {"setup": 0.7, "bar_path": 0.8}, "trend_slope": 0.0}


async def _three_rep_deadlift_set(orch: CoachingOrchestrator) -> list[CoachingEvent]:
    await _rep(orch, 1, highlights=["clean", "best_rep_so_far"], features=_features(0.55))
    await _rep(orch, 2, highlights=["clean", "best_rep_so_far"], features=_features(0.60))
    await _rep(orch, 3, _fault("deadlift_bar_drift", "moderate"), features=_features(0.48, 20.4))
    return _drain(orch)


def _recap_instructions(orch: CoachingOrchestrator) -> str:
    return orch._generate_llm.call_args.args[0]


class TestWaitsForDiagnosis:
    def test_deadlift_recap_waits_for_its_diagnosis(self):
        async def _run():
            orch = _deadlift_orchestrator()
            orch._diagnosis_wait_s = 2.0

            async def _arrive_late():
                await asyncio.sleep(0.05)
                orch.set_diagnosis_data(_diagnosis_implicating("bar_drift"), _deadlift_scoring(), set_number=1)

            arrival = asyncio.create_task(_arrive_late())
            diagnosis, _ = await orch._await_set_diagnosis(1)
            await arrival
            return diagnosis

        assert asyncio.run(_run()) is not None

    def test_profile_without_the_flag_keeps_todays_no_wait(self):
        orch = _orchestrator()
        orch.set_exercise("Barbell Overhead Press", is_squat=False)
        orch.apply_cue_config(ExerciseCueConfig(profile="overhead_press"))
        assert orch._expects_diagnosis() is False

    def test_exercise_recap_snapshot_carries_the_flag(self):
        async def _run():
            orch = _deadlift_orchestrator(advance_set_fn=AsyncMock(return_value=None))
            orch.reset_set(target_reps=1, total_sets=1)
            await _rep(orch, 1, features=_features(0.5))
            return _drain(orch)

        recap = asyncio.run(_run())[0]
        assert recap.event_type == "llm_exercise_recap"
        assert recap.data["waits_for_diagnosis"] is True
        assert recap.data["is_squat"] is False


class TestDeadliftRecap:
    def test_set_summary_has_bar_speed_and_best_rep(self):
        async def _run():
            orch = _deadlift_orchestrator(advance_set_fn=AsyncMock(return_value=3))
            orch.reset_set(target_reps=3, total_sets=3)
            return await _three_rep_deadlift_set(orch)

        recap = asyncio.run(_run())[0].data["deadlift"]
        assert recap["fastest_rep"] == 2
        assert recap["best_rep"] == 2
        assert recap["last_rep_speed_loss_pct"] == pytest.approx(20.4, abs=BAR_SPEED_TOLERANCE)

    def test_recap_names_bar_speed_main_fault_and_best_rep_without_depth(self):
        async def _run():
            orch = _deadlift_orchestrator(advance_set_fn=AsyncMock(return_value=3))
            orch.reset_set(target_reps=3, total_sets=3)
            orch._diagnosis_wait_s = 0.01
            recap = (await _three_rep_deadlift_set(orch))[0]
            await orch._speak_llm_set_recap(recap.data)
            return _recap_instructions(orch)

        instructions = asyncio.run(_run())
        assert "BAR SPEED: fastest on rep 2; the last rep was 20 percent slower" in instructions
        assert "MAIN FAULT: bar drifting away from the legs on 1 of 3 reps." in instructions
        assert "BEST REP: rep 2" in instructions
        for depth_word in ("depth", "deepest", "shallow"):
            assert depth_word not in instructions.lower()

    def test_numeric_delta_comes_from_the_diagnosis_explanation(self):
        async def _run():
            orch = _deadlift_orchestrator(advance_set_fn=AsyncMock(return_value=3))
            orch.reset_set(target_reps=3, total_sets=3)
            recap = (await _three_rep_deadlift_set(orch))[0]
            orch.set_diagnosis_data(_diagnosis_implicating("bar_drift"), _deadlift_scoring(), set_number=1)
            await orch._speak_llm_set_recap(recap.data)
            return _recap_instructions(orch)

        instructions = asyncio.run(_run())
        assert "TOP ISSUE: Start with the bar about 4 cm closer to your shins." in instructions
        assert "ADJUSTMENT:" not in instructions
        assert "FORM SCORE: 75 out of 100 (setup: 70, bar path: 80)" in instructions

    def test_exercise_recap_carries_the_last_sets_bar_speed(self):
        async def _run():
            orch = _deadlift_orchestrator(advance_set_fn=AsyncMock(return_value=None))
            orch.reset_set(target_reps=3, total_sets=1)
            orch._diagnosis_wait_s = 0.01
            recap = (await _three_rep_deadlift_set(orch))[0]
            await orch._speak_llm_exercise_recap(recap.data)
            return _recap_instructions(orch)

        instructions = asyncio.run(_run())
        assert "Last set — BAR SPEED: fastest on rep 2" in instructions
        assert "Last set — BEST REP: rep 2" in instructions

    def test_squat_recap_has_no_deadlift_lines(self):
        async def _run():
            orch = _orchestrator(advance_set_fn=AsyncMock(return_value=2))
            orch.reset_set(target_reps=2, total_sets=3)
            orch._diagnosis_wait_s = 0.01
            await orch.on_rep_complete(1, "parallel", True, [], max_depth_angle=95.0,
                                       features={"concentric_velocity_mps": 0.5})
            await orch.on_rep_complete(2, "parallel", True, [], max_depth_angle=100.0,
                                       features={"concentric_velocity_mps": 0.6})
            recap = _drain(orch)[-1]
            await orch._speak_llm_set_recap(recap.data)
            return recap.data, _recap_instructions(orch)

        data, instructions = asyncio.run(_run())
        assert "deadlift" not in data
        assert "Best rep: rep 2, their deepest clean rep." in instructions
        assert "BAR SPEED" not in instructions


# =============================================================================
# §5.8 PROGRESS QUERIES
# =============================================================================


class _FakeDb:
    def close(self) -> None:
        pass


class TestProgressQueries:
    @pytest.mark.parametrize("exercise_name,profile", [
        ("Barbell Deadlift", "deadlift"),
        ("Barbell Back Squat", "squat"),
    ])
    def test_queries_pass_the_active_profiles_exercise(self, monkeypatch, exercise_name, profile):
        import db.biomechanics_persistence as persistence
        import db.database as database

        calls: dict[str, str] = {}
        monkeypatch.setattr(database, "SessionLocal", _FakeDb)
        monkeypatch.setattr(
            persistence, "get_progress_baseline",
            lambda db, user_id, exercise="squat": calls.setdefault("baseline", exercise) and None,
        )
        monkeypatch.setattr(
            persistence, "get_multi_session_fault_trends",
            lambda db, user_id, exercise="squat": calls.setdefault("trends", exercise) and None,
        )
        monkeypatch.setattr(persistence, "get_cue_effectiveness", lambda db, user_id: [])
        service = CoachingService(session=None, state=_StubState(exercise_name))

        asyncio.run(service._fetch_progress_baseline("u1"))

        assert calls == {"baseline": profile, "trends": profile}

    def test_wording_names_the_exercise(self):
        baseline = {"days_ago": 2, "mean_score": 0.7, "total_reps": 15, "total_sets": 3}
        assert "last deadlift session was 2 days ago" in build_greeting_progress_line(baseline, exercise="deadlift")
        assert build_progress_report([], None, exercise="deadlift").startswith("No deadlift sessions recorded")
        rows = [{"mean_score": 0.7}, {"mean_score": 0.72}]
        assert build_progress_report(rows, baseline, exercise="deadlift").startswith("2 deadlift sets recorded.")

    def test_squat_wording_is_unchanged(self):
        baseline = {"days_ago": 2, "mean_score": 0.7, "total_reps": 15, "total_sets": 3}
        rows = [{"mean_score": 0.7}, {"mean_score": 0.72}]
        assert build_greeting_progress_line(baseline) == build_greeting_progress_line(baseline, exercise="squat")
        assert "last squat session was 2 days ago" in build_greeting_progress_line(baseline)
        assert build_progress_report([], None).startswith("No squat sessions recorded")
        assert build_progress_report(rows, baseline).startswith("2 squat sets recorded.")


# =============================================================================
# THE DEMO-α LOOP THROUGH THE SERVICE
# =============================================================================


class TestDemoAlphaLoop:
    def test_guide_the_feet_pull_then_one_cue_at_the_floor(self):
        """Stand at the bar (closed loop), set up, pull a rep with bar drift,
        and hear the cue once the bar is back on the floor."""
        service, orch, recorder = _service(DEADLIFT_CACHE_CUES)
        orch._feedback_interval = 0.0
        orch._play_cached = AsyncMock()

        def _frame_message(phase: str, bar_midfoot_cm: float | None = None) -> dict:
            return {
                "type": "frame_data", "joint_angles": {"knee_flexion_l": 20.0, "knee_flexion_r": 20.0},
                "fps": 30.0, "frame_index": 1, "rep_phase": phase, "deadlift_phase": phase,
                "bar_midfoot_live_cm": bar_midfoot_cm, "bar_source": "bar",
            }

        async def _run():
            for offset_cm in (18.0, 6.0, 0.5):
                await service._handle_message(_frame_message("stance", offset_cm))
                await asyncio.sleep(0)
            for phase in ("setup", "pull", "top", "lower"):
                await service._handle_message(_frame_message(phase))
            await service._handle_message({
                "type": "rep_complete", "rep_number": 1, "max_depth_angle": None,
                "depth_category": "n/a", "depth_target_met": True, "is_clean": False,
                "faults_in_rep": ["deadlift_bar_drift"], "set_number": 1, "highlights": [],
                "features": _features(0.55), "faults_detailed": [_fault("deadlift_bar_drift", "moderate")],
            })
            before_floor = _drain(orch)
            await service._handle_message(_frame_message("floor"))
            return before_floor, _drain(orch)

        before_floor, at_floor = asyncio.run(_run())
        guidance = [call.args[0] for call in orch._play_cached.await_args_list if call.args[0] != "rep_1"]
        assert guidance == ["deadlift_step_closer", "deadlift_closer", "adjust_good"]
        assert before_floor == []
        assert _cue_keys(at_floor) == ["deadlift_bar_close"]
        assert recorder.reps[0]["features"]["dl_schema"] == 1


class TestExerciseFaultSets:
    def test_deadlift_symptoms_name_deadlift_faults(self):
        from agent.services.coaching_orchestrator import DEADLIFT_SYMPTOM_FAULT_TYPES

        orch = _deadlift_orchestrator()
        assert orch._fault_sets.symptom_fault_types is DEADLIFT_SYMPTOM_FAULT_TYPES
        assert orch._fault_sets.symptom_fault_types["hip_shift"] == "deadlift_hip_shift"
        assert orch._fault_sets.safety_fault_type is None

    def test_the_squat_keeps_its_sets(self):
        from agent.services.coaching_orchestrator import SQUAT_FAULT_SETS

        assert _orchestrator()._fault_sets is SQUAT_FAULT_SETS

    def test_one_camera_mutes_side_view_deadlift_faults(self):
        orch = _deadlift_orchestrator()
        orch._side_view_observable = False
        assert orch._is_unseen_side_view("deadlift_hips_shoot")
        assert not orch._is_unseen_side_view("deadlift_bar_drift")
