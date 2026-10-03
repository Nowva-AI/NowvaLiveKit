"""
Tests for TeachingAgent assessment flow.

Covers: assessment result handling (pass/fail/demo), handoff to
CalibrationAgent, correction prompt building, height threshold,
on_exit cleanup, and assessment logging integration.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.agents.teaching_agent import TeachingAgent, _TALL_THRESHOLD_CM


# =============================================================================
# HELPERS
# =============================================================================


def _make_state(height_cm=170):
    state = MagicMock()
    state.get_user.return_value = {
        "id": "test-user",
        "name": "Test",
        "height_cm": height_cm,
    }
    return state


def _make_userdata():
    ud = MagicMock()
    ud.audio_cue_service = MagicMock()
    ud.audio_cue_service.play_cue = AsyncMock()
    ud.audio_cue_service.cache_cues = AsyncMock()
    # No coaching_service by default — TeachingAgent creates it in on_enter
    ud.coaching_service = None
    return ud


def _make_agent(height_cm=170, exercise="squat"):
    state = _make_state(height_cm)
    userdata = _make_userdata()
    agent = TeachingAgent(state=state, userdata=userdata, exercise=exercise)

    agent._say = AsyncMock()
    agent._truncate_context_for_handoff = AsyncMock()

    mock_session = MagicMock()
    mock_session.update_agent = MagicMock()
    agent._mock_session = mock_session

    return agent


def _assessment_result(passed=False, round_num=1, immediate_causes=None, scoring=None):
    return {
        "passed": passed,
        "round": round_num,
        "diagnosis": {
            "immediate_causes": immediate_causes or [],
            "session_causes": [],
            "contextual_notes": [],
        },
        "scoring": scoring or {"mean_score": 0.75},
        "demo": {"available": False, "cues": []},
    }


# =============================================================================
# 1. Height threshold
# =============================================================================


class TestHeightThreshold:
    def test_tall_at_185(self):
        agent = _make_agent(height_cm=185)
        assert agent.user_is_tall is True

    def test_not_tall_at_184(self):
        agent = _make_agent(height_cm=184)
        assert agent.user_is_tall is False

    def test_tall_above_185(self):
        agent = _make_agent(height_cm=200)
        assert agent.user_is_tall is True

    def test_not_tall_below_185(self):
        agent = _make_agent(height_cm=170)
        assert agent.user_is_tall is False

    def test_none_height_defaults_false(self):
        state = MagicMock()
        state.get_user.return_value = {"id": "u", "name": "N", "height_cm": None}
        userdata = _make_userdata()
        agent = TeachingAgent(state=state, userdata=userdata)
        assert agent.user_is_tall is False


# =============================================================================
# 2. Assessment result — fail triggers correction speech
# =============================================================================


class TestAssessmentFail:
    def test_fail_triggers_say(self):
        async def _run():
            agent = _make_agent()
            msg = _assessment_result(
                passed=False,
                immediate_causes=[{
                    "cause_id": "knee_valgus",
                    "explanation": "Knees caving inward",
                    "parameter_delta": {"stance_width": "+5cm"},
                }],
            )
            await agent._on_assessment_result(msg, demo_was_played=False)

            agent._say.assert_awaited_once()
            prompt = agent._say.call_args[0][0]
            assert "Knees caving inward" in prompt

        asyncio.run(_run())

    def test_fail_does_not_handoff(self):
        async def _run():
            agent = _make_agent()
            msg = _assessment_result(passed=False)
            await agent._on_assessment_result(msg, demo_was_played=False)

            assert agent._handed_off is False

        asyncio.run(_run())


# =============================================================================
# 3. Assessment result — pass triggers handoff to CalibrationAgent
# =============================================================================


class TestAssessmentPass:
    def test_pass_hands_off(self):
        async def _run():
            agent = _make_agent()
            msg = _assessment_result(passed=True)

            with patch.object(
                type(agent), 'session',
                new_callable=lambda: property(lambda self: self._mock_session),
            ):
                await agent._on_assessment_result(msg, demo_was_played=False)

            assert agent._handed_off is True
            agent._say.assert_awaited_once()
            agent._mock_session.update_agent.assert_called_once()

            from agent.agents.calibration_agent import CalibrationAgent
            new_agent = agent._mock_session.update_agent.call_args[0][0]
            assert isinstance(new_agent, CalibrationAgent)

        asyncio.run(_run())

    def test_pass_prompt_includes_score(self):
        async def _run():
            agent = _make_agent()
            msg = _assessment_result(passed=True, scoring={"mean_score": 0.92})

            with patch.object(
                type(agent), 'session',
                new_callable=lambda: property(lambda self: self._mock_session),
            ):
                await agent._on_assessment_result(msg, demo_was_played=False)

            prompt = agent._say.call_args[0][0]
            assert "92" in prompt

        asyncio.run(_run())


class TestAssessmentProceedsAnyway:
    def test_unresolved_final_round_hands_off_with_one_focus(self):
        async def _run():
            agent = _make_agent()
            msg = _assessment_result(
                passed=False, round_num=2,
                immediate_causes=[{
                    "cause_id": "knee_track_cue",
                    "explanation": "Your knees drift inside your toes on the way up.",
                }],
            )
            msg["final_round"] = True
            msg["proceed_anyway"] = True

            with patch.object(
                type(agent), 'session',
                new_callable=lambda: property(lambda self: self._mock_session),
            ):
                await agent._on_assessment_result(msg, demo_was_played=False)

            assert agent._handed_off is True
            prompt = agent._say.call_args[0][0]
            assert "knees drift inside your toes" in prompt
            assert "passed" not in prompt

        asyncio.run(_run())


# =============================================================================
# 4. Demo acknowledgment
# =============================================================================


class TestDemoAcknowledgment:
    def test_demo_played_gets_encouragement(self):
        async def _run():
            agent = _make_agent()
            msg = _assessment_result(passed=False)
            await agent._on_assessment_result(msg, demo_was_played=True)

            agent._say.assert_awaited_once()
            prompt = agent._say.call_args[0][0]
            assert "demo" in prompt.lower() or "difference" in prompt.lower()

        asyncio.run(_run())

    def test_demo_not_played_gets_correction(self):
        async def _run():
            agent = _make_agent()
            msg = _assessment_result(
                passed=False,
                immediate_causes=[{
                    "cause_id": "trunk_lean",
                    "explanation": "Excessive forward lean",
                    "parameter_delta": None,
                }],
            )
            await agent._on_assessment_result(msg, demo_was_played=False)

            prompt = agent._say.call_args[0][0]
            assert "Excessive forward lean" in prompt

        asyncio.run(_run())


# =============================================================================
# 5. on_exit cleanup
# =============================================================================


class TestOnExit:
    def test_exit_without_handoff_stops_assessment(self):
        async def _run():
            agent = _make_agent()
            mock_coaching = MagicMock()
            agent.userdata.coaching_service = mock_coaching

            await agent.on_exit()

            mock_coaching.stop_assessment.assert_called_once_with(False)

        asyncio.run(_run())

    def test_exit_after_handoff_does_not_double_stop(self):
        async def _run():
            agent = _make_agent()
            agent._handed_off = True
            mock_coaching = MagicMock()
            agent.userdata.coaching_service = mock_coaching

            await agent.on_exit()

            mock_coaching.stop_assessment.assert_not_called()

        asyncio.run(_run())


# =============================================================================
# 6. Correction prompt building
# =============================================================================


class TestCorrectionPrompt:
    def test_includes_main_issue(self):
        agent = _make_agent()
        diagnosis = {
            "immediate_causes": [{
                "cause_id": "ankle_mobility",
                "explanation": "Limited ankle dorsiflexion",
                "parameter_delta": {"heel_lift": "consider"},
            }],
        }
        prompt = agent._build_correction_prompt(diagnosis, {"mean_score": 0.6}, 2)

        assert "Limited ankle dorsiflexion" in prompt
        assert "heel_lift" not in prompt

    def test_adjustment_is_spoken_in_body_units_not_raw_parameters(self):
        agent = _make_agent()
        diagnosis = {
            "immediate_causes": [{
                "cause_id": "narrow_stance",
                "explanation": "Your stance is narrow",
                "parameter_delta": {
                    "__foot_target_delta": [0.0, 0.0, -0.03, 0.0, 0.0, 0.03],
                    "__target_stance_ratio": 1.4,
                    "__width_increase_per_side_m": 0.03,
                },
            }],
        }
        prompt = agent._build_correction_prompt(diagnosis, {"mean_score": 0.6}, 2)

        assert "each foot about 3 centimeters wider" in prompt
        assert "__" not in prompt

    def test_correction_prompt_has_no_verbatim_example_lines(self):
        agent = _make_agent()
        prompt = agent._build_correction_prompt(
            {"immediate_causes": [{"cause_id": "x", "explanation": "test", "parameter_delta": None}]},
            {"mean_score": 0.5},
            1,
        )
        assert "e.g." not in prompt
        assert "widen your stance a bit" not in prompt
        assert "Let me see that again" not in prompt

    def test_includes_secondary_issue(self):
        agent = _make_agent()
        diagnosis = {
            "immediate_causes": [
                {"cause_id": "a", "explanation": "Primary issue", "parameter_delta": None},
                {"cause_id": "b", "explanation": "Secondary issue", "parameter_delta": None},
            ],
        }
        prompt = agent._build_correction_prompt(diagnosis, {"mean_score": 0.5}, 1)

        assert "Primary issue" in prompt
        assert "Secondary issue" in prompt

    def test_empty_causes_still_builds_prompt(self):
        agent = _make_agent()
        prompt = agent._build_correction_prompt(
            {"immediate_causes": []}, {"mean_score": 0.0}, 1,
        )
        assert "exercise" in prompt.lower() or "adjustment" in prompt.lower()

    def test_no_assessment_word_in_prompt(self):
        agent = _make_agent()
        prompt = agent._build_correction_prompt(
            {"immediate_causes": [{"cause_id": "x", "explanation": "test", "parameter_delta": None}]},
            {"mean_score": 0.5},
            1,
        )
        assert "assessment" not in prompt.lower() or "Do NOT say" in prompt


# =============================================================================
# 7. Assessment logging integration
# =============================================================================


class TestAssessmentLogging:
    def test_start_logging_on_enter_with_session_dir(self):
        async def _run():
            agent = _make_agent()
            mock_coaching = MagicMock()
            agent.userdata.coaching_service = mock_coaching

            with patch.dict("os.environ", {"NOWVA_SESSION_OUTPUT_DIR": "/tmp/test_session"}):
                agent._start_assessment_logging()

            mock_coaching.start_assessment.assert_called_once()
            call_kwargs = mock_coaching.start_assessment.call_args[1]
            assert call_kwargs["session_dir"] == "/tmp/test_session"
            assert call_kwargs["user_height_cm"] == 170.0

        asyncio.run(_run())

    def test_no_logging_without_session_dir(self):
        async def _run():
            agent = _make_agent()
            mock_coaching = MagicMock()
            agent.userdata.coaching_service = mock_coaching

            with patch.dict("os.environ", {}, clear=True):
                agent._start_assessment_logging()

            mock_coaching.start_assessment.assert_not_called()

        asyncio.run(_run())

    def test_recommendations_logged_on_fail(self):
        async def _run():
            agent = _make_agent()
            mock_logger = MagicMock()
            mock_coaching = MagicMock()
            mock_coaching.assessment_logger = mock_logger
            agent.userdata.coaching_service = mock_coaching

            msg = _assessment_result(
                passed=False,
                immediate_causes=[{
                    "cause_id": "knee_valgus",
                    "explanation": "Knees caving",
                    "parameter_delta": None,
                }],
            )
            await agent._on_assessment_result(msg, demo_was_played=False)

            mock_logger.set_outgoing_recommendations_latest.assert_called_once()
            recs = mock_logger.set_outgoing_recommendations_latest.call_args[0][0]
            assert len(recs) == 1
            assert recs[0].fault_type == "knee_valgus"

        asyncio.run(_run())

    def test_no_recommendations_logged_on_pass(self):
        async def _run():
            agent = _make_agent()
            mock_logger = MagicMock()
            mock_coaching = MagicMock()
            mock_coaching.assessment_logger = mock_logger
            agent.userdata.coaching_service = mock_coaching

            msg = _assessment_result(passed=True)

            with patch.object(
                type(agent), 'session',
                new_callable=lambda: property(lambda self: self._mock_session),
            ):
                await agent._on_assessment_result(msg, demo_was_played=False)

            mock_logger.set_outgoing_recommendations_latest.assert_not_called()

        asyncio.run(_run())


# =============================================================================
# 8. CoachingService reuse in on_enter
# =============================================================================


class TestCoachingServiceReuse:
    def test_on_enter_reuses_existing_service(self):
        async def _run():
            agent = _make_agent()
            mock_coaching = MagicMock()
            agent.userdata.coaching_service = mock_coaching

            with patch.dict("os.environ", {}, clear=True):
                await agent.on_enter()

            mock_coaching.set_assessment_result_callback.assert_called_once_with(
                agent._on_assessment_result
            )
            assert agent.userdata.coaching_service is mock_coaching

        asyncio.run(_run())

    def test_on_enter_creates_service_when_absent(self):
        async def _run():
            agent = _make_agent()  # userdata.coaching_service is None

            with patch.dict("os.environ", {}, clear=True), patch(
                "agent.services.coaching_service.CoachingService"
            ) as mock_service_cls:
                mock_service_cls.return_value.start = AsyncMock()
                with patch.object(
                    type(agent), 'session',
                    new_callable=lambda: property(lambda self: self._mock_session),
                ):
                    await agent.on_enter()

            mock_service_cls.assert_called_once()
            assert agent.userdata.coaching_service is mock_service_cls.return_value

        asyncio.run(_run())
