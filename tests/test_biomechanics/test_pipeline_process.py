"""
Tests for pipeline_process helpers around athlete params: extraction from the
pipeline's body calibration, the None guard on the calibration_complete IPC
payload, and camera calibration inputs (user height, refined-file load order).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biomechanics.pipeline_process import (
    FALLBACK_USER_HEIGHT_M,
    MAX_ASSESSMENT_ROUNDS,
    _adopt_measured_athlete_params,
    _adopt_provisional_athlete_params,
    _assessment_result_message,
    _build_calibration_complete_message,
    _extract_athlete_params,
    _resolve_user_height_m,
    refined_calibration_path,
    select_calibration_file,
)
from biomechanics.triangulation.calibration import CalibrationResult, TPoseCalibrator, rig_calibration_path
from biomechanics.utils.segment_lengths import SegmentLengthEstimator

HEIGHT_TOLERANCE_M = 1e-9
FACTORY_TIMESTAMP = "2026-09-18T10:00:00"
REFINED_TIMESTAMP = "2026-09-18T11:00:00"
OTHER_TIMESTAMP = "2026-09-01T09:00:00"

ATHLETE_PARAMS = {
    "shoulder_width_m": 0.39,
    "femur_avg_m": 0.46,
    "torso_avg_m": 0.55,
    "hip_width_m": 0.27,
    "tibia_avg_m": 0.44,
    "foot_avg_m": 0.21,
}
BASELINE = {"peakDorsi": 33.0, "peakKneeFlex": 118.0}
CAL_PROFILE = {
    "knee_valgus": {"mild": 12.0, "moderate": 17.0, "severe": 24.0},
    "defaults": {"knee_valgus": {"mild": 10.0}},
}


class _StubPipeline:
    def __init__(self, body_calibration: SegmentLengthEstimator) -> None:
        self.body_calibration = body_calibration


class _ParamsRecorder:
    """Stands in for SessionTracker and IPCBridge: records set_athlete_params calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def set_athlete_params(self, params: dict, baseline: dict) -> None:
        self.calls.append((params, baseline))


class TestExtractAthleteParams:
    def test_none_while_body_unmeasured(self) -> None:
        pipeline = _StubPipeline(SegmentLengthEstimator())

        assert _extract_athlete_params(pipeline) is None

    def test_measured_body_gives_legacy_keys(self) -> None:
        pipeline = _StubPipeline(SegmentLengthEstimator.from_athlete_params(ATHLETE_PARAMS))

        params = _extract_athlete_params(pipeline)

        assert set(params) == set(ATHLETE_PARAMS)
        assert params["femur_avg_m"] == pytest.approx(ATHLETE_PARAMS["femur_avg_m"], abs=HEIGHT_TOLERANCE_M)


class TestAdoptMeasuredAthleteParams:
    def test_unmeasured_body_touches_nothing(self) -> None:
        tracker = _ParamsRecorder()
        bridge = _ParamsRecorder()

        adopted = _adopt_measured_athlete_params(
            _StubPipeline(SegmentLengthEstimator()), tracker, bridge, BASELINE,
        )

        assert adopted is None
        assert tracker.calls == []
        assert bridge.calls == []

    def test_measured_body_wires_tracker_and_bridge(self) -> None:
        tracker = _ParamsRecorder()
        bridge = _ParamsRecorder()
        pipeline = _StubPipeline(SegmentLengthEstimator.from_athlete_params(ATHLETE_PARAMS))

        adopted = _adopt_measured_athlete_params(pipeline, tracker, bridge, BASELINE)

        assert adopted is not None
        assert tracker.calls == [(adopted, BASELINE)]
        assert bridge.calls == [(adopted, BASELINE)]


class _PartlyMeasuredBody:
    """A body measurement still running: no final params, a provisional estimate (or none)."""

    def __init__(self, provisional: dict | None) -> None:
        self._provisional = provisional

    def to_athlete_params(self) -> dict | None:
        return None

    def provisional_athlete_params(self) -> dict | None:
        return self._provisional


class TestAdoptProvisionalAthleteParams:
    def test_assessment_diagnoses_on_the_best_estimate(self) -> None:
        """A one-rep assessment that outruns body measurement once passed with an empty diagnosis."""
        tracker = _ParamsRecorder()
        bridge = _ParamsRecorder()

        adopted = _adopt_provisional_athlete_params(
            _StubPipeline(_PartlyMeasuredBody(ATHLETE_PARAMS)), tracker, bridge, BASELINE,
        )

        assert adopted == ATHLETE_PARAMS
        assert tracker.calls == [(ATHLETE_PARAMS, BASELINE)]
        assert bridge.calls == [(ATHLETE_PARAMS, BASELINE)]

    def test_nothing_measured_touches_nothing(self) -> None:
        tracker = _ParamsRecorder()
        bridge = _ParamsRecorder()

        adopted = _adopt_provisional_athlete_params(
            _StubPipeline(_PartlyMeasuredBody(None)), tracker, bridge, BASELINE,
        )

        assert adopted is None
        assert tracker.calls == [] and bridge.calls == []


DIAGNOSIS = {"confidence": 0.8, "immediate_causes": [{"cause_id": "stance_toe_mismatch"}]}
SCORING = {"mean_score": 0.7}
DEMO = {"available": False, "cues": []}


class TestAssessmentResultMessage:
    def test_clean_round_passes_and_ends_the_assessment(self) -> None:
        message = _assessment_result_message(1, False, DIAGNOSIS, SCORING, DEMO, body_measurement="complete")

        assert message["type"] == "assessment_result"
        assert message["passed"] is True
        assert message["final_round"] is True
        assert message["proceed_anyway"] is False

    def test_issues_before_the_cap_ask_for_another_round(self) -> None:
        message = _assessment_result_message(1, True, DIAGNOSIS, SCORING, DEMO, body_measurement="complete")

        assert message["passed"] is False
        assert message["final_round"] is False
        assert message["proceed_anyway"] is False

    def test_issues_on_the_last_round_proceed_to_the_workout(self) -> None:
        """Real data: 28% of sessions never passed; the top cue is carried into the set instead."""
        message = _assessment_result_message(
            MAX_ASSESSMENT_ROUNDS, True, DIAGNOSIS, SCORING, DEMO, body_measurement="provisional",
        )

        assert message["passed"] is False
        assert message["final_round"] is True
        assert message["proceed_anyway"] is True
        assert message["round"] == MAX_ASSESSMENT_ROUNDS
        assert message["diagnosis"] == DIAGNOSIS
        assert message["scoring"] == SCORING
        assert message["demo"] == DEMO
        assert message["body_measurement"] == "provisional"


class TestCalibrationCompleteMessage:
    def test_missing_params_are_omitted_not_sent_as_none(self) -> None:
        message = _build_calibration_complete_message("squat", {"trunk_flexion": 40.0}, CAL_PROFILE, None, BASELINE)

        assert message["type"] == "calibration_complete"
        assert "athlete_params" not in message
        assert "baseline" not in message

    def test_measured_params_are_included_with_baseline(self) -> None:
        message = _build_calibration_complete_message("squat", {}, CAL_PROFILE, ATHLETE_PARAMS, BASELINE)

        assert message["athlete_params"] == ATHLETE_PARAMS
        assert message["baseline"] == BASELINE

    def test_profile_is_sent_as_is(self) -> None:
        message = _build_calibration_complete_message("squat", {}, CAL_PROFILE, None, BASELINE)

        assert message["thresholds"] == CAL_PROFILE


class TestResolveUserHeight:
    def test_profile_height_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOWVA_USER_HEIGHT_M", "1.60")

        assert _resolve_user_height_m(175.0) == pytest.approx(1.75, abs=HEIGHT_TOLERANCE_M)

    def test_env_fallback_when_profile_has_no_height(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOWVA_USER_HEIGHT_M", "1.60")

        assert _resolve_user_height_m(None) == pytest.approx(1.60, abs=HEIGHT_TOLERANCE_M)

    def test_constant_fallback_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOWVA_USER_HEIGHT_M", raising=False)

        assert _resolve_user_height_m(0.0) == pytest.approx(FALLBACK_USER_HEIGHT_M, abs=HEIGHT_TOLERANCE_M)


def _write_rig_file(path: Path, timestamp: str, source_timestamp: str = "") -> Path:
    TPoseCalibrator.save_calibration(
        CalibrationResult(timestamp=timestamp, source_timestamp=source_timestamp), str(path),
    )
    return path


class TestCalibrationFileSelection:
    def test_refined_path_is_a_sibling_of_the_factory_file(self, tmp_path: Path) -> None:
        factory_path = rig_calibration_path([0, 1, 2], tmp_path)

        assert refined_calibration_path(factory_path) == tmp_path / "rig_calibration_cams_0-1-2_refined.json"

    def test_no_files_selects_nothing(self, tmp_path: Path) -> None:
        assert select_calibration_file(rig_calibration_path([0, 1, 2], tmp_path)) is None

    def test_factory_file_alone_is_selected(self, tmp_path: Path) -> None:
        factory_path = _write_rig_file(rig_calibration_path([0, 1, 2], tmp_path), FACTORY_TIMESTAMP)

        assert select_calibration_file(factory_path) == factory_path

    def test_refined_file_derived_from_the_factory_file_wins(self, tmp_path: Path) -> None:
        factory_path = _write_rig_file(rig_calibration_path([0, 1, 2], tmp_path), FACTORY_TIMESTAMP)
        refined_path = _write_rig_file(refined_calibration_path(factory_path), REFINED_TIMESTAMP, FACTORY_TIMESTAMP)

        assert select_calibration_file(factory_path) == refined_path

    def test_refined_file_from_another_factory_calibration_is_ignored(self, tmp_path: Path) -> None:
        factory_path = _write_rig_file(rig_calibration_path([0, 1, 2], tmp_path), FACTORY_TIMESTAMP)
        _write_rig_file(refined_calibration_path(factory_path), REFINED_TIMESTAMP, OTHER_TIMESTAMP)

        assert select_calibration_file(factory_path) == factory_path

    def test_refined_file_without_lineage_is_ignored(self, tmp_path: Path) -> None:
        factory_path = _write_rig_file(rig_calibration_path([0, 1, 2], tmp_path), FACTORY_TIMESTAMP)
        _write_rig_file(refined_calibration_path(factory_path), REFINED_TIMESTAMP)

        assert select_calibration_file(factory_path) == factory_path

    def test_refined_file_without_a_factory_file_is_ignored(self, tmp_path: Path) -> None:
        factory_path = rig_calibration_path([0, 1, 2], tmp_path)
        _write_rig_file(refined_calibration_path(factory_path), REFINED_TIMESTAMP, FACTORY_TIMESTAMP)

        assert select_calibration_file(factory_path) is None


class TestSwitchExercise:
    def test_switch_moves_pipeline_tracker_and_agent_to_the_new_exercise(self, monkeypatch, mock_ipc_client):
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.coaching.session_tracker import SessionTracker
        from biomechanics.config import BiomechanicsConfig
        from biomechanics.pipeline import BiomechanicsPipeline
        from biomechanics.pipeline_process import _activate_profile
        from biomechanics.profiles.overhead_press import OverheadPressProfile

        monkeypatch.setattr(OverheadPressProfile, "coaching_ready", True)
        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
        pipeline = BiomechanicsPipeline(BiomechanicsConfig(), defer_capture=True)
        bridge = IPCBridge(mock_ipc_client)
        session_tracker = SessionTracker(bridge)

        _activate_profile(pipeline, session_tracker, bridge, "Barbell Overhead Press")

        assert pipeline.profile.name == "overhead_press"
        assert not session_tracker.diagnosis_enabled
        cache_cues = [m for m in mock_ipc_client.messages if m["type"] == "cache_cues"][-1]
        assert cache_cues["exercise_name"] == "Barbell Overhead Press"
        assert cache_cues["profile"] == "overhead_press"
        assert "press_lockout" in cache_cues["cues"]
        assert "knees_out" not in cache_cues["cues"]


class _ViewRecordingProvider:
    def __init__(self) -> None:
        self.view_recording: list[bool] = []

    def set_view_recording(self, enabled: bool) -> None:
        self.view_recording.append(enabled)


class TestActivateProfile:
    """One activation path at startup and on every set_exercise (SHARED-PLUMBING.md §9)."""

    def _parts(self, monkeypatch, mock_ipc_client):
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.coaching.session_tracker import SessionTracker
        from biomechanics.config import BiomechanicsConfig
        from biomechanics.pipeline import BiomechanicsPipeline

        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
        pipeline = BiomechanicsPipeline(BiomechanicsConfig(), defer_capture=True)
        pipeline._multi_camera_provider = _ViewRecordingProvider()
        bridge = IPCBridge(mock_ipc_client)
        return pipeline, SessionTracker(bridge), bridge

    def test_startup_keeps_the_pipeline_built_for_the_exercise(self, monkeypatch, mock_ipc_client):
        from biomechanics.pipeline_process import _activate_profile

        pipeline, tracker, bridge = self._parts(monkeypatch, mock_ipc_client)
        rule_engine = pipeline._rule_engine
        _activate_profile(pipeline, tracker, bridge, "Barbell Back Squat", startup=True)

        assert pipeline._rule_engine is rule_engine
        assert tracker.diagnosis_enabled and tracker.reports_depth
        assert tracker.set_diagnosis is None
        assert pipeline._multi_camera_provider.view_recording == [True]
        cache_cues = [m for m in mock_ipc_client.messages if m["type"] == "cache_cues"]
        assert [m["profile"] for m in cache_cues] == ["squat"]

    def test_a_switch_rebuilds_the_pipeline_and_applies_the_metadata(self, monkeypatch, mock_ipc_client):
        from biomechanics.pipeline_process import _activate_profile
        from biomechanics.profiles.untracked_variant import UntrackedVariantProfile

        pipeline, tracker, bridge = self._parts(monkeypatch, mock_ipc_client)
        rule_engine = pipeline._rule_engine
        _activate_profile(pipeline, tracker, bridge, "Barbell Sumo Deadlift", {"grip": "mixed"})

        assert pipeline._rule_engine is not rule_engine
        assert isinstance(pipeline.profile, UntrackedVariantProfile)
        assert not tracker.diagnosis_enabled
        assert tracker.exercise_meta == {"grip": "mixed"}
        assert pipeline._multi_camera_provider.view_recording == [False]

    def test_a_gated_exercise_runs_untracked(self, monkeypatch, mock_ipc_client):
        from biomechanics.pipeline_process import _activate_profile

        pipeline, tracker, bridge = self._parts(monkeypatch, mock_ipc_client)
        _activate_profile(pipeline, tracker, bridge, "Barbell Romanian Deadlift")

        assert pipeline.profile.name == "untracked"
        assert [m for m in mock_ipc_client.messages if m["type"] == "cache_cues"][-1]["profile"] == "untracked"


class TestShallowRepDelivery:
    def test_a_squat_shallow_rep_keeps_its_depth_class_and_cue(self, mock_ipc_client):
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.pipeline_process import _send_shallow_rep
        from biomechanics.utils.types import PipelineFrame

        bridge = IPCBridge(mock_ipc_client)
        _send_shallow_rep(bridge, PipelineFrame(frame_index=0, timestamp=0.0, shallow_rep_class=1), set_number=1)

        message = mock_ipc_client.messages[-1]
        assert (message["depth_class"], message["cue"]) == (1, "deeper")

    def test_an_analyser_rejection_carries_the_profiles_cue_and_label(self, mock_ipc_client):
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.pipeline_process import _send_shallow_rep
        from biomechanics.utils.types import FaultEvent, FaultSeverity, PipelineFrame

        fault = FaultEvent(
            fault_type="rdl_partial", severity=FaultSeverity.MODERATE, severity_score=1.5, message="m",
            details={"shallow_rep": True, "range_label": "knee"},
        )
        frame = PipelineFrame(
            frame_index=0, timestamp=0.0, faults=[fault], analyzer_shallow_rep=True, shallow_rep_cue="rdl_lower",
        )
        _send_shallow_rep(IPCBridge(mock_ipc_client), frame, set_number=2)

        message = mock_ipc_client.messages[-1]
        assert message["cue"] == "rdl_lower"
        assert message["depth_class"] is None
        assert message["label"] == "knee"
        assert message["set_number"] == 2


class TestCameraPitchStatus:
    def test_a_revoked_mount_and_a_far_off_level_camera_each_send_camera_pitch(self, mock_ipc_client):
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.pipeline_process import _send_camera_pitch_status
        from biomechanics.utils.types import PipelineFrame

        bridge = IPCBridge(mock_ipc_client)
        _send_camera_pitch_status(bridge, PipelineFrame(frame_index=0, timestamp=0.0, pitch_certification_revoked_deg=4.0))
        _send_camera_pitch_status(bridge, PipelineFrame(frame_index=1, timestamp=0.1, camera_pitch_untrusted_deg=-12.0))

        statuses = [message for message in mock_ipc_client.messages if message["type"] == "pipeline_status"]
        assert [message["status"] for message in statuses] == ["camera_pitch", "camera_pitch"]
        assert [message["latency_ms"] for message in statuses] == [{"delta_deg": 4.0}, {"pitch_deg": -12.0}]

    def test_an_ordinary_frame_sends_nothing(self, mock_ipc_client):
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.pipeline_process import _send_camera_pitch_status
        from biomechanics.utils.types import PipelineFrame

        _send_camera_pitch_status(IPCBridge(mock_ipc_client), PipelineFrame(frame_index=0, timestamp=0.0))
        assert mock_ipc_client.messages == []


class TestReadinessHint:
    def test_the_profiles_hint_names_its_own_setup_failure(self, monkeypatch):
        from biomechanics.config import BiomechanicsConfig
        from biomechanics.pipeline import BiomechanicsPipeline
        from biomechanics.pipeline_process import GATE_FAILURE_HINTS, _readiness_hint
        from biomechanics.utils.standing_gate import PROFILE_CHECK_FAILURE

        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
        pipeline = BiomechanicsPipeline(BiomechanicsConfig(), defer_capture=True)
        pipeline._readiness_gate.last_failure = "visibility"
        assert _readiness_hint(pipeline, "HOLD STILL") == GATE_FAILURE_HINTS["visibility"]

        pipeline._readiness_gate.last_failure = PROFILE_CHECK_FAILURE
        assert _readiness_hint(pipeline, "HOLD STILL") == "HOLD STILL"
        monkeypatch.setattr(type(pipeline.profile), "readiness_hint", "HANDS IN VIEW")
        assert _readiness_hint(pipeline, "HOLD STILL") == "HANDS IN VIEW"
