"""
Tests for pipeline_process helpers around athlete params: extraction from the
pipeline's body calibration, the None guard on the calibration_complete IPC
payload, and T-pose calibration inputs (user height, per-rig path).
"""

from __future__ import annotations

import pytest

from biomechanics.pipeline_process import (
    FALLBACK_USER_HEIGHT_M,
    RIG_CALIBRATION_DIR,
    _adopt_measured_athlete_params,
    _build_calibration_complete_message,
    _extract_athlete_params,
    _resolve_user_height_m,
    _rig_calibration_path,
)
from biomechanics.utils.segment_lengths import SegmentLengthEstimator

HEIGHT_TOLERANCE_M = 1e-9

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

    def test_defaults_are_stripped_from_thresholds(self) -> None:
        message = _build_calibration_complete_message("squat", {}, CAL_PROFILE, None, BASELINE)

        assert message["thresholds"] == {"knee_valgus": CAL_PROFILE["knee_valgus"]}


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


class TestRigCalibrationPath:
    def test_path_is_per_camera_set_under_rig_dir(self) -> None:
        three_camera_path = _rig_calibration_path([0, 1, 2])
        two_camera_path = _rig_calibration_path([0, 2])

        assert three_camera_path.parent == RIG_CALIBRATION_DIR
        assert three_camera_path != two_camera_path
        assert three_camera_path.suffix == ".json"
