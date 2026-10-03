"""Tests for the --valgus knee debug recorder and its HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biomechanics.faults.rules.depth import DepthRule
from biomechanics.faults.rules.knee_tracking import KneeTrackingRule
from biomechanics.faults.rules.knee_valgus import KneeValgusRule
from biomechanics.utils.types import (
    FaultEvent,
    FaultSeverity,
    JointAngles,
    PipelineFrame,
    RepData,
    RepPhase,
)
from biomechanics.viz.valgus_debug import (
    MAX_RETIME_FPS,
    MIN_RETIME_FPS,
    TEMPLATE_PATH,
    ValgusDebugRecorder,
    _is_knee_related,
    _read_valgus_thresholds,
    build_recorder,
)

FRAME_HEIGHT = 120
FRAME_WIDTH = 160
# The squat's knee rule judges each rep once, when the rep ends.
REP_END_FRAME = 16
# Data keys the recorder no longer writes; the report page crashed reading "fallback".
REMOVED_REPORT_KEYS = ('"fallback"', "fault_cooldown_frames", "max_valgus", "affected_side", "metric_source")


class _FakeRepCounter:
    def __init__(self) -> None:
        self.phase = "standing"
        self.rep_count = 0


class _FakeRuleEngine:
    def __init__(self) -> None:
        self.rules = [DepthRule(), KneeTrackingRule()]


class _FakePipeline:
    def __init__(self) -> None:
        self.rep_counter = _FakeRepCounter()
        self.is_ready = True
        self._rule_engine = _FakeRuleEngine()


class _FakeIPCClient:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.client_socket = "sentinel"

    def send_message(self, message: dict) -> None:
        self.sent.append(message)

    def disconnect(self) -> None:
        self.sent.append({"type": "_disconnected"})


def _display() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _angles(valgus_l: float = 5.0, valgus_r: float = -3.0, foot_conf: float = 0.9) -> JointAngles:
    return JointAngles(
        knee_valgus_l=valgus_l,
        knee_valgus_r=valgus_r,
        foot_confidence_l=foot_conf,
        foot_confidence_r=foot_conf,
        knee_ankle_sep_ratio=0.85,
        hip_adduction_l=7.0,
        hip_adduction_r=6.0,
        knee_flexion_l=95.0,
        knee_flexion_r=94.0,
    )


def _frame(index: int, angles: JointAngles | None = None, faults: list | None = None) -> PipelineFrame:
    return PipelineFrame(
        frame_index=index,
        timestamp=float(index),
        joint_angles=angles,
        faults=faults or [],
    )


def _valgus_fault(frame_index: int = 3) -> FaultEvent:
    """A squat knee-tracking verdict as the rule engine emits it at rep end."""
    return FaultEvent(
        fault_type="knee_valgus",
        severity=FaultSeverity.MODERATE,
        severity_score=1.8,
        message="Knees caving in",
        frame_index=frame_index,
        rep_number=1,
        details={
            "side": "left",
            "phase": "ascent",
            "is_drift": False,
            "value": 14.0,
            "unit": "deg",
            "knee_valgus_l": 14.0,
            "knee_valgus_r": 4.0,
            "observability": "observable",
        },
    )


@pytest.fixture
def recorder(tmp_path: Path) -> ValgusDebugRecorder:
    return ValgusDebugRecorder(str(tmp_path), nominal_fps=30.0, multi_camera=False)


class TestBuildRecorder:
    def test_disabled_without_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("NOWVA_VALGUS_DEBUG", raising=False)
        assert build_recorder(str(tmp_path), 30.0, False) is None

    def test_enabled_with_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NOWVA_VALGUS_DEBUG", "1")
        built = build_recorder(str(tmp_path), 30.0, False)
        assert built is not None
        assert built.path.is_dir()

    def test_falsy_env_value_stays_disabled(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NOWVA_VALGUS_DEBUG", "0")
        assert build_recorder(str(tmp_path), 30.0, False) is None


class TestIPCTap:
    def test_tap_forwards_and_records(self, recorder: ValgusDebugRecorder) -> None:
        client = _FakeIPCClient()
        tapped = recorder.tap(client)
        tapped.send_message({"type": "fault", "fault_type": "knee_valgus"})

        assert client.sent == [{"type": "fault", "fault_type": "knee_valgus"}]
        assert len(recorder._ipc) == 1
        assert recorder._ipc[0]["type"] == "fault"
        assert recorder._ipc[0]["knee"] is True

    def test_tap_delegates_other_attributes(self, recorder: ValgusDebugRecorder) -> None:
        client = _FakeIPCClient()
        tapped = recorder.tap(client)

        assert tapped.client_socket == "sentinel"
        tapped.disconnect()
        assert client.sent[-1]["type"] == "_disconnected"

    def test_non_knee_message_flagged_false(self, recorder: ValgusDebugRecorder) -> None:
        recorder.tap(_FakeIPCClient()).send_message({"type": "rest_complete"})
        assert recorder._ipc[0]["knee"] is False


class TestIsKneeRelated:
    def test_detects_valgus_payload(self) -> None:
        assert _is_knee_related({"type": "fault", "fault_type": "knee_valgus"})

    def test_detects_nested_angle_keys(self) -> None:
        assert _is_knee_related({"type": "frame_data", "joint_angles": {"knee_valgus_l": 3.0}})

    def test_detects_hip_adduction_payload(self) -> None:
        assert _is_knee_related({"type": "x", "hip_adduction_l": 2.0})

    def test_rejects_unrelated(self) -> None:
        assert not _is_knee_related({"type": "rest_complete"})


class TestReadValgusThresholds:
    def test_reads_squat_knee_tracking_thresholds(self) -> None:
        thresholds = _read_valgus_thresholds(_FakeRuleEngine())
        assert thresholds == {"primary": {"mild": 8.0, "moderate": 13.0, "severe": 18.0}}

    def test_has_no_fallback_thresholds(self) -> None:
        assert "fallback" not in _read_valgus_thresholds(_FakeRuleEngine())

    def test_reads_lunge_per_frame_valgus_rule(self) -> None:
        engine = _FakeRuleEngine()
        engine.rules = [KneeValgusRule(mild_threshold=6.0, moderate_threshold=10.0, severe_threshold=16.0)]
        thresholds = _read_valgus_thresholds(engine)
        assert thresholds == {"primary": {"mild": 6.0, "moderate": 10.0, "severe": 16.0}}

    def test_returns_none_without_valgus_rule(self) -> None:
        engine = _FakeRuleEngine()
        engine.rules = [DepthRule()]
        assert _read_valgus_thresholds(engine) is None


class TestRecordFrame:
    def test_records_knee_series(self, recorder: ValgusDebugRecorder) -> None:
        pipeline = _FakePipeline()
        recorder.record_frame(_frame(0, _angles()), _display(), pipeline, resting=False)

        assert recorder.frame_count == 1
        assert recorder._series["vl"] == [5.0]
        assert recorder._series["vr"] == [-3.0]
        assert recorder._series["kasr"] == [0.85]
        assert recorder._series["fcl"] == [0.9]
        assert recorder._series["phase"] == ["standing"]

    def test_missing_angles_record_none(self, recorder: ValgusDebugRecorder) -> None:
        recorder.record_frame(_frame(0, None), _display(), _FakePipeline(), resting=False)

        assert recorder.frame_count == 1
        assert recorder._series["vl"] == [None]
        assert recorder._series["kasr"] == [None]

    def test_series_stay_aligned_across_frames(self, recorder: ValgusDebugRecorder) -> None:
        pipeline = _FakePipeline()
        for i in range(5):
            angles = _angles() if i % 2 == 0 else None
            recorder.record_frame(_frame(i, angles), _display(), pipeline, resting=False)

        lengths = {key: len(values) for key, values in recorder._series.items()}
        assert set(lengths.values()) == {5}

    def test_records_faults_with_details(self, recorder: ValgusDebugRecorder) -> None:
        frame = _frame(3, _angles(), faults=[_valgus_fault()])
        recorder.record_frame(frame, _display(), _FakePipeline(), resting=False)

        assert len(recorder._faults) == 1
        fault = recorder._faults[0]
        assert fault["knee"] is True
        assert fault["severity"] == "moderate"
        assert fault["details"]["value"] == 14.0
        assert fault["details"]["side"] == "left"
        assert fault["details"]["phase"] == "ascent"
        assert fault["i"] == 0

    def test_boolean_fault_details_stay_boolean(self, recorder: ValgusDebugRecorder) -> None:
        frame = _frame(3, _angles(), faults=[_valgus_fault()])
        recorder.record_frame(frame, _display(), _FakePipeline(), resting=False)

        assert recorder._faults[0]["details"]["is_drift"] is False

    def test_records_rep_events(self, recorder: ValgusDebugRecorder) -> None:
        frame = _frame(4, _angles())
        frame.rep_data = RepData(rep_number=2, start_time=0.0, end_time=1.0,
                                 start_frame=0, end_frame=4)
        recorder.record_frame(frame, _display(), _FakePipeline(), resting=False)

        assert recorder._rep_events == [{"i": 0, "rep": 2}]

    def test_rep_phase_enum_serializes_to_plain_string(self, recorder: ValgusDebugRecorder) -> None:
        # The report's "rule can fire" logic compares phase to the literal
        # "bottom", so the enum must not be stored as "RepPhase.BOTTOM".
        pipeline = _FakePipeline()
        pipeline.rep_counter.phase = RepPhase.BOTTOM
        recorder.record_frame(_frame(0, _angles()), _display(), pipeline, resting=False)

        assert recorder._series["phase"] == ["bottom"]

    def test_samples_thresholds_on_first_frame(self, recorder: ValgusDebugRecorder) -> None:
        recorder.record_frame(_frame(0, _angles()), _display(), _FakePipeline(), resting=False)

        assert len(recorder._threshold_history) == 1
        assert recorder._threshold_history[0]["primary"]["severe"] == 18.0

    def test_threshold_rescale_appends_new_step(self, recorder: ValgusDebugRecorder) -> None:
        pipeline = _FakePipeline()
        recorder.record_frame(_frame(0, _angles()), _display(), pipeline, resting=False)
        pipeline._rule_engine.rules[-1].mild_threshold = 6.0

        for i in range(1, 32):
            recorder.record_frame(_frame(i, _angles()), _display(), pipeline, resting=False)

        assert len(recorder._threshold_history) == 2
        assert recorder._threshold_history[1]["primary"]["mild"] == 6.0
        assert recorder._threshold_history[1]["i"] == 30


class TestIPCOverflow:
    def test_caps_recorded_messages(self, recorder: ValgusDebugRecorder, monkeypatch) -> None:
        monkeypatch.setattr("biomechanics.viz.valgus_debug.MAX_IPC_MESSAGES", 3)
        tapped = recorder.tap(_FakeIPCClient())
        for _ in range(5):
            tapped.send_message({"type": "frame_data"})

        assert len(recorder._ipc) == 3
        assert recorder._ipc_overflow == 2


class TestRetimeClamp:
    def _recorder_at(self, tmp_path: Path, times: list[float]) -> ValgusDebugRecorder:
        rec = ValgusDebugRecorder(str(tmp_path), nominal_fps=30.0, multi_camera=False)
        rec._series["t"] = times
        return rec

    def test_unthrottled_run_keeps_nominal_fps(self, tmp_path: Path) -> None:
        # 240 frames in 0.3s -> ~800 fps, far outside a real capture rate.
        rec = self._recorder_at(tmp_path, [i * 0.00125 for i in range(240)])
        assert rec._measured_fps() > MAX_RETIME_FPS
        assert rec._retime_video(rec._measured_fps()) == 30.0

    def test_stalled_run_keeps_nominal_fps(self, tmp_path: Path) -> None:
        rec = self._recorder_at(tmp_path, [0.0, 600.0])
        assert rec._measured_fps() < MIN_RETIME_FPS
        assert rec._retime_video(rec._measured_fps()) == 30.0

    def test_near_nominal_rate_is_not_retimed(self, tmp_path: Path) -> None:
        rec = self._recorder_at(tmp_path, [i / 29.0 for i in range(60)])
        assert rec._retime_video(rec._measured_fps()) == 30.0

    def test_missing_video_keeps_nominal_fps(self, tmp_path: Path) -> None:
        rec = self._recorder_at(tmp_path, [i / 15.0 for i in range(60)])
        assert not rec._video_path.exists()
        assert rec._retime_video(rec._measured_fps()) == 30.0


class TestFinalize:
    def test_no_frames_returns_none(self, recorder: ValgusDebugRecorder) -> None:
        assert recorder.finalize("Barbell Back Squat") is None

    def test_writes_report_and_data(self, recorder: ValgusDebugRecorder) -> None:
        pipeline = _FakePipeline()
        tapped = recorder.tap(_FakeIPCClient())
        for i in range(20):
            pipeline.rep_counter.phase = "bottom" if 8 <= i <= 12 else "descending"
            faults = [_valgus_fault(i)] if i == REP_END_FRAME else []
            frame = _frame(i, _angles(), faults)
            if i == REP_END_FRAME:
                frame.rep_data = RepData(rep_number=1, start_time=0.0, end_time=float(i),
                                         start_frame=0, end_frame=i)
            recorder.record_frame(frame, _display(), pipeline, resting=False)
            tapped.send_message({"type": "frame_data", "joint_angles": {"knee_valgus_l": 5.0}})

        report = recorder.finalize("Barbell Back Squat")

        assert report is not None and report.exists()
        html = report.read_text()
        assert "__DATA_JSON__" not in html
        assert "__TITLE__" not in html
        assert "__VIDEO_B64__" not in html

        data = json.loads((recorder.path / "data.json").read_text())
        assert data["n_frames"] == 20
        assert data["mode"] == "2D knees-over-toes (single camera)"
        assert len(data["faults"]) == 1
        assert data["faults"][0]["knee"] is True
        assert data["faults"][0]["i"] == REP_END_FRAME
        assert data["reps"] == [{"i": REP_END_FRAME, "rep": 1}]
        assert all("fallback" not in step for step in data["thresholds"])
        assert len(data["ipc"]) == 20
        assert data["series"]["phase"][10] == "bottom"
        assert data["foot_confidence_threshold"] == 0.3

    def test_multi_camera_mode_label(self, tmp_path: Path) -> None:
        rec = ValgusDebugRecorder(str(tmp_path), nominal_fps=30.0, multi_camera=True)
        rec.record_frame(_frame(0, _angles()), _display(), _FakePipeline(), resting=False)
        rec.finalize("Barbell Back Squat")

        data = json.loads((rec.path / "data.json").read_text())
        assert data["mode"] == "3D knees-over-toes (triangulated)"
        assert data["multi_camera"] is True

    def test_template_reads_no_removed_keys(self) -> None:
        template = TEMPLATE_PATH.read_text()
        for key in REMOVED_REPORT_KEYS:
            assert key not in template

    def test_report_is_self_contained(self, recorder: ValgusDebugRecorder) -> None:
        for i in range(5):
            recorder.record_frame(_frame(i, _angles()), _display(), _FakePipeline(), resting=False)
        report = recorder.finalize("Barbell Back Squat")

        html = report.read_text()
        assert "http://" not in html
        assert "https://" not in html
