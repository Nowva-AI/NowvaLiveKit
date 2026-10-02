"""
Tests for Coaching Module

Tests CueCache, IPCBridge, and SessionTracker with a mock IPC client.
Verifies cue mapping, rate limiting, message formatting, and set detection.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.coaching.cue_cache import (
    GENERIC_POSITIVE_CUE_KEYS,
    POSITIVE_CUE_KEYS,
    SQUAT_CUES,
    CueCache,
)
from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.coaching.session_tracker import SessionTracker
from biomechanics.config import CoachingConfig, IPCConfig
from biomechanics.utils.types import (
    FaultEvent,
    FaultSeverity,
    JointAngles,
    PipelineFrame,
    RepData,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def coaching_config():
    """CoachingConfig with short set timeout for faster tests."""
    return CoachingConfig(min_cue_gap_seconds=2.0, set_timeout_seconds=5.0)


@pytest.fixture
def ipc_config():
    return IPCConfig(frame_send_interval=10, fault_cooldown_seconds=3.0)


@pytest.fixture
def cue_cache(coaching_config):
    return CueCache(coaching_config)


@pytest.fixture
def ipc_bridge(mock_ipc_client, coaching_config, ipc_config):
    return IPCBridge(mock_ipc_client, coaching_config, ipc_config)


@pytest.fixture
def session_tracker(ipc_bridge, coaching_config):
    return SessionTracker(ipc_bridge, coaching_config)


# =============================================================================
# HELPERS
# =============================================================================


def make_rep(
    rep_number,
    start_time=0.0,
    duration=2.5,
    depth=95.0,
    faults=None,
):
    """Create a RepData for testing."""
    return RepData(
        rep_number=rep_number,
        start_time=start_time,
        end_time=start_time + duration,
        start_frame=0,
        end_frame=75,
        max_depth_angle=depth,
        min_depth_angle=15.0,
        descent_time=duration * 0.48,
        ascent_time=duration * 0.4,
        faults=faults or [],
    )


def make_fault(fault_type="knee_valgus", severity=FaultSeverity.MODERATE, timestamp=0.0):
    """Create a FaultEvent for testing."""
    return FaultEvent(
        fault_type=fault_type,
        severity=severity,
        severity_score=float({"none": 0, "mild": 1, "moderate": 2, "severe": 3}[severity.value]),
        message=f"Test fault: {fault_type}",
        timestamp=timestamp,
        rep_number=1,
    )


# =============================================================================
# TEST CUE CACHE
# =============================================================================


class TestCueCache:
    """Test CueCache cue management and rate limiting."""

    def test_prepare_squat_cues(self, cue_cache):
        """prepare_for_exercise with 'squat' should load SQUAT_CUES."""
        cues = cue_cache.prepare_for_exercise("squat")
        assert "knees_out" in cues
        assert "deeper" in cues
        assert "rep_1" in cues
        assert "rep_20" in cues
        assert cue_cache.current_exercise == "squat"

    def test_prepare_barbell_back_squat(self, cue_cache):
        """Every squat name resolves to the squat profile's cues."""
        cues = cue_cache.prepare_for_exercise("Barbell Back Squat")
        assert cues == SQUAT_CUES
        assert cue_cache.profile_name == "squat"
        assert cue_cache.current_exercise == "barbell_back_squat"

    def test_prepare_romanian_deadlift_gets_its_own_cues(self, cue_cache):
        """The program library's name resolves to the RDL profile, whose cues
        are its own: nothing borrowed from the squat."""
        cues = cue_cache.prepare_for_exercise("Barbell Romanian Deadlift")
        assert cue_cache.profile_name == "romanian_deadlift"
        assert "rdl_flat_back" in cues
        assert "knees_out" not in cues
        assert "chest_up" not in cues
        assert "great_depth" not in cues

    def test_prepare_overhead_press_gets_press_cues(self, cue_cache):
        cues = cue_cache.prepare_for_exercise("Barbell Overhead Press")
        assert {"press_lockout", "press_elbows", "press_bar_path", "press_even"} <= set(cues)
        assert "knees_out" not in cues
        assert "rep_1" in cues

    def test_overhead_press_lockout_fault_gets_the_press_cue(self, cue_cache):
        """The squat also has a lockout fault; the press must not say "stand tall"."""
        cue_cache.prepare_for_exercise("Barbell Overhead Press")
        assert cue_cache.get_cue_for_fault("lockout", 10.0) == "press_lockout"

    def test_prepare_untracked_exercise_gets_only_praise_and_counts(self, cue_cache):
        cues = cue_cache.prepare_for_exercise("Barbell Bench Press")
        assert cue_cache.profile_name == "untracked"
        corrections = {key for key in cues if not key.startswith("rep_")}
        assert corrections == set(GENERIC_POSITIVE_CUE_KEYS)
        assert cue_cache.get_cue_for_fault("knee_valgus", 10.0) is None

    def test_get_cue_for_fault_mapping(self, cue_cache):
        """Should map fault types to correct cue keys."""
        cue_cache.prepare_for_exercise("squat")
        assert cue_cache.get_cue_for_fault("knee_valgus", 10.0) == "knees_out"

    def test_get_cue_for_fault_rate_limited(self, cue_cache):
        """A fault that doesn't outrank the last one waits out the gap."""
        cue_cache.prepare_for_exercise("squat")
        cue1 = cue_cache.get_cue_for_fault("knee_valgus", 10.0)
        assert cue1 == "knees_out"
        # Only 1s later, and asymmetry ranks below knee valgus.
        cue2 = cue_cache.get_cue_for_fault("bilateral_asymmetry", 11.0)
        assert cue2 is None

    def test_higher_priority_fault_preempts_the_gap(self, cue_cache):
        """Knee cave outranks every other squat fault and must not be starved
        by a lower-priority fault that claimed the slot first."""
        cue_cache.prepare_for_exercise("squat")
        assert cue_cache.get_cue_for_fault("hip_shift", 10.0) == "even_it_out"
        assert cue_cache.get_cue_for_fault("knee_valgus", 11.0) == "knees_out"

    def test_get_cue_for_fault_after_gap(self, cue_cache):
        """Cue should be returned after sufficient time gap."""
        cue_cache.prepare_for_exercise("squat")
        cue_cache.get_cue_for_fault("knee_valgus", 10.0)
        cue = cue_cache.get_cue_for_fault("forward_lean", 12.5)
        assert cue == "chest_up"

    def test_get_cue_for_unknown_fault(self, cue_cache):
        """Unknown fault type should return None."""
        cue_cache.prepare_for_exercise("squat")
        assert cue_cache.get_cue_for_fault("unknown_fault", 10.0) is None

    def test_get_cue_fault_not_in_exercise_cues(self, cue_cache):
        """Fault maps to cue key not in current exercise cues → None."""
        cue_cache.prepare_for_exercise("deadlift")
        # "depth" → "deeper", but "deeper" is not in DEADLIFT_CUES
        assert cue_cache.get_cue_for_fault("depth", 10.0) is None

    def test_get_rep_cue_valid(self, cue_cache):
        """Should return rep cue key for valid rep numbers."""
        cue_cache.prepare_for_exercise("squat")
        assert cue_cache.get_rep_cue(1) == "rep_1"
        assert cue_cache.get_rep_cue(20) == "rep_20"

    def test_get_rep_cue_out_of_range(self, cue_cache):
        """Should return None for rep > 20."""
        cue_cache.prepare_for_exercise("squat")
        assert cue_cache.get_rep_cue(21) is None

    def test_get_positive_cue(self, cue_cache):
        """Should return one of the positive cue keys."""
        cue_cache.prepare_for_exercise("squat")
        positive = cue_cache.get_positive_cue()
        assert positive in POSITIVE_CUE_KEYS

    def test_get_positive_cue_deadlift(self, cue_cache):
        """Deadlift praise never includes the squat-only great_depth."""
        cue_cache.prepare_for_exercise("deadlift")
        positive = cue_cache.get_positive_cue()
        assert positive in GENERIC_POSITIVE_CUE_KEYS

    def test_returned_dict_is_copy(self, cue_cache):
        """Returned dict should not mutate internal state."""
        cues = cue_cache.prepare_for_exercise("squat")
        cues["injected"] = "bad"
        assert "injected" not in cue_cache.cues


# =============================================================================
# TEST IPC BRIDGE
# =============================================================================


class TestIPCBridge:
    """Test IPCBridge message formatting and throttling."""

    def test_prepare_exercise_sends_cache_cues(self, ipc_bridge, mock_ipc_client):
        """prepare_exercise should send cache_cues message."""
        ipc_bridge.prepare_exercise("squat")
        msg = mock_ipc_client.messages[-1]
        assert msg["type"] == "cache_cues"
        assert msg["exercise_name"] == "squat"
        assert "knees_out" in msg["cues"]

    def test_send_frame_data_throttled(self, ipc_bridge, mock_ipc_client):
        """Should only send every Nth frame."""
        for i in range(25):
            frame = PipelineFrame(
                frame_index=i,
                timestamp=i * 0.033,
                joint_angles=JointAngles(knee_flexion_l=90.0, knee_flexion_r=90.0),
                latency_ms={"pose": 8.0, "ik": 4.0},
            )
            ipc_bridge.send_frame_data(frame)

        frame_msgs = [m for m in mock_ipc_client.messages if m["type"] == "frame_data"]
        # counter starts at 0, increments to 1..25; fires at 10 and 20
        assert len(frame_msgs) == 2

    def test_send_frame_data_skips_no_angles(self, ipc_bridge, mock_ipc_client):
        """Should not send when joint_angles is None."""
        ipc_bridge.frame_counter = 9  # next increment → 10
        frame = PipelineFrame(frame_index=10, timestamp=0.33)
        ipc_bridge.send_frame_data(frame)
        frame_msgs = [m for m in mock_ipc_client.messages if m["type"] == "frame_data"]
        assert len(frame_msgs) == 0

    def test_send_fault_with_cue(self, ipc_bridge, mock_ipc_client):
        """send_fault should include cue from cue_cache."""
        ipc_bridge.prepare_exercise("squat")
        fault = make_fault("knee_valgus", timestamp=10.0)
        ipc_bridge.send_fault(fault)
        fault_msgs = [m for m in mock_ipc_client.messages if m["type"] == "fault"]
        assert len(fault_msgs) == 1
        assert fault_msgs[0]["cue"] == "knees_out"
        assert fault_msgs[0]["severity"] == "moderate"

    def test_send_fault_cooldown_same_type(self, ipc_bridge, mock_ipc_client):
        """Same fault_type within cooldown should be suppressed."""
        ipc_bridge.prepare_exercise("squat")
        ipc_bridge.send_fault(make_fault("knee_valgus", timestamp=10.0))
        ipc_bridge.send_fault(make_fault("knee_valgus", timestamp=11.0))  # 1s < 3s cooldown
        fault_msgs = [m for m in mock_ipc_client.messages if m["type"] == "fault"]
        assert len(fault_msgs) == 1

    def test_send_fault_different_types_no_shared_cooldown(self, ipc_bridge, mock_ipc_client):
        """Different fault types should not share cooldown."""
        ipc_bridge.prepare_exercise("squat")
        ipc_bridge.send_fault(make_fault("knee_valgus", timestamp=10.0))
        ipc_bridge.send_fault(make_fault("forward_lean", severity=FaultSeverity.MILD, timestamp=10.5))
        fault_msgs = [m for m in mock_ipc_client.messages if m["type"] == "fault"]
        assert len(fault_msgs) == 2

    def test_send_shallow_rep_message(self, ipc_bridge, mock_ipc_client):
        """Shallow rep should carry the depth class and the deeper cue."""
        fault = make_fault("depth", severity=FaultSeverity.MODERATE, timestamp=10.0)
        fault.details = {
            "depth_ratio": 0.62, "target_ratio": 0.0, "depth_cm_above_parallel": 28.6,
            "category": "quarter", "shallow_rep": True,
        }
        ipc_bridge.send_shallow_rep(1, fault=fault, set_number=2)

        msg = mock_ipc_client.messages[-1]
        assert msg["type"] == "shallow_rep"
        assert msg["depth_class"] == 1
        assert msg["depth_class_name"] == "Quarter"
        assert msg["cue"] == "deeper"
        assert msg["fault_type"] == "depth"
        assert msg["severity"] == "moderate"
        assert msg["depth_ratio"] == 0.62
        assert msg["depth_cm_above_parallel"] == 28.6
        assert msg["category"] == "quarter"
        assert msg["set_number"] == 2

    def test_send_shallow_rep_ignores_fault_cooldown(self, ipc_bridge, mock_ipc_client):
        """Every shallow rep must reach the agent — no rate limiting."""
        fault = make_fault("depth", timestamp=10.0)
        for _ in range(4):
            ipc_bridge.send_shallow_rep(1, fault=fault)
        shallow_msgs = [m for m in mock_ipc_client.messages if m["type"] == "shallow_rep"]
        assert len(shallow_msgs) == 4

    def test_send_shallow_rep_without_fault(self, ipc_bridge, mock_ipc_client):
        """Message is still useful when no depth rule produced a fault."""
        ipc_bridge.send_shallow_rep(2)
        msg = mock_ipc_client.messages[-1]
        assert msg["type"] == "shallow_rep"
        assert msg["depth_class_name"] == "Half"
        assert "fault_type" not in msg

    def test_send_rep_complete_messages(self, ipc_bridge, mock_ipc_client):
        """Should send rep_complete and legacy rep_count (no play_cue — orchestrator handles those)."""
        ipc_bridge.prepare_exercise("squat")
        rep = make_rep(rep_number=1, depth=95.0)
        ipc_bridge.send_rep_complete(rep)

        types = [m["type"] for m in mock_ipc_client.messages]
        assert "rep_complete" in types
        assert "rep_count" in types
        assert "play_cue" not in types  # Orchestrator handles cue dispatch now

        rep_msg = next(m for m in mock_ipc_client.messages if m["type"] == "rep_complete")
        assert rep_msg["depth_category"] == "parallel"
        assert rep_msg["rep_number"] == 1
        assert rep_msg["is_clean"] is True

    def test_send_rep_complete_no_play_cue_messages(self, ipc_bridge, mock_ipc_client):
        """IPCBridge should not send play_cue — CoachingOrchestrator handles cue dispatch."""
        ipc_bridge.prepare_exercise("squat")
        rep = make_rep(rep_number=1, depth=105.0)
        ipc_bridge.send_rep_complete(rep)

        play_cues = [m for m in mock_ipc_client.messages if m["type"] == "play_cue"]
        assert len(play_cues) == 0

    def test_send_set_complete(self, ipc_bridge, mock_ipc_client):
        """Should compute and send set summary."""
        ipc_bridge.prepare_exercise("squat")
        reps = [make_rep(1, depth=90.0), make_rep(2, depth=100.0), make_rep(3, depth=95.0)]
        ipc_bridge.send_set_complete(1, reps)

        set_msg = next(m for m in mock_ipc_client.messages if m["type"] == "set_complete")
        assert set_msg["set_number"] == 1
        assert set_msg["total_reps"] == 3
        assert 90.0 <= set_msg["avg_depth"] <= 100.0
        assert set_msg["clean_reps"] == 3

    def test_send_set_complete_empty_reps(self, ipc_bridge, mock_ipc_client):
        """Should not send anything for empty rep list."""
        initial_count = len(mock_ipc_client.messages)
        ipc_bridge.send_set_complete(1, [])
        assert len(mock_ipc_client.messages) == initial_count

    def test_send_pipeline_status(self, ipc_bridge, mock_ipc_client):
        """Should send pipeline status message."""
        ipc_bridge.send_pipeline_status("running", {"pose": 8.0, "ik": 4.0})
        msg = mock_ipc_client.messages[-1]
        assert msg["type"] == "pipeline_status"
        assert msg["status"] == "running"
        assert msg["latency_ms"]["pose"] == 8.0

    def test_depth_category_comes_from_hip_height_when_measured(self):
        """105° of knee flexion is not 'below parallel' when the hip is 12 cm above the knee."""
        rep = RepData(
            rep_number=1, start_time=0.0, end_time=2.0, start_frame=0, end_frame=60,
            max_depth_angle=105.0, features={"depth_ratio": 0.26},
        )
        assert IPCBridge._rep_depth_category(rep) == "half"

    def test_depth_category_falls_back_to_knee_angle_without_features(self):
        rep = RepData(
            rep_number=1, start_time=0.0, end_time=2.0, start_frame=0, end_frame=60,
            max_depth_angle=45.0,
        )
        assert IPCBridge._rep_depth_category(rep) == "quarter"


# =============================================================================
# TEST SESSION TRACKER
# =============================================================================


class TestSessionTracker:
    """Test SessionTracker set boundary detection and session stats."""

    def test_first_rep_starts_set(self, session_tracker, mock_ipc_client):
        """First rep should start set 1."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0))
        assert session_tracker.current_set_number == 1
        assert session_tracker.set_active is True
        assert session_tracker.total_reps == 1

    def test_consecutive_reps_same_set(self, session_tracker):
        """Reps within timeout should stay in same set."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0))
        session_tracker.on_rep_complete(make_rep(2, start_time=3.0))
        session_tracker.on_rep_complete(make_rep(3, start_time=6.0))
        assert session_tracker.current_set_number == 1
        assert len(session_tracker.current_set_reps) == 3

    def test_timeout_triggers_new_set(self, session_tracker, mock_ipc_client):
        """Gap > set_timeout between reps should end set and start new one."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0))
        session_tracker.on_rep_complete(make_rep(2, start_time=3.0))
        # Rep 3 arrives after >5s gap (timeout is 5s in fixture)
        session_tracker.on_rep_complete(make_rep(3, start_time=11.0))

        assert session_tracker.current_set_number == 2
        assert session_tracker.total_sets == 1
        set_msgs = [m for m in mock_ipc_client.messages if m["type"] == "set_complete"]
        assert len(set_msgs) == 1
        assert set_msgs[0]["total_reps"] == 2

    def test_check_set_timeout_expired(self, session_tracker):
        """check_set_timeout should detect timeout and end set."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0))
        # end_time = 2.5, so we need current_time > 2.5 + 5.0 = 7.5
        ended = session_tracker.check_set_timeout(current_time=8.0)
        assert ended is True
        assert session_tracker.set_active is False
        assert session_tracker.total_sets == 1

    def test_check_set_timeout_not_expired(self, session_tracker):
        """check_set_timeout should return False if not expired."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0))
        # end_time = 2.5, gap = 4.0 - 2.5 = 1.5 < 5.0 timeout
        ended = session_tracker.check_set_timeout(current_time=4.0)
        assert ended is False
        assert session_tracker.set_active is True

    def test_force_end_set(self, session_tracker, mock_ipc_client):
        """force_end_set should immediately end the current set."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0))
        session_tracker.on_rep_complete(make_rep(2, start_time=3.0))
        session_tracker.force_end_set()
        assert session_tracker.set_active is False
        set_msgs = [m for m in mock_ipc_client.messages if m["type"] == "set_complete"]
        assert len(set_msgs) == 1

    def test_session_stats(self, session_tracker):
        """session_stats should aggregate across all reps."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0, depth=90.0))
        session_tracker.on_rep_complete(make_rep(2, start_time=3.0, depth=100.0))
        stats = session_tracker.session_stats
        assert stats["total_reps"] == 2
        assert stats["avg_depth"] == 95.0
        assert stats["clean_rep_percentage"] == 100.0

    def test_reset(self, session_tracker):
        """reset should clear all session state."""
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0))
        session_tracker.reset()
        assert session_tracker.total_reps == 0
        assert session_tracker.total_sets == 0
        assert session_tracker.current_set_number == 0
        assert session_tracker.all_reps == []

    def test_multiple_sets_full_workflow(self, session_tracker, mock_ipc_client):
        """Full workflow: 2 sets with timeout between them."""
        # Set 1: 3 reps
        session_tracker.on_rep_complete(make_rep(1, start_time=0.0, depth=90.0))
        session_tracker.on_rep_complete(make_rep(2, start_time=2.5, depth=95.0))
        session_tracker.on_rep_complete(make_rep(3, start_time=5.0, depth=92.0))

        # Timeout detected
        session_tracker.check_set_timeout(current_time=12.0)

        # Set 2: 2 reps
        session_tracker.on_rep_complete(make_rep(4, start_time=15.0, depth=100.0))
        session_tracker.on_rep_complete(make_rep(5, start_time=17.5, depth=98.0))
        session_tracker.force_end_set()

        assert session_tracker.total_sets == 2
        assert session_tracker.total_reps == 5
        set_msgs = [m for m in mock_ipc_client.messages if m["type"] == "set_complete"]
        assert len(set_msgs) == 2
        assert set_msgs[0]["total_reps"] == 3
        assert set_msgs[1]["total_reps"] == 2


# =============================================================================
# TEST DIAGNOSIS INTEGRATION
# =============================================================================


def _squat_bottom_kpts_mediapipe() -> list[list[float]]:
    return [
        [0.00, -0.25, 0.20],   # 0  nose
        [0.02, -0.27, 0.19],   # 1  left_eye
        [-0.02, -0.27, 0.19],  # 2  right_eye
        [0.05, -0.25, 0.16],   # 3  left_ear
        [-0.05, -0.25, 0.16],  # 4  right_ear
        [0.18, -0.15, 0.12],   # 5  left_shoulder
        [-0.18, -0.15, 0.12],  # 6  right_shoulder
        [0.22, 0.05, 0.18],    # 7  left_elbow
        [-0.22, 0.05, 0.18],   # 8  right_elbow
        [0.20, 0.10, 0.22],    # 9  left_wrist
        [-0.20, 0.10, 0.22],   # 10 right_wrist
        [0.12, 0.00, 0.00],    # 11 left_hip
        [-0.12, 0.00, 0.00],   # 12 right_hip
        [0.12, 0.02, 0.30],    # 13 left_knee
        [-0.12, 0.02, 0.30],   # 14 right_knee
        [0.14, 0.38, 0.10],    # 15 left_ankle
        [-0.14, 0.38, 0.10],   # 16 right_ankle
        [0.12, 0.40, 0.16],    # 17 left_foot_index
        [-0.12, 0.40, 0.16],   # 18 right_foot_index
    ]


def _squat_bottom_angles() -> dict[str, float]:
    return {
        "hip_flexion_l": 95.0,
        "hip_flexion_r": 93.0,
        "hip_adduction_l": 2.0,
        "hip_adduction_r": 1.5,
        "hip_rotation_l": 5.0,
        "hip_rotation_r": 4.0,
        "knee_flexion_l": 110.0,
        "knee_flexion_r": 108.0,
        "ankle_dorsiflexion_l": 28.0,
        "ankle_dorsiflexion_r": 26.0,
        "knee_valgus_l": 3.5,
        "knee_valgus_r": 2.0,
        "foot_confidence_l": 0.8,
        "foot_confidence_r": 0.7,
        "shoulder_flexion_l": 40.0,
        "shoulder_flexion_r": 38.0,
        "shoulder_abduction_l": 15.0,
        "shoulder_abduction_r": 14.0,
        "elbow_flexion_l": 90.0,
        "elbow_flexion_r": 88.0,
        "wrist_y_l": -10.0,
        "wrist_y_r": -10.0,
        "wrist_x_l": 5.0,
        "wrist_x_r": 5.0,
        "trunk_flexion": 145.0,
        "trunk_lateral_flexion": 1.0,
        "trunk_rotation": 2.0,
        "pelvis_tilt": 12.0,
        "pelvis_list": 0.5,
        "pelvis_rotation": 1.0,
    }


def _default_athlete_params() -> dict:
    return {
        "shoulder_width_m": 0.40,
        "hip_width_m": 0.30,
        "femur_avg_m": 0.42,
        "torso_avg_m": 0.45,
        "tibia_avg_m": 0.43,
        "foot_avg_m": 0.26,
    }


class TestDiagnosisIntegration:

    def test_diagnosis_skipped_without_athlete_params(self, session_tracker, mock_ipc_client):
        """No set_athlete_params → no diagnosis_complete, but set_complete still fires."""
        session_tracker.on_rep_complete(
            make_rep(1, start_time=0.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )
        session_tracker.force_end_set()

        types = [m["type"] for m in mock_ipc_client.messages]
        assert "set_complete" in types
        assert "diagnosis_complete" not in types

    def test_diagnosis_runs_on_set_end(self, session_tracker, mock_ipc_client):
        """With athlete params, completing a set should send diagnosis_complete."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})

        for i in range(1, 4):
            session_tracker.on_rep_complete(
                make_rep(i, start_time=(i - 1) * 3.0),
                bottom_kpts=_squat_bottom_kpts_mediapipe(),
                bottom_angles=_squat_bottom_angles(),
            )
        session_tracker.force_end_set()

        diag_msgs = [m for m in mock_ipc_client.messages if m["type"] == "diagnosis_complete"]
        assert len(diag_msgs) == 1

        msg = diag_msgs[0]
        assert msg["set_number"] == 1
        assert "diagnosis" in msg
        assert "scoring" in msg
        assert "confidence" in msg["diagnosis"]
        assert "detected_symptoms" in msg["diagnosis"]
        assert "immediate_causes" in msg["diagnosis"]
        assert "session_causes" in msg["diagnosis"]
        assert "combined_perturbation" in msg["diagnosis"]
        assert "mean_score" in msg["scoring"]
        assert "best_rep" in msg["scoring"]
        assert "worst_rep" in msg["scoring"]
        assert "trend_slope" in msg["scoring"]

    def test_rep_kinematic_buffer_cleared_between_sets(self, session_tracker, mock_ipc_client):
        """Second set's diagnosis should only include reps from the second set."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})

        # Set 1: 2 reps
        session_tracker.on_rep_complete(
            make_rep(1, start_time=0.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )
        session_tracker.on_rep_complete(
            make_rep(2, start_time=3.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )
        session_tracker.force_end_set()

        # Set 2: 3 reps
        for i in range(3, 6):
            session_tracker.on_rep_complete(
                make_rep(i, start_time=20.0 + (i - 3) * 3.0),
                bottom_kpts=_squat_bottom_kpts_mediapipe(),
                bottom_angles=_squat_bottom_angles(),
            )
        session_tracker.force_end_set()

        diag_msgs = [m for m in mock_ipc_client.messages if m["type"] == "diagnosis_complete"]
        assert len(diag_msgs) == 2

        # First diagnosis should have had 2 reps, second should have had 3
        # We can verify via the scoring — best_rep/worst_rep should be from the correct set
        second_diag = diag_msgs[1]
        assert second_diag["set_number"] == 2
        assert second_diag["scoring"]["best_rep"] in [3, 4, 5]
        assert second_diag["scoring"]["worst_rep"] in [3, 4, 5]

    def test_diagnosis_complete_includes_per_dimension_scoring(self, session_tracker, mock_ipc_client):
        """diagnosis_complete message should include per-dimension score averages."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})

        for i in range(1, 4):
            session_tracker.on_rep_complete(
                make_rep(i, start_time=(i - 1) * 3.0),
                bottom_kpts=_squat_bottom_kpts_mediapipe(),
                bottom_angles=_squat_bottom_angles(),
            )
        session_tracker.force_end_set()

        diag_msgs = [m for m in mock_ipc_client.messages if m["type"] == "diagnosis_complete"]
        assert len(diag_msgs) == 1

        per_dim = diag_msgs[0]["scoring"]["per_dimension"]
        assert set(per_dim.keys()) == {
            "depth", "trunk_control", "knee_tracking", "symmetry", "tempo",
        }
        for key, value in per_dim.items():
            assert 0.0 <= value <= 1.0, f"{key} score {value} out of [0, 1] range"

    def test_reset_clears_kinematic_buffer(self, session_tracker, mock_ipc_client):
        """reset() should clear the kinematic buffer."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})

        session_tracker.on_rep_complete(
            make_rep(1, start_time=0.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )
        assert len(session_tracker._rep_kinematic_buffer) == 1

        session_tracker.reset()
        assert len(session_tracker._rep_kinematic_buffer) == 0

    def test_missing_bottom_kpts_excluded_from_buffer(self, session_tracker, mock_ipc_client):
        """Reps with None bottom_kpts are excluded from kinematic buffer."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})

        session_tracker.on_rep_complete(
            make_rep(1, start_time=0.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )
        session_tracker.on_rep_complete(
            make_rep(2, start_time=3.0),
            bottom_kpts=None,
            bottom_angles=_squat_bottom_angles(),
        )
        session_tracker.on_rep_complete(
            make_rep(3, start_time=6.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )
        assert len(session_tracker._rep_kinematic_buffer) == 2

        session_tracker.force_end_set()
        diag_msgs = [m for m in mock_ipc_client.messages if m["type"] == "diagnosis_complete"]
        assert len(diag_msgs) == 1

    def test_missing_bottom_angles_excluded_from_buffer(self, session_tracker, mock_ipc_client):
        """Reps with None bottom_angles are excluded from kinematic buffer."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})

        session_tracker.on_rep_complete(
            make_rep(1, start_time=0.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=None,
        )
        session_tracker.on_rep_complete(
            make_rep(2, start_time=3.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )
        assert len(session_tracker._rep_kinematic_buffer) == 1

    def test_all_reps_missing_data_skips_diagnosis(self, session_tracker, mock_ipc_client):
        """If all reps have missing bottom frame data, diagnosis is skipped."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})

        session_tracker.on_rep_complete(
            make_rep(1, start_time=0.0),
            bottom_kpts=None,
            bottom_angles=None,
        )
        session_tracker.on_rep_complete(
            make_rep(2, start_time=3.0),
            bottom_kpts=None,
            bottom_angles=_squat_bottom_angles(),
        )
        session_tracker.force_end_set()

        types = [m["type"] for m in mock_ipc_client.messages]
        assert "set_complete" in types
        assert "diagnosis_complete" not in types


class TestExerciseChange:
    def _squat_rep(self, session_tracker, rep_number: int, start_time: float) -> None:
        session_tracker.on_rep_complete(
            make_rep(rep_number, start_time=start_time),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )

    def test_exercise_without_diagnosis_sends_no_diagnosis(self, session_tracker, mock_ipc_client):
        """The squat diagnosis engine must not judge an overhead press."""
        session_tracker.set_athlete_params(_default_athlete_params(), {})
        session_tracker.on_exercise_changed(diagnosis_enabled=False)

        self._squat_rep(session_tracker, 1, start_time=0.0)
        session_tracker.force_end_set()

        types = [m["type"] for m in mock_ipc_client.messages]
        assert "set_complete" in types
        assert "diagnosis_complete" not in types
        assert session_tracker.build_on_demand_demo() is None

    def test_switch_mid_set_closes_the_old_exercise_set(self, session_tracker, mock_ipc_client):
        self._squat_rep(session_tracker, 1, start_time=0.0)
        session_tracker.on_exercise_changed(diagnosis_enabled=False)

        set_msgs = [m for m in mock_ipc_client.messages if m["type"] == "set_complete"]
        assert len(set_msgs) == 1
        assert not session_tracker.set_active
        assert session_tracker.get_last_rep_snapshot() is None

    def test_set_numbers_restart_for_the_new_exercise(self, session_tracker, mock_ipc_client):
        self._squat_rep(session_tracker, 1, start_time=0.0)
        session_tracker.force_end_set()
        session_tracker.on_exercise_changed(diagnosis_enabled=True)

        self._squat_rep(session_tracker, 1, start_time=60.0)

        rep_msgs = [m for m in mock_ipc_client.messages if m["type"] == "rep_complete"]
        assert rep_msgs[-1]["set_number"] == 1


class TestBottomFrameTracking:

    def _complete_rep(self, session_tracker, rep_number: int) -> None:
        session_tracker.on_rep_complete(
            make_rep(rep_number, start_time=(rep_number - 1) * 3.0),
            bottom_kpts=_squat_bottom_kpts_mediapipe(),
            bottom_angles=_squat_bottom_angles(),
        )

    def test_bottom_frame_for_rep_returns_viewer_kpts(self, session_tracker, mock_ipc_client):
        session_tracker.set_athlete_params(_default_athlete_params(), {})
        self._complete_rep(session_tracker, 1)
        self._complete_rep(session_tracker, 2)

        frame = session_tracker.bottom_frame_for_rep(1)

        assert frame is not None
        assert len(frame) == 19
        assert len(frame[0]) == 3

    def test_unknown_rep_falls_back_to_latest_frame(self, session_tracker, mock_ipc_client):
        session_tracker.set_athlete_params(_default_athlete_params(), {})
        self._complete_rep(session_tracker, 1)

        assert session_tracker.bottom_frame_for_rep(99) is not None

    def test_empty_buffer_returns_none(self, session_tracker, mock_ipc_client):
        assert session_tracker.bottom_frame_for_rep(1) is None

    def test_reset_rep_buffers_clears_frames(self, session_tracker, mock_ipc_client):
        session_tracker.set_athlete_params(_default_athlete_params(), {})
        self._complete_rep(session_tracker, 1)

        session_tracker.reset_rep_buffers()

        assert session_tracker.bottom_frame_for_rep(1) is None
