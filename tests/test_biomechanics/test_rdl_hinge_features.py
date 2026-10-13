"""
RdlRepAnalyzer and its per-frame samples (docs/exercises/rdl.md §3.2-§4.2),
driven with FrameContexts built from synthetic poses in the production frame
(Y-down, forward = -Z), calling the protocol in the pipeline's order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.analysis.hinge_features import (
    EXERCISE_NAME,
    HIP_SOURCE_MEASURED,
    HIP_SOURCE_SETUP,
    REJECT_IGNORE,
    REJECT_PARTIAL,
    HingeRepFeatures,
    RdlRepAnalyzer,
    build_hinge_sample,
    descent_crossings,
)
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.observability import SINGLE_CAMERA, TRIANGULATED
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
from biomechanics.utils.frame_context import (
    GRAVITY_BODY,
    GRAVITY_CAMERA_PITCH,
    GRAVITY_MEASURED,
    MOUNT_INCLINOMETER,
    FrameContext,
)
from biomechanics.utils.geometry import WORLD_UP, level_camera_frame
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Skeleton3D
from test_rdl_poses import (
    CLEAN_BOTTOM,
    dropping_hand,
    NUM_KEYPOINTS,
    STANDING,
    RdlPose,
    bar_offset_ratio,
    bar_travel,
    camera_view,
    cosine_crossing_time_s,
    cosine_move,
    hip_sink_pct,
    pose_at_drop,
)

CONFIDENCE = 0.9
LOW_HIP_CONFIDENCE = 0.3
FPS_RATES = (12.0, 30.0)
DESCENT_S = 2.0
HOLD_S = 0.3
ASCENT_S = 1.2
STAND_S = 1.5
# The counter enters a rep a few frames after the motion starts.
REP_START_LAG_S = 0.25
CROSSING_TOLERANCE_S = 0.05
RATIO_TOLERANCE = 0.005
HINGE_TOLERANCE_DEG = 1.0
TILT_TOLERANCE_DEG = 0.5
PITCH_TOLERANCE_DEG = 0.5
HAND_DROP_DEG = 7.0
GRADUAL_DROP_DEG = 12.0
YAW_DEG = 25.0
BAR_AHEAD_M = 0.12
MEASURED_UP_TILT_DEG = 3.0
MOUNT_PITCH_DEG = 2.0
KNOCKED_PITCH_DEG = 6.0
TURNAROUND_DROP = 0.2
TURNAROUND_DESCENT_S = 1.5
TRIPWIRE_STANDING_FRAMES = 40
TOO_FEW_STANDING_FRAMES = 30
GROSS_PITCH_DEG = 13.0
OFFSET_TOLERANCE = 0.01
MIN_SHOULDER_DROP_CM = 35.0
TRIP_DELTA_TOLERANCE_DEG = 0.1

_SOLVER = AnalyticalIKSolver()


def _config() -> BiomechanicsConfig:
    return BiomechanicsConfig()


def _analyzer() -> RdlRepAnalyzer:
    config = _config()
    return RdlRepAnalyzer(config.faults.rdl, config.readiness_gate)


def _ctx(
    points: np.ndarray,
    timestamp: float = 0.0,
    capture_mode: str = SINGLE_CAMERA,
    confidences: np.ndarray | None = None,
    with_angles: bool = True,
    **fields,
) -> FrameContext:
    if confidences is None:
        confidences = np.full(NUM_KEYPOINTS, CONFIDENCE)
    skeleton = Skeleton3D.from_numpy(points, confidences=confidences, timestamp=timestamp)
    return FrameContext(
        skeleton_3d=skeleton,
        angles=_SOLVER.solve(skeleton) if with_angles else None,
        timestamp=timestamp,
        phase="idle",
        in_rep=False,
        capture_mode=capture_mode,
        raw_confidences=confidences,
        **fields,
    )


class _Driver:
    """Feeds poses to an analyser on a clock, starting and finishing reps
    where the pipeline would."""

    def __init__(self, analyzer: RdlRepAnalyzer, fps: float, capture_mode: str = SINGLE_CAMERA, **fields) -> None:
        self.analyzer = analyzer
        self.fps = fps
        self.capture_mode = capture_mode
        self.fields = fields
        self.time_s = 0.0
        self.rep_count = 0
        self.view_fn = None

    def feed(self, poses: list[RdlPose], confidences: np.ndarray | None = None) -> None:
        for pose in poses:
            points = pose.points()
            if self.view_fn is not None:
                points = self.view_fn(points)
            self.analyzer.observe(_ctx(points, self.time_s, self.capture_mode, confidences, **self.fields))
            self.time_s += 1.0 / self.fps

    def rep(
        self,
        bottom: RdlPose = CLEAN_BOTTOM,
        top: RdlPose = STANDING,
        stand_s: float = STAND_S,
        down_s: float = DESCENT_S,
        hold_s: float = HOLD_S,
        up_s: float = ASCENT_S,
        lag_s: float = REP_START_LAG_S,
        piston: bool = False,
        rep_confidences: np.ndarray | None = None,
        finish_at: RdlPose | None = None,
    ) -> HingeRepFeatures:
        self.feed([top] * int(round(stand_s * self.fps)))
        descent = cosine_move(top, bottom, down_s, self.fps)
        lag_frames = int(round(lag_s * self.fps))
        if piston:
            self.analyzer.on_rep_start(piston=True)
            lag_frames = 0
        else:
            self.feed(descent[:lag_frames])
            self.analyzer.on_rep_start(piston=False)
        self.feed(descent[lag_frames:], rep_confidences)
        self.feed([bottom] * int(round(hold_s * self.fps)), rep_confidences)
        self.feed(cosine_move(bottom, finish_at or top, up_s, self.fps), rep_confidences)
        self.rep_count += 1
        return self.analyzer.finish_rep(self.rep_count)


def rdl_features(**overrides) -> HingeRepFeatures:
    rdl = _config().faults.rdl
    fields = dict(
        rep_number=1,
        hip_sink_bands=(rdl.hip_sink.mild, rdl.hip_sink.moderate, rdl.hip_sink.severe),
        count_floor_travel=rdl.rom.count_floor_travel,
        partial_min_travel=rdl.rom.partial_min_travel,
        partial_min_duration_s=rdl.rom.partial_min_duration_s,
        fallback_floor_deg=rdl.rom.fallback_floor_deg,
        partial_bands=(rdl.partial.moderate, rdl.partial.severe),
    )
    fields.update(overrides)
    return HingeRepFeatures(**fields)


def _tilted_about_x(points: np.ndarray, degrees: float) -> np.ndarray:
    return level_camera_frame(points, degrees)


class TestFrameSample:
    def test_heights_are_cm_above_the_ankle_midpoint(self):
        points = CLEAN_BOTTOM.points()
        sample = build_hinge_sample(_ctx(points), _config().readiness_gate, in_rep=True)
        ankle_y = points[CK.LEFT_ANKLE][1]
        assert sample.shoulder_cm == pytest.approx((ankle_y - points[CK.LEFT_SHOULDER][1]) * 100.0, abs=1e-6)
        assert sample.hip_cm == pytest.approx((ankle_y - points[CK.LEFT_HIP][1]) * 100.0, abs=1e-6)
        assert sample.knee_cm == pytest.approx((ankle_y - points[CK.LEFT_KNEE][1]) * 100.0, abs=1e-6)

    def test_right_hand_lower_reads_positive_tilt(self):
        sample = build_hinge_sample(
            _ctx(RdlPose(tilt_deg=HAND_DROP_DEG).points()), _config().readiness_gate, in_rep=False,
        )
        assert sample.bar_tilt_signed_deg == pytest.approx(HAND_DROP_DEG, abs=TILT_TOLERANCE_DEG)

    def test_a_bar_on_the_legs_reads_its_offset_and_a_bar_ahead_reads_more(self):
        readiness = _config().readiness_gate
        on_legs = build_hinge_sample(_ctx(CLEAN_BOTTOM.points(), capture_mode=TRIANGULATED), readiness, True)
        ahead = build_hinge_sample(
            _ctx(RdlPose(75.0, 20.0, 5.0, bar_forward_m=BAR_AHEAD_M).points(), capture_mode=TRIANGULATED),
            readiness, True,
        )
        assert on_legs.bar_offset_ratio == pytest.approx(bar_offset_ratio(CLEAN_BOTTOM.points()), abs=OFFSET_TOLERANCE)
        assert ahead.bar_offset_ratio > on_legs.bar_offset_ratio

    @pytest.mark.parametrize("gravity_source, tilt_deg", [(GRAVITY_BODY, 0.0), (GRAVITY_MEASURED, MEASURED_UP_TILT_DEG)])
    def test_upright_trunk_reads_zero_pitch_whatever_the_up(self, gravity_source: str, tilt_deg: float):
        readiness = _config().readiness_gate
        up = _tilted_about_x(WORLD_UP[np.newaxis, :], tilt_deg)[0]
        standing = _tilted_about_x(RdlPose(hinge_deg=0.0, knee_flex_deg=0.0).points(), tilt_deg)
        ahead = _tilted_about_x(RdlPose(bar_forward_m=BAR_AHEAD_M).points(), tilt_deg)
        fields = dict(capture_mode=TRIANGULATED, up=up, gravity_source=gravity_source)
        assert build_hinge_sample(_ctx(standing, **fields), readiness, False).trunk_pitch_signed_deg == pytest.approx(
            0.0, abs=PITCH_TOLERANCE_DEG,
        )
        assert build_hinge_sample(_ctx(ahead, **fields), readiness, False).bar_offset_ratio > 0.0

    def test_leaning_forward_reads_positive_trunk_pitch(self):
        sample = build_hinge_sample(_ctx(CLEAN_BOTTOM.points()), _config().readiness_gate, in_rep=True)
        assert sample.trunk_pitch_signed_deg == pytest.approx(CLEAN_BOTTOM.hinge_deg, abs=PITCH_TOLERANCE_DEG)

    def test_a_certified_mount_is_levelled_by_its_measured_pitch(self):
        readiness = _config().readiness_gate
        world = CLEAN_BOTTOM.points()
        seen = camera_view(world, MOUNT_PITCH_DEG)
        certified = dict(pitch_certified=True, mount_source=MOUNT_INCLINOMETER, mount_pitch_deg=MOUNT_PITCH_DEG)
        levelled = build_hinge_sample(_ctx(seen, **certified), readiness, True)
        level = build_hinge_sample(_ctx(world), readiness, True)
        assert levelled.hip_cm == pytest.approx(level.hip_cm, abs=1e-6)
        assert levelled.shoulder_cm == pytest.approx(level.shoulder_cm, abs=1e-6)

    def test_an_uncertified_camera_never_uses_the_pitch_corrected_up(self):
        readiness = _config().readiness_gate
        points = CLEAN_BOTTOM.points()
        tilted_up = _tilted_about_x(WORLD_UP[np.newaxis, :], MOUNT_PITCH_DEG)[0]
        corrected = build_hinge_sample(
            _ctx(points, up=tilted_up, gravity_source=GRAVITY_CAMERA_PITCH), readiness, True,
        )
        assert corrected.hip_cm == pytest.approx(build_hinge_sample(_ctx(points), readiness, True).hip_cm, abs=1e-9)


class TestRepSignal:
    def test_signal_rises_through_the_descent(self):
        analyzer = _analyzer()
        signals = []
        for index, pose in enumerate(cosine_move(STANDING, CLEAN_BOTTOM, DESCENT_S, 12.0)):
            analyzer.observe(_ctx(pose.points(), index / 12.0))
            signals.append(analyzer.rep_signal)
        assert signals == sorted(signals)
        assert signals[-1] - signals[0] > MIN_SHOULDER_DROP_CM

    def test_signal_is_nan_without_a_shoulder(self):
        analyzer = _analyzer()
        confidences = np.full(NUM_KEYPOINTS, CONFIDENCE)
        confidences[CK.LEFT_SHOULDER] = 0.0
        analyzer.observe(_ctx(STANDING.points(), confidences=confidences))
        assert math.isnan(analyzer.rep_signal)


class TestRepFeatures:
    @pytest.mark.parametrize("fps", FPS_RATES)
    def test_clean_rep_features(self, fps: float):
        config = _config()
        features = _Driver(_analyzer(), fps).rep()

        assert features.exercise == EXERCISE_NAME
        assert features.rep_number == 1
        assert features.hinge_hip_source == HIP_SOURCE_MEASURED
        assert features.hip_sink_ratio == pytest.approx(hip_sink_pct(20.0, 5.0) / 100.0, abs=RATIO_TOLERANCE)
        assert features.bar_travel_ratio == pytest.approx(bar_travel(75.0, 20.0, 5.0), abs=RATIO_TOLERANCE)
        assert features.hinge_deg == pytest.approx(CLEAN_BOTTOM.hinge_deg, abs=HINGE_TOLERANCE_DEG)
        assert features.max_depth_angle == features.hinge_deg
        assert features.range_label == "at the knee"
        assert features.bar_tilt_deg == pytest.approx(0.0, abs=TILT_TOLERANCE_DEG)
        assert features.lockout_pitch_deg == pytest.approx(0.0, abs=PITCH_TOLERANCE_DEG)
        assert not features.started_at_turnaround
        rdl = config.faults.rdl
        assert features.hip_sink_bands == (rdl.hip_sink.mild, rdl.hip_sink.moderate, rdl.hip_sink.severe)
        assert features.count_floor_travel == rdl.rom.count_floor_travel
        assert features.partial_bands == (rdl.partial.moderate, rdl.partial.severe)
        assert features.eccentric_time_s == features.descent_time_s

    @pytest.mark.parametrize("fps", FPS_RATES)
    def test_cosine_descent_reads_its_5_to_95_time(self, fps: float):
        features = _Driver(_analyzer(), fps).rep(down_s=DESCENT_S)
        assert features.descent_time_s == pytest.approx(cosine_crossing_time_s(DESCENT_S), abs=CROSSING_TOLERANCE_S)
        assert features.descent_time_s == pytest.approx(1.43, abs=CROSSING_TOLERANCE_S)

    def test_a_late_rep_start_still_reads_the_whole_descent_from_the_pre_roll(self):
        features = _Driver(_analyzer(), 30.0).rep(lag_s=0.4)
        assert features.descent_time_s == pytest.approx(cosine_crossing_time_s(DESCENT_S), abs=CROSSING_TOLERANCE_S)

    def test_crossings_are_downward_and_inside_the_window(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        # An upward pass before i_top is never read.
        smoothed = [90.0, 100.0, 95.0, 92.0, 90.0, 89.0]
        start, end = descent_crossings(times, smoothed, i_top=1, i_low=5, top_cm=100.0, drop_cm=10.0)
        assert start == pytest.approx(1.1)
        assert end == pytest.approx(3.75)

    def test_one_camera_needs_visible_hips_at_the_bottom(self):
        confidences = np.full(NUM_KEYPOINTS, CONFIDENCE)
        confidences[[CK.LEFT_HIP, CK.RIGHT_HIP]] = LOW_HIP_CONFIDENCE
        features = _Driver(_analyzer(), 12.0).rep(bottom=RdlPose(70.0, 60.0, 15.0), rep_confidences=confidences)
        assert features.hinge_hip_source == HIP_SOURCE_SETUP
        assert math.isnan(features.hip_sink_ratio)
        assert math.isfinite(features.hinge_deg)

    def test_rig_keeps_two_view_hips_at_the_confidence_cap(self):
        confidences = np.full(NUM_KEYPOINTS, CONFIDENCE)
        confidences[[CK.LEFT_HIP, CK.RIGHT_HIP]] = LOW_HIP_CONFIDENCE
        features = _Driver(_analyzer(), 30.0, TRIANGULATED).rep(
            bottom=RdlPose(70.0, 60.0, 15.0), rep_confidences=confidences,
        )
        assert features.hinge_hip_source == HIP_SOURCE_MEASURED
        assert features.hip_sink_ratio == pytest.approx(hip_sink_pct(60.0, 15.0) / 100.0, abs=RATIO_TOLERANCE)

    @pytest.mark.parametrize("drop_deg, side", [(HAND_DROP_DEG, "right"), (-HAND_DROP_DEG, "left")])
    def test_a_dropped_hand_reads_its_tilt_and_side(self, drop_deg: float, side: str):
        driver = _Driver(_analyzer(), 30.0)
        driver.view_fn = dropping_hand(drop_deg)
        features = driver.rep()
        assert features.bar_tilt_deg == pytest.approx(HAND_DROP_DEG, abs=TILT_TOLERANCE_DEG)
        assert features.bar_tilt_side == side

    @pytest.mark.parametrize("fps", FPS_RATES)
    def test_a_gradual_drop_reads_in_full_against_the_standing_pre_roll(self, fps: float):
        driver = _Driver(_analyzer(), fps)
        driver.view_fn = dropping_hand(GRADUAL_DROP_DEG)
        assert driver.rep().bar_tilt_deg == pytest.approx(GRADUAL_DROP_DEG, abs=TILT_TOLERANCE_DEG)

    def test_a_grip_asymmetry_present_from_the_start_cancels(self):
        driver = _Driver(_analyzer(), 30.0)
        driver.view_fn = lambda points: dropping_hand(HAND_DROP_DEG)(points, constant=True)
        assert driver.rep().bar_tilt_deg == pytest.approx(0.0, abs=TILT_TOLERANCE_DEG)

    def test_a_yawed_lifter_reads_no_tilt(self):
        driver = _Driver(_analyzer(), 30.0)
        driver.view_fn = lambda points: _yawed(points, YAW_DEG)
        features = driver.rep()
        assert features.bar_tilt_deg == pytest.approx(0.0, abs=TILT_TOLERANCE_DEG)

    def test_collapsed_hand_separation_skips_tilt(self):
        driver = _Driver(_analyzer(), 30.0)
        driver.view_fn = lambda points: _collapse_hands_at_depth(dropping_hand(HAND_DROP_DEG)(points))
        features = driver.rep()
        assert math.isnan(features.bar_tilt_deg)
        assert features.bar_tilt_side is None

    def test_a_bar_off_the_legs_raises_the_offset(self):
        on_legs = _Driver(_analyzer(), 30.0, TRIANGULATED).rep()
        off = _Driver(_analyzer(), 30.0, TRIANGULATED).rep(bottom=RdlPose(75.0, 20.0, 5.0, bar_forward_m=BAR_AHEAD_M))
        assert on_legs.bar_offset_ratio_p80 == pytest.approx(bar_offset_ratio(CLEAN_BOTTOM.points()), abs=OFFSET_TOLERANCE)
        assert off.bar_offset_ratio_p80 > _config().faults.rdl.bar_drift.moderate


def _yawed(points: np.ndarray, degrees: float) -> np.ndarray:
    yaw = math.radians(degrees)
    rotation = np.array([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]])
    return points @ rotation.T


def _collapse_hands_at_depth(points: np.ndarray) -> np.ndarray:
    # Below mid-shin height the right wrist is mis-tracked onto the left.
    collapsed = points.copy()
    if points[CK.LEFT_WRIST][1] > points[CK.LEFT_KNEE][1] - 0.15:
        collapsed[CK.RIGHT_WRIST, 0] = points[CK.LEFT_WRIST, 0] - 0.05
    return collapsed


class TestSetups:
    def test_setups_survive_reset_set_and_a_new_analyser_starts_empty(self):
        analyzer = _analyzer()
        _Driver(analyzer, 30.0).rep()
        assert analyzer.setup_count == 1
        analyzer.reset_set()
        assert analyzer.setup_count == 1
        assert _analyzer().setup_count == 0

    def test_the_setup_is_the_idle_frames_near_the_highest_hip(self):
        analyzer = _analyzer()
        driver = _Driver(analyzer, 30.0)
        # A shuffle dips the hips before the lifter stands tall.
        driver.feed([RdlPose(10.0, 30.0, 10.0)] * 20)
        features = driver.rep()
        assert features.hip_sink_ratio == pytest.approx(hip_sink_pct(20.0, 5.0) / 100.0, abs=RATIO_TOLERANCE)

    def test_standing_back_up_between_reps_adds_a_setup(self):
        analyzer = _analyzer()
        driver = _Driver(analyzer, 30.0)
        driver.rep()
        driver.rep()
        assert analyzer.setup_count == 2


class TestTurnaroundStarts:
    def test_a_piston_start_reads_its_descent_from_the_previous_reps_tail(self):
        analyzer = _analyzer()
        driver = _Driver(analyzer, 30.0, TRIANGULATED)
        turnaround = pose_at_drop(STANDING, CLEAN_BOTTOM, TURNAROUND_DROP)
        first = driver.rep(finish_at=turnaround)
        piston = driver.rep(top=turnaround, stand_s=0.0, down_s=TURNAROUND_DESCENT_S, piston=True)

        assert analyzer.setup_count == 1
        assert piston.started_at_turnaround
        assert math.isnan(piston.lockout_pitch_deg)
        assert piston.descent_time_s == pytest.approx(
            cosine_crossing_time_s(TURNAROUND_DESCENT_S), abs=CROSSING_TOLERANCE_S,
        )
        # References still come from the real standing setup.
        assert piston.hip_sink_ratio == pytest.approx(first.hip_sink_ratio, abs=RATIO_TOLERANCE)
        assert piston.hinge_deg == pytest.approx(first.hinge_deg, abs=HINGE_TOLERANCE_DEG)

    def test_a_smooth_turnaround_the_counter_ended_at_the_top_is_a_turnaround(self):
        analyzer = _analyzer()
        driver = _Driver(analyzer, 30.0, TRIANGULATED)
        turnaround = pose_at_drop(STANDING, CLEAN_BOTTOM, TURNAROUND_DROP)
        driver.rep(finish_at=turnaround)
        # Two idle frames settled at the turnaround, then the counter starts a rep.
        second = driver.rep(top=turnaround, stand_s=2.0 / 30.0, down_s=TURNAROUND_DESCENT_S, lag_s=0.0)

        assert analyzer.setup_count == 1
        assert second.started_at_turnaround
        assert math.isnan(second.lockout_pitch_deg)
        assert second.descent_time_s == pytest.approx(
            cosine_crossing_time_s(TURNAROUND_DESCENT_S), abs=CROSSING_TOLERANCE_S,
        )


class TestRejection:
    def test_counted_partial_and_ignored_reps(self):
        analyzer = _analyzer()
        assert analyzer.rejection_reason(rdl_features(bar_travel_ratio=0.35)) is None
        assert analyzer.rejection_reason(rdl_features(bar_travel_ratio=0.29, duration_s=1.5)) == REJECT_PARTIAL
        assert analyzer.rejection_reason(rdl_features(bar_travel_ratio=0.29, duration_s=0.8)) == REJECT_IGNORE
        assert analyzer.rejection_reason(rdl_features(bar_travel_ratio=0.14, duration_s=1.5)) == REJECT_IGNORE

    def test_unseen_wrists_fall_back_to_the_hinge_floor(self):
        analyzer = _analyzer()
        assert analyzer.rejection_reason(rdl_features(hinge_deg=55.0)) is None
        assert analyzer.rejection_reason(rdl_features(hinge_deg=45.0)) == REJECT_IGNORE
        assert analyzer.rejection_reason(rdl_features()) == REJECT_IGNORE

    def test_rejection_reason_is_pure(self):
        analyzer = _analyzer()
        partial = rdl_features(bar_travel_ratio=0.29, duration_s=1.5)
        assert [analyzer.rejection_reason(partial) for _ in range(3)] == [REJECT_PARTIAL] * 3

    def test_a_rejected_rep_keeps_the_setups(self):
        analyzer = _analyzer()
        _Driver(analyzer, 30.0).rep()
        analyzer.discard_rep()
        assert analyzer.setup_count == 1


class TestReadiness:
    def test_hands_below_the_hips_pass(self):
        assert _analyzer().is_ready(_ctx(STANDING.points(), with_angles=False))

    def test_hands_at_the_hips_or_unseen_fail(self):
        analyzer = _analyzer()
        raised = STANDING.points()
        raised[[CK.LEFT_WRIST, CK.RIGHT_WRIST], 1] = raised[CK.LEFT_HIP, 1]
        assert not analyzer.is_ready(_ctx(raised, with_angles=False))
        confidences = np.full(NUM_KEYPOINTS, CONFIDENCE)
        confidences[CK.RIGHT_WRIST] = 0.2
        assert not analyzer.is_ready(_ctx(STANDING.points(), confidences=confidences, with_angles=False))

    def test_readiness_reads_no_analyser_state(self):
        analyzer = _analyzer()
        ctx = _ctx(STANDING.points(), with_angles=False)
        before = analyzer.is_ready(ctx)
        _Driver(analyzer, 30.0).rep()
        assert analyzer.is_ready(ctx) == before


def _standing_set(analyzer: RdlRepAnalyzer, pitch_deg: float, frames: int, **fields) -> list[float | None]:
    deltas = []
    for index in range(frames):
        points = camera_view(STANDING.points(), pitch_deg)
        analyzer.observe(_ctx(points, index / 30.0, **fields))
        deltas.append(analyzer.pitch_tripwire_delta())
    analyzer.reset_set()
    return deltas


class TestPitchTripwire:
    CERTIFIED = dict(pitch_certified=True, mount_source=MOUNT_INCLINOMETER, mount_pitch_deg=MOUNT_PITCH_DEG)

    def test_a_knocked_camera_trips_once_the_set_average_is_in(self):
        analyzer = _analyzer()
        config = _config().faults.rdl.camera
        for _ in range(config.reference_sets):
            assert not any(_standing_set(analyzer, MOUNT_PITCH_DEG, TRIPWIRE_STANDING_FRAMES, **self.CERTIFIED))
        deltas = _standing_set(analyzer, KNOCKED_PITCH_DEG, TRIPWIRE_STANDING_FRAMES, **self.CERTIFIED)
        tripped = [index for index, delta in enumerate(deltas) if delta is not None]
        assert tripped == [config.delta_min_frames - 1]
        assert deltas[tripped[0]] == pytest.approx(KNOCKED_PITCH_DEG - MOUNT_PITCH_DEG, abs=TRIP_DELTA_TOLERANCE_DEG)

    def test_an_unmoved_camera_never_trips(self):
        analyzer = _analyzer()
        for _ in range(6):
            assert not any(_standing_set(analyzer, MOUNT_PITCH_DEG, TRIPWIRE_STANDING_FRAMES, **self.CERTIFIED))

    def test_sets_with_too_few_standing_frames_build_no_reference(self):
        analyzer = _analyzer()
        for _ in range(4):
            _standing_set(analyzer, MOUNT_PITCH_DEG, TOO_FEW_STANDING_FRAMES, **self.CERTIFIED)
        assert not any(_standing_set(analyzer, KNOCKED_PITCH_DEG, TRIPWIRE_STANDING_FRAMES, **self.CERTIFIED))

    def test_an_uncertified_camera_has_no_delta_tripwire(self):
        analyzer = _analyzer()
        for _ in range(3):
            _standing_set(analyzer, MOUNT_PITCH_DEG, TRIPWIRE_STANDING_FRAMES)
        assert not any(_standing_set(analyzer, KNOCKED_PITCH_DEG, TRIPWIRE_STANDING_FRAMES))

    def test_an_uncertified_camera_far_off_level_marks_the_session_untrusted(self):
        analyzer = _analyzer()
        features = _Driver(analyzer, 30.0, camera_pitch_deg=GROSS_PITCH_DEG).rep()
        assert features.camera_pitch_untrusted
        level = _Driver(_analyzer(), 30.0, camera_pitch_deg=MOUNT_PITCH_DEG).rep()
        assert not level.camera_pitch_untrusted
