"""Tests for per-rep quality scoring."""

import math

import pytest

from biomechanics.diagnosis.rep_scoring import (
    DEPTH_DECAY_RATIO,
    DEPTH_TARGET_TOLERANCE_RATIO,
    LOADED_WINDOW_RATIO,
    WEIGHT_DEPTH,
    WEIGHT_KNEES,
    WEIGHT_TRUNK,
    score_depth,
    score_knee_tracking,
    score_rep,
    score_set,
    score_symmetry,
    score_tempo,
    score_trunk_control,
)
from biomechanics.diagnosis.types import (
    RepKinematicSummary,
    RepTrajectory,
    RepTrajectorySample,
)

# Femur length used across the fixtures, in the units each layer expects.
FEMUR_M = 0.40
FEMUR_CM = FEMUR_M * 100.0
# The trunk pitch the default rep's balance model asks for (diagnosis.lean_model).
BALANCED_PITCH_DEG = 35.0
SCORE_TOLERANCE = 0.01


def _make_rep(**overrides) -> RepKinematicSummary:
    defaults = dict(
        rep_number=1,
        trunk_pitch_at_bottom=35.0,
        knee_valgus_l=0.0,
        knee_valgus_r=0.0,
        ankle_df_l_max=20.0,
        ankle_df_r_max=20.0,
        hip_y_l_at_bottom=45.0,
        hip_y_r_at_bottom=45.0,
        knee_y_l_at_bottom=45.0,
        knee_y_r_at_bottom=45.0,
        stance_width_ratio=1.0,
        foot_direction_angle_l=20.0,
        foot_direction_angle_r=20.0,
        depth_class_int=4,
        descent_time_s=2.0,
        ascent_time_s=1.0,
        expected_pitch_reference=BALANCED_PITCH_DEG,
        expected_pitch_athlete=BALANCED_PITCH_DEG,
        expected_pitch_with_ankles=BALANCED_PITCH_DEG,
    )
    defaults.update(overrides)
    return RepKinematicSummary(**defaults)


def _make_sample(**overrides) -> RepTrajectorySample:
    defaults = dict(
        trunk_pitch=30.0,
        knee_valgus_l=0.0,
        knee_valgus_r=0.0,
        hip_y_l=45.0,
        hip_y_r=45.0,
        knee_y_l=45.0,
        knee_y_r=45.0,
    )
    defaults.update(overrides)
    return RepTrajectorySample(**defaults)


def _descent_trajectory(
    bottom_offset_cm: float = 0.0, **bottom_overrides
) -> RepTrajectory:
    """A rep descending from standing to a bottom `bottom_offset_cm` above the knee.

    Only the bottom frames carry the fault overrides, so tests exercise the
    loaded-window filter rather than assuming the whole rep is scored.
    """
    standing = [
        _make_sample(hip_y_l=45.0 + FEMUR_CM, hip_y_r=45.0 + FEMUR_CM, trunk_pitch=0.0)
        for _ in range(10)
    ]
    bottom = [
        _make_sample(
            hip_y_l=45.0 + bottom_offset_cm,
            hip_y_r=45.0 + bottom_offset_cm,
            **bottom_overrides,
        )
        for _ in range(10)
    ]
    return RepTrajectory(samples=standing + bottom)


def _default_anthro() -> dict:
    return {
        "femur_torso_ratio": 1.0,
        "hip_width": 0.30,
        "shoulder_width": 0.40,
        "femur_length_avg": FEMUR_M,
    }


def _default_rom() -> dict:
    return {
        "peak_dorsiflexion": 35.0,
        "avg_depth": 120.0,
    }


class TestDepthScore:
    def test_parallel_scores_one(self):
        # hip joint centre level with knee joint centre → parallel → 1.0
        rep = _make_rep(
            hip_y_l_at_bottom=45.0, hip_y_r_at_bottom=45.0,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
        )
        assert score_depth(rep, _default_anthro(), _default_rom()) == 1.0

    def test_hip_below_knee_still_capped(self):
        rep = _make_rep(
            hip_y_l_at_bottom=35.0, hip_y_r_at_bottom=35.0,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
        )
        assert score_depth(rep, _default_anthro(), _default_rom()) == 1.0

    def test_half_femur_above_parallel(self):
        # hip 20cm above knee on a 40cm femur → ratio 0.5, which is 0.42
        # past the parallel tolerance → 1 - 0.42/0.75
        rep = _make_rep(
            hip_y_l_at_bottom=65.0, hip_y_r_at_bottom=65.0,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
        )
        score = score_depth(rep, _default_anthro(), _default_rom())
        expected = 1.0 - (0.5 - DEPTH_TARGET_TOLERANCE_RATIO) / DEPTH_DECAY_RATIO
        assert score == pytest.approx(expected, abs=SCORE_TOLERANCE)

    def test_just_above_parallel_within_tolerance_scores_one(self):
        hip_cm = 45.0 + DEPTH_TARGET_TOLERANCE_RATIO * FEMUR_CM
        rep = _make_rep(hip_y_l_at_bottom=hip_cm, hip_y_r_at_bottom=hip_cm)
        assert score_depth(rep, _default_anthro(), _default_rom()) == pytest.approx(1.0)

    def test_depth_is_scored_against_the_athletes_target(self):
        # An athlete whose calibrated target sits 0.2 femurs above parallel
        # is not penalised for reaching it; the default target is parallel.
        rep = _make_rep(depth_ratio=0.25)
        own_target = {**_default_rom(), "depth_target_ratio": 0.2}
        assert score_depth(rep, _default_anthro(), own_target) == pytest.approx(1.0)
        expected = 1.0 - (0.25 - DEPTH_TARGET_TOLERANCE_RATIO) / DEPTH_DECAY_RATIO
        assert score_depth(rep, _default_anthro(), _default_rom()) == pytest.approx(
            expected, abs=SCORE_TOLERANCE
        )

    def test_measured_depth_ratio_preferred_over_bottom_heights(self):
        # The whole-rep depth_ratio is what the fault rules judged; the
        # single bottom frame's heights (here: standing) must not override it.
        rep = _make_rep(
            depth_ratio=-0.1,
            hip_y_l_at_bottom=45.0 + FEMUR_CM, hip_y_r_at_bottom=45.0 + FEMUR_CM,
        )
        assert score_depth(rep, _default_anthro(), _default_rom()) == pytest.approx(1.0)

    def test_sample_depth_ratio_used_directly(self):
        # Samples carrying a measured depth ratio are scored on it, not on
        # their (here: parallel-looking) heights.
        samples = [_make_sample(depth_ratio=0.5) for _ in range(10)]
        trajectory = RepTrajectory(samples=samples)
        score = score_depth(_make_rep(), _default_anthro(), _default_rom(), trajectory)
        expected = 1.0 - (0.5 - DEPTH_TARGET_TOLERANCE_RATIO) / DEPTH_DECAY_RATIO
        assert score == pytest.approx(expected, abs=SCORE_TOLERANCE)

    def test_standing_scores_zero(self):
        # hip a full femur above the knee → thigh vertical → 0.0
        rep = _make_rep(
            hip_y_l_at_bottom=45.0 + FEMUR_CM, hip_y_r_at_bottom=45.0 + FEMUR_CM,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
        )
        assert score_depth(rep, _default_anthro(), _default_rom()) == 0.0

    def test_missing_standing_frame_does_not_zero_depth(self):
        # The at_top fields are unused now, so a rep with no standing frame
        # still scores on its own merits rather than being capped.
        rep = _make_rep(
            hip_y_l_at_top=0.0, hip_y_r_at_top=0.0,
            knee_y_l_at_top=0.0, knee_y_r_at_top=0.0,
        )
        assert score_depth(rep, _default_anthro(), _default_rom()) == 1.0

    def test_translation_invariant(self):
        # Shifting every keypoint by a constant must not change the score.
        rep = _make_rep(
            hip_y_l_at_bottom=55.0, hip_y_r_at_bottom=55.0,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
        )
        shifted = _make_rep(
            hip_y_l_at_bottom=-45.0, hip_y_r_at_bottom=-45.0,
            knee_y_l_at_bottom=-55.0, knee_y_r_at_bottom=-55.0,
        )
        assert score_depth(rep, _default_anthro(), _default_rom()) == pytest.approx(
            score_depth(shifted, _default_anthro(), _default_rom())
        )

    def test_trajectory_uses_deepest_frames(self):
        trajectory = _descent_trajectory(bottom_offset_cm=0.0)
        rep = _make_rep(
            hip_y_l_at_bottom=45.0 + FEMUR_CM, hip_y_r_at_bottom=45.0 + FEMUR_CM,
        )
        # The summary alone says "standing"; the trajectory says "parallel".
        assert score_depth(rep, _default_anthro(), _default_rom()) == 0.0
        assert score_depth(
            rep, _default_anthro(), _default_rom(), trajectory
        ) == pytest.approx(1.0, abs=0.02)

    def test_single_deep_frame_does_not_fake_depth(self):
        # 19 quarter-squat frames plus one spike to parallel should not score
        # as a parallel rep.
        shallow = [_make_sample(hip_y_l=75.0, hip_y_r=75.0) for _ in range(19)]
        spike = [_make_sample(hip_y_l=45.0, hip_y_r=45.0)]
        trajectory = RepTrajectory(samples=shallow + spike)
        score = score_depth(_make_rep(), _default_anthro(), _default_rom(), trajectory)
        assert score < 0.4


class TestTrunkControlScore:
    """Lean is judged against the pitch the athlete's build and ankles need to
    keep the shoulders over midfoot, and only leaning past it costs score."""

    def test_balanced_lean_scores_one(self):
        rep = _make_rep(trunk_pitch_at_bottom=BALANCED_PITCH_DEG)
        assert score_trunk_control(rep, _default_anthro(), _default_rom()) == 1.0

    def test_within_tolerance(self):
        rep = _make_rep(trunk_pitch_at_bottom=BALANCED_PITCH_DEG + 3.0)
        assert score_trunk_control(rep, _default_anthro(), _default_rom()) == 1.0

    def test_moderate_excess(self):
        # 13° past balance → 1-(13-3)/20 = 0.5
        rep = _make_rep(trunk_pitch_at_bottom=BALANCED_PITCH_DEG + 13.0)
        score = score_trunk_control(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.5, abs=SCORE_TOLERANCE)

    def test_extreme_excess(self):
        rep = _make_rep(trunk_pitch_at_bottom=60.0)
        assert score_trunk_control(rep, _default_anthro(), _default_rom()) == 0.0

    def test_more_upright_than_needed_is_not_penalised(self):
        # Regression: the trunk score was two-sided and docked a rep for
        # sitting more upright than expected, which is never a fault.
        rep = _make_rep(trunk_pitch_at_bottom=BALANCED_PITCH_DEG - 20.0)
        assert score_trunk_control(rep, _default_anthro(), _default_rom()) == 1.0

    def test_ankle_adjusted_expectation_is_preferred(self):
        # Stiff ankles need 45°; a 48° rep is within tolerance of that even
        # though it leans 13° past the free-ankle expectation.
        rep = _make_rep(
            trunk_pitch_at_bottom=48.0,
            expected_pitch_with_ankles=45.0,
            expected_pitch_athlete=BALANCED_PITCH_DEG,
            expected_pitch_reference=BALANCED_PITCH_DEG,
        )
        assert score_trunk_control(rep, _default_anthro(), _default_rom()) == 1.0

    def test_falls_back_to_athlete_then_reference_expectation(self):
        athlete_only = _make_rep(
            trunk_pitch_at_bottom=BALANCED_PITCH_DEG + 13.0,
            expected_pitch_with_ankles=math.nan,
        )
        reference_only = _make_rep(
            trunk_pitch_at_bottom=BALANCED_PITCH_DEG + 13.0,
            expected_pitch_with_ankles=math.nan,
            expected_pitch_athlete=math.nan,
        )
        for rep in (athlete_only, reference_only):
            score = score_trunk_control(rep, _default_anthro(), _default_rom())
            assert score == pytest.approx(0.5, abs=SCORE_TOLERANCE)

    def test_unknown_expectation_is_not_scored(self):
        rep = _make_rep(
            trunk_pitch_at_bottom=60.0,
            expected_pitch_with_ankles=math.nan,
            expected_pitch_athlete=math.nan,
            expected_pitch_reference=math.nan,
        )
        assert math.isnan(score_trunk_control(rep, _default_anthro(), _default_rom()))

    def test_hip_shoot_costs_trunk_score(self):
        # Chest dropping 11° out of the hole → 1-(11-4)/14 = 0.5
        rep = _make_rep(hip_shoot_deg=11.0)
        score = score_trunk_control(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.5, abs=SCORE_TOLERANCE)

    def test_small_hip_shoot_is_tolerated(self):
        rep = _make_rep(hip_shoot_deg=4.0)
        assert score_trunk_control(rep, _default_anthro(), _default_rom()) == 1.0

    def test_lean_outside_the_loaded_window_is_excluded(self):
        # The athlete hinges over at the top (resetting their grip); only the
        # bottom of the rep is judged against the balance model.
        standing = [
            _make_sample(hip_y_l=45.0 + FEMUR_CM, hip_y_r=45.0 + FEMUR_CM, trunk_pitch=70.0)
            for _ in range(10)
        ]
        bottom = [_make_sample(trunk_pitch=BALANCED_PITCH_DEG) for _ in range(10)]
        trajectory = RepTrajectory(samples=standing + bottom)
        score = score_trunk_control(
            _make_rep(), _default_anthro(), _default_rom(), trajectory
        )
        assert score == 1.0

    def test_lean_during_bottom_is_caught(self):
        trajectory = _descent_trajectory(trunk_pitch=BALANCED_PITCH_DEG + 13.0)
        score = score_trunk_control(
            _make_rep(), _default_anthro(), _default_rom(), trajectory
        )
        assert score == pytest.approx(0.5, abs=SCORE_TOLERANCE)


class TestKneeTrackingScore:
    def test_perfect_alignment(self):
        rep = _make_rep(knee_valgus_l=0.0, knee_valgus_r=0.0)
        assert score_knee_tracking(rep, _default_anthro(), _default_rom()) == 1.0

    def test_both_within_zone(self):
        rep = _make_rep(knee_valgus_l=3.5, knee_valgus_r=-2.0)
        assert score_knee_tracking(rep, _default_anthro(), _default_rom()) == 1.0

    def test_one_bad_other_perfect(self):
        rep = _make_rep(knee_valgus_l=10.0, knee_valgus_r=0.0)
        score = score_knee_tracking(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.75, abs=0.01)

    def test_knees_out_is_not_scored_like_valgus(self):
        # Regression: abs() penalised knees pushed out — the very correction
        # the knee cue prescribes — exactly as much as knees caving in.
        rep_valgus = _make_rep(knee_valgus_l=8.0, knee_valgus_r=0.0)
        rep_knees_out = _make_rep(knee_valgus_l=-8.0, knee_valgus_r=0.0)
        assert score_knee_tracking(rep_valgus, _default_anthro(), _default_rom()) < 1.0
        assert score_knee_tracking(rep_knees_out, _default_anthro(), _default_rom()) == 1.0

    def test_knees_far_out_is_penalised(self):
        # 21° out is 6° past the 15° lateral tolerance → that knee 0.5
        rep = _make_rep(knee_valgus_l=-21.0, knee_valgus_r=0.0)
        score = score_knee_tracking(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.75, abs=SCORE_TOLERANCE)

    def test_both_extreme(self):
        rep = _make_rep(knee_valgus_l=16.0, knee_valgus_r=16.0)
        assert score_knee_tracking(rep, _default_anthro(), _default_rom()) == 0.0

    def test_sustained_valgus_in_the_hole_is_caught(self):
        trajectory = _descent_trajectory(knee_valgus_l=10.0, knee_valgus_r=10.0)
        score = score_knee_tracking(
            _make_rep(), _default_anthro(), _default_rom(), trajectory
        )
        assert score == pytest.approx(0.5, abs=0.01)

    def test_single_spiked_frame_is_rejected(self):
        # One mistracked frame out of twenty must not sink the rep.
        clean = [_make_sample() for _ in range(19)]
        spike = [_make_sample(knee_valgus_l=40.0, knee_valgus_r=40.0)]
        trajectory = RepTrajectory(samples=clean + spike)
        score = score_knee_tracking(
            _make_rep(), _default_anthro(), _default_rom(), trajectory
        )
        assert score == 1.0


class TestSymmetryScore:
    def test_perfect_symmetry(self):
        rep = _make_rep(hip_y_l_at_bottom=45.0, hip_y_r_at_bottom=45.0)
        assert score_symmetry(rep, _default_anthro(), _default_rom()) == 1.0

    def test_moderate_asymmetry(self):
        rep = _make_rep(hip_y_l_at_bottom=45.0, hip_y_r_at_bottom=48.5)
        score = score_symmetry(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.5, abs=0.01)

    def test_extreme_asymmetry(self):
        rep = _make_rep(hip_y_l_at_bottom=45.0, hip_y_r_at_bottom=51.0)
        assert score_symmetry(rep, _default_anthro(), _default_rom()) == 0.0

    def test_hip_shift_within_tolerance(self):
        rep = _make_rep(hip_shift_ratio=0.03)
        assert score_symmetry(rep, _default_anthro(), _default_rom()) == 1.0

    def test_hip_shift_moderate(self):
        # 0.105 is 0.075 past the 0.03 tolerance → 1 - 0.075/0.15 = 0.5
        rep = _make_rep(hip_shift_ratio=0.105)
        score = score_symmetry(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.5, abs=SCORE_TOLERANCE)

    def test_hip_shift_direction_does_not_matter(self):
        left = score_symmetry(_make_rep(hip_shift_ratio=-0.105), _default_anthro(), _default_rom())
        right = score_symmetry(_make_rep(hip_shift_ratio=0.105), _default_anthro(), _default_rom())
        assert left == pytest.approx(right)

    def test_hip_shift_preferred_over_pelvic_level(self):
        # A measured shift is what coaches watch; a tilted pelvis at the
        # bottom frame only counts when the shift was not measured.
        rep = _make_rep(hip_y_l_at_bottom=45.0, hip_y_r_at_bottom=51.0, hip_shift_ratio=0.0)
        assert score_symmetry(rep, _default_anthro(), _default_rom()) == 1.0

    def test_sustained_hip_drop_is_caught(self):
        samples = [
            _make_sample(hip_y_l=45.0, hip_y_r=48.5) for _ in range(10)
        ]
        trajectory = RepTrajectory(samples=samples)
        score = score_symmetry(
            _make_rep(), _default_anthro(), _default_rom(), trajectory
        )
        assert score == pytest.approx(0.5, abs=0.01)


class TestTempoScore:
    def test_ideal_tempo(self):
        rep = _make_rep(descent_time_s=2.0, ascent_time_s=1.0)
        assert score_tempo(rep, _default_anthro(), _default_rom()) == 1.0

    def test_dive_bombed_descent(self):
        # 0.25s descent → 0.75s under the 1.0s floor → eccentric 0.5, and the
        # clean ascent must not average that away.
        rep = _make_rep(descent_time_s=0.25, ascent_time_s=1.0)
        score = score_tempo(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.5, abs=0.01)

    def test_hard_but_not_grinding_ascent_is_clean(self):
        # A tough rep that takes 2.75s to stand up is still inside the window.
        rep = _make_rep(descent_time_s=2.0, ascent_time_s=2.75)
        assert score_tempo(rep, _default_anthro(), _default_rom()) == 1.0

    def test_grinding_ascent(self):
        # 3.75s ascent → 0.75s over the 3.0s ceiling → 0.5
        rep = _make_rep(descent_time_s=2.0, ascent_time_s=3.75)
        score = score_tempo(rep, _default_anthro(), _default_rom())
        assert score == pytest.approx(0.5, abs=0.01)

    def test_too_slow_descent_penalised(self):
        rep = _make_rep(descent_time_s=6.0, ascent_time_s=1.0)
        assert score_tempo(rep, _default_anthro(), _default_rom()) == 0.0

    def test_weakest_phase_decides(self):
        rep = _make_rep(descent_time_s=0.1, ascent_time_s=8.0)
        assert score_tempo(rep, _default_anthro(), _default_rom()) == 0.0

    def test_untimed_rep_is_not_scored(self):
        """Neither penalised nor credited: an unmeasured dimension leaves the composite."""
        rep = _make_rep(descent_time_s=0.0, ascent_time_s=0.0)
        assert math.isnan(score_tempo(rep, _default_anthro(), _default_rom()))
        assert math.isfinite(score_rep(rep, _default_anthro(), _default_rom()).composite_score)

    def test_unmeasured_knees_earn_no_credit(self):
        """Unmeasured knees used to score 1.0 and hand every rep a free quarter of the composite."""
        anthro, rom = _default_anthro(), _default_rom()
        poor_depth = dict(depth_ratio=0.6)
        measured = score_rep(_make_rep(knee_valgus_l=0.0, knee_valgus_r=0.0, **poor_depth), anthro, rom)
        unmeasured = score_rep(
            _make_rep(knee_valgus_l=math.nan, knee_valgus_r=math.nan, **poor_depth), anthro, rom,
        )
        assert math.isnan(unmeasured.knee_tracking_score)
        assert unmeasured.composite_score < measured.composite_score


class TestLoadedWindow:
    def test_window_covers_the_bottom_only(self):
        # A frame a full femur above the deepest point is outside the window;
        # one within LOADED_WINDOW_RATIO femurs is inside.
        inside_offset = (LOADED_WINDOW_RATIO * FEMUR_CM) - 1.0
        samples = [
            _make_sample(hip_y_l=45.0, hip_y_r=45.0),
            _make_sample(
                hip_y_l=45.0 + inside_offset, hip_y_r=45.0 + inside_offset,
                knee_valgus_l=10.0, knee_valgus_r=10.0,
            ),
            _make_sample(
                hip_y_l=45.0 + FEMUR_CM, hip_y_r=45.0 + FEMUR_CM,
                knee_valgus_l=40.0, knee_valgus_r=40.0,
            ),
        ]
        trajectory = RepTrajectory(samples=samples)
        score = score_knee_tracking(
            _make_rep(), _default_anthro(), _default_rom(), trajectory
        )
        # The 40° standing frame is excluded; the 10° loaded frame is not.
        assert 0.0 < score < 1.0


class TestCompositeScore:
    def test_perfect_rep(self):
        rep = _make_rep(
            trunk_pitch_at_bottom=30.0,
            knee_valgus_l=0.0, knee_valgus_r=0.0,
            hip_y_l_at_bottom=45.0, hip_y_r_at_bottom=45.0,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
            descent_time_s=2.0, ascent_time_s=1.0,
        )
        result = score_rep(rep, _default_anthro(), _default_rom())
        assert result.composite_score == pytest.approx(1.0, abs=0.01)

    def test_all_scores_bounded(self):
        rep = _make_rep()
        result = score_rep(rep, _default_anthro(), _default_rom())
        assert 0.0 <= result.depth_score <= 1.0
        assert 0.0 <= result.trunk_control_score <= 1.0
        assert 0.0 <= result.knee_tracking_score <= 1.0
        assert 0.0 <= result.symmetry_score <= 1.0
        assert 0.0 <= result.tempo_score <= 1.0
        assert 0.0 <= result.composite_score <= 1.0

    def test_weights_sum_to_one(self):
        from biomechanics.diagnosis.rep_scoring import (
            WEIGHT_DEPTH,
            WEIGHT_KNEES,
            WEIGHT_SYMMETRY,
            WEIGHT_TEMPO,
            WEIGHT_TRUNK,
        )
        total = (
            WEIGHT_DEPTH + WEIGHT_TRUNK + WEIGHT_KNEES + WEIGHT_SYMMETRY + WEIGHT_TEMPO
        )
        assert total == pytest.approx(1.0)

    def test_position_control_outweighs_depth(self):
        # Regression: depth was 42% of the composite, so a deep rep with knee
        # cave outscored a controlled rep a few centimeters short of parallel.
        assert WEIGHT_TRUNK > WEIGHT_DEPTH
        assert WEIGHT_KNEES > WEIGHT_DEPTH
        controlled_short = _make_rep(depth_ratio=0.2)
        deep_with_cave = _make_rep(depth_ratio=-0.2, knee_valgus_l=12.0, knee_valgus_r=12.0)
        controlled_score = score_rep(controlled_short, _default_anthro(), _default_rom())
        caving_score = score_rep(deep_with_cave, _default_anthro(), _default_rom())
        assert controlled_score.composite_score > caving_score.composite_score

    def test_tempo_moves_the_composite(self):
        good = _make_rep(descent_time_s=2.0, ascent_time_s=1.0)
        rushed = _make_rep(descent_time_s=0.2, ascent_time_s=0.2)
        good_score = score_rep(good, _default_anthro(), _default_rom())
        rushed_score = score_rep(rushed, _default_anthro(), _default_rom())
        assert rushed_score.composite_score < good_score.composite_score


class TestSetScoreSummary:
    def test_best_and_worst_identification(self):
        good_rep = _make_rep(
            rep_number=2, trunk_pitch_at_bottom=30.0,
            knee_valgus_l=0.0, knee_valgus_r=0.0,
            hip_y_l_at_bottom=45.0, hip_y_r_at_bottom=45.0,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
        )
        bad_rep = _make_rep(
            rep_number=3, trunk_pitch_at_bottom=55.0,
            knee_valgus_l=14.0, knee_valgus_r=12.0,
            hip_y_l_at_bottom=70.0, hip_y_r_at_bottom=70.0,
            knee_y_l_at_bottom=45.0, knee_y_r_at_bottom=45.0,
        )
        result = score_set([good_rep, bad_rep], _default_anthro(), _default_rom())
        assert result.best_rep_number == 2
        assert result.worst_rep_number == 3

    def test_single_rep_slope_zero(self):
        rep = _make_rep(rep_number=2)
        result = score_set([rep], _default_anthro(), _default_rom())
        assert result.trend_slope == 0.0

    def test_degrading_trend_negative_slope(self):
        reps = [
            _make_rep(rep_number=2, trunk_pitch_at_bottom=30.0),
            _make_rep(rep_number=3, trunk_pitch_at_bottom=35.0,
                      hip_y_l_at_bottom=55.0, hip_y_r_at_bottom=55.0),
            _make_rep(rep_number=4, trunk_pitch_at_bottom=50.0,
                      hip_y_l_at_bottom=70.0, hip_y_r_at_bottom=70.0,
                      knee_valgus_l=12.0),
        ]
        result = score_set(reps, _default_anthro(), _default_rom())
        assert result.trend_slope < 0

    def test_mean_score_computed(self):
        result = score_set(
            [_make_rep(rep_number=2), _make_rep(rep_number=3)],
            _default_anthro(),
            _default_rom(),
        )
        score_a = result.per_rep_scores[0].composite_score
        score_b = result.per_rep_scores[1].composite_score
        assert result.mean_score == pytest.approx((score_a + score_b) / 2.0, abs=0.01)

    def test_trajectories_are_matched_per_rep(self):
        reps = [_make_rep(rep_number=1), _make_rep(rep_number=2)]
        trajectories = [
            _descent_trajectory(knee_valgus_l=14.0, knee_valgus_r=14.0),
            _descent_trajectory(),
        ]
        result = score_set(reps, _default_anthro(), _default_rom(), trajectories)
        assert result.worst_rep_number == 1
        assert result.best_rep_number == 2

    def test_missing_trajectories_fall_back_to_summary(self):
        reps = [_make_rep(rep_number=1), _make_rep(rep_number=2)]
        with_none = score_set(reps, _default_anthro(), _default_rom(), [None, None])
        without = score_set(reps, _default_anthro(), _default_rom())
        assert with_none.mean_score == without.mean_score
