"""Tests for demo_builder pose stack and cue metadata construction."""

from __future__ import annotations

import numpy as np
import pytest

from biomechanics.diagnosis.demo_builder import (
    CORRECTOR_CAUSE_ORDER,
    GENERIC_MAGNITUDE_TEXT,
    _prepare_observed_kpts,
    build_demo_data,
    build_pose_stack,
    order_demo_causes,
    summarize_cue_magnitude,
)
from biomechanics.diagnosis.keypoint_corrector import (
    ANKLE_L,
    ANKLE_R,
    FOOT_L,
    FOOT_R,
    HIP_L,
    HIP_R,
    KNEE_L,
    KNEE_R,
    KeypointCorrector,
)
from biomechanics.diagnosis.types import DiagnosisResult, HypothesizedCause

STACK_TOL = 1e-5


def _make_19pt_skeleton() -> list[list[float]]:
    kpts = np.zeros((19, 3))

    kpts[0] = [0.0, 1.6, 0.0]
    kpts[5] = [0.0, 1.2, -0.18]
    kpts[6] = [0.0, 1.2, 0.18]
    kpts[7] = [0.15, 1.0, -0.20]
    kpts[8] = [0.15, 1.0, 0.20]
    kpts[9] = [0.25, 0.85, -0.20]
    kpts[10] = [0.25, 0.85, 0.20]

    kpts[HIP_L] = [0.0, 0.50, -0.15]
    kpts[HIP_R] = [0.0, 0.50, 0.15]
    kpts[KNEE_L] = [0.10, 0.30, -0.13]
    kpts[KNEE_R] = [0.10, 0.30, 0.13]
    kpts[ANKLE_L] = [0.0, 0.0, -0.15]
    kpts[ANKLE_R] = [0.0, 0.0, 0.15]
    kpts[FOOT_L] = [0.15, 0.0, -0.15]
    kpts[FOOT_R] = [0.15, 0.0, 0.15]

    return kpts.tolist()


def _make_cause(cause_id: str, parameter_delta: dict | None) -> HypothesizedCause:
    return HypothesizedCause(
        cause_id=cause_id,
        tier=1,
        score=0.8,
        evidence_score=0.7,
        prior=0.3,
        implicated_by=["test"],
        parameter_delta=parameter_delta,
        explanation=f"{cause_id} explanation",
    )


def _make_diagnosis(cause_ids: list[str]) -> DiagnosisResult:
    delta_by_id: dict[str, dict] = {
        "narrow_stance": {
            "__foot_target_delta": [0.0, 0.0, -0.04, 0.0, 0.0, 0.04],
            "__width_increase_per_side_m": 0.04,
        },
        "narrow_foot_angle": {"L_ankle.ry": 0.15, "R_ankle.ry": -0.15},
        "knee_track_cue": {"L_hip.ry": -0.08, "R_hip.ry": 0.08},
        "weight_shift_cue": {"pelvis.tx": 0.02},
        "depth_cue_unfamiliar": {"depth_target": "parallel"},
    }
    causes = [_make_cause(cid, delta_by_id.get(cid)) for cid in cause_ids]
    return DiagnosisResult(
        set_id="test",
        detected_symptoms=[],
        immediate_causes=causes,
        session_causes=[],
        longterm_causes=[],
        contextual_notes=[],
        combined_perturbation={},
        confidence=0.8,
    )


def _stance_width(pose: np.ndarray) -> float:
    return float(np.linalg.norm(pose[ANKLE_R] - pose[ANKLE_L]))


class TestOrderDemoCauses:

    def test_orders_by_corrector_application_order(self) -> None:
        diagnosis = _make_diagnosis(["knee_track_cue", "narrow_stance"])

        ordered = order_demo_causes(diagnosis)

        assert [c.cause_id for c in ordered] == ["narrow_stance", "knee_track_cue"]

    def test_excludes_unsupported_cause_ids(self) -> None:
        diagnosis = _make_diagnosis(["narrow_stance"])
        diagnosis.immediate_causes.append(_make_cause("brace_trunk_cue", None))

        ordered = order_demo_causes(diagnosis)

        assert [c.cause_id for c in ordered] == ["narrow_stance"]

    def test_empty_list_when_no_immediate_causes(self) -> None:
        diagnosis = _make_diagnosis([])

        assert order_demo_causes(diagnosis) == []


class TestBuildPoseStack:

    def test_stack_length_is_cause_count_plus_one(self) -> None:
        diagnosis = _make_diagnosis(["narrow_stance", "knee_track_cue"])

        pose_stack = build_pose_stack(_make_19pt_skeleton(), diagnosis)

        assert pose_stack is not None
        assert pose_stack.shape == (3, 19, 3)

    def test_first_pose_equals_observed(self) -> None:
        observed = _make_19pt_skeleton()
        diagnosis = _make_diagnosis(["narrow_stance"])

        pose_stack = build_pose_stack(observed, diagnosis)

        assert pose_stack is not None
        np.testing.assert_allclose(pose_stack[0], np.array(observed), atol=STACK_TOL)

    def test_stance_width_increases_with_stance_cue(self) -> None:
        diagnosis = _make_diagnosis(["narrow_stance"])

        pose_stack = build_pose_stack(_make_19pt_skeleton(), diagnosis)

        assert pose_stack is not None
        assert _stance_width(pose_stack[1]) > _stance_width(pose_stack[0])

    def test_corrections_accumulate_across_prefixes(self) -> None:
        diagnosis = _make_diagnosis(["narrow_stance", "knee_track_cue"])

        pose_stack = build_pose_stack(_make_19pt_skeleton(), diagnosis)

        assert pose_stack is not None
        widened = _stance_width(pose_stack[1])
        assert widened > _stance_width(pose_stack[0])
        assert _stance_width(pose_stack[2]) == pytest.approx(widened, abs=0.02)

    def test_returns_none_when_no_supported_causes(self) -> None:
        diagnosis = _make_diagnosis([])

        assert build_pose_stack(_make_19pt_skeleton(), diagnosis) is None

    def test_final_pose_matches_one_shot_all_causes(self) -> None:
        """The last stack entry must be the same all-causes correction VVS produces."""
        observed = _make_19pt_skeleton()
        diagnosis = _make_diagnosis(
            ["narrow_stance", "narrow_foot_angle", "knee_track_cue"]
        )

        pose_stack = build_pose_stack(observed, diagnosis)
        assert pose_stack is not None

        # VVS calls correct() without precomputed bone lengths; the corrector
        # canonicalizes internally and locks lengths from the canonical pose.
        prepared = _prepare_observed_kpts(observed)
        one_shot = KeypointCorrector().correct(prepared, diagnosis)
        assert one_shot is not None

        np.testing.assert_allclose(
            pose_stack[-1], np.asarray(one_shot, dtype=np.float32), atol=STACK_TOL,
        )

    def test_bone_lengths_constant_across_stack(self) -> None:
        """Rigid bones must have identical lengths in every stack entry.

        This is the invariant that keeps the choreographer's lerps from
        stretching bones: if endpoints agree in length, any interpolated
        pose shrinks a bone at most transiently (chord vs arc), and the
        endpoints themselves never disagree.
        """
        observed = np.asarray(_make_19pt_skeleton())
        # Realistic capture noise: asymmetric hips, knees, ankles
        observed[HIP_L][1] += 0.03
        observed[KNEE_R][1] -= 0.02
        observed[ANKLE_L][0] += 0.01
        diagnosis = _make_diagnosis([
            "narrow_stance", "narrow_foot_angle", "knee_track_cue",
            "weight_shift_cue", "depth_cue_unfamiliar",
        ])

        pose_stack = build_pose_stack(
            observed.tolist(), diagnosis, rom={"peak_dorsiflexion": 35.0},
        )
        assert pose_stack is not None

        rigid_bones = [
            (HIP_L, KNEE_L), (KNEE_L, ANKLE_L),
            (HIP_R, KNEE_R), (KNEE_R, ANKLE_R),
            (HIP_L, HIP_R),
            (ANKLE_L, FOOT_L), (ANKLE_R, FOOT_R),
        ]
        stack = pose_stack.astype(np.float64)
        for j_a, j_b in rigid_bones:
            lengths = [
                float(np.linalg.norm(pose[j_b] - pose[j_a])) for pose in stack
            ]
            assert max(lengths) - min(lengths) < 1e-4, (
                f"bone ({j_a},{j_b}) length varies across stack: {lengths}"
            )

    def test_cue_marginal_change_is_isolated(self) -> None:
        """Adjacent poses differ only by the marginal effect of one cue."""
        diagnosis = _make_diagnosis(["narrow_stance", "knee_track_cue"])

        pose_stack = build_pose_stack(_make_19pt_skeleton(), diagnosis)
        assert pose_stack is not None

        planted = [ANKLE_L, ANKLE_R, FOOT_L, FOOT_R, HIP_L, HIP_R]
        np.testing.assert_allclose(
            pose_stack[2][planted], pose_stack[1][planted], atol=1e-3,
        )

        knee_span_before = abs(pose_stack[1][KNEE_R][2] - pose_stack[1][KNEE_L][2])
        knee_span_after = abs(pose_stack[2][KNEE_R][2] - pose_stack[2][KNEE_L][2])
        assert knee_span_after > knee_span_before + 0.005


class TestSummarizeCueMagnitude:

    def test_narrow_stance_reports_centimeters_per_foot(self) -> None:
        """Athletes move their feet in centimeters, not shoulder-width ratios
        (the keypoint-based ratio also read 10-20% wide)."""
        text = summarize_cue_magnitude(
            "narrow_stance",
            {
                "__foot_target_delta": [0.0, 0.0, -0.04, 0.0, 0.0, 0.04],
                "__target_stance_ratio": 1.44,
                "__width_increase_per_side_m": 0.04,
            },
        )

        assert text == "each foot about 4 centimeters wider"
        assert "shoulder width" not in text

    def test_narrow_stance_without_per_side_width_is_generic(self) -> None:
        text = summarize_cue_magnitude(
            "narrow_stance", {"__target_stance_ratio": 1.44},
        )

        assert text == GENERIC_MAGNITUDE_TEXT

    def test_narrow_stance_already_wide_enough_is_generic(self) -> None:
        text = summarize_cue_magnitude(
            "narrow_stance", {"__width_increase_per_side_m": 0.0},
        )

        assert text == GENERIC_MAGNITUDE_TEXT

    def test_foot_angle_reports_degrees(self) -> None:
        text = summarize_cue_magnitude(
            "narrow_foot_angle", {"L_ankle.ry": 0.35, "R_ankle.ry": -0.35},
        )

        assert "20" in text
        assert "degrees" in text

    def test_knee_track_reports_degrees(self) -> None:
        text = summarize_cue_magnitude(
            "knee_track_cue", {"L_hip.ry": -0.14, "R_hip.ry": 0.14},
        )

        assert "8" in text
        assert "degrees" in text

    def test_weight_shift_reports_centimeters(self) -> None:
        text = summarize_cue_magnitude("weight_shift_cue", {"pelvis.tx": -0.03})

        assert "3 centimeters" in text

    def test_depth_cue_reports_parallel(self) -> None:
        assert "parallel" in summarize_cue_magnitude("depth_cue_unfamiliar", None)

    def test_missing_delta_returns_generic_text(self) -> None:
        assert summarize_cue_magnitude("narrow_stance", None) == GENERIC_MAGNITUDE_TEXT

    def test_unknown_cause_returns_generic_text(self) -> None:
        assert summarize_cue_magnitude("mystery_cause", {"x": 1.0}) == GENERIC_MAGNITUDE_TEXT


class TestBuildDemoData:

    def test_cues_match_stack_segments(self) -> None:
        diagnosis = _make_diagnosis(["narrow_stance", "knee_track_cue"])

        demo = build_demo_data(_make_19pt_skeleton(), diagnosis)

        assert demo is not None
        assert len(demo.cues) == 2
        assert demo.pose_stack.shape[0] == len(demo.cues) + 1
        assert demo.cues[0].cause_id == "narrow_stance"
        assert demo.cues[0].cue_index == 0
        assert demo.cues[1].cue_index == 1

    def test_returns_none_without_supported_causes(self) -> None:
        diagnosis = _make_diagnosis([])

        assert build_demo_data(_make_19pt_skeleton(), diagnosis) is None
