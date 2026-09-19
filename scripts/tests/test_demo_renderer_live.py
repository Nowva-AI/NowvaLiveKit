"""Visual smoke test for the demo choreographer — no agent, no pipeline, no IPC.

Drives DemoChoreographer through a scripted timeline in a cv2 window using a
synthetic narrow-stance diagnosis. Uses the webcam for the morph-in/out if
available, otherwise a synthetic background. Press q to quit.

Run: python scripts/tests/test_demo_renderer_live.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import numpy as np

from biomechanics.diagnosis.demo_builder import build_demo_data
from biomechanics.diagnosis.types import DiagnosisResult, HypothesizedCause
from biomechanics.viz.demo_renderer import DemoChoreographer

NUM_KEYPOINTS = 21  # COCO-17 + toes + heels
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SECONDS_PER_CUE = 5.0
START_DELAY_SECONDS = 1.5


def _squat_bottom_pose() -> list[list[float]]:
    pose = np.zeros((NUM_KEYPOINTS, 3))
    pose[0] = [0.0, 1.25, 0.0]
    pose[5] = [0.05, 1.0, -0.18]
    pose[6] = [0.05, 1.0, 0.18]
    pose[7] = [0.25, 0.9, -0.20]
    pose[8] = [0.25, 0.9, 0.20]
    pose[9] = [0.40, 0.85, -0.18]
    pose[10] = [0.40, 0.85, 0.18]
    pose[11] = [-0.05, 0.45, -0.14]
    pose[12] = [-0.05, 0.45, 0.14]
    pose[13] = [0.20, 0.35, -0.12]
    pose[14] = [0.20, 0.35, 0.12]
    pose[15] = [0.05, 0.0, -0.14]
    pose[16] = [0.05, 0.0, 0.14]
    pose[17] = [0.30, 0.0, -0.14]
    pose[18] = [0.30, 0.0, 0.14]
    pose[19] = [-0.02, 0.0, -0.14]
    pose[20] = [-0.02, 0.0, 0.14]
    return pose.tolist()


def _make_cause(cause_id: str, delta: dict, explanation: str) -> HypothesizedCause:
    return HypothesizedCause(
        cause_id=cause_id,
        tier=1,
        score=0.85,
        evidence_score=0.8,
        prior=0.3,
        implicated_by=["smoke_test"],
        parameter_delta=delta,
        explanation=explanation,
    )


def _make_diagnosis() -> DiagnosisResult:
    return DiagnosisResult(
        set_id="smoke_test",
        detected_symptoms=[],
        immediate_causes=[
            _make_cause(
                "narrow_stance",
                {"__foot_target_delta": [0.0, 0.0, -0.06, 0.0, 0.0, 0.06]},
                "stance too narrow for femur length",
            ),
            _make_cause(
                "narrow_foot_angle",
                {"L_ankle.ry": 0.30, "R_ankle.ry": -0.30},
                "toes pointing straight ahead",
            ),
            _make_cause(
                "knee_track_cue",
                {"L_hip.ry": -0.12, "R_hip.ry": 0.12},
                "knees caving inward",
            ),
        ],
        session_causes=[],
        longterm_causes=[],
        contextual_notes=[],
        combined_perturbation={},
        confidence=0.85,
    )


def main() -> None:
    demo = build_demo_data(_squat_bottom_pose(), _make_diagnosis())
    if demo is None:
        print("build_demo_data returned None — corrector rejected the synthetic pose")
        return
    print(f"Pose stack: {demo.pose_stack.shape}, cues: {[c.cause_id for c in demo.cues]}")

    capture = cv2.VideoCapture(0)
    use_camera = capture.isOpened()
    print(f"Camera: {'yes' if use_camera else 'no — synthetic background'}")

    choreographer = DemoChoreographer()
    window_name = "Demo Renderer Smoke Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

    start_time = time.monotonic()
    cue_count = len(demo.cues)
    fired = {"start": False, "finish": False}
    cues_fired = 0

    while True:
        if use_camera:
            ok, frame = capture.read()
            if not ok:
                frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 90, dtype=np.uint8)
        else:
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 90, dtype=np.uint8)
            stripe_y = int((time.monotonic() * 120) % FRAME_HEIGHT)
            cv2.line(frame, (0, stripe_y), (FRAME_WIDTH, stripe_y), (140, 140, 140), 30)

        elapsed = time.monotonic() - start_time

        if not fired["start"] and elapsed >= START_DELAY_SECONDS:
            choreographer.start(demo, live_skeleton_px=None)
            fired["start"] = True
            print("→ demo_start")
        next_cue_at = START_DELAY_SECONDS + 1.0 + cues_fired * SECONDS_PER_CUE
        if fired["start"] and cues_fired < cue_count and elapsed >= next_cue_at:
            choreographer.advance_cue(cues_fired)
            print(f"→ demo_cue {cues_fired} ({demo.cues[cues_fired].cause_id})")
            cues_fired += 1
        finish_at = START_DELAY_SECONDS + 1.0 + cue_count * SECONDS_PER_CUE
        if not fired["finish"] and cues_fired == cue_count and elapsed >= finish_at:
            choreographer.finish()
            fired["finish"] = True
            print("→ demo_end")

        display = choreographer.render(frame)
        cv2.imshow(window_name, display)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
        if fired["finish"] and not choreographer.is_active and elapsed > finish_at + 4.0:
            print("Morph-out complete — done")
            break

    if use_camera:
        capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
