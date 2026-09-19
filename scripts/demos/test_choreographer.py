#!/usr/bin/env python3
"""Standalone choreographer test: calibrate, assess, show demo, save outputs.

Live mode (default):
    python scripts/demos/test_choreographer.py
    python scripts/demos/test_choreographer.py --camera 0 --reps 2

Replay mode:
    python scripts/demos/test_choreographer.py --replay
    python scripts/demos/test_choreographer.py --replay path/to/session.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import numpy as np

from biomechanics.config import load_pipeline_config
from biomechanics.diagnosis.bridge import build_anthro_dict, build_rom_dict
from biomechanics.diagnosis.demo_builder import build_demo_data
from biomechanics.diagnosis.engine import HypothesisEngine
from biomechanics.diagnosis.rep_scoring import score_set
from biomechanics.diagnosis.types import SetFeatures
from biomechanics.pipeline import BiomechanicsPipeline
from biomechanics.viz import draw_skeleton, draw_fps, FPSCounter, precreate_window, animate_window_fullscreen
from biomechanics.viz.demo_renderer import (
    MORPH_IN_SECONDS,
    MORPH_OUT_SECONDS,
    YOYO_HOLD_SECONDS,
    YOYO_TRAVEL_SECONDS,
    SETTLE_SECONDS,
    FINAL_HOLD_SECONDS,
)
from biomechanics.viz.demo_ws_bridge import DemoWSBridge, DEMO_WS_PORT
from biomechanics.diagnosis.bridge import build_frame_from_live_pipeline, build_rep_kinematic_summary


BASELINE_STUB = {"peakDorsi": 35.0, "peakKneeFlex": 120.0}
# After the last assessment rep, how long to keep running the pipeline for the
# session-scoped body measurement to complete before giving up.
MEASUREMENT_WAIT_S = 5.0


def _play_demo(demo, cycles_per_cue: int = 2) -> None:
    yoyo_period = 2.0 * (YOYO_TRAVEL_SECONDS + YOYO_HOLD_SECONDS)
    dwell = cycles_per_cue * yoyo_period

    bridge = DemoWSBridge()
    bridge.start()
    time.sleep(0.5)

    url = f"http://localhost:{DEMO_WS_PORT}"
    print(f"\n  Opening {url} ...")
    webbrowser.open(url)
    time.sleep(2.0)

    bridge.send_init(demo)
    time.sleep(0.5)

    print("  Sending demo_start")
    bridge.send_event({"type": "demo_start"})
    time.sleep(MORPH_IN_SECONDS + 0.2)

    for i, cue in enumerate(demo.cues):
        print(f"  Cue {i}: {cue.cause_id} — {cue.magnitude_text}")
        bridge.send_event({"type": "demo_cue", "cue_index": i})
        time.sleep(SETTLE_SECONDS + dwell)

    print("  Sending demo_end")
    bridge.send_event({"type": "demo_end"})

    done = bridge.wait_done(
        timeout=SETTLE_SECONDS + FINAL_HOLD_SECONDS + MORPH_OUT_SECONDS + 3.0,
    )
    print("  Demo complete." if done else "  Timed out waiting for done.")

    print("\n  Press Enter to close viewer...")
    try:
        input()
    except KeyboardInterrupt:
        pass

    bridge.stop()


def _save_session(
    out_dir: Path,
    athlete_params: dict,
    rep_kinematics: list,
    bottom_frames: list,
    diagnosis_result,
    score_summary,
    demo,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "timestamp": datetime.now().isoformat(),
        "athlete_params": athlete_params,
        "baseline": BASELINE_STUB,
        "rep_kinematics": [rk.model_dump() for rk in rep_kinematics],
        "bottom_frames": bottom_frames,
        "diagnosis": {
            "confidence": diagnosis_result.confidence,
            "immediate_causes": [c.model_dump() for c in diagnosis_result.immediate_causes],
            "session_causes": [c.model_dump() for c in diagnosis_result.session_causes],
        },
        "scoring": {
            "mean_score": score_summary.mean_score,
            "worst_rep": score_summary.worst_rep_number,
            "per_rep": [
                {"rep": s.rep_number, "score": s.composite_score}
                for s in score_summary.per_rep_scores
            ],
        },
        "demo": {
            "available": demo is not None,
            "cues": [cue.model_dump() for cue in demo.cues] if demo else [],
            "pose_stack": demo.pose_stack.tolist() if demo else None,
        },
    }
    path = out_dir / "session.json"
    with open(path, "w") as f:
        json.dump(session, f, indent=2)
    print(f"\n  Saved session to {path}")
    return path


def run_live(camera_id: int, target_reps: int) -> None:
    config = load_pipeline_config()
    config.capture.device_id = camera_id

    pipeline = BiomechanicsPipeline(config, exercise_name="squat")

    fps_counter = FPSCounter()
    window_name = "Nowva — Choreographer Test"
    precreate_window(window_name)
    animated = False

    # Phase 1: wait for the readiness gate. Body measurement accumulates inside
    # the pipeline during the assessment reps, not here.
    print("\n  Waiting for readiness gate...")
    try:
        while not pipeline.is_ready:
            result = pipeline.process_frame()
            frame = pipeline.last_frame
            if frame is not None:
                display = frame.copy()
                if result.skeleton_2d is not None:
                    draw_skeleton(display, result.skeleton_2d)
                fps_counter.update()
                draw_fps(display, fps_counter.fps)

                current, required = pipeline._readiness_gate.progress
                text = f"CALIBRATING  {current}/{required}"
                h, w = display.shape[:2]
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                x = w - ts[0] - 15
                cv2.rectangle(display, (x - 5, 5), (x + ts[0] + 5, 35), (0, 0, 0), -1)
                cv2.putText(display, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                if not animated:
                    animate_window_fullscreen(window_name)
                    animated = True
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    cv2.destroyAllWindows()
                    pipeline.release()
                    return
    except KeyboardInterrupt:
        print("\nStopped by user")
        cv2.destroyAllWindows()
        pipeline.release()
        return

    # Phase 2: collect assessment reps. Bottom frames wait until the body
    # measurement completes, then get their kinematic summaries.
    athlete_params = None
    bottom_frame_dicts: list[tuple[int, dict]] = []
    reps_done = 0
    last_rep_time = time.time()
    pipeline.rep_counter.reset()

    print(f"\n  Collecting {target_reps} rep(s)... do your squats!")

    try:
        while True:
            result = pipeline.process_frame()

            if athlete_params is None:
                athlete_params = pipeline.body_calibration.to_athlete_params()

            if pipeline.is_ready and result.rep_data is not None:
                bottom_kpts, bottom_angles = pipeline.consume_bottom_frame()
                reps_done += 1
                last_rep_time = time.time()
                depth = result.rep_data.max_depth_angle
                print(f"  Rep {reps_done}/{target_reps}  depth={depth:.1f}°")
                if bottom_kpts is not None and bottom_angles is not None:
                    bottom_frame_dicts.append(
                        (reps_done, build_frame_from_live_pipeline(bottom_kpts, bottom_angles))
                    )

            measurement_overdue = time.time() - last_rep_time > MEASUREMENT_WAIT_S
            if reps_done >= target_reps and (athlete_params is not None or measurement_overdue):
                break

            frame = pipeline.last_frame
            if frame is not None:
                display = frame.copy()
                if result.skeleton_2d is not None:
                    draw_skeleton(display, result.skeleton_2d)
                fps_counter.update()
                draw_fps(display, fps_counter.fps)

                if pipeline.body_calibration.is_complete:
                    text = f"ASSESSMENT  Rep {reps_done}/{target_reps}"
                else:
                    measured, required = pipeline.body_calibration.progress
                    text = f"MEASURING BODY {measured}/{required}  Rep {reps_done}/{target_reps}"
                h, w = display.shape[:2]
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                x = w - ts[0] - 15
                cv2.rectangle(display, (x - 5, 5), (x + ts[0] + 5, 35), (0, 0, 0), -1)
                cv2.putText(display, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
    except KeyboardInterrupt:
        print("\nStopped by user")

    cv2.destroyAllWindows()
    pipeline.release()

    if athlete_params is None:
        print("  ERROR: Body measurement never completed")
        return

    print(
        f"  Athlete params: shoulder={athlete_params['shoulder_width_m']:.3f}m "
        f"femur={athlete_params['femur_avg_m']:.3f}m "
        f"tibia={athlete_params['tibia_avg_m']:.3f}m"
    )
    anthro = build_anthro_dict(athlete_params)
    rom = build_rom_dict(athlete_params, BASELINE_STUB)

    rep_kinematics = []
    bottom_frames = []
    for rep_number, frame_dict in bottom_frame_dicts:
        rep_kinematics.append(build_rep_kinematic_summary(frame_dict, athlete_params, rep_number))
        bottom_frames.append({"rep_number": rep_number, "kpts": frame_dict["kpts"]})

    if not rep_kinematics:
        print("  No reps collected — nothing to diagnose")
        return

    # Phase 3: diagnose
    set_features = SetFeatures(
        user_id=0,
        set_id="choreo_test",
        rep_count=len(rep_kinematics),
        per_rep_kinematics=rep_kinematics,
        anthropometry=anthro,
        rom=rom,
    )
    diagnosis_result = HypothesisEngine().diagnose(set_features)
    score_summary = score_set(rep_kinematics, anthro, rom)

    print(f"\n  Diagnosis: confidence={diagnosis_result.confidence:.2f}")
    print(f"  Immediate causes: {len(diagnosis_result.immediate_causes)}")
    for cause in diagnosis_result.immediate_causes:
        print(f"    - {cause.cause_id}: {cause.explanation}")
    print(f"  Form score: {round(score_summary.mean_score * 100)}/100")

    # Build demo
    demo = None
    if diagnosis_result.immediate_causes:
        worst_kpts = bottom_frames[score_summary.worst_rep_number - 1]["kpts"]
        demo = build_demo_data(worst_kpts, diagnosis_result, anthro=anthro, rom=rom)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("output") / f"choreo_test_{timestamp}"
    _save_session(
        out_dir, athlete_params, rep_kinematics, bottom_frames,
        diagnosis_result, score_summary, demo,
    )

    # Phase 4: play demo
    if demo is not None:
        print(f"\n  Playing demo with {len(demo.cues)} cue(s)...")
        _play_demo(demo)
    else:
        print("\n  No issues found — no demo to play")


def run_replay(session_path: str | None) -> None:
    if session_path is None:
        candidates = sorted(Path("output").glob("choreo_test_*/session.json"))
        if not candidates:
            print("ERROR: No saved sessions found in output/")
            sys.exit(1)
        session_path = str(candidates[-1])

    print(f"  Loading {session_path}")
    with open(session_path) as f:
        session = json.load(f)

    demo_section = session.get("demo")
    if not demo_section or not demo_section.get("available"):
        print("ERROR: Session has no demo data")
        sys.exit(1)

    from biomechanics.diagnosis.demo_builder import DemoData, DemoCue
    from biomechanics.diagnosis.types import (
        DiagnosisResult, DetectedSymptom, HypothesizedCause,
    )

    diag_raw = session["diagnosis"]
    diagnosis = DiagnosisResult(
        set_id="replay",
        detected_symptoms=[],
        immediate_causes=[HypothesizedCause(**c) for c in diag_raw["immediate_causes"]],
        session_causes=[HypothesizedCause(**c) for c in diag_raw.get("session_causes", [])],
        longterm_causes=[],
        contextual_notes=[],
        combined_perturbation={},
        confidence=diag_raw["confidence"],
    )

    scoring = session.get("scoring", {})
    worst_rep = scoring.get("worst_rep", 1)
    worst_kpts = session["bottom_frames"][worst_rep - 1]["kpts"]

    anthro = session.get("athlete_params")
    baseline = session.get("baseline")
    rom = {"peak_dorsiflexion": baseline["peakDorsi"]} if baseline else None

    demo = build_demo_data(worst_kpts, diagnosis, anthro=anthro, rom=rom)
    if demo is None:
        print("ERROR: build_demo_data returned None")
        sys.exit(1)

    print(f"  Pose stack: {demo.pose_stack.shape}")
    print(f"  Cues ({len(demo.cues)}):")
    for cue in demo.cues:
        print(f"    [{cue.cue_index}] {cue.cause_id} — {cue.magnitude_text}")

    _play_demo(demo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the choreographer standalone")
    parser.add_argument("--replay", nargs="?", const="__latest__", default=None,
                        help="Replay a saved session (default: latest)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--reps", type=int, default=1, help="Assessment reps to collect")
    args = parser.parse_args()

    if args.replay is not None:
        run_replay(None if args.replay == "__latest__" else args.replay)
    else:
        run_live(args.camera, args.reps)


if __name__ == "__main__":
    main()
