"""
Calibrated workout test — calibration phase + workout with personalized thresholds.

Usage:
    python test_calibrated_workout.py [--camera 0] [--exercise "Barbell Back Squat"]

Phase 1: Perform 5 bodyweight calibration squats (no fault reporting).
         Peaks are recorded to build a biomechanical profile.
Phase 2: Perform 3 sets of 5 reps with calibrated fault thresholds.
         Fault cues appear on the dashboard time-series charts.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from biomechanics.pipeline import BiomechanicsPipeline
from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.coaching.session_tracker import SessionTracker
from biomechanics.config import load_pipeline_config

from biomechanics.viz import draw_skeleton, draw_fps, FPSCounter
from biomechanics.viz.set_plots import make_output_dir
from biomechanics.analysis.set_finalizer import SetDataCollector, finalize_set
from biomechanics.viz.html_dashboard import generate_session_dashboard

from biomechanics.calibration import (
    CalibrationTracker,
    build_calibration_profile,
    apply_calibration_to_rule_engine,
    extract_thresholds_from_rule_engine,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CALIBRATION_REPS = 5
TARGET_SETS = 1
TARGET_REPS = 6
REST_SECONDS = 10
EXERCISE_NAME = "Barbell Back Squat"


# ---------------------------------------------------------------------------
# Mock IPC client — prints messages instead of sending over socket
# ---------------------------------------------------------------------------
class MockIPCClient:
    """Drop-in replacement for IPCClient that logs to console."""

    def __init__(self):
        self.messages = []

    def send_message(self, msg: dict):
        self.messages.append(msg)
        msg_type = msg.get("type", "")
        if msg_type == "rep_complete":
            rep = msg["rep_number"]
            depth = msg["max_depth_angle"]
            clean = "clean" if msg["is_clean"] else "faults: " + ", ".join(msg["faults_in_rep"])
            print(f"  [REP {rep}] depth={depth}° | {clean}")
        elif msg_type == "set_complete":
            s = msg["set_number"]
            total = msg["total_reps"]
            clean = msg["clean_reps"]
            avg_d = msg["avg_depth"]
            print(f"\n  === SET {s} COMPLETE: {total} reps ({clean} clean), avg depth {avg_d}° ===\n")
        elif msg_type == "fault":
            print(f"  [FAULT] {msg['fault_type']} ({msg['severity']}) — {msg['message']}")
        elif msg_type in ("rep_count", "frame_data", "cache_cues"):
            pass  # skip noisy messages
        else:
            print(f"  [IPC] {msg_type}: {msg}")


def save_calibration_report(peaks: dict, profile: dict, out_dir: str):
    """Save calibration_profile.json and calibration_profile.md."""
    # JSON
    cal_json = {
        "calibration_reps": CALIBRATION_REPS,
        "peaks": peaks,
        "profile": {k: v for k, v in profile.items() if k != "defaults"},
        "defaults": profile.get("defaults", {}),
    }
    json_path = str(Path(out_dir) / "calibration_profile.json")
    with open(json_path, "w") as f:
        json.dump(cal_json, f, indent=2)
    print(f"  Saved: {json_path}")

    # Markdown
    defaults = profile.get("defaults", {})
    md_lines = [
        f"# Calibration Profile ({CALIBRATION_REPS} reps)",
        "",
        "## Observed Peaks",
        "",
        f"| Signal | Value |",
        f"|--------|-------|",
        f"| Trunk Flexion (peak) | {peaks['trunk_flexion']:.1f}° |",
        *[f"| Hip Adduction Rep {i+1} | {p:.1f}° |" for i, p in enumerate(peaks.get('hip_adduction_per_rep', []))],
        f"| **Hip Adduction (avg → band)** | **±{peaks['hip_adduction']:.1f}°** |",
        f"| Bilateral Asymmetry (peak) | {peaks['asymmetry']:.1f}° |",
        f"| Peak Dorsiflexion | {peaks['peak_dorsiflexion']:.1f}° |",
        f"| **Avg Squat Depth** | **{peaks.get('avg_depth', 0):.1f}°** |",
        *[f"| Squat Depth Rep {i+1} | {d:.1f}° |" for i, d in enumerate(peaks.get('depth_per_rep', []))],
        "",
        "## Calibrated Thresholds",
        "",
        "| Fault | Mild | Moderate | Severe | Default Mild | Default Moderate | Default Severe |",
        "|-------|------|----------|--------|-------------|-----------------|----------------|",
        f"| Knee Valgus | {profile['knee_valgus']['mild']:.1f}° | {profile['knee_valgus']['moderate']:.1f}° | {profile['knee_valgus']['severe']:.1f}° | {defaults['knee_valgus']['mild']:.1f}° | {defaults['knee_valgus']['moderate']:.1f}° | {defaults['knee_valgus']['severe']:.1f}° |",
        f"| Forward Lean | {profile['forward_lean']['mild']:.1f}° | {profile['forward_lean']['moderate']:.1f}° | {profile['forward_lean']['severe']:.1f}° | {defaults['forward_lean']['mild']:.1f}° | {defaults['forward_lean']['moderate']:.1f}° | {defaults['forward_lean']['severe']:.1f}° |",
        f"| Bilateral Asymmetry | {profile['bilateral_asymmetry']['mild']:.1f}° | {profile['bilateral_asymmetry']['moderate']:.1f}° | {profile['bilateral_asymmetry']['severe']:.1f}° | {defaults['bilateral_asymmetry']['mild']:.1f}° | {defaults['bilateral_asymmetry']['moderate']:.1f}° | {defaults['bilateral_asymmetry']['severe']:.1f}° |",
        "",
        "### Depth Thresholds",
        "",
        f"| Category | Calibrated | Default |",
        f"|----------|-----------|---------|",
        f"| Deep Squat (below parallel) | {profile.get('depth', {}).get('parallel_threshold', '-')}° | {defaults.get('depth', {}).get('parallel_threshold', '-')}° |",
        f"| Parallel | {profile.get('depth', {}).get('half_threshold', '-')}° | {defaults.get('depth', {}).get('half_threshold', '-')}° |",
        f"| Half Squat | {profile.get('depth', {}).get('quarter_threshold', '-')}° | {defaults.get('depth', {}).get('quarter_threshold', '-')}° |",
        "",
    ]
    md_path = str(Path(out_dir) / "calibration_profile.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_test():
    camera_id = 0
    config_path = None

    # Parse CLI args
    args = sys.argv[1:]
    i = 0
    exercise = EXERCISE_NAME
    while i < len(args):
        if args[i] == "--camera" and i + 1 < len(args):
            camera_id = int(args[i + 1])
            i += 2
        elif args[i] == "--exercise" and i + 1 < len(args):
            exercise = args[i + 1]
            i += 2
        else:
            i += 1

    config = load_pipeline_config(config_path)
    config.capture.device_id = camera_id
    config.bilstm.enabled = True

    pipeline = BiomechanicsPipeline(config)
    ipc_client = MockIPCClient()
    bridge = IPCBridge(ipc_client)
    session_tracker = SessionTracker(bridge, config=config.coaching)

    bridge.prepare_exercise(exercise)

    print(f"\n{'='*60}")
    print(f"  CALIBRATED WORKOUT TEST: {exercise}")
    print(f"  Phase 1: {CALIBRATION_REPS} calibration squats")
    print(f"  Phase 2: {TARGET_SETS} sets x {TARGET_REPS} reps (calibrated)")
    print(f"  Rest: {REST_SECONDS}s between sets")
    print(f"{'='*60}")
    print(f"\nStand in frame and wait for CALIBRATING indicator.")
    print("Press 'q' to stop early.\n")

    fps_counter = FPSCounter()
    window_name = f"Nowva Calibrated — {exercise}"
    out_dir = make_output_dir()

    set_collector = SetDataCollector()
    set_summaries = []
    set_plot_data = []

    # -----------------------------------------------------------------------
    # Phase 1: Calibration
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PHASE 1: CALIBRATION ({CALIBRATION_REPS} bodyweight squats)")
    print(f"{'='*60}\n")

    tracker = CalibrationTracker(target_reps=CALIBRATION_REPS)

    try:
        while not tracker.is_complete:
            result = pipeline.process_frame()

            if pipeline.is_ready and result.skeleton_3d is not None:
                set_collector.record_frame(result, result.skeleton_3d)

                if result.joint_angles is not None:
                    tracker.record_frame(result.joint_angles)

                # Count reps but do NOT report faults
                if result.rep_data is not None:
                    depth = result.rep_data.max_depth_angle
                    tracker.on_rep_complete(depth)
                    print(f"  [CAL REP {tracker.reps_completed}/{CALIBRATION_REPS}] depth={depth:.1f}° | peak adduction={tracker.rep_adduction_peaks[-1]:.1f}°")

            # Display
            frame = pipeline.last_frame
            if frame is not None:
                display = frame.copy()
                if result.skeleton_2d is not None:
                    draw_skeleton(display, result.skeleton_2d)
                fps_counter.update()
                draw_fps(display, fps_counter.fps)

                h, w = display.shape[:2]
                if pipeline.is_ready:
                    text = f"CALIBRATING  Rep {tracker.reps_completed}/{CALIBRATION_REPS}"
                    color = (0, 200, 255)
                else:
                    current, required = pipeline._readiness_gate.progress
                    text, color = f"WAITING {current}/{required}", (0, 200, 255)

                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                x = w - ts[0] - 15
                cv2.rectangle(display, (x - 5, 5), (x + ts[0] + 5, 35), (0, 0, 0), -1)
                cv2.putText(display, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.putText(display, f"CALIBRATION  {tracker.reps_completed}/{CALIBRATION_REPS}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped early by user (pressed 'q')")
                    if tracker.reps_completed == 0:
                        cv2.destroyAllWindows()
                        pipeline.release()
                        return

            # Throttle
            total_ms = sum(result.latency_ms.values())
            target_ms = 1000.0 / config.target_fps
            if total_ms < target_ms:
                time.sleep((target_ms - total_ms) / 1000.0)

        # Finalize calibration set as set 0
        cal_plot_result = finalize_set(set_collector, 0, out_dir)

        # Build calibration profile from tracker
        peaks = tracker.get_peaks()
        avg_peak_adduction = peaks["hip_adduction"]
        profile = build_calibration_profile(peaks, config)

        # Print calibration summary
        print(f"\n{'='*60}")
        print(f"  CALIBRATION PROFILE ({tracker.reps_completed} reps)")
        print(f"{'='*60}")
        print(f"  Peak trunk flexion:       {peaks['trunk_flexion']:.1f}°")
        for i, p in enumerate(peaks['hip_adduction_per_rep']):
            print(f"  Hip adduction rep {i+1}:      {p:.1f}°")
        print(f"  Avg hip adduction (band): {avg_peak_adduction:.1f}°")
        print(f"  Peak asymmetry:           {peaks['asymmetry']:.1f}°")
        print(f"  Peak dorsiflexion:        {peaks['peak_dorsiflexion']:.1f}°")
        print(f"  {'─'*56}")
        kv = profile["knee_valgus"]
        print(f"  Knee Valgus band: ±{avg_peak_adduction:.1f}°  thresholds: {kv['mild']:.1f}° / {kv['moderate']:.1f}° / {kv['severe']:.1f}°")
        fl = profile["forward_lean"]
        print(f"  Forward Lean thresholds:  {fl['mild']:.1f}° / {fl['moderate']:.1f}° / {fl['severe']:.1f}°")
        ba = profile["bilateral_asymmetry"]
        print(f"  Asymmetry thresholds:     {ba['mild']:.1f}° / {ba['moderate']:.1f}° / {ba['severe']:.1f}°")
        print(f"{'='*60}\n")

        # Apply to rule engine
        apply_calibration_to_rule_engine(pipeline._rule_engine, profile)
        set_collector.thresholds = extract_thresholds_from_rule_engine(pipeline._rule_engine)

        # Save calibration report (JSON + MD)
        save_calibration_report(peaks, profile, out_dir)

        # Reset for workout phase — widen the standing gate based on
        # the knee flexion observed during calibration so the gate
        # doesn't reject the user's natural standing posture.
        pipeline.reset_readiness_gate()
        gate_knee = tracker.standing_knee_flexion + 10.0  # 10° margin above observed max
        pipeline._readiness_gate.max_knee_flexion_deg = gate_knee
        print(f"  [GATE] Standing knee threshold raised to {gate_knee:.1f}° (observed max: {tracker.standing_knee_flexion:.1f}°)")
        pipeline.rep_counter.reset()
        if pipeline._bilstm is not None:
            pipeline._bilstm.reset()

    except KeyboardInterrupt:
        print("\nStopped during calibration")
        cv2.destroyAllWindows()
        pipeline.release()
        return

    # -----------------------------------------------------------------------
    # Phase 2: Workout
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PHASE 2: WORKOUT ({TARGET_SETS} sets x {TARGET_REPS} reps)")
    print(f"{'='*60}")
    print("Stand in frame and wait for COLLECTING indicator.\n")

    completed_sets = 0
    reps_this_set = 0
    resting = False
    rest_end_time = 0.0

    try:
        while True:
            result = pipeline.process_frame()

            if pipeline.is_ready and result.skeleton_3d is not None and not resting:
                set_collector.record_frame(result, result.skeleton_3d)

            if not resting:
                bridge.send_frame_data(result)
                for fault in result.faults:
                    bridge.send_fault(fault)
                    set_collector.record_fault(fault)
                if result.rep_data:
                    session_tracker.on_rep_complete(result.rep_data)
                    reps_this_set += 1

                    if reps_this_set >= TARGET_REPS:
                        session_tracker.force_end_set()
                        if session_tracker.last_set_summary:
                            set_summaries.append(session_tracker.last_set_summary)
                        completed_sets += 1
                        print(f"\n  Set {completed_sets}/{TARGET_SETS} done ({reps_this_set} reps)")

                        plot_result = finalize_set(set_collector, completed_sets, out_dir)
                        if plot_result is not None:
                            plot_result["valgus_band"] = avg_peak_adduction
                            set_plot_data.append(plot_result)

                        if completed_sets >= TARGET_SETS:
                            print(f"\nAll {TARGET_SETS} sets complete!")
                            break
                        resting = True
                        rest_end_time = time.time() + REST_SECONDS
                        reps_this_set = 0
                        pipeline.reset_readiness_gate()
                        pipeline.rep_counter.reset()
                        if pipeline._bilstm is not None:
                            pipeline._bilstm.reset()
                        print(f"[REST] {REST_SECONDS}s rest before set {completed_sets + 1}...")

            # Display
            frame = pipeline.last_frame
            if frame is not None:
                display = frame.copy()
                if result.skeleton_2d is not None:
                    draw_skeleton(display, result.skeleton_2d)
                fps_counter.update()
                draw_fps(display, fps_counter.fps)

                h, w = display.shape[:2]
                if pipeline.is_ready:
                    text, color = "COLLECTING", (0, 255, 0)
                else:
                    current, required = pipeline._readiness_gate.progress
                    text, color = f"WAITING {current}/{required}", (0, 200, 255)
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                x = w - ts[0] - 15
                cv2.rectangle(display, (x - 5, 5), (x + ts[0] + 5, 35), (0, 0, 0), -1)
                cv2.putText(display, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if resting:
                    remaining = max(0.0, rest_end_time - time.time())
                    if remaining <= 0:
                        resting = False
                        pipeline.rep_counter.reset()
                        if pipeline._bilstm is not None:
                            pipeline._bilstm.reset()
                        print(f"[REST] Done — starting set {completed_sets + 1}")
                    else:
                        overlay = display.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                        secs = int(remaining)
                        timer_text = f"{secs // 60}:{secs % 60:02d}" if secs >= 60 else str(secs)
                        cv2.putText(display, "REST", (w // 2 - 80, h // 2 - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
                        ts2 = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)[0]
                        cv2.putText(display, timer_text, ((w - ts2[0]) // 2, h // 2 + 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 6)
                else:
                    set_num = completed_sets + 1
                    cv2.putText(display, f"Set {set_num}/{TARGET_SETS}  Reps: {reps_this_set}/{TARGET_REPS}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped early by user (pressed 'q')")
                    break

            # Throttle
            total_ms = sum(result.latency_ms.values())
            target_ms = 1000.0 / config.target_fps
            if total_ms < target_ms:
                time.sleep((target_ms - total_ms) / 1000.0)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cv2.destroyAllWindows()
        pipeline.release()

        # Finalize any in-progress set
        if set_collector.has_enough_data() and not resting:
            if session_tracker.set_active:
                session_tracker.force_end_set()
                if session_tracker.last_set_summary:
                    set_summaries.append(session_tracker.last_set_summary)
            completed_sets += 1
            plot_result = finalize_set(set_collector, completed_sets, out_dir)
            if plot_result is not None:
                plot_result["valgus_band"] = avg_peak_adduction
                set_plot_data.append(plot_result)

        # Session summary
        stats = session_tracker.session_stats
        print(f"\n{'='*60}")
        print(f"  SESSION SUMMARY")
        print(f"  Total reps: {stats['total_reps']}")
        print(f"  Total sets: {stats['total_sets']}")
        print(f"  Avg depth: {stats['avg_depth']:.1f}°")
        print(f"  Clean rep %: {stats['clean_rep_percentage']:.1f}%")
        print(f"{'='*60}")

        # Save session-level summary
        session_export = {
            "exercise": exercise,
            "calibration_reps": CALIBRATION_REPS,
            "target_sets": TARGET_SETS,
            "target_reps": TARGET_REPS,
            "weight": "bodyweight",
            "calibration_peaks": peaks,
            "session_stats": stats,
            "set_summaries": set_summaries,
        }
        summary_path = str(Path(out_dir) / "session_summary.json")
        with open(summary_path, "w") as f:
            json.dump(session_export, f, indent=2, default=str)
        print(f"  Saved: {summary_path}")

        # Save IPC message log
        ipc_path = str(Path(out_dir) / "ipc_messages.json")
        with open(ipc_path, "w") as f:
            json.dump(ipc_client.messages, f, indent=2, default=str)
        print(f"  Saved: {ipc_path}")

        # Generate session-level interactive dashboard
        if set_plot_data:
            session_info = {
                "exercise": exercise,
                "target_sets": TARGET_SETS,
                "target_reps": TARGET_REPS,
                "calibration_reps": CALIBRATION_REPS,
            }
            generate_session_dashboard(set_plot_data, session_info, out_dir, auto_open=True)

        print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    run_test()
