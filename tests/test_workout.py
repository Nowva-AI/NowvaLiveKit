"""
Standalone workout test — bypasses main.py and voice agent entirely.

Usage:
    python test_workout.py [--camera 0] [--exercise "Barbell Back Squat"]

Runs 3 sets of 6 reps (bodyweight) with full biomechanics pipeline.
Generates per-set plots and saves per-set data to output/ when done.
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_SETS = 3
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

    print(f"\n{'='*50}")
    print(f"  WORKOUT TEST: {exercise}")
    print(f"  {TARGET_SETS} sets × {TARGET_REPS} reps (bodyweight)")
    print(f"  Rest: {REST_SECONDS}s between sets")
    print(f"{'='*50}")
    print(f"\nStand in frame and wait for COLLECTING indicator.")
    print("Press 'q' to stop early.\n")

    fps_counter = FPSCounter()
    window_name = f"Nowva Test — {exercise}"

    # Output directory (created once, all sets go here)
    out_dir = make_output_dir()

    # Per-set data collection via shared module
    set_collector = SetDataCollector()

    # Session-level accumulators
    set_summaries = []
    set_plot_data = []

    # Set tracking
    completed_sets = 0
    reps_this_set = 0
    resting = False
    rest_end_time = 0.0

    try:
        while True:
            result = pipeline.process_frame()

            # Only collect data once readiness gate has passed (30/30)
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

                    # Auto-end set when target reps reached
                    if reps_this_set >= TARGET_REPS:
                        session_tracker.force_end_set()
                        if session_tracker.last_set_summary:
                            set_summaries.append(session_tracker.last_set_summary)
                        completed_sets += 1
                        print(f"\n  Set {completed_sets}/{TARGET_SETS} done ({reps_this_set} reps)")

                        # Save this set's plots and data
                        plot_result = finalize_set(set_collector, completed_sets, out_dir)
                        if plot_result is not None:
                            set_plot_data.append(plot_result)

                        if completed_sets >= TARGET_SETS:
                            print(f"\nAll {TARGET_SETS} sets complete!")
                            break
                        # Start rest and reset for next set
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

                # Readiness indicator
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
                        # Draw rest overlay
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

        # Finalize any in-progress set (quit early / ctrl-c)
        if set_collector.has_enough_data() and not resting:
            if session_tracker.set_active:
                session_tracker.force_end_set()
                if session_tracker.last_set_summary:
                    set_summaries.append(session_tracker.last_set_summary)
            completed_sets += 1
            plot_result = finalize_set(set_collector, completed_sets, out_dir)
            if plot_result is not None:
                set_plot_data.append(plot_result)

        # Print session summary
        stats = session_tracker.session_stats
        print(f"\n{'='*50}")
        print(f"  SESSION SUMMARY")
        print(f"  Total reps: {stats['total_reps']}")
        print(f"  Total sets: {stats['total_sets']}")
        print(f"  Avg depth: {stats['avg_depth']:.1f}°")
        print(f"  Clean rep %: {stats['clean_rep_percentage']:.1f}%")
        print(f"{'='*50}")

        # Save session-level summary
        session_export = {
            "exercise": exercise,
            "target_sets": TARGET_SETS,
            "target_reps": TARGET_REPS,
            "weight": "bodyweight",
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
            }
            generate_session_dashboard(set_plot_data, session_info, out_dir, auto_open=True)

        print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    run_test()
