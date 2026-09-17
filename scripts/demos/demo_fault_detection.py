#!/usr/bin/env python3
"""
Live Fault Detection Demo with Derivative-Based Rep Detection

Uses webcam + MediaPipe + IK + Fault Detection to provide real-time
squat form feedback. Features derivative-based rep counting and tempo analysis.

Usage:
    python scripts/demo_fault_detection.py

Controls:
    'q' - Quit (shows timeseries plot)
    'r' - Reset rep counter
    's' - Toggle skeleton display
    'v' - Start/Stop video recording
    'd' - Toggle debug mode (show raw angles)
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import time
import numpy as np

from biomechanics.pose.mediapipe_fallback import MediaPipePoseEstimator
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
from biomechanics.faults import RuleEngine, RepCounter, RepCounterConfig
from biomechanics.utils.types import FaultSeverity
from biomechanics.utils.derivatives import DerivativeTracker


# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_BLUE = (255, 100, 100)
COLOR_CYAN = (255, 255, 0)


def severity_color(severity: FaultSeverity) -> tuple:
    """Get color for fault severity."""
    if severity == FaultSeverity.SEVERE:
        return COLOR_RED
    elif severity == FaultSeverity.MODERATE:
        return COLOR_ORANGE
    elif severity == FaultSeverity.MILD:
        return COLOR_YELLOW
    return COLOR_GREEN


def draw_skeleton(frame, skeleton_2d, color=COLOR_GREEN):
    """Draw 2D skeleton on frame."""
    if skeleton_2d is None:
        return

    # COCO skeleton connections
    connections = [
        (5, 6),   # shoulders
        (5, 7), (7, 9),   # left arm
        (6, 8), (8, 10),  # right arm
        (5, 11), (6, 12), # torso sides
        (11, 12),  # hips
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16),  # right leg
    ]

    # Draw connections
    for i, j in connections:
        kp1 = skeleton_2d.keypoints[i]
        kp2 = skeleton_2d.keypoints[j]

        if kp1.confidence > 0.3 and kp2.confidence > 0.3:
            pt1 = (int(kp1.x), int(kp1.y))
            pt2 = (int(kp2.x), int(kp2.y))
            cv2.line(frame, pt1, pt2, color, 2)

    # Draw keypoints
    for kp in skeleton_2d.keypoints:
        if kp.confidence > 0.3:
            cv2.circle(frame, (int(kp.x), int(kp.y)), 5, color, -1)


def draw_info_panel(frame, rep_counter, angles, derivatives, faults, recent_faults, fps, is_recording, debug_mode):
    """Draw info panel on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background for panel
    panel_width = 400
    panel_height = 450 if debug_mode else 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 35
    line_height = 25

    # Title
    cv2.putText(frame, "SQUAT FORM ANALYZER", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    y += line_height + 10

    # Recording indicator
    if is_recording:
        cv2.circle(frame, (w - 30, 30), 12, COLOR_RED, -1)
        cv2.putText(frame, "REC", (w - 70, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)

    # Rep count
    cv2.putText(frame, f"Reps: {rep_counter.rep_count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
    y += line_height

    # State/Phase with color coding
    phase = rep_counter.phase.upper()
    phase_colors = {
        "IDLE": COLOR_WHITE,
        "DESCENDING": COLOR_CYAN,
        "BOTTOM": COLOR_YELLOW,
        "ASCENDING": COLOR_GREEN,
    }
    state_color = phase_colors.get(phase, COLOR_WHITE)
    cv2.putText(frame, f"Phase: {phase}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    y += line_height + 5

    # Joint angles
    if angles:
        cv2.putText(frame, "Joint Angles:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
        y += line_height - 5

        hip_avg = (angles.hip_flexion_l + angles.hip_flexion_r) / 2
        knee_avg = angles.avg_knee_flexion
        trunk = angles.trunk_flexion

        # Color code based on thresholds
        hip_color = COLOR_YELLOW if hip_avg >= rep_counter.config.entry_hip_angle else COLOR_WHITE
        knee_color = COLOR_GREEN if knee_avg >= rep_counter.config.min_depth_knee_angle else COLOR_WHITE
        cv2.putText(frame, f"  Hip: {hip_avg:.1f}°  Knee: {knee_avg:.1f}°", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hip_color, 1)
        y += line_height - 5

        cv2.putText(frame, f"  Trunk: {trunk:.1f}°", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        y += line_height

        # Debug mode - show more angles and velocity
        if debug_mode:
            cv2.putText(frame, f"  Hip L/R: {angles.hip_flexion_l:.1f}/{angles.hip_flexion_r:.1f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1)
            y += line_height - 8

            # Show velocity if available
            if derivatives is not None:
                vel = derivatives.avg_knee_velocity
                vel_color = COLOR_CYAN if vel < 0 else COLOR_GREEN if vel > 0 else COLOR_WHITE
                cv2.putText(frame, f"  Velocity: {vel:.1f} deg/s", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, vel_color, 1)
                y += line_height - 8

                accel = derivatives.avg_knee_acceleration
                cv2.putText(frame, f"  Accel: {accel:.1f} deg/s²", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)
                y += line_height - 8

            cv2.putText(frame, f"  Entry: {rep_counter.config.entry_knee_angle}°  Depth: {rep_counter.config.min_depth_knee_angle}°", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1)
            y += line_height

    # Current faults
    if faults:
        cv2.putText(frame, "Current Faults:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)
        y += line_height - 5

        for fault in faults[:3]:  # Show up to 3
            color = severity_color(fault.severity)
            cv2.putText(frame, f"  {fault.fault_type}: {fault.severity.value}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += line_height - 5

    # Recent faults (last 5 seconds)
    y += 10
    cv2.putText(frame, "Recent Faults:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    y += line_height - 5

    if recent_faults:
        for fault_type, count in list(recent_faults.items())[:3]:
            cv2.putText(frame, f"  {fault_type}: {count}x", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
            y += line_height - 5
    else:
        cv2.putText(frame, "  None - Good form!", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 60 if is_recording else 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    # Controls hint
    cv2.putText(frame, "V=Record  R=Reset  D=Debug  Q=Quit", (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)


def draw_fault_alert(frame, fault):
    """Draw large fault alert on screen."""
    h, w = frame.shape[:2]

    color = severity_color(fault.severity)

    # Draw alert box at bottom
    alert_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - alert_h - 30), (w, h - 30), color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Alert text
    cv2.putText(frame, fault.message, (20, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
    cv2.putText(frame, f"Severity: {fault.severity.value.upper()}", (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)


def draw_go_deeper_alert(frame):
    """Draw 'GO DEEPER!' alert for insufficient depth."""
    h, w = frame.shape[:2]

    # Draw orange banner
    alert_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - alert_h - 30), (w, h - 30), COLOR_ORANGE, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, "GO DEEPER!", (w//2 - 120, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
    cv2.putText(frame, "Rep not counted - didn't hit depth", (w//2 - 180, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)


def draw_rep_complete(frame, rep_data):
    """Draw rep completion overlay."""
    h, w = frame.shape[:2]

    # Draw green banner
    banner_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2 - banner_h//2), (w, h//2 + banner_h//2), COLOR_GREEN, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Rep info
    depth_category = "BELOW PARALLEL" if rep_data.max_depth_angle >= 100 else \
                     "PARALLEL" if rep_data.max_depth_angle >= 90 else \
                     "HALF" if rep_data.max_depth_angle >= 60 else "QUARTER"

    cv2.putText(frame, f"REP {rep_data.rep_number} COMPLETE!",
                (w//2 - 150, h//2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
    cv2.putText(frame, f"Depth: {rep_data.max_depth_angle:.0f}° ({depth_category})",
                (w//2 - 150, h//2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)


def plot_timeseries(timeseries_data, output_dir):
    """Plot knee angle timeseries and save/show it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot")
        return

    if not timeseries_data['timestamps']:
        print("No data to plot")
        return

    # Convert to numpy arrays
    timestamps = np.array(timeseries_data['timestamps'])
    timestamps = timestamps - timestamps[0]  # Start from 0

    knee_angles = np.array(timeseries_data['knee_angles'])
    velocities = np.array(timeseries_data['velocities'])
    phases = timeseries_data['phases']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot 1: Knee angle over time
    ax1 = axes[0]

    # Color code by phase
    phase_colors = {
        "idle": "#808080",      # gray
        "descending": "#00BFFF", # cyan
        "bottom": "#FFD700",     # yellow
        "ascending": "#00FF00",  # green
    }

    # Plot each phase segment with different colors
    for i in range(len(timestamps) - 1):
        color = phase_colors.get(phases[i], "#808080")
        ax1.plot(timestamps[i:i+2], knee_angles[i:i+2], color=color, linewidth=2)

    ax1.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='Depth threshold (95°)')
    ax1.set_ylabel('Knee Flexion Angle (°)', fontsize=12)
    ax1.set_title('Knee Angle Timeseries', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(knee_angles) * 1.1 if len(knee_angles) > 0 else 120)

    # Plot 2: Velocity over time
    # Sign convention: positive velocity = angle increasing (descending/flexing)
    #                  negative velocity = angle decreasing (ascending/extending)
    ax2 = axes[1]
    ax2.plot(timestamps, velocities, 'b-', linewidth=1, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Too fast eccentric (>100°/s)')
    ax2.axhline(y=-15, color='orange', linestyle='--', alpha=0.5, label='Stalling threshold')
    ax2.fill_between(timestamps, velocities, 0,
                     where=(np.array(velocities) > 0), alpha=0.3, color='cyan', label='Descending')
    ax2.fill_between(timestamps, velocities, 0,
                     where=(np.array(velocities) < 0), alpha=0.3, color='green', label='Ascending')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Velocity (°/sec)', fontsize=12)
    ax2.set_title('Knee Angle Velocity', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add phase legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', label='Idle'),
        Patch(facecolor='#00BFFF', label='Descending'),
        Patch(facecolor='#FFD700', label='Bottom'),
        Patch(facecolor='#00FF00', label='Ascending'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', title='Phase')

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"timeseries_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Timeseries plot saved to: {plot_path}")

    # Also save data as CSV
    csv_path = output_dir / f"timeseries_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("timestamp,knee_angle,velocity,phase\n")
        for i in range(len(timestamps)):
            f.write(f"{timestamps[i]:.3f},{knee_angles[i]:.2f},{velocities[i]:.2f},{phases[i]}\n")
    print(f"Timeseries data saved to: {csv_path}")

    # Show plot
    plt.show()


def main():
    print("=" * 50)
    print("  SQUAT FORM ANALYZER - Live Demo")
    print("  (Derivative-Based Rep Detection)")
    print("=" * 50)
    print("\nInitializing...")

    # Initialize components
    pose_estimator = MediaPipePoseEstimator(model_complexity=1)
    ik_solver = AnalyticalIKSolver()
    rule_engine = RuleEngine()

    # Derivative tracker for velocity/acceleration
    derivative_tracker = DerivativeTracker(smoothing_alpha=0.3)

    # Rep counter with new thresholds
    rep_config = RepCounterConfig(
        entry_knee_angle=30.0,      # Knee flexion to start rep
        exit_knee_angle=25.0,       # Knee flexion to end rep
        entry_hip_angle=20.0,       # Hip flexion backup entry
        exit_hip_angle=18.0,        # Hip flexion backup exit
        min_depth_knee_angle=95.0,  # Knee must reach this for valid rep
        min_rep_duration_frames=15, # ~0.5 sec at 30fps
    )
    rep_counter = RepCounter(rep_config)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get actual resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30

    print("\nCamera opened successfully!")
    print(f"Resolution: {frame_width}x{frame_height}")
    print("\nControls:")
    print("  'q' - Quit (shows timeseries plot)")
    print("  'r' - Reset rep counter")
    print("  's' - Toggle skeleton display")
    print("  'v' - Start/Stop video recording")
    print("  'd' - Toggle debug mode")
    print("\nStand back so your full body is visible.")
    print("Start squatting to see form analysis!\n")

    # State
    show_skeleton = True
    debug_mode = True  # Start with debug on to see angles
    recent_faults = {}
    fault_decay_time = 5.0
    fault_timestamps = []
    rep_complete_time = 0
    go_deeper_time = 0  # Time when "go deeper" feedback was triggered
    last_rep_data = None
    frame_times = []
    derivatives = None

    # Video recording
    is_recording = False
    video_writer = None
    output_dir = Path(__file__).parent.parent / "recordings"

    # Timeseries data collection
    timeseries_data = {
        'timestamps': [],
        'knee_angles': [],
        'velocities': [],
        'phases': [],
    }
    session_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # Get pose estimation (2D and 3D)
        skeleton_2d, skeleton_3d = pose_estimator.estimate_both(frame)

        angles = None
        faults = []
        rep_data = None
        feedback = None

        if skeleton_3d is not None:
            # Compute joint angles (no post-IK angle filter, as in production)
            angles = ik_solver.solve(skeleton_3d)

            # Compute derivatives (velocity, acceleration)
            derivatives = derivative_tracker.update(angles)

            # Run fault detection
            faults = rule_engine.evaluate(
                angles,
                in_rep=rep_counter.in_rep,
                rep_number=rep_counter.rep_count + 1
            )

            # Update rep counter with derivatives
            rep_data, feedback = rep_counter.update(angles, derivatives, faults)

            if rep_data is not None:
                last_rep_data = rep_data
                rep_complete_time = time.time()
                print(f"REP {rep_data.rep_number} - Depth: {rep_data.max_depth_angle:.1f}°")

                # Check depth faults at rep completion
                depth_faults = rule_engine.evaluate_rep_complete(
                    rep_data.max_depth_angle, angles, rep_data.rep_number
                )
                faults.extend(depth_faults)

            if feedback == "go_deeper":
                go_deeper_time = time.time()
                print("GO DEEPER! - Rep not counted (didn't hit depth)")

            # Track recent faults
            now = time.time()
            for fault in faults:
                fault_timestamps.append((now, fault.fault_type))
                recent_faults[fault.fault_type] = recent_faults.get(fault.fault_type, 0) + 1

            # Decay old faults
            fault_timestamps = [(t, ft) for t, ft in fault_timestamps if now - t < fault_decay_time]
            recent_faults = {}
            for t, ft in fault_timestamps:
                recent_faults[ft] = recent_faults.get(ft, 0) + 1

            # Collect timeseries data
            timeseries_data['timestamps'].append(time.time() - session_start_time)
            timeseries_data['knee_angles'].append(angles.avg_knee_flexion)
            timeseries_data['velocities'].append(derivatives.avg_knee_velocity if derivatives else 0)
            timeseries_data['phases'].append(rep_counter.phase)

        # Draw skeleton
        if show_skeleton and skeleton_2d is not None:
            skel_color = COLOR_RED if faults else (COLOR_YELLOW if rep_counter.in_rep else COLOR_GREEN)
            draw_skeleton(frame, skeleton_2d, skel_color)

        # Calculate FPS
        frame_times.append(time.time() - frame_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

        # Draw info panel
        draw_info_panel(frame, rep_counter, angles, derivatives, faults, recent_faults, fps, is_recording, debug_mode)

        # Draw fault alert (most severe)
        if faults:
            worst_fault = max(faults, key=lambda f: f.severity_score)
            if worst_fault.severity in (FaultSeverity.MODERATE, FaultSeverity.SEVERE):
                draw_fault_alert(frame, worst_fault)

        # Draw "GO DEEPER!" alert
        if time.time() - go_deeper_time < 2.0:
            draw_go_deeper_alert(frame)

        # Draw rep complete overlay (briefly)
        elif last_rep_data and time.time() - rep_complete_time < 1.5:
            draw_rep_complete(frame, last_rep_data)

        # Write frame if recording
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        # Show frame
        cv2.imshow("Squat Form Analyzer", frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rep_counter.reset()
            rule_engine.reset()
            derivative_tracker.reset()
            recent_faults.clear()
            fault_timestamps.clear()
            # Reset timeseries but keep session
            timeseries_data = {
                'timestamps': [],
                'knee_angles': [],
                'velocities': [],
                'phases': [],
            }
            session_start_time = time.time()
            print("Rep counter reset!")
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('v'):
            if not is_recording:
                # Start recording
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"squat_session_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps_cap, (frame_width, frame_height)
                )
                is_recording = True
                print(f"Recording started: {output_path}")
            else:
                # Stop recording
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                is_recording = False
                print("Recording saved!")

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print("Recording saved!")

    cap.release()
    cv2.destroyAllWindows()
    pose_estimator.release()

    print("\n" + "=" * 50)
    print("  Session Summary")
    print("=" * 50)
    print(f"  Total Reps: {rep_counter.rep_count}")
    if is_recording:
        print(f"  Video saved to: {output_dir}")
    print("=" * 50)

    # Plot timeseries on quit
    if len(timeseries_data['timestamps']) > 10:
        print("\nGenerating timeseries plot...")
        plot_timeseries(timeseries_data, output_dir)
    else:
        print("\nNot enough data for timeseries plot (need at least 10 frames)")


if __name__ == "__main__":
    main()
