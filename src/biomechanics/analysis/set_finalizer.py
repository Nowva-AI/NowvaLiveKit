"""
Per-set data collection, plot generation, and export.

Shared between pose_estimation_process.py (main pipeline) and test_workout.py.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from biomechanics.utils.json_safe import nan_to_none
from biomechanics.utils.types import PipelineFrame, Skeleton3D, CocoKeypoints, FaultEvent
from biomechanics.utils.geometry import (
    WORLD_UP,
    joint_angle_3_points,
    calculate_trunk_angle,
)
from biomechanics.analysis.rep_segmenter import segment_set, plot_segmentation, write_set_report
from biomechanics.viz.html_dashboard import generate_set_dashboard


# ---------------------------------------------------------------------------
# Signal smoothing
# ---------------------------------------------------------------------------
def smooth_1d(signal, median_window=5, sma_window=3, sma_start=0):
    """Median filter to kill impulse noise, then simple moving average.

    Args:
        signal: 1-D numpy array.
        median_window: Odd window size for the median pre-filter.
        sma_window: Window size for the moving average smoothing pass.
        sma_start: Index at which SMA begins. Samples before this index
            are kept as-is (only median-filtered).
    """
    from scipy.signal import medfilt

    if median_window % 2 == 0:
        median_window += 1

    cleaned = medfilt(signal, kernel_size=median_window)

    if sma_start >= len(cleaned):
        return cleaned

    tail = cleaned[sma_start:]
    kernel = np.ones(sma_window) / sma_window
    pad_l = sma_window // 2
    pad_r = sma_window - 1 - pad_l
    padded = np.pad(tail, (pad_l, pad_r), mode="edge")
    smoothed_tail = np.convolve(padded, kernel, mode="valid")

    out = cleaned.copy()
    out[sma_start:] = smoothed_tail
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def save_set_plots(label, timestamps, hip_mid, vel_mid, knee_angles,
                   hip_angles, trunk_angles, rep_events, out_dir):
    """Generate and save hip position, velocity, and joint angle plots.

    ``label`` names the scope in plot titles (e.g. "Set 1", "Rep 3").
    """
    t_rel = timestamps - timestamps[0]
    pos_cm = hip_mid * 100.0
    vel_cm = vel_mid * 100.0

    # --- Hip position ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, pos_cm, label="Hip Height (rel. ankle)", color="#2196F3", linewidth=1.5)
    for ts_val, rep_num in rep_events:
        rt = ts_val - timestamps[0]
        ax.axvline(x=rt, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"Rep {rep_num}", (rt, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(5, -15), fontsize=8, color="green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hip Height Above Ankles (cm)")
    ax.set_title(f"Hip Position — {label}")
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pos_path = str(Path(out_dir) / "hip_position.png")
    fig.savefig(pos_path, dpi=150)
    print(f"  Saved: {pos_path}")
    plt.close(fig)

    # --- Hip velocity ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, vel_cm, label="Hip Midpoint Velocity", color="#2196F3", linewidth=1.0, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    for ts_val, rep_num in rep_events:
        rt = ts_val - timestamps[0]
        ax.axvline(x=rt, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Velocity (cm/s)")
    ax.set_title(f"Hip Velocity — {label}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    vel_path = str(Path(out_dir) / "hip_velocity.png")
    fig.savefig(vel_path, dpi=150)
    print(f"  Saved: {vel_path}")
    plt.close(fig)

    # --- Joint angles ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, knee_angles, label="Knee Angle", color="#2196F3", linewidth=1.5)
    ax.plot(t_rel, hip_angles, label="Hip Flexion", color="#FF5722", linewidth=1.5)
    ax.plot(t_rel, trunk_angles, label="Trunk Lean", color="#4CAF50", linewidth=1.5)
    for ts_val, rep_num in rep_events:
        rt = ts_val - timestamps[0]
        ax.axvline(x=rt, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(f"Joint Angles — {label}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    angles_path = str(Path(out_dir) / "joint_angles.png")
    fig.savefig(angles_path, dpi=150)
    print(f"  Saved: {angles_path}")
    plt.close(fig)


def save_pipeline_angles_plot(label, t_rel, knee_flexion, hip_flexion,
                              trunk_flexion, rep_events, timestamps, out_dir):
    """Generate and save pipeline joint angles (IK solver + One Euro filter)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, knee_flexion, label="Knee Angle", color="#2196F3", linewidth=1.5)
    ax.plot(t_rel, hip_flexion, label="Hip Flexion", color="#FF5722", linewidth=1.5)
    ax.plot(t_rel, trunk_flexion, label="Trunk Lean", color="#4CAF50", linewidth=1.5)
    for ts_val, rep_num in rep_events:
        rt = ts_val - timestamps[0]
        ax.axvline(x=rt, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(f"Pipeline Joint Angles (IK + One Euro) — {label}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = str(Path(out_dir) / "pipeline_angles.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def save_hip_adduction_plot(label, t_rel, hip_adduction_l, hip_adduction_r,
                            rep_events, timestamps, out_dir):
    """Generate and save hip adduction (knee valgus proxy) plot for a set."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, hip_adduction_l, label="Left Hip Adduction", color="#2196F3", linewidth=1.5)
    ax.plot(t_rel, hip_adduction_r, label="Right Hip Adduction", color="#F44336", linewidth=1.5)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    for ts_val, rep_num in rep_events:
        rt = ts_val - timestamps[0]
        ax.axvline(x=rt, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"Rep {rep_num}", (rt, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(5, -15), fontsize=8, color="green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hip Adduction (°)")
    ax.set_title(f"Hip Adduction (Knee Valgus Proxy) — {label}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = str(Path(out_dir) / "hip_adduction.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def save_knee_valgus_plot(label, t_rel, valgus_l, valgus_r, kasr,
                          rep_events, timestamps, thresholds, out_dir):
    """Generate knee valgus plot with threshold bands and KASR secondary axis."""
    multi = os.getenv("NOWVA_MULTI_CAMERA", "false").lower() == "true"
    mode_label = "3D Abduction" if multi else "2D FPPA"

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t_rel, valgus_l, label=f"Left Valgus ({mode_label})", color="#2196F3", linewidth=1.5)
    ax1.plot(t_rel, valgus_r, label=f"Right Valgus ({mode_label})", color="#F44336", linewidth=1.5)
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

    if thresholds:
        kv = thresholds.get("knee_valgus", {})
        mild = kv.get("mild", 0)
        moderate = kv.get("moderate", 0)
        severe = kv.get("severe", 0)
        if mild:
            ax1.axhline(y=mild, color="#FFC107", linewidth=1, linestyle="--", alpha=0.7, label=f"Mild ({mild:.0f}°)")
        if moderate:
            ax1.axhline(y=moderate, color="#FF9800", linewidth=1, linestyle="--", alpha=0.7, label=f"Moderate ({moderate:.0f}°)")
        if severe:
            ax1.axhline(y=severe, color="#F44336", linewidth=1, linestyle="--", alpha=0.7, label=f"Severe ({severe:.0f}°)")

    for ts_val, rep_num in rep_events:
        rt = ts_val - timestamps[0]
        ax1.axvline(x=rt, color="green", linestyle="--", alpha=0.5)
        ax1.annotate(f"Rep {rep_num}", (rt, ax1.get_ylim()[1]),
                     textcoords="offset points", xytext=(5, -15), fontsize=8, color="green")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Knee Valgus (°)")
    ax1.set_title(f"Knee Valgus ({mode_label}) — {label}")

    ax2 = ax1.twinx()
    ax2.plot(t_rel, kasr, label="KASR", color="#4CAF50", linewidth=1, alpha=0.6, linestyle=":")
    ax2.axhline(y=1.0, color="#4CAF50", linewidth=0.5, linestyle="-", alpha=0.3)
    ax2.set_ylabel("Knee/Ankle Sep. Ratio")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path = str(Path(out_dir) / "knee_valgus.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def save_bilateral_asymmetry_plot(label, t_rel, bilateral_asymmetry,
                                  rep_events, timestamps, out_dir):
    """Generate and save bilateral asymmetry plot for a set."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, bilateral_asymmetry, label="Bilateral Asymmetry (max of knee/hip)",
            color="#9C27B0", linewidth=1.5)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    for ts_val, rep_num in rep_events:
        rt = ts_val - timestamps[0]
        ax.axvline(x=rt, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"Rep {rep_num}", (rt, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(5, -15), fontsize=8, color="green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Asymmetry (°)")
    ax.set_title(f"Bilateral Asymmetry — {label}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = str(Path(out_dir) / "bilateral_asymmetry.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-set data collector
# ---------------------------------------------------------------------------
class SetDataCollector:
    """Accumulates per-set frame data for post-set analysis and plotting."""

    def __init__(self):
        self.timestamps: list = []
        self.hip_mid_y: list = []
        self.knee_angles: list = []
        self.hip_angles: list = []
        self.trunk_angles: list = []
        self.pipeline_knee: list = []
        self.pipeline_hip: list = []
        self.pipeline_trunk: list = []
        self.hip_adduction_l: list = []
        self.hip_adduction_r: list = []
        self.knee_valgus_l: list = []
        self.knee_valgus_r: list = []
        self.knee_ankle_sep_ratio: list = []
        self.bilateral_asymmetry: list = []
        self.rep_events: list = []
        self.fault_events: list = []
        self.frames_data: list = []
        self.thresholds: dict | None = None

    def record_frame(self, result: PipelineFrame, skeleton_3d: Skeleton3D) -> None:
        """Record one frame of data.

        Call only when pipeline.is_ready, skeleton_3d is not None,
        and not resting / not workout_finished.
        """
        t = result.timestamp

        hip_l = skeleton_3d.get_keypoint(CocoKeypoints.LEFT_HIP)
        hip_r = skeleton_3d.get_keypoint(CocoKeypoints.RIGHT_HIP)
        knee_l = skeleton_3d.get_keypoint(CocoKeypoints.LEFT_KNEE)
        knee_r = skeleton_3d.get_keypoint(CocoKeypoints.RIGHT_KNEE)
        ankle_l = skeleton_3d.get_keypoint(CocoKeypoints.LEFT_ANKLE)
        ankle_r = skeleton_3d.get_keypoint(CocoKeypoints.RIGHT_ANKLE)
        shoulder_l = skeleton_3d.get_keypoint(CocoKeypoints.LEFT_SHOULDER)
        shoulder_r = skeleton_3d.get_keypoint(CocoKeypoints.RIGHT_SHOULDER)

        self.timestamps.append(t)

        # Hip midpoint Y relative to ankle midpoint Y
        hip_mid_y = (hip_l.y + hip_r.y) / 2.0
        ankle_mid_y = (ankle_l.y + ankle_r.y) / 2.0
        self.hip_mid_y.append(hip_mid_y - ankle_mid_y)

        # 3D midpoints for joint angle computation
        shoulder_mid = np.array([
            (shoulder_l.x + shoulder_r.x) / 2,
            (shoulder_l.y + shoulder_r.y) / 2,
            (shoulder_l.z + shoulder_r.z) / 2,
        ])
        hip_mid = np.array([
            (hip_l.x + hip_r.x) / 2,
            (hip_l.y + hip_r.y) / 2,
            (hip_l.z + hip_r.z) / 2,
        ])
        knee_mid = np.array([
            (knee_l.x + knee_r.x) / 2,
            (knee_l.y + knee_r.y) / 2,
            (knee_l.z + knee_r.z) / 2,
        ])
        ankle_mid = np.array([
            (ankle_l.x + ankle_r.x) / 2,
            (ankle_l.y + ankle_r.y) / 2,
            (ankle_l.z + ankle_r.z) / 2,
        ])

        self.knee_angles.append(joint_angle_3_points(hip_mid, knee_mid, ankle_mid))
        self.hip_angles.append(joint_angle_3_points(shoulder_mid, hip_mid, knee_mid))
        self.trunk_angles.append(
            calculate_trunk_angle(shoulder_mid, hip_mid, vertical=WORLD_UP)
        )

        # Pipeline joint angles (IK solver + One Euro filter)
        if result.joint_angles is not None:
            avg_knee = (result.joint_angles.knee_flexion_l + result.joint_angles.knee_flexion_r) / 2.0
            avg_hip = (result.joint_angles.hip_flexion_l + result.joint_angles.hip_flexion_r) / 2.0
            self.pipeline_knee.append(avg_knee)
            self.pipeline_hip.append(avg_hip)
            self.pipeline_trunk.append(result.joint_angles.trunk_flexion)
            self.hip_adduction_l.append(result.joint_angles.hip_adduction_l)
            self.hip_adduction_r.append(result.joint_angles.hip_adduction_r)
            self.knee_valgus_l.append(result.joint_angles.knee_valgus_l)
            self.knee_valgus_r.append(result.joint_angles.knee_valgus_r)
            self.knee_ankle_sep_ratio.append(result.joint_angles.knee_ankle_sep_ratio)
            self.bilateral_asymmetry.append(
                max(result.joint_angles.knee_asymmetry, result.joint_angles.hip_asymmetry)
            )
        else:
            self.pipeline_knee.append(0.0)
            self.pipeline_hip.append(0.0)
            self.pipeline_trunk.append(0.0)
            self.hip_adduction_l.append(0.0)
            self.hip_adduction_r.append(0.0)
            self.knee_valgus_l.append(0.0)
            self.knee_valgus_r.append(0.0)
            self.knee_ankle_sep_ratio.append(1.0)
            self.bilateral_asymmetry.append(0.0)

        # Rep events
        if result.rep_data is not None:
            self.rep_events.append((t, result.rep_data.rep_number))

        # Raw 3D keypoint positions for JSON export
        self.frames_data.append({
            "timestamp": t,
            "frame_index": result.frame_index,
            "hip_l": {"x": hip_l.x, "y": hip_l.y, "z": hip_l.z},
            "hip_r": {"x": hip_r.x, "y": hip_r.y, "z": hip_r.z},
            "knee_l": {"x": knee_l.x, "y": knee_l.y, "z": knee_l.z},
            "knee_r": {"x": knee_r.x, "y": knee_r.y, "z": knee_r.z},
            "ankle_l": {"x": ankle_l.x, "y": ankle_l.y, "z": ankle_l.z},
            "ankle_r": {"x": ankle_r.x, "y": ankle_r.y, "z": ankle_r.z},
            "shoulder_l": {"x": shoulder_l.x, "y": shoulder_l.y, "z": shoulder_l.z},
            "shoulder_r": {"x": shoulder_r.x, "y": shoulder_r.y, "z": shoulder_r.z},
        })

    def record_fault(self, fault: FaultEvent) -> None:
        """Record a fault event for dashboard visualization."""
        self.fault_events.append({
            "timestamp": fault.timestamp,
            "frame_index": fault.frame_index,
            "fault_type": fault.fault_type,
            "severity": fault.severity.value,
            "severity_score": fault.severity_score,
            "message": fault.message,
            "rep_number": fault.rep_number,
        })

    def has_enough_data(self) -> bool:
        return len(self.timestamps) > 10

    def reset(self):
        self.timestamps.clear()
        self.hip_mid_y.clear()
        self.knee_angles.clear()
        self.hip_angles.clear()
        self.trunk_angles.clear()
        self.pipeline_knee.clear()
        self.pipeline_hip.clear()
        self.pipeline_trunk.clear()
        self.hip_adduction_l.clear()
        self.hip_adduction_r.clear()
        self.knee_valgus_l.clear()
        self.knee_valgus_r.clear()
        self.knee_ankle_sep_ratio.clear()
        self.bilateral_asymmetry.clear()
        self.rep_events.clear()
        self.fault_events.clear()
        self.frames_data.clear()
        # Note: thresholds intentionally NOT cleared — they persist across sets


# ---------------------------------------------------------------------------
# Set finalization
# ---------------------------------------------------------------------------
def _finalize_reps(frames_data, plot_export, seg_result, set_dir):
    for rep in seg_result.get("reps", []):
        rep_number = rep["rep_number"]
        rep_dir = str(Path(set_dir) / f"rep_{rep_number}")
        os.makedirs(rep_dir, exist_ok=True)
        label = f"Rep {rep_number}"
        s, e = rep["start_idx"], rep["end_idx"] + 1

        ts = np.array(plot_export["timestamps"][s:e])
        t_rel = ts - ts[0]
        hip_mid = np.array(plot_export["hip_position_cm"][s:e]) / 100.0
        vel_mid = np.array(plot_export["hip_velocity_cm_s"][s:e]) / 100.0
        knee_ang = np.array(plot_export["knee_angle_deg"][s:e])
        hip_ang = np.array(plot_export["hip_angle_deg"][s:e])
        trunk_ang = np.array(plot_export["trunk_angle_deg"][s:e])

        save_set_plots(
            label, ts, hip_mid, vel_mid,
            knee_ang, hip_ang, trunk_ang, [], rep_dir,
        )

        if plot_export["pipeline_knee_flexion_deg"]:
            pipe_knee = np.array(plot_export["pipeline_knee_flexion_deg"][s:e])
            pipe_hip = np.array(plot_export["pipeline_hip_flexion_deg"][s:e])
            pipe_trunk = np.array(plot_export["pipeline_trunk_flexion_deg"][s:e])
            adduction_l = np.array(plot_export["hip_adduction_l_deg"][s:e])
            adduction_r = np.array(plot_export["hip_adduction_r_deg"][s:e])
            valgus_l = np.array(plot_export["knee_valgus_l_deg"][s:e])
            valgus_r = np.array(plot_export["knee_valgus_r_deg"][s:e])
            kasr = np.array(plot_export["knee_ankle_sep_ratio"][s:e])
            bilat_asym = np.array(plot_export["bilateral_asymmetry_deg"][s:e])

            save_pipeline_angles_plot(
                label, t_rel, 180.0 - pipe_knee, pipe_hip, pipe_trunk,
                [], ts, rep_dir,
            )
            save_hip_adduction_plot(label, t_rel, adduction_l, adduction_r, [], ts, rep_dir)
            save_knee_valgus_plot(
                label, t_rel, valgus_l, valgus_r, kasr,
                [], ts, plot_export["thresholds"], rep_dir,
            )
            save_bilateral_asymmetry_plot(label, t_rel, bilat_asym, [], ts, rep_dir)

        rep_faults = [
            fe for fe in plot_export["fault_events"]
            if rep["start_time_s"] <= fe["time_s"] <= rep["end_time_s"]
        ]
        metrics = {**rep, "fault_events": rep_faults}
        metrics_path = str(Path(rep_dir) / "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(nan_to_none(metrics), f, indent=2)
        print(f"  Saved: {metrics_path}")

        rep_export = {"rep_number": rep_number, "frames": frames_data[s:e]}
        data_path = str(Path(rep_dir) / "data.json")
        with open(data_path, "w") as f:
            json.dump(nan_to_none(rep_export), f, indent=2, default=str)
        print(f"  Saved: {data_path}")


def finalize_set(
    collector: SetDataCollector,
    set_number: int,
    out_dir: str,
) -> Optional[dict]:
    """Finalize a set: smooth data, generate plots, save JSON, run segmentation.

    Writes into ``<out_dir>/set_<set_number>/``, with per-rep breakdowns in
    ``rep_<n>/`` subfolders. Returns the plot_export dict on success, or None
    if not enough data. The collector is reset after finalization.
    """
    if not collector.has_enough_data():
        print(f"  Set {set_number}: not enough data for plots ({len(collector.timestamps)} frames)")
        collector.reset()
        return None

    set_dir = str(Path(out_dir) / f"set_{set_number}")
    os.makedirs(set_dir, exist_ok=True)
    set_label = f"Set {set_number}"

    timestamps = np.array(collector.timestamps)
    raw_mid = np.array(collector.hip_mid_y)

    # SMA starts at 1.5 s into the set
    t_rel = timestamps - timestamps[0]
    sma_start = int(np.searchsorted(t_rel, 1.5))

    # Smooth hip position and compute velocity
    hip_mid = smooth_1d(raw_mid, sma_window=2, sma_start=sma_start)
    raw_vel = np.gradient(hip_mid, timestamps)
    vel_mid = smooth_1d(raw_vel, median_window=1, sma_window=3, sma_start=sma_start)

    # Smooth joint angles
    knee_ang = smooth_1d(np.array(collector.knee_angles), sma_window=3, sma_start=sma_start)
    hip_ang = smooth_1d(np.array(collector.hip_angles), sma_window=3, sma_start=sma_start)
    trunk_ang = smooth_1d(np.array(collector.trunk_angles), sma_window=3, sma_start=sma_start)

    # Generate set plots (hip position, hip velocity, joint angles)
    save_set_plots(
        set_label, timestamps, hip_mid, vel_mid,
        knee_ang, hip_ang, trunk_ang, collector.rep_events, set_dir,
    )

    # Smooth and plot pipeline joint angles
    pipe_knee = pipe_hip = pipe_trunk = None
    adduction_l = adduction_r = bilat_asym = None
    valgus_l = valgus_r = kasr = None
    if collector.pipeline_knee:
        pipe_knee = smooth_1d(np.array(collector.pipeline_knee), sma_window=3, sma_start=sma_start)
        pipe_hip = smooth_1d(np.array(collector.pipeline_hip), sma_window=3, sma_start=sma_start)
        pipe_trunk = smooth_1d(np.array(collector.pipeline_trunk), sma_window=3, sma_start=sma_start)
        adduction_l = smooth_1d(np.array(collector.hip_adduction_l), sma_window=3, sma_start=sma_start)
        adduction_r = smooth_1d(np.array(collector.hip_adduction_r), sma_window=3, sma_start=sma_start)
        bilat_asym = smooth_1d(np.array(collector.bilateral_asymmetry), sma_window=3, sma_start=sma_start)
        save_pipeline_angles_plot(
            set_label, t_rel, 180.0 - pipe_knee, pipe_hip, pipe_trunk,
            collector.rep_events, timestamps, set_dir,
        )
        save_hip_adduction_plot(
            set_label, t_rel, adduction_l, adduction_r,
            collector.rep_events, timestamps, set_dir,
        )
        valgus_l = smooth_1d(np.array(collector.knee_valgus_l), sma_window=3, sma_start=sma_start)
        valgus_r = smooth_1d(np.array(collector.knee_valgus_r), sma_window=3, sma_start=sma_start)
        kasr = np.array(collector.knee_ankle_sep_ratio)
        save_knee_valgus_plot(
            set_label, t_rel, valgus_l, valgus_r, kasr,
            collector.rep_events, timestamps, collector.thresholds, set_dir,
        )
        save_bilateral_asymmetry_plot(
            set_label, t_rel, bilat_asym,
            collector.rep_events, timestamps, set_dir,
        )

    # Save per-set raw JSON data
    set_export = {
        "set_number": set_number,
        "rep_events": [{"timestamp": t, "rep_number": r} for t, r in collector.rep_events],
        "frames": collector.frames_data,
    }
    data_path = str(Path(set_dir) / "data.json")
    with open(data_path, "w") as f:
        json.dump(nan_to_none(set_export), f, indent=2, default=str)
    print(f"  Saved: {data_path}")

    # Save smoothed plot data
    plot_export = {
        "set_number": set_number,
        "rep_events": [{"timestamp": t, "rep_number": r} for t, r in collector.rep_events],
        "timestamps": t_rel.tolist(),
        "hip_position_cm": (hip_mid * 100.0).tolist(),
        "hip_velocity_cm_s": (vel_mid * 100.0).tolist(),
        "knee_angle_deg": knee_ang.tolist(),
        "hip_angle_deg": hip_ang.tolist(),
        "trunk_angle_deg": trunk_ang.tolist(),
        "pipeline_knee_flexion_deg": pipe_knee.tolist() if pipe_knee is not None else [],
        "pipeline_hip_flexion_deg": pipe_hip.tolist() if pipe_hip is not None else [],
        "pipeline_trunk_flexion_deg": pipe_trunk.tolist() if pipe_trunk is not None else [],
        "hip_adduction_l_deg": adduction_l.tolist() if adduction_l is not None else [],
        "hip_adduction_r_deg": adduction_r.tolist() if adduction_r is not None else [],
        "knee_valgus_l_deg": valgus_l.tolist() if valgus_l is not None else [],
        "knee_valgus_r_deg": valgus_r.tolist() if valgus_r is not None else [],
        "knee_ankle_sep_ratio": kasr.tolist() if kasr is not None else [],
        "bilateral_asymmetry_deg": bilat_asym.tolist() if bilat_asym is not None else [],
        "fault_events": [
            {**fe, "time_s": fe["timestamp"] - timestamps[0]}
            for fe in collector.fault_events
        ],
        "thresholds": collector.thresholds,
    }
    plot_path = str(Path(set_dir) / "plot_data.json")
    with open(plot_path, "w") as f:
        json.dump(nan_to_none(plot_export), f, indent=2)
    print(f"  Saved: {plot_path}")

    # Rep segmentation analysis, plot, and report
    seg_result = segment_set(plot_export)
    plot_segmentation(
        plot_export, seg_result,
        save_path=Path(set_dir) / "segmentation.png",
    )
    write_set_report(
        seg_result, set_number=set_number,
        save_path=Path(set_dir) / "report.md",
    )

    # Generate interactive HTML dashboard
    generate_set_dashboard(plot_export, seg_result, set_dir, set_number)

    # Per-rep breakdowns from segmentation boundaries
    _finalize_reps(collector.frames_data, plot_export, seg_result, set_dir)

    collector.reset()
    return plot_export
