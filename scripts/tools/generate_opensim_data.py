#!/usr/bin/env python3
"""
Synthetic Squat Data Generator using OpenSim

Generates labeled training data for the BiLSTM rep counter by:
1. Creating randomized squat motion trajectories (depth, speed, asymmetry)
2. Using OpenSim Rajagopal 2015 model for forward kinematics
3. Mapping body positions to COCO 17 keypoints
4. Extracting features via LandmarkFeatureExtractor
5. Windowing into 30-frame sequences with per-frame labels

Usage:
    /Users/naiahoard/miniforge3/envs/biomech/bin/python scripts/generate_opensim_data.py \
        --num-sessions 200 --output data/dataset.npz

Requires: opensim (conda install -c opensim-org opensim)
"""

import argparse
import multiprocessing as mp
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

# Add src/ to path for biomechanics imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import opensim as osim

from biomechanics.utils.types import Skeleton3D, DEPTH_CLASS_NAMES, depth_class_from_angle
from biomechanics.ml.feature_extractor import LandmarkFeatureExtractor


# =============================================================================
# Constants
# =============================================================================

RAJAGOPAL_MODEL_PATH = (
    Path("/Users/naiahoard/miniforge3/envs/biomech/share/doc/OpenSim/Code/Python")
    / "OpenSenseExample" / "Rajagopal_2015.osim"
)

FPS = 30
WINDOW_SIZE = 30
STRIDE = 5

# COCO 17 + foot_index + heel keypoint indices (21-keypoint skeleton layout)
NUM_KEYPOINTS = 21
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
L_FOOT_INDEX, R_FOOT_INDEX = 17, 18
L_HEEL, R_HEEL = 19, 20

# 5-class labels are computed via depth_class_from_angle() from types.py


# =============================================================================
# OpenSim Model Wrapper
# =============================================================================

class OpenSimSquatModel:
    """Wraps the Rajagopal 2015 model for fast forward kinematics."""

    # OpenSim body names we need
    _BODY_NAMES = [
        "pelvis", "torso",
        "femur_r", "femur_l",
        "tibia_r", "tibia_l",
        "talus_r", "talus_l",
        "calcn_r", "calcn_l",
        "toes_r", "toes_l",
        "humerus_r", "humerus_l",
        "radius_r", "radius_l",
        "hand_r", "hand_l",
    ]

    # Coordinate names for squat DOFs
    _COORD_NAMES = [
        "hip_flexion_r", "hip_flexion_l",
        "knee_angle_r", "knee_angle_l",
        "ankle_angle_r", "ankle_angle_l",
    ]

    def __init__(self, model_path: str):
        self._model = osim.Model(str(model_path))
        self._state = self._model.initSystem()

        # Cache coordinate and body references for speed
        coord_set = self._model.getCoordinateSet()
        self._coords = {
            name: coord_set.get(name) for name in self._COORD_NAMES
        }

        body_set = self._model.getBodySet()
        self._bodies = {
            name: body_set.get(name) for name in self._BODY_NAMES
        }

    def set_pose(self, hip_flex_r: float, hip_flex_l: float,
                 knee_r: float, knee_l: float,
                 ankle_r: float, ankle_l: float):
        """Set joint angles (radians) and realize positions."""
        self._coords["hip_flexion_r"].setValue(self._state, hip_flex_r)
        self._coords["hip_flexion_l"].setValue(self._state, hip_flex_l)
        self._coords["knee_angle_r"].setValue(self._state, knee_r)
        self._coords["knee_angle_l"].setValue(self._state, knee_l)
        self._coords["ankle_angle_r"].setValue(self._state, ankle_r)
        self._coords["ankle_angle_l"].setValue(self._state, ankle_l)
        self._model.realizePosition(self._state)

    def get_body_pos(self, name: str) -> np.ndarray:
        """Get body center position in ground frame [x, y, z]."""
        p = self._bodies[name].getTransformInGround(self._state).p()
        return np.array([p[0], p[1], p[2]])

    def extract_coco17(self, limb_scales: dict = None) -> np.ndarray:
        """
        Map OpenSim body positions to COCO 17 + foot_index + heel keypoints.

        Args:
            limb_scales: Optional per-segment scaling to simulate different body
                proportions. Keys: 'torso', 'thigh', 'shin', 'arm'.
                Values are float multipliers (1.0 = default).

        Returns (21, 3) array in meters, Y-up coordinate system.
        """
        kpts = np.zeros((NUM_KEYPOINTS, 3))

        # Reference positions
        pelvis = self.get_body_pos("pelvis")
        torso = self.get_body_pos("torso")

        # Apply torso scale: stretch torso-pelvis offset
        if limb_scales and "torso" in limb_scales:
            torso_offset = torso - pelvis
            torso = pelvis + torso_offset * limb_scales["torso"]

        # Head keypoints (approximate offsets from torso top)
        kpts[NOSE]  = torso + np.array([0.02, 0.42, 0.0])
        kpts[L_EYE] = torso + np.array([0.02, 0.44, -0.03])
        kpts[R_EYE] = torso + np.array([0.02, 0.44, 0.03])
        kpts[L_EAR] = torso + np.array([0.0, 0.42, -0.06])
        kpts[R_EAR] = torso + np.array([0.0, 0.42, 0.06])

        # Shoulders: use humerus positions (near shoulder joint)
        # Scale shoulder offset from torso
        hum_l = self.get_body_pos("humerus_l")
        hum_r = self.get_body_pos("humerus_r")
        kpts[L_SHOULDER] = hum_l
        kpts[R_SHOULDER] = hum_r

        # Elbows + wrists
        rad_l = self.get_body_pos("radius_l")
        rad_r = self.get_body_pos("radius_r")
        arm_scale = limb_scales.get("arm", 1.0) if limb_scales else 1.0
        kpts[L_ELBOW] = hum_l + (rad_l - hum_l) * 0.5 * arm_scale
        kpts[R_ELBOW] = hum_r + (rad_r - hum_r) * 0.5 * arm_scale
        kpts[L_WRIST] = hum_l + (self.get_body_pos("hand_l") - hum_l) * arm_scale
        kpts[R_WRIST] = hum_r + (self.get_body_pos("hand_r") - hum_r) * arm_scale

        # Hips: femur body position (near hip joint)
        fem_l = self.get_body_pos("femur_l")
        fem_r = self.get_body_pos("femur_r")
        kpts[L_HIP] = fem_l
        kpts[R_HIP] = fem_r

        # Knees: scale thigh length (hip → knee segment)
        tib_l = self.get_body_pos("tibia_l")
        tib_r = self.get_body_pos("tibia_r")
        thigh_scale = limb_scales.get("thigh", 1.0) if limb_scales else 1.0
        kpts[L_KNEE] = fem_l + (tib_l - fem_l) * thigh_scale
        kpts[R_KNEE] = fem_r + (tib_r - fem_r) * thigh_scale

        # Ankles: scale shin length (knee → ankle segment)
        tal_l = self.get_body_pos("talus_l")
        tal_r = self.get_body_pos("talus_r")
        shin_scale = limb_scales.get("shin", 1.0) if limb_scales else 1.0
        kpts[L_ANKLE] = kpts[L_KNEE] + (tal_l - tib_l) * shin_scale
        kpts[R_ANKLE] = kpts[R_KNEE] + (tal_r - tib_r) * shin_scale

        # Foot index: toes body position (distal foot segment)
        kpts[L_FOOT_INDEX] = self.get_body_pos("toes_l")
        kpts[R_FOOT_INDEX] = self.get_body_pos("toes_r")

        # Heels: calcaneus body position
        kpts[L_HEEL] = self.get_body_pos("calcn_l")
        kpts[R_HEEL] = self.get_body_pos("calcn_r")

        return kpts


# =============================================================================
# Trajectory Generation
# =============================================================================

def generate_squat_trajectory(
    num_reps: int,
    max_knee_flex_deg: float,
    rep_duration: float,
    pause_duration: float,
    asymmetry_deg: float,
    fps: int = FPS,
) -> dict:
    """
    Generate joint angle trajectories for a squat session.

    Returns dict with arrays for each DOF (in radians) and per-frame labels.
    """
    max_knee_rad = np.radians(max_knee_flex_deg)
    asym_rad = np.radians(asymmetry_deg)

    # Coupled ratios (realistic squat kinematics)
    hip_knee_ratio = np.random.uniform(0.45, 0.75)  # hip flexes ~45-75% of knee
    ankle_knee_ratio = np.random.uniform(0.08, 0.18)  # small dorsiflexion

    frames_per_rep = int(rep_duration * fps)
    frames_pause = int(pause_duration * fps)

    all_knee_r, all_knee_l = [], []
    all_hip_r, all_hip_l = [], []
    all_ankle_r, all_ankle_l = [], []
    all_labels = []

    for _ in range(num_reps):
        # Squat phase: sin²(π * t / T) gives smooth descent-ascent
        t = np.linspace(0, 1, frames_per_rep)
        profile = np.sin(np.pi * t) ** 2

        knee_r = max_knee_rad * profile
        knee_l = (max_knee_rad + asym_rad) * profile
        hip_r = hip_knee_ratio * knee_r
        hip_l = hip_knee_ratio * knee_l
        ankle_r = ankle_knee_ratio * knee_r
        ankle_l = ankle_knee_ratio * knee_l

        # 5-class depth label per frame
        avg_knee_deg = np.degrees((knee_r + knee_l) / 2)
        labels = np.array([depth_class_from_angle(d) for d in avg_knee_deg], dtype=np.int64)

        all_knee_r.append(knee_r)
        all_knee_l.append(knee_l)
        all_hip_r.append(hip_r)
        all_hip_l.append(hip_l)
        all_ankle_r.append(ankle_r)
        all_ankle_l.append(ankle_l)
        all_labels.append(labels)

        # Pause at top
        if frames_pause > 0:
            all_knee_r.append(np.zeros(frames_pause))
            all_knee_l.append(np.zeros(frames_pause))
            all_hip_r.append(np.zeros(frames_pause))
            all_hip_l.append(np.zeros(frames_pause))
            all_ankle_r.append(np.zeros(frames_pause))
            all_ankle_l.append(np.zeros(frames_pause))
            all_labels.append(np.zeros(frames_pause, dtype=np.int64))

    return {
        "knee_r": np.concatenate(all_knee_r),
        "knee_l": np.concatenate(all_knee_l),
        "hip_r": np.concatenate(all_hip_r),
        "hip_l": np.concatenate(all_hip_l),
        "ankle_r": np.concatenate(all_ankle_r),
        "ankle_l": np.concatenate(all_ankle_l),
        "labels": np.concatenate(all_labels),
    }


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_session_features(
    osim_model: OpenSimSquatModel,
    trajectory: dict,
    body_scale: float,
    limb_scales: dict,
    noise_std: float,
    extractor: LandmarkFeatureExtractor,
) -> tuple:
    """
    Run OpenSim FK + feature extraction for an entire session.

    Returns (features, labels) where features is (N_frames, 14) and
    labels is (N_frames,).
    """
    n_frames = len(trajectory["labels"])
    features = np.zeros((n_frames, extractor.FEATURE_DIM))
    labels = trajectory["labels"]

    for i in range(n_frames):
        # Set pose
        osim_model.set_pose(
            hip_flex_r=trajectory["hip_r"][i],
            hip_flex_l=trajectory["hip_l"][i],
            knee_r=trajectory["knee_r"][i],
            knee_l=trajectory["knee_l"][i],
            ankle_r=trajectory["ankle_r"][i],
            ankle_l=trajectory["ankle_l"][i],
        )

        # Extract COCO 17 keypoints with per-limb body proportion scaling
        kpts = osim_model.extract_coco17(limb_scales=limb_scales)

        # Apply overall body scale
        kpts *= body_scale

        # Center at hip midpoint (like MediaPipe world coords)
        hip_mid = (kpts[L_HIP] + kpts[R_HIP]) / 2
        kpts -= hip_mid

        # Add noise
        if noise_std > 0:
            kpts += np.random.randn(*kpts.shape) * noise_std

        # Convert to Skeleton3D and extract features
        skeleton = Skeleton3D.from_numpy(kpts, timestamp=i / FPS, frame_index=i)
        features[i] = extractor.extract(skeleton)

    return features, labels


# =============================================================================
# Windowing
# =============================================================================

def window_sequences(features: np.ndarray, labels: np.ndarray,
                     window_size: int = WINDOW_SIZE,
                     stride: int = STRIDE) -> tuple:
    """
    Slide fixed-length windows over a session's frames.

    Returns (sequences, seq_labels) of shapes:
        (N_windows, window_size, feature_dim) and (N_windows, window_size)
    """
    n_frames = len(features)
    if n_frames < window_size:
        return np.empty((0, window_size, features.shape[1])), np.empty((0, window_size), dtype=np.int64)

    starts = range(0, n_frames - window_size + 1, stride)
    sequences = np.array([features[s:s + window_size] for s in starts])
    seq_labels = np.array([labels[s:s + window_size] for s in starts])
    return sequences, seq_labels


# =============================================================================
# Debug Visualization
# =============================================================================

def save_debug_plot(trajectory: dict, output_path: str):
    """Save a trajectory visualization for sanity checking."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping debug plot")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    knee_deg = np.degrees((trajectory["knee_r"] + trajectory["knee_l"]) / 2)
    labels = trajectory["labels"]
    frames = np.arange(len(knee_deg))

    ax.plot(frames, knee_deg, "k-", linewidth=1.5, label="Avg knee flexion (deg)")

    # Color bands for each depth class
    class_colors = {
        0: "#2ecc71",  # Standing — green
        1: "#f1c40f",  # Quarter — yellow
        2: "#e67e22",  # Half — orange
        3: "#3498db",  # Parallel — blue
        4: "#9b59b6",  # Deep — purple
    }
    for c, color in class_colors.items():
        name = DEPTH_CLASS_NAMES.get(c, f"Class {c}")
        mask = labels == c
        if mask.any():
            ax.fill_between(frames, 0, knee_deg.max() * 1.05, where=mask,
                            alpha=0.2, color=color, label=f"{name} ({c})")

    # Threshold lines
    for lo, _ in [(40, 60), (60, 80), (80, 100), (100, 130)]:
        ax.axhline(y=lo, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Knee Flexion (deg)")
    ax.set_title("Sample Squat Session — Trajectory + 5-Class Labels")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 140)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Debug plot saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic squat training data via OpenSim")
    p.add_argument("--num-sessions", type=int, default=300,
                    help="Number of squat sessions to generate (default: 300)")
    p.add_argument("--output", type=str, default="data/dataset.npz",
                    help="Output path for .npz file")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--model", type=str, default=str(RAJAGOPAL_MODEL_PATH),
                    help="Path to OpenSim .osim model file")
    return p.parse_args()


def _generate_session_params(session_idx: int, depth_bins, depth_weights, seed: int):
    """Pre-generate all random parameters for a single session (picklable)."""
    rng = np.random.RandomState(seed + session_idx)

    body_scale = rng.uniform(0.85, 1.15)
    limb_scales = {
        "torso": rng.uniform(0.88, 1.12),
        "thigh": rng.uniform(0.90, 1.10),
        "shin":  rng.uniform(0.90, 1.10),
        "arm":   rng.uniform(0.88, 1.12),
    }

    if session_idx < 10:
        max_knee_flex = rng.uniform(5, 15)
    else:
        bin_idx = rng.choice(len(depth_bins), p=depth_weights)
        lo, hi, _ = depth_bins[bin_idx]
        max_knee_flex = rng.uniform(lo, hi)

    rep_duration = rng.uniform(0.8, 5.0)
    num_reps = rng.randint(5, 25)
    noise_std = rng.uniform(0.0, 0.015)
    asymmetry = rng.uniform(0, 8)
    pause = rng.uniform(0.0, 0.5)

    return {
        "session_idx": session_idx,
        "body_scale": body_scale,
        "limb_scales": limb_scales,
        "max_knee_flex": max_knee_flex,
        "rep_duration": rep_duration,
        "num_reps": num_reps,
        "noise_std": noise_std,
        "asymmetry": asymmetry,
        "pause": pause,
    }


# Global per-worker state (initialized once per process)
_worker_model = None
_worker_extractor = None


def _init_worker(model_path: str):
    """Initialize OpenSim model and feature extractor in each worker process."""
    global _worker_model, _worker_extractor
    _worker_model = OpenSimSquatModel(model_path)
    _worker_extractor = LandmarkFeatureExtractor()


def _process_session(params: dict) -> tuple:
    """
    Process a single session in a worker process.

    Returns (session_idx, sequences, seq_labels, n_frames) or
            (session_idx, None, None, 0) if no sequences produced.
    """
    trajectory = generate_squat_trajectory(
        num_reps=params["num_reps"],
        max_knee_flex_deg=params["max_knee_flex"],
        rep_duration=params["rep_duration"],
        pause_duration=params["pause"],
        asymmetry_deg=params["asymmetry"],
    )

    features, labels = extract_session_features(
        _worker_model, trajectory, params["body_scale"],
        params["limb_scales"], params["noise_std"], _worker_extractor,
    )

    seq, seq_labels = window_sequences(features, labels)

    if len(seq) > 0:
        return (params["session_idx"], seq, seq_labels, len(features), trajectory)
    return (params["session_idx"], None, None, 0, trajectory)


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, mp.cpu_count() - 2)  # leave 2 cores for system
    print(f"Using {num_workers} worker processes (of {mp.cpu_count()} CPUs)")
    print(f"OpenSim model: {args.model}")
    print(f"Sessions: {args.num_sessions}")

    # Depth categories
    depth_bins = [
        (100, 130, 0.35),
        (80, 100, 0.30),
        (60, 80, 0.20),
        (40, 60, 0.15),
    ]
    depth_weights = np.array([w for _, _, w in depth_bins])
    depth_weights /= depth_weights.sum()

    # Pre-generate all session parameters (deterministic from seed)
    print("Pre-generating session parameters...")
    all_params = [
        _generate_session_params(i, depth_bins, depth_weights, args.seed)
        for i in range(args.num_sessions)
    ]

    t_start = time.time()
    all_sequences = []
    all_labels = []
    total_frames = 0
    debug_trajectory_saved = False

    print(f"Processing {args.num_sessions} sessions across {num_workers} workers...")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(args.model,),
    ) as pool:
        for result in pool.imap_unordered(_process_session, all_params):
            session_idx, seq, seq_labels, n_frames, trajectory = result

            # Save debug plot for first session that comes back
            if not debug_trajectory_saved and trajectory is not None:
                debug_path = str(output_path.parent / "debug_trajectory.png")
                save_debug_plot(trajectory, debug_path)
                debug_trajectory_saved = True

            if seq is not None:
                all_sequences.append(seq)
                all_labels.append(seq_labels)
                total_frames += n_frames

            done = len(all_sequences)
            if done % 10 == 0 and done > 0:
                elapsed = time.time() - t_start
                fps_rate = total_frames / elapsed if elapsed > 0 else 0
                print(
                    f"  Completed {done}/{args.num_sessions} sessions | "
                    f"Sequences: {sum(len(s) for s in all_sequences):,} | "
                    f"Frames: {total_frames:,} | "
                    f"{fps_rate:.0f} frames/sec"
                )

    # Concatenate all sessions
    sequences = np.concatenate(all_sequences, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    # Print statistics
    elapsed = time.time() - t_start
    n_total = labels.size

    print(f"\n{'='*60}")
    print(f"Dataset generated in {elapsed:.1f}s")
    print(f"  Sequences:  {sequences.shape[0]:,}")
    print(f"  Shape:      sequences={sequences.shape}, labels={labels.shape}")
    print(f"  Frames:     {total_frames:,} total")
    print(f"  Per-class distribution:")
    for c in range(int(labels.max()) + 1):
        count = (labels == c).sum()
        pct = count / n_total * 100
        name = DEPTH_CLASS_NAMES.get(c, f"Class {c}")
        print(f"    {c} ({name:>9s}): {count:>10,} ({pct:.1f}%)")
    print(f"  Features:   min={sequences.min():.3f}, max={sequences.max():.3f}, "
          f"mean={sequences.mean():.3f}, std={sequences.std():.3f}")
    print(f"  Any NaN:    {np.isnan(sequences).any()}")
    print(f"  Any Inf:    {np.isinf(sequences).any()}")

    # Save
    np.savez_compressed(
        str(output_path),
        sequences=sequences,
        labels=labels,
    )
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
