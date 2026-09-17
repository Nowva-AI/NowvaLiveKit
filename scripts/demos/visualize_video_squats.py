#!/usr/bin/env python3
"""
Live-capture squat visualizer.

Flow:
  1. Run the production BiomechanicsPipeline (single-camera) with a live preview
  2. Wait for the pipeline readiness gate to pass
  3. Record until 5 reps are detected (pipeline rep counter; body measured during reps)
  4. Save video, generate 3D replay HTML, open in browser

Usage:
    python scripts/visualize_video_squats.py
    python scripts/visualize_video_squats.py --output my_session.mp4 --camera 0
    python scripts/visualize_video_squats.py --refit
    python scripts/visualize_video_squats.py --refit recordings/squat_20260101_120000.session.json
    python scripts/visualize_video_squats.py --shoe-size-eur 46
"""

import argparse
import json
import os
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import numpy as np

from biomechanics.config import load_pipeline_config
from biomechanics.pipeline import BiomechanicsPipeline


def eur_size_to_foot_length_m(eur_size: float) -> float:
    """ISO/TS 19407: foot length in metres from EU shoe size."""
    return (eur_size - 2) * 0.00667


COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (16, 18),
]

TARGET_REPS = 5
VIEWER_KEYPOINTS = 19  # the Three.js replay draws COCO-17 + toes; heels are not rendered
SESSION_VERSION = 1
LAST_SESSION_POINTER = "last_session.path"


def draw_skeleton(frame, skeleton_2d, color=(0, 255, 0)):
    if skeleton_2d is None:
        return
    for i, j in COCO_CONNECTIONS:
        kp1, kp2 = skeleton_2d.keypoints[i], skeleton_2d.keypoints[j]
        if kp1.confidence > 0.3 and kp2.confidence > 0.3:
            cv2.line(frame, (int(kp1.x), int(kp1.y)),
                     (int(kp2.x), int(kp2.y)), color, 2)
    for kp in skeleton_2d.keypoints:
        if kp.confidence > 0.3:
            cv2.circle(frame, (int(kp.x), int(kp.y)), 5, color, -1)


def draw_banner(frame, text, color=(40, 40, 40), text_color=(255, 255, 255)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)


def draw_status(frame, lines, y_start=80):
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def extract_frame_data(skeleton_3d, angles, frame_idx):
    """Convert a single frame's 3D skeleton + angles to viewer format."""
    kpts_mp = skeleton_3d.to_numpy()[:VIEWER_KEYPOINTS]
    kpts_vis = np.zeros_like(kpts_mp)
    kpts_vis[:, 0] = kpts_mp[:, 2]   # vis_x = mp_z
    kpts_vis[:, 1] = -kpts_mp[:, 1]  # vis_y = -mp_y
    kpts_vis[:, 2] = -kpts_mp[:, 0]  # vis_z = -mp_x
    return {
        "kpts": kpts_vis.tolist(),
        "angles": {
            "knee_flex": angles.avg_knee_flexion,
            "knee_flex_l": angles.knee_flexion_l,
            "knee_flex_r": angles.knee_flexion_r,
            "trunk_flexion": angles.trunk_flexion,
            "knee_valgus_l": angles.knee_valgus_l,
            "knee_valgus_r": angles.knee_valgus_r,
            "dorsi_l": angles.ankle_dorsiflexion_l,
            "dorsi_r": angles.ankle_dorsiflexion_r,
            "hip_flex_l": angles.hip_flexion_l,
            "hip_flex_r": angles.hip_flexion_r,
            "shoulder_flex_l": angles.shoulder_flexion_l,
            "shoulder_flex_r": angles.shoulder_flexion_r,
            "elbow_flex_l": angles.elbow_flexion_l,
            "elbow_flex_r": angles.elbow_flexion_r,
        },
        "frame": frame_idx,
    }


def ground_and_center(rep_frames):
    for frame in rep_frames:
        if frame is None:
            continue
        kpts = np.array(frame["kpts"])
        foot_y = kpts[[15, 16, 17, 18], 1].min()
        kpts[:, 1] -= foot_y
        hip_mid_x = (kpts[11][0] + kpts[12][0]) / 2
        hip_mid_z = (kpts[11][2] + kpts[12][2]) / 2
        kpts[:, 0] -= hip_mid_x
        kpts[:, 2] -= hip_mid_z
        frame["kpts"] = kpts.tolist()


def add_phase_to_rep(rep_frames):
    """Add normalized phase (0=standing, ~1=max depth) to each frame."""
    max_kf = max(
        (f["angles"]["knee_flex"] for f in rep_frames if f is not None),
        default=1.0,
    )
    for f in rep_frames:
        if f is None:
            continue
        f["phase"] = f["angles"]["knee_flex"] / max(max_kf, 1.0)


def compute_baseline(rep_frames):
    peak_trunk_offset = 0.0
    peak_valgus = 0.0
    peak_knee_flex = 0.0
    peak_dorsi = 0.0
    for f in rep_frames:
        if f is None:
            continue
        a = f["angles"]
        peak_trunk_offset = max(peak_trunk_offset, 180 - a["trunk_flexion"])
        peak_valgus = max(peak_valgus, abs(a["knee_valgus_l"]), abs(a["knee_valgus_r"]))
        peak_knee_flex = max(peak_knee_flex, a["knee_flex"])
        peak_dorsi = max(peak_dorsi, a["dorsi_l"], a["dorsi_r"])
    return {
        "peakTrunkOffset": round(peak_trunk_offset, 2),
        "peakValgus": round(peak_valgus, 2),
        "peakKneeFlex": round(peak_knee_flex, 2),
        "peakDorsi": round(peak_dorsi, 2),
        "leanThresholds": {
            "mild": round(peak_trunk_offset + 10, 1),
            "moderate": round(peak_trunk_offset + 15, 1),
            "severe": round(peak_trunk_offset + 20, 1),
        },
        "valgusThresholds": {
            "mild": round(peak_valgus + 5, 1),
            "moderate": round(peak_valgus + 10, 1),
            "severe": round(peak_valgus + 15, 1),
        },
    }


def compute_athlete_params(frames_data, rep_boundaries, measured):
    """Reverse-compute athlete parameters for the synthetic visualizer.

    measured is SegmentLengthEstimator.to_athlete_params() (metres). Arms are
    not measured; they are assumed to follow the overall body scale."""
    if measured is None:
        return None

    REF_TORSO = 0.50
    REF_THIGH = 0.42
    REF_SHIN = 0.40
    REF_UPPER_ARM = 0.30
    REF_FOREARM = 0.26
    REF_SHOULDER_W = 0.36
    REF_FOOT = 0.26

    hip_width = measured["hip_width_m"]
    raw_torso = measured["torso_avg_m"] / REF_TORSO
    raw_thigh = measured["femur_avg_m"] / REF_THIGH
    raw_shin = measured["tibia_avg_m"] / REF_SHIN
    body_scale = (raw_torso + raw_thigh + raw_shin) / 3.0
    torso_ratio = raw_torso / body_scale
    thigh_ratio = raw_thigh / body_scale
    shin_ratio = raw_shin / body_scale

    upper_arm_avg = REF_UPPER_ARM * body_scale
    forearm_avg = REF_FOREARM * body_scale
    upper_arm_ratio = 1.0
    forearm_ratio = 1.0
    arm_ratio = 1.0

    shoulder_width = measured["shoulder_width_m"]
    shoulder_width_ratio = (shoulder_width / REF_SHOULDER_W) / body_scale

    foot_avg = measured["foot_avg_m"]
    foot_ratio = (foot_avg / REF_FOOT) / body_scale

    # Stance width & toe-out from standing frames (before first rep)
    first_rep_start = rep_boundaries[0][0] if rep_boundaries else len(frames_data)
    standing_frames = [f for f in frames_data[:first_rep_start] if f is not None]
    sample = standing_frames[-10:] if len(standing_frames) >= 10 else standing_frames

    stance_widths = []
    toe_outs = []

    for f in sample:
        kpts = np.array(f["kpts"])
        if len(kpts) < VIEWER_KEYPOINTS:
            continue

        l_ankle, r_ankle = kpts[15], kpts[16]
        ankle_dx = l_ankle[0] - r_ankle[0]
        ankle_dz = l_ankle[2] - r_ankle[2]
        ankle_xz_dist = np.sqrt(ankle_dx**2 + ankle_dz**2)
        stance_widths.append(ankle_xz_dist / hip_width)

        # Toe-out: ankle→foot_index projected onto ground plane vs forward
        # vis_x = mp_z which points backward (away from person), so forward = -x
        forward = np.array([-1.0, 0.0])
        for ankle_idx, foot_idx in [(15, 17), (16, 18)]:
            vec = kpts[foot_idx] - kpts[ankle_idx]
            vec_xz = np.array([vec[0], vec[2]])
            norm = np.linalg.norm(vec_xz)
            if norm > 1e-6:
                cos_a = np.clip(np.dot(vec_xz, forward) / norm, -1, 1)
                toe_outs.append(np.degrees(np.arccos(cos_a)))

    stance_width = float(np.median(stance_widths)) if stance_widths else 1.2
    toe_out = float(np.median(toe_outs)) if toe_outs else 15.0

    # Find peak depth frame and extract angles at that point
    peak_dorsi_ratio = 0.15
    max_knee = 0.0
    peak_trunk_offset = 0.0
    peak_valgus = 0.0
    peak_shoulder_flex = 0.0
    peak_elbow_flex = 0.0
    for f in frames_data:
        if f is None:
            continue
        a = f["angles"]
        if a["knee_flex"] > max_knee:
            max_knee = a["knee_flex"]
            avg_dorsi = (a["dorsi_l"] + a["dorsi_r"]) / 2.0
            if max_knee > 5:
                peak_dorsi_ratio = avg_dorsi / max_knee
            peak_trunk_offset = 180 - a["trunk_flexion"]
            peak_valgus = max(abs(a["knee_valgus_l"]), abs(a["knee_valgus_r"]))
            peak_shoulder_flex = (a["shoulder_flex_l"] + a["shoulder_flex_r"]) / 2.0
            peak_elbow_flex = (a["elbow_flex_l"] + a["elbow_flex_r"]) / 2.0

    return _round_athlete_params({
        "bodyScale": body_scale,
        "torsoRatio": torso_ratio,
        "thighRatio": thigh_ratio,
        "shinRatio": shin_ratio,
        "armRatio": arm_ratio,
        "upperArmRatio": upper_arm_ratio,
        "forearmRatio": forearm_ratio,
        "shoulderWidthRatio": shoulder_width_ratio,
        "footRatio": foot_ratio,
        "stanceWidth": stance_width,
        "toeOut": toe_out,
        "dorsiRatio": peak_dorsi_ratio,
        "maxKneeFlex": max_knee,
        "forwardLean": peak_trunk_offset,
        "kneeValgus": peak_valgus,
        "shoulderFlex": peak_shoulder_flex,
        "elbowFlex": peak_elbow_flex,
        "hip_width_m": hip_width,
        "femur_avg_m": measured["femur_avg_m"],
        "tibia_avg_m": measured["tibia_avg_m"],
        "torso_avg_m": measured["torso_avg_m"],
        "upper_arm_avg_m": upper_arm_avg,
        "forearm_avg_m": forearm_avg,
        "shoulder_width_m": shoulder_width,
        "foot_avg_m": foot_avg,
    })


def _round_athlete_params(params):
    return {
        "bodyScale": round(params["bodyScale"], 3),
        "torsoRatio": round(params["torsoRatio"], 3),
        "thighRatio": round(params["thighRatio"], 3),
        "shinRatio": round(params["shinRatio"], 3),
        "armRatio": round(params["armRatio"], 3),
        "upperArmRatio": round(params["upperArmRatio"], 3),
        "forearmRatio": round(params["forearmRatio"], 3),
        "shoulderWidthRatio": round(params["shoulderWidthRatio"], 3),
        "footRatio": round(params["footRatio"], 3),
        "stanceWidth": round(params["stanceWidth"], 2),
        "toeOut": round(params["toeOut"], 1),
        "dorsiRatio": round(params["dorsiRatio"], 3),
        "maxKneeFlex": round(params["maxKneeFlex"], 1),
        "forwardLean": round(params["forwardLean"], 1),
        "kneeValgus": round(params["kneeValgus"], 1),
        "shoulderFlex": round(params["shoulderFlex"], 1),
        "elbowFlex": round(params["elbowFlex"], 1),
        "hip_width_m": round(params["hip_width_m"], 4),
        "femur_avg_m": round(params["femur_avg_m"], 4),
        "tibia_avg_m": round(params["tibia_avg_m"], 4),
        "torso_avg_m": round(params["torso_avg_m"], 4),
        "upper_arm_avg_m": round(params["upper_arm_avg_m"], 4),
        "forearm_avg_m": round(params["forearm_avg_m"], 4),
        "shoulder_width_m": round(params["shoulder_width_m"], 4),
        "foot_avg_m": round(params["foot_avg_m"], 4),
    }


def process_captured_reps(frames_data, rep_boundaries, measured):
    """Turn raw capture into baseline, replay reps, and athlete params."""
    rep_frame_slices = []
    for start, end in rep_boundaries:
        rep_slice = [frame for frame in frames_data[start:end + 1] if frame is not None]
        rep_frame_slices.append(rep_slice)

    for rep_frames in rep_frame_slices:
        ground_and_center(rep_frames)
        add_phase_to_rep(rep_frames)

    baseline = compute_baseline(rep_frame_slices[0])
    athlete_params = compute_athlete_params(frames_data, rep_boundaries, measured)
    replay_reps = rep_frame_slices[1:]
    return baseline, replay_reps, athlete_params


def save_session(
    session_path,
    *,
    source_video,
    fps,
    baseline,
    athlete_params,
    replay_reps,
    rep_count,
):
    """Persist processed session data for --refit."""
    session_path = Path(session_path)
    payload = {
        "version": SESSION_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(source_video),
        "fps": fps,
        "baseline": baseline,
        "athlete_params": athlete_params,
        "replay_reps": replay_reps,
        "rep_count": rep_count,
    }
    session_path.write_text(json.dumps(payload, indent=2))
    pointer_path = session_path.parent / LAST_SESSION_POINTER
    pointer_path.write_text(str(session_path.resolve()) + "\n")
    return session_path


def resolve_last_session_path(recordings_dir):
    """Find the most recent .session.json from pointer or directory scan."""
    recordings_dir = Path(recordings_dir)
    pointer_path = recordings_dir / LAST_SESSION_POINTER
    if pointer_path.exists():
        pointed = Path(pointer_path.read_text().strip())
        if pointed.exists():
            return pointed

    candidates = sorted(
        recordings_dir.glob("*.session.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None


def load_session(session_path):
    """Load a saved session; exit with a helpful message if missing/invalid."""
    session_path = Path(session_path)
    if not session_path.exists():
        print(f"ERROR: Session file not found: {session_path}")
        sys.exit(1)

    try:
        payload = json.loads(session_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid session JSON ({session_path}): {exc}")
        sys.exit(1)

    version = payload.get("version")
    if version != SESSION_VERSION:
        print(
            f"ERROR: Unsupported session version {version!r} "
            f"(expected {SESSION_VERSION}). Re-run a full capture first."
        )
        sys.exit(1)

    required = ("baseline", "replay_reps", "fps")
    missing = [key for key in required if key not in payload]
    if missing:
        print(f"ERROR: Session file missing fields: {', '.join(missing)}")
        sys.exit(1)

    replay_reps = payload["replay_reps"]
    if not replay_reps:
        print("ERROR: Session has no replay reps. Re-run a full capture first.")
        sys.exit(1)

    return payload


def run_refit(session_path, html_path, open_browser, shoe_size_eur=46):
    """Regenerate HTML from a prior session without webcam capture."""
    payload = load_session(session_path)
    baseline = payload["baseline"]
    replay_reps = payload["replay_reps"]
    fps = payload["fps"]
    athlete_params = payload.get("athlete_params")

    print("=" * 50)
    print("  SQUAT REFIT (from saved session)")
    print("=" * 50)
    print(f"  Session → {session_path}")
    if payload.get("source_video"):
        print(f"  Source  → {payload['source_video']}")
    print(f"  Reps    → {payload.get('rep_count', len(replay_reps) + 1)} total "
          f"({len(replay_reps)} replay)")
    print(f"  HTML    → {html_path}")
    print("=" * 50)

    if athlete_params:
        print("  Athlete params (from session):")
        print(f"    Stance width: {athlete_params['stanceWidth']}x  "
              f"Toe-out: {athlete_params['toeOut']}°")
        print(f"    Dorsi ratio: {athlete_params['dorsiRatio']}  "
              f"Body scale: {athlete_params['bodyScale']}")
    else:
        print("  WARNING: No athlete params in session; sandbox sliders won't pre-fill.")

    foot_length_m = eur_size_to_foot_length_m(shoe_size_eur)
    print(f"  Shoe size: EU {shoe_size_eur} → foot length {foot_length_m * 100:.1f} cm")

    html = build_html(
        baseline, replay_reps, fps, athlete_params, foot_length_m=foot_length_m,
    )
    html_path.write_text(html)
    print(f"\nHTML saved: {html_path}")

    if open_browser:
        webbrowser.open(f"file://{html_path.resolve()}")


def run_capture(camera_id, video_output_path):
    """Run the production pipeline: readiness gate → record reps → return data."""
    os.environ["NOWVA_MULTI_CAMERA"] = "false"
    config = load_pipeline_config()
    config.capture.device_id = camera_id
    fps = float(config.target_fps)
    frame_budget_s = 1.0 / fps

    pipeline = BiomechanicsPipeline(config, exercise_name="squat")
    pipeline.preload_pose_model()

    print(f"Camera {camera_id}: pipeline paced at {fps:.0f} fps")
    print("Stand back so your full body is visible.")
    print("Recording starts automatically once the readiness gate passes. Q=quit\n")

    # State
    state = "calibrating"  # calibrating → recording → done
    video_writer = None
    frames_data = []
    reps = []
    rep_boundaries = []
    current_rep_start = None
    prev_in_rep = False
    rec_frame_idx = 0
    rec_start_time = None

    def _shutdown():
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        pipeline.release()

    while True:
        frame_start = time.time()
        result = pipeline.process_frame()
        frame = pipeline.last_frame
        if frame is None:
            time.sleep(frame_budget_s)
            continue

        display = frame.copy()
        angles = result.joint_angles

        # --- CALIBRATING: the pipeline's readiness gate decides ---
        if state == "calibrating":
            if pipeline.is_ready:
                state = "recording"
                rec_start_time = time.time()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (w, h))
                print("Readiness gate passed — recording started automatically.")
                print(f"Recording → {video_output_path}")

            draw_skeleton(display, result.skeleton_2d, (100, 200, 255))
            passes, required = pipeline._readiness_gate.progress
            failure = pipeline._readiness_gate.last_failure
            draw_banner(display, f"WAITING FOR READINESS GATE  {passes}/{required}",
                        (40, 60, 80), (200, 220, 255))
            status = [
                (f"Gate check: {failure or 'passing'}", (180, 180, 200)),
                ("Stand still with full body visible", (150, 150, 170)),
            ]
            draw_status(display, status)

        # --- RECORDING ---
        elif state == "recording":
            elapsed = time.time() - rec_start_time
            video_writer.write(frame)

            if result.skeleton_3d is not None and angles is not None:
                in_rep = pipeline.rep_counter.in_rep
                if in_rep and not prev_in_rep:
                    current_rep_start = rec_frame_idx
                if not in_rep and prev_in_rep and current_rep_start is not None:
                    rep_boundaries.append((current_rep_start, rec_frame_idx))
                    current_rep_start = None
                prev_in_rep = in_rep

                if result.rep_data is not None:
                    reps.append(result.rep_data)
                    print(f"  Rep {result.rep_data.rep_number}: depth={result.rep_data.max_depth_angle:.1f}°")

                frames_data.append(extract_frame_data(result.skeleton_3d, angles, rec_frame_idx))
            else:
                frames_data.append(None)

            rec_frame_idx += 1

            # Check if we have enough reps
            if len(reps) >= TARGET_REPS:
                state = "done"
                print(f"\n{TARGET_REPS} reps captured!")

            skel_color = (0, 200, 255) if pipeline.rep_counter.in_rep else (0, 255, 0)
            draw_skeleton(display, result.skeleton_2d, skel_color)

            phase_name = pipeline.rep_counter.phase.upper()
            draw_banner(display,
                        f"RECORDING  Rep {len(reps)}/{TARGET_REPS}  "
                        f"[{phase_name}]  {elapsed:.1f}s",
                        (80, 20, 20), (255, 100, 100))
            if pipeline.body_calibration.is_complete:
                body_line = "Body measured"
            else:
                measured, required = pipeline.body_calibration.progress
                body_line = f"Measuring body: {measured}/{required}"
            status = [(body_line, (200, 200, 220))]
            if angles is not None:
                status.append((f"Knee: {angles.avg_knee_flexion:.1f}°  "
                               f"Trunk: {angles.trunk_flexion:.1f}°",
                               (200, 200, 220)))
            draw_status(display, status)

        # --- DONE ---
        elif state == "done":
            draw_banner(display, f"DONE — {len(reps)} reps captured. Processing...",
                        (20, 60, 20), (100, 255, 100))
            cv2.imshow("Squat Capture", display)
            cv2.waitKey(500)
            break

        cv2.imshow("Squat Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit.")
            _shutdown()
            sys.exit(0)

        # Pace to the pipeline's target fps so video frames match frames_data
        remaining_s = frame_budget_s - (time.time() - frame_start)
        if remaining_s > 0:
            time.sleep(remaining_s)

    measured = pipeline.body_calibration.to_athlete_params()
    if measured is None:
        print("WARNING: body measurement did not complete; sandbox sliders won't pre-fill.")
    _shutdown()

    return frames_data, reps, rep_boundaries, fps, measured


def build_html(
    baseline,
    replay_reps_data,
    fps,
    athlete_params=None,
    foot_length_m=None,
    diagnosis_data=None,
):
    if foot_length_m is None:
        foot_length_m = eur_size_to_foot_length_m(46)
    data_json = json.dumps({
        "baseline": baseline,
        "reps": replay_reps_data,
        "fps": fps,
        "athleteParams": athlete_params,
        "footLengthM": foot_length_m,
        "shoeSizeEur": round(
            foot_length_m / 0.00667 + 2,
            1,
        ),
        "diagnosis": diagnosis_data,
    })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Squat Video Replay</title>
<script type="importmap">
{{
    "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
    }}
}}
</script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a1a; color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    overflow: hidden; height: 100vh; display: flex;
}}
#scene-container {{ flex: 1; position: relative; z-index: 1; min-width: 0; }}
#three-canvas {{ display: block; width: 100%; height: 100%; }}
#controls {{
    width: 380px; flex-shrink: 0; position: relative; z-index: 20;
    background: #12122a; border-left: 1px solid #2a2a4a;
    overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px;
    pointer-events: auto;
}}
h1 {{ font-size: 18px; font-weight: 600; color: #a0a0ff; margin-bottom: 4px; }}
.section {{
    background: #1a1a35; border: 1px solid #2a2a4a; border-radius: 8px; padding: 14px;
    transition: opacity 0.3s;
}}
.section-title {{
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}}
.section-title .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.baseline .section-title {{ color: #2ecc71; }}  .baseline .dot {{ background: #2ecc71; }}
.angles .section-title {{ color: #a0a0ff; }}    .angles .dot {{ background: #a0a0ff; }}
.faults .section-title {{ color: #ff6b6b; }}    .faults .dot {{ background: #ff6b6b; }}
.playback .section-title {{ color: #ffd93d; }}  .playback .dot {{ background: #ffd93d; }}
.mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.8; }}
.lbl {{ color: #888; }} .val {{ color: #a0a0ff; }}
.val-green {{ color: #2ecc71; font-weight: 600; }}
.severity-indicator {{
    font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 4px; font-weight: 600;
}}
.sev-none {{ background: #1a3a2a; color: #2ecc71; }}
.sev-mild {{ background: #3a3a1a; color: #f1c40f; }}
.sev-moderate {{ background: #3a2a1a; color: #e67e22; }}
.sev-severe {{ background: #3a1a1a; color: #e74c3c; }}
.anim-controls {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
.anim-controls button {{
    width: 32px; height: 32px; border-radius: 50%; border: 1px solid #3a3a5a;
    background: #2a2a4a; color: #e0e0e0; font-size: 14px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}}
.anim-controls button:hover {{ background: #3a3a5a; }}
.slider-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
.slider-row label {{ font-size: 12px; color: #b0b0cc; min-width: 90px; flex-shrink: 0; }}
.slider-row input[type="range"] {{
    flex: 1; -webkit-appearance: none; height: 4px; background: #2a2a4a;
    border-radius: 2px; outline: none;
}}
.slider-row input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: #6060ff; cursor: pointer;
}}
.slider-row .value {{
    font-size: 12px; color: #8080cc; min-width: 42px; text-align: right;
    font-family: 'SF Mono', 'Fira Code', monospace;
}}
#info-overlay {{
    position: absolute; top: 12px; left: 12px;
    background: rgba(18, 18, 42, 0.85); border: 1px solid #2a2a4a;
    border-radius: 8px; padding: 10px 14px; font-size: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace; line-height: 1.6;
    pointer-events: none;
}}
#info-overlay .lbl {{ color: #888; }} #info-overlay .val {{ color: #a0a0ff; }}
#ground-label {{
    position: absolute; bottom: 12px; left: 12px; font-size: 11px;
    color: #555; pointer-events: none;
}}
.rep-btn {{
    padding: 6px 14px; border: 1px solid #3a3a5a; border-radius: 6px;
    background: #2a2a4a; color: #b0b0cc; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
}}
.rep-btn:hover {{ background: #3a3a5a; }}
.rep-btn.active {{ background: #4040cc; color: white; border-color: #6060ff; }}
.athlete .section-title {{ color: #c090ff; }} .athlete .dot {{ background: #c090ff; }}
.sandbox .section-title {{ color: #4ecdc4; }} .sandbox .dot {{ background: #4ecdc4; }}
.barbell-s .section-title {{ color: #f0a040; }} .barbell-s .dot {{ background: #f0a040; }}
.mode-tab {{
    padding: 5px 14px; border: 1px solid #3a3a5a; border-radius: 6px;
    background: #2a2a4a; color: #b0b0cc; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
}}
.mode-tab:hover {{ background: #3a3a5a; }}
.mode-tab.active {{ background: #4040cc; color: white; border-color: #6060ff; }}
.threshold-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 8px 0; }}
.threshold-bar .band {{ height: 100%; }}
.threshold-legend {{ display: flex; gap: 8px; font-size: 10px; margin-bottom: 4px; }}
.threshold-legend span {{ padding: 1px 6px; border-radius: 3px; }}
.legend-ok {{ background: #1a3a2a; color: #2ecc71; }}
.legend-mild {{ background: #3a3a1a; color: #f1c40f; }}
.legend-moderate {{ background: #3a2a1a; color: #e67e22; }}
.legend-severe {{ background: #3a1a1a; color: #e74c3c; }}
.balance-ok {{ color: #2ecc71; font-weight: 600; }}
.balance-bad {{ color: #ff4444; font-weight: 600; }}
.btn-primary {{
    padding: 10px 16px; border: none; border-radius: 8px; cursor: pointer;
    background: linear-gradient(135deg, #5050cc, #4040aa); color: white;
    font-size: 14px; font-weight: 600; transition: background 0.2s;
    pointer-events: auto; position: relative; z-index: 1;
}}
.btn-primary:hover {{ background: linear-gradient(135deg, #6060dd, #5050bb); }}
.btn-primary:disabled {{ opacity: 0.45; cursor: not-allowed; }}
.balance-section .section-title {{ color: #f0a040; }}
.balance-section .dot {{ background: #f0a040; }}
#sb-balance-header {{ display: none; }}
#sb-balance-header.visible {{ display: block; }}
/* ── Diagnosis Panel ── */
.diagnosis .section-title {{ color: #4ecdc4; }} .diagnosis .dot {{ background: #4ecdc4; }}
.confidence-badge {{
    display: inline-block; background: #1a1a35; border: 1px solid #2a2a4a;
    border-radius: 6px; padding: 4px 12px; font-size: 12px; font-weight: 600;
}}
.tier-section {{
    background: #1a1a35; border: 1px solid #2a2a4a; border-radius: 8px;
    overflow: hidden; margin-bottom: 8px;
}}
.tier-header {{
    padding: 10px 12px; cursor: pointer; display: flex; align-items: center;
    gap: 8px; user-select: none; transition: background 0.15s;
}}
.tier-header:hover {{ background: #222245; }}
.tier-icon {{
    font-size: 10px; transition: transform 0.2s; display: inline-block;
    width: 14px; text-align: center; color: #888;
}}
.tier-icon.expanded {{ transform: rotate(90deg); }}
.tier-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }}
.tier-count {{ font-size: 11px; color: #777; margin-left: auto; }}
.tier-body {{ padding: 0 12px 12px; display: none; }}
.tier-body.visible {{ display: block; }}
.tier-section[data-tier="1"] .tier-label {{ color: #4ecdc4; }}
.tier-section[data-tier="2"] .tier-label {{ color: #f1c40f; }}
.tier-section[data-tier="3"] .tier-label {{ color: #e67e22; }}
.tier-section[data-tier="0"] .tier-label {{ color: #a0a0ff; }}
.cause-card {{
    background: #0f0f25; border: 1px solid #252545; border-radius: 6px;
    padding: 10px 12px; margin-top: 8px;
}}
.cause-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }}
.cause-id {{ font-size: 12px; font-weight: 600; color: #ccc; }}
.score-bar-container {{
    width: 50px; height: 6px; background: #1a1a35; border-radius: 3px;
    overflow: hidden; flex-shrink: 0;
}}
.score-bar-fill {{ height: 100%; border-radius: 3px; }}
.cause-explanation {{ font-size: 11px; color: #aaa; line-height: 1.5; }}
.cause-evidence {{ font-size: 10px; color: #666; margin-top: 4px; }}
.diag-morph-controls {{
    display: flex; align-items: center; gap: 8px; margin-top: 8px;
}}
.diag-morph-controls button {{
    width: 32px; height: 32px; border-radius: 50%; border: 1px solid #3a3a5a;
    background: #2a2a4a; color: #e0e0e0; font-size: 14px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}}
.no-issues {{ color: #2ecc71; font-size: 13px; padding: 12px; text-align: center; }}
/* ── Quality score ── */
.quality-composite {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 10px; }}
.quality-number {{ font-size: 28px; font-weight: 700; line-height: 1; }}
.quality-label {{ font-size: 11px; color: #888; }}
.quality-sub-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }}
.quality-sub-label {{ font-size: 11px; color: #888; width: 90px; flex-shrink: 0; }}
.quality-sub-bar-bg {{ flex: 1; height: 8px; background: #1a1a35; border-radius: 4px; overflow: hidden; }}
.quality-sub-bar {{ height: 100%; border-radius: 4px; min-width: 2px; transition: width 0.3s; }}
.quality-sub-val {{ font-size: 10px; color: #aaa; width: 36px; text-align: right; flex-shrink: 0; }}
.set-score-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; font-size: 12px; }}
.set-score-mean {{ font-size: 22px; font-weight: 700; }}
.set-score-detail {{ font-size: 11px; color: #888; }}
.set-score-best {{ color: #2ecc71; }}
.set-score-trend {{ font-size: 13px; }}
/* ── Sparkline bars ── */
.sparkline-group {{ margin-bottom: 14px; }}
.sparkline-label {{ font-size: 11px; font-weight: 600; color: #888; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
.sparkline-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }}
.sparkline-rep {{ font-size: 10px; color: #666; width: 26px; text-align: right; flex-shrink: 0; }}
.sparkline-bar-bg {{ flex: 1; height: 10px; background: #1a1a35; border-radius: 5px; overflow: hidden; }}
.sparkline-bar {{ height: 100%; border-radius: 5px; min-width: 2px; }}
.sparkline-val {{ font-size: 10px; color: #aaa; width: 44px; text-align: right; flex-shrink: 0; }}
/* ── Fault map grid ── */
.fault-map-row {{ display: flex; align-items: center; gap: 4px; margin-bottom: 4px; }}
.fault-map-header {{ font-size: 10px; color: #666; margin-bottom: 8px; }}
.fault-map-label {{ font-size: 11px; color: #aaa; min-width: 90px; max-width: 120px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.fault-map-cell {{ width: 28px; text-align: center; flex-shrink: 0; font-size: 10px; }}
.fault-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; }}
.fault-dot.present {{ background: #e74c3c; box-shadow: 0 0 4px #e74c3c66; }}
.fault-dot.absent {{ background: #1a1a35; border: 1px solid #2a2a4a; }}
/* ── Performance trend ── */
.trend-line {{ font-size: 11px; margin-bottom: 5px; line-height: 1.5; }}
.trend-bad {{ color: #e74c3c; }}
.trend-good {{ color: #2ecc71; }}
.trend-neutral {{ color: #4ecdc4; }}
.trend-warn {{ color: #f1c40f; }}
</style>
</head>
<body>
<div id="scene-container">
    <canvas id="three-canvas"></canvas>
    <div id="info-overlay"></div>
    <div id="ground-label">Drag to orbit | Scroll to zoom</div>
</div>
<div id="controls">
    <div>
        <h1>Squat Replay</h1>
        <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; align-items:center;">
            <button class="mode-tab active" id="tab-replay">Replay</button>
            <button class="mode-tab" id="tab-sandbox">Sandbox</button>
            <button class="mode-tab" id="tab-diagnosis" style="display:none">Diagnosis</button>
            <button type="button" id="sb-balance-btn" class="btn-primary" style="display:none; min-width:100px;">Balance</button>
        </div>
        <div id="sb-balance-header">
            <p class="mono" id="sb-balance-status" style="margin-top:6px; font-size:11px; color:#888">
                Click Balance to enable parameter controls
            </p>
        </div>
    </div>

    <!-- ======== REPLAY PANEL ======== -->
    <div id="replay-panel">
        <div class="section baseline">
            <div class="section-title"><span class="dot"></span> Baseline (Rep 1)</div>
            <div class="mono" id="baseline-info"></div>
        </div>
        <div class="section athlete">
            <div class="section-title"><span class="dot"></span> Athlete Stats</div>
            <div class="mono" id="athlete-info"></div>
        </div>
        <div class="section playback">
            <div class="section-title"><span class="dot"></span> Playback</div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;" id="rep-buttons"></div>
            <div class="anim-controls">
                <button id="play-btn" title="Play/Pause">&#9646;&#9646;</button>
                <input type="range" id="frame-scrubber" min="0" max="100" value="0" step="1">
                <span class="value" id="frame-val" style="min-width:50px;">0/0</span>
            </div>
            <div class="slider-row">
                <label>Speed</label>
                <input type="range" id="speed-slider" min="0.1" max="3.0" value="1.0" step="0.1">
                <span class="value" id="speed-val">1.0x</span>
            </div>
        </div>
        <div class="section angles">
            <div class="section-title"><span class="dot"></span> Live Angles</div>
            <div class="mono" id="angles-info"></div>
        </div>
        <div class="section faults">
            <div class="section-title"><span class="dot"></span> Fault Classification</div>
            <div class="mono" id="faults-info"></div>
        </div>
    </div>

    <!-- ======== SANDBOX PANEL ======== -->
    <div id="sandbox-panel" style="display:none">
        <div class="section">
            <div class="section-title"><span class="dot"></span> Depth Check</div>
            <div class="mono" id="sb-depth-check"></div>
        </div>
        <div class="section balance-section">
            <div class="section-title"><span class="dot"></span> Balance</div>
            <p class="mono" style="font-size:11px; color:#888">Click Balance to lock COM over mid-foot and enable correction sliders.</p>
        </div>
        <div class="section sandbox">
            <div class="section-title"><span class="dot"></span> Stance</div>
            <div class="slider-row"><label>Stance width</label><input type="range" id="sb-stance-width" min="0.8" max="2.5" value="1.2" step="0.05" disabled><span class="value" id="sb-stance-width-val">1.20x</span></div>
            <div class="slider-row"><label>Toe-out angle</label><input type="range" id="sb-toe-out" min="0" max="45" value="15" step="1" disabled><span class="value" id="sb-toe-out-val">15°</span></div>
        </div>
        <div class="section sandbox">
            <div class="section-title"><span class="dot"></span> Dorsiflexion (delta)</div>
            <div class="slider-row"><label>Both ankles &Delta;</label><input type="range" id="sb-d-dorsi" min="-20" max="20" value="0" step="0.5" disabled><span class="value" id="sb-d-dorsi-val">0°</span></div>
        </div>
        <div class="section sandbox">
            <div class="section-title"><span class="dot"></span> Knee Depth (delta)</div>
            <div class="slider-row"><label>Both knees &Delta;</label><input type="range" id="sb-d-knee-flex" min="-40" max="40" value="0" step="0.5" disabled><span class="value" id="sb-d-knee-flex-val">0&deg;</span></div>
        </div>
        <div class="section barbell-s">
            <div class="section-title"><span class="dot"></span> Barbell</div>
            <div class="slider-row"><label>Weight</label><input type="range" id="sb-barbell-weight" min="0" max="200" value="0" step="5" disabled><span class="value" id="sb-barbell-weight-val">0 kg</span></div>
            <div class="slider-row"><label>Body mass</label><input type="range" id="sb-body-mass" min="40" max="150" value="75" step="1" disabled><span class="value" id="sb-body-mass-val">75 kg</span></div>
        </div>
        <div class="section">
            <div class="section-title"><span class="dot"></span> Trunk Lean</div>
            <div class="mono" id="sb-trunk-lean-status"></div>
        </div>
        <div class="section playback">
            <div class="section-title"><span class="dot"></span> Playback</div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;" id="sb-rep-buttons"></div>
            <div class="anim-controls">
                <button id="sb-play-btn" title="Play/Pause">&#9646;&#9646;</button>
                <input type="range" id="sb-frame-scrubber" min="0" max="100" value="0" step="1">
                <span class="value" id="sb-frame-val" style="min-width:50px;">0/0</span>
            </div>
            <div class="slider-row">
                <label>Speed</label>
                <input type="range" id="sb-speed-slider" min="0.1" max="3.0" value="1.0" step="0.1">
                <span class="value" id="sb-speed-val">1.0x</span>
            </div>
        </div>
        <div class="section angles">
            <div class="section-title"><span class="dot"></span> Live Angles</div>
            <div class="mono" id="sb-angles-info"></div>
        </div>
    </div>

    <!-- ======== DIAGNOSIS PANEL ======== -->
    <div id="diagnosis-panel" style="display:none">
        <!-- Rep selector -->
        <div class="section diagnosis">
            <div class="section-title"><span class="dot"></span> Diagnosis</div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;" id="diag-rep-selector"></div>
        </div>

        <!-- Per-rep view (hidden by default, shown when a rep is selected) -->
        <div id="diag-rep-view" style="display:none">
            <div class="section diagnosis">
                <div class="confidence-badge" id="diag-rep-confidence">--</div>
                <div id="diag-rep-symptoms" style="margin-top:8px; font-size:11px; color:#888;"></div>
            </div>
            <div class="section diagnosis">
                <div class="section-title"><span class="dot"></span> Quality Score</div>
                <div id="diag-rep-score"></div>
            </div>
            <div id="diag-rep-tiers"></div>
            <div class="section diagnosis" id="diag-voice-cues-section" style="display:none">
                <div class="section-title"><span class="dot"></span> Voice Cues (LLM Input)</div>
                <pre id="diag-voice-cues" class="mono" style="font-size:11px; color:#e0e0e0; background:#1a1a2e; padding:10px; border-radius:6px; white-space:pre-wrap; max-height:300px; overflow-y:auto;"></pre>
            </div>
            <div class="section diagnosis" id="diag-morph-section" style="display:none">
                <div class="section-title"><span class="dot"></span> Correction Preview</div>
                <div class="diag-morph-controls">
                    <button id="diag-play-btn" title="Play/Pause">&#9646;&#9646;</button>
                    <span class="mono" style="font-size:11px; color:#888;">Ghost = current form &nbsp;|&nbsp; <span style="color:#4ecdc4;">Teal = corrected</span></span>
                </div>
            </div>
        </div>

        <!-- Set Overview (shown by default) -->
        <div id="diag-set-view">
            <div class="section diagnosis">
                <div class="confidence-badge" id="diag-set-confidence">--</div>
                <div id="diag-set-symptoms" style="margin-top:8px; font-size:11px; color:#888;"></div>
            </div>
            <div class="section diagnosis">
                <div class="section-title"><span class="dot"></span> Set Quality</div>
                <div id="diag-set-score"></div>
            </div>
            <div class="section diagnosis">
                <div class="section-title"><span class="dot"></span> Rep Comparison</div>
                <div id="diag-sparklines"></div>
            </div>
            <div class="section diagnosis" id="diag-fault-section" style="display:none">
                <div class="section-title"><span class="dot"></span> Fault Map</div>
                <div id="diag-fault-map"></div>
            </div>
            <div class="section diagnosis" id="diag-trend-section">
                <div class="section-title"><span class="dot"></span> Performance Trend</div>
                <div id="diag-trend" class="mono"></div>
            </div>
            <div id="diag-set-tiers"></div>
        </div>

        <!-- Playback (always visible in diagnosis mode) -->
        <div class="section playback">
            <div class="section-title"><span class="dot"></span> Playback</div>
            <div class="anim-controls">
                <button id="diag-anim-play-btn" title="Play/Pause">&#9646;&#9646;</button>
                <input type="range" id="diag-frame-scrubber" min="0" max="100" value="0" step="1">
                <span class="value" id="diag-frame-val" style="min-width:50px;">0/0</span>
            </div>
            <div class="slider-row">
                <label>Speed</label>
                <input type="range" id="diag-speed-slider" min="0.1" max="3.0" value="1.0" step="0.1">
                <span class="value" id="diag-speed-val">1.0x</span>
            </div>
        </div>
    </div>
</div>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const DATA = {data_json};
const BONE_CONNECTIONS = [
    [0,1],[0,2],[1,3],[2,4],
    [0,5],[0,6],
    [5,6],[5,7],[7,9],[6,8],[8,10],
    [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],
    [15,17],[16,18],
];
const baselineData = DATA.baseline;
let LEAN_T = baselineData.leanThresholds;
let VALG_T = baselineData.valgusThresholds;

// FK reference body dimensions
const REF = {{
    hip_width: 0.22, thigh_len: 0.42, shin_len: 0.40,
    torso_len: 0.50, shoulder_width: 0.36, upper_arm: 0.30,
    forearm: 0.26, head_offset: 0.22, neck_len: 0.10, foot_len: 0.26,
}};
const SEGMENT_MASS = {{
    head: {{ joints: null, frac: 0.081 }},
    trunk: {{ joints: null, frac: 0.497 }},
    upper_arm_l: {{ joints: [5,7], frac: 0.028 }}, upper_arm_r: {{ joints: [6,8], frac: 0.028 }},
    forearm_l: {{ joints: [7,9], frac: 0.022 }}, forearm_r: {{ joints: [8,10], frac: 0.022 }},
    thigh_l: {{ joints: [11,13], frac: 0.100 }}, thigh_r: {{ joints: [12,14], frac: 0.100 }},
    shank_l: {{ joints: [13,15], frac: 0.047 }}, shank_r: {{ joints: [14,16], frac: 0.047 }},
    foot_l: {{ joints: [15,null], frac: 0.014 }}, foot_r: {{ joints: [16,null], frac: 0.014 }},
}};

// ======== Populate baseline info ========
document.getElementById('baseline-info').innerHTML = `
    <span class="lbl">Peak trunk offset:</span> <span class="val-green">${{baselineData.peakTrunkOffset.toFixed(1)}}°</span><br>
    <span class="lbl">Peak knee flex:</span> <span class="val-green">${{baselineData.peakKneeFlex.toFixed(1)}}°</span><br>
    <span class="lbl">Peak dorsiflexion:</span> <span class="val-green">${{baselineData.peakDorsi.toFixed(1)}}°</span><br>
    <span class="lbl">Baseline valgus:</span> <span class="val-green">${{baselineData.peakValgus.toFixed(1)}}°</span><br>
    <hr style="border-color:#2a2a4a; margin:6px 0">
    <span class="lbl">Lean thresholds:</span> <span class="val">${{LEAN_T.mild}}° / ${{LEAN_T.moderate}}° / ${{LEAN_T.severe}}°</span><br>
    <span class="lbl">Valgus thresholds:</span> <span class="val">${{VALG_T.mild}}° / ${{VALG_T.moderate}}° / ${{VALG_T.severe}}°</span>
`;

const FOOT_LEN_M = DATA.footLengthM || 0.2937;
const SHOE_SIZE_EUR = DATA.shoeSizeEur || 46;
const HEEL_OFFSET = 0.06;
const BALANCE_FRAC = 0.35;
const BALANCE_MARGIN_MIN = 0.10;
const HIP_IR_COUPLING = 0.75;

const AP = DATA.athleteParams;
if (AP) {{
    document.getElementById('athlete-info').innerHTML = `
        <span class="lbl">Body scale:</span> <span class="val">${{AP.bodyScale.toFixed(2)}}</span>
        <span class="lbl" style="margin-left:8px">Torso:</span> <span class="val">${{AP.torsoRatio.toFixed(2)}}</span>
        <span class="lbl" style="margin-left:8px">Thigh:</span> <span class="val">${{AP.thighRatio.toFixed(2)}}</span>
        <span class="lbl" style="margin-left:8px">Shin:</span> <span class="val">${{AP.shinRatio.toFixed(2)}}</span><br>
        <span class="lbl">Stance:</span> <span class="val">${{AP.stanceWidth.toFixed(2)}}x</span>
        <span class="lbl" style="margin-left:8px">Toe-out:</span> <span class="val">${{AP.toeOut.toFixed(1)}}°</span>
        <span class="lbl" style="margin-left:8px">Dorsi:</span> <span class="val">${{AP.dorsiRatio.toFixed(3)}}</span><br>
        <span class="lbl">Lean:</span> <span class="val">${{AP.forwardLean.toFixed(1)}}°</span>
        <span class="lbl" style="margin-left:8px">Max knee:</span> <span class="val">${{AP.maxKneeFlex.toFixed(1)}}°</span><br>
        <span class="lbl">Shoe (EU):</span> <span class="val">${{SHOE_SIZE_EUR}}</span>
        <span class="lbl" style="margin-left:8px">Foot len:</span> <span class="val">${{(FOOT_LEN_M * 100).toFixed(1)}} cm</span>
    `;
    // Pre-populate sandbox sliders
    const sliderInit = {{
        'sb-stance-width': [AP.stanceWidth, 'sb-stance-width-val', 'x', 2],
        'sb-toe-out': [AP.toeOut, 'sb-toe-out-val', '°', 0],
        'sb-d-dorsi': [0, 'sb-d-dorsi-val', '°', 1],
        'sb-d-knee-flex': [0, 'sb-d-knee-flex-val', '°', 1],
    }};
    for (const [id, [val, valId, suf, dec]] of Object.entries(sliderInit)) {{
        const el = document.getElementById(id);
        if (el) {{
            el.value = val;
            const vEl = document.getElementById(valId);
            if (vEl) vEl.textContent = dec > 0 ? parseFloat(val).toFixed(dec) + suf : Math.round(val) + suf;
        }}
    }}
}} else {{
    document.getElementById('athlete-info').innerHTML = '<span class="lbl">Not available</span>';
}}


// ======== THREE.JS SETUP ========
const canvas = document.getElementById('three-canvas');
const container = document.getElementById('scene-container');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0a0a1a);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 50);
camera.position.set(2.0, 1.0, 2.0);
camera.lookAt(0, 0.7, 0);
const orbitCtrl = new OrbitControls(camera, canvas);
orbitCtrl.target.set(0, 0.7, 0);
orbitCtrl.enableDamping = true;
orbitCtrl.dampingFactor = 0.08;
scene.add(new THREE.AmbientLight(0x404060, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(3, 5, 3);
scene.add(dl);
scene.add(new THREE.GridHelper(4, 20, 0x222244, 0x1a1a30));

// Joint and bone materials
const matN = new THREE.MeshPhongMaterial({{ color: 0x40e0a0, emissive: 0x103020 }});
const matF = new THREE.MeshPhongMaterial({{ color: 0xff4444, emissive: 0x401010 }});
const matB = new THREE.MeshPhongMaterial({{ color: 0x3090d0, emissive: 0x102030 }});
const matBF = new THREE.MeshPhongMaterial({{ color: 0xff6666, emissive: 0x301010 }});
const sGeo = new THREE.SphereGeometry(0.018, 12, 8);
const jm = [], js = [];
for (let i = 0; i < 19; i++) {{ const m = new THREE.Mesh(sGeo, matN.clone()); scene.add(m); jm.push(m); js.push('n'); }}
const bm = [];
for (const [a, b] of BONE_CONNECTIONS) {{
    const g = new THREE.CylinderGeometry(0.006, 0.006, 1, 6); g.translate(0, 0.5, 0);
    const m = new THREE.Mesh(g, matB.clone()); scene.add(m); bm.push({{ mesh: m, a, b }});
}}

// COM / BOS / Ghost / Midfoot visuals (sandbox mode)
const comSphere = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 7), new THREE.MeshPhongMaterial({{ color: 0xff2222, emissive: 0x550000, transparent: true, opacity: 0.9 }}));
scene.add(comSphere); comSphere.visible = false;
const comDiscGeo = new THREE.CircleGeometry(0.025, 16);
const comDisc = new THREE.Mesh(comDiscGeo, new THREE.MeshBasicMaterial({{ color: 0xff3333, transparent: true, opacity: 0.7 }}));
comDisc.rotation.x = -Math.PI / 2; comDisc.position.y = 0.001; scene.add(comDisc); comDisc.visible = false;
const comLineGeo = new THREE.BufferGeometry();
comLineGeo.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0, 0,2,0], 3));
const comLineMat = new THREE.LineDashedMaterial({{ color: 0xff3333, dashSize: 0.03, gapSize: 0.02, transparent: true, opacity: 0.6 }});
const comLine = new THREE.Line(comLineGeo, comLineMat); scene.add(comLine); comLine.visible = false;
const bosGeo = new THREE.BufferGeometry();
const bosLineMat = new THREE.LineBasicMaterial({{ color: 0x44ff88, transparent: true, opacity: 0.5 }});
const bosLine = new THREE.LineLoop(bosGeo, bosLineMat); bosLine.position.y = 0.002; scene.add(bosLine); bosLine.visible = false;
const ghostTorsoGeo = new THREE.BufferGeometry();
ghostTorsoGeo.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0, 0,1,0], 3));
const ghostTorsoLine = new THREE.Line(ghostTorsoGeo, new THREE.LineDashedMaterial({{ color: 0x00e5ff, dashSize: 0.04, gapSize: 0.03, transparent: true, opacity: 0.55 }}));
scene.add(ghostTorsoLine); ghostTorsoLine.visible = false;
const midfootLineGeo = new THREE.BufferGeometry();
midfootLineGeo.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0, 0,2,0], 3));
const midfootLine = new THREE.Line(midfootLineGeo, new THREE.LineDashedMaterial({{ color: 0xf0a040, dashSize: 0.025, gapSize: 0.02, transparent: true, opacity: 0.45 }}));
scene.add(midfootLine); midfootLine.visible = false;
const midfootDiscGeo = new THREE.CircleGeometry(0.018, 12);
const midfootDisc = new THREE.Mesh(midfootDiscGeo, new THREE.MeshBasicMaterial({{ color: 0xf0a040, transparent: true, opacity: 0.5 }}));
midfootDisc.rotation.x = -Math.PI / 2; midfootDisc.position.y = 0.001; scene.add(midfootDisc); midfootDisc.visible = false;

// Barbell
const barbellGroup = new THREE.Group(); scene.add(barbellGroup);
const barMat = new THREE.MeshPhongMaterial({{ color: 0x888899, emissive: 0x111122 }});
const plateMat = new THREE.MeshPhongMaterial({{ color: 0x333344, emissive: 0x080810 }});
const barGeo = new THREE.CylinderGeometry(0.014, 0.014, 1.6, 10);
const barMesh = new THREE.Mesh(barGeo, barMat); barMesh.rotation.x = Math.PI / 2; barbellGroup.add(barMesh);
const plateMeshes = [];
for (let i = 0; i < 4; i++) {{
    const pGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.04, 16);
    const pMesh = new THREE.Mesh(pGeo, plateMat.clone()); pMesh.rotation.x = Math.PI / 2;
    barbellGroup.add(pMesh); plateMeshes.push(pMesh);
}}
barbellGroup.visible = false;

// ======== STATE ========
let viewMode = 'replay'; // 'replay', 'sandbox', or 'diagnosis'
let curRep = 0, curFrame = 0, playing = true, lastT = performance.now(), speed = 1.0, frameAcc = 0;
const reps = DATA.reps, dataFps = DATA.fps;
let repFilter = -1;
let _barbellPos = null;

// ======== REPLAY CONTROLS ========
const rbc = document.getElementById('rep-buttons');
const allBtn = document.createElement('button'); allBtn.className='rep-btn active'; allBtn.textContent='All'; allBtn.dataset.idx='-1'; rbc.appendChild(allBtn);
for (let i = 0; i < reps.length; i++) {{
    const b = document.createElement('button'); b.className='rep-btn'; b.textContent=`Rep ${{i+2}}`; b.dataset.idx=String(i); rbc.appendChild(b);
}}
function onRepClick(e) {{
    if (!e.target.classList.contains('rep-btn')) return;
    e.currentTarget.querySelectorAll('.rep-btn').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    repFilter = parseInt(e.target.dataset.idx);
    curRep = repFilter === -1 ? 0 : repFilter;
    curFrame = 0;
}}
rbc.addEventListener('click', onRepClick);

const scrub = document.getElementById('frame-scrubber'), fv = document.getElementById('frame-val');
scrub.addEventListener('input', () => {{ curFrame = parseInt(scrub.value); playing = false; document.getElementById('play-btn').innerHTML = '&#9654;'; }});
document.getElementById('play-btn').addEventListener('click', () => {{ playing = !playing; document.getElementById('play-btn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;'; }});
const ss = document.getElementById('speed-slider'), sv = document.getElementById('speed-val');
ss.addEventListener('input', () => {{ speed = parseFloat(ss.value); sv.textContent = speed.toFixed(1) + 'x'; }});

// ======== SANDBOX REP BUTTONS (shared frame timing) ========
const sbRbc = document.getElementById('sb-rep-buttons');
const sbAllBtn = document.createElement('button'); sbAllBtn.className='rep-btn active'; sbAllBtn.textContent='All'; sbAllBtn.dataset.idx='-1'; sbRbc.appendChild(sbAllBtn);
for (let i = 0; i < reps.length; i++) {{
    const b = document.createElement('button'); b.className='rep-btn'; b.textContent=`Rep ${{i+2}}`; b.dataset.idx=String(i); sbRbc.appendChild(b);
}}
sbRbc.addEventListener('click', onRepClick);
const sbScrub = document.getElementById('sb-frame-scrubber'), sbFv = document.getElementById('sb-frame-val');
sbScrub.addEventListener('input', () => {{ curFrame = parseInt(sbScrub.value); playing = false; document.getElementById('sb-play-btn').innerHTML = '&#9654;'; }});
document.getElementById('sb-play-btn').addEventListener('click', () => {{ playing = !playing; document.getElementById('sb-play-btn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;'; }});
const sbSs = document.getElementById('sb-speed-slider'), sbSv = document.getElementById('sb-speed-val');
sbSs.addEventListener('input', () => {{ speed = parseFloat(sbSs.value); sbSv.textContent = speed.toFixed(1) + 'x'; }});

// ======== SANDBOX SLIDER BINDINGS ========
let _balanceLocked = false;
let _balanceLeanOffsetDeg = 0;
let _balanceHipShiftX = 0;
let _balanceHipShiftZ = 0;
const BALANCE_TARGET_EPS = 0.002;
const BALANCE_LEAN_OFFSET_MAX_DEG = 50;
const LOWER_BODY_SEG_KEYS = ['foot_l', 'foot_r', 'shank_l', 'shank_r', 'thigh_l', 'thigh_r'];
const baselineStanceWidth = AP ? AP.stanceWidth : 1.2;
const baselineToeOut = AP ? AP.toeOut : 15;

const SB_PARAM_IDS = [
    'sb-stance-width', 'sb-toe-out',
    'sb-barbell-weight', 'sb-body-mass',
    'sb-d-dorsi', 'sb-d-knee-flex',
];

function setSandboxParamsEnabled(enabled) {{
    for (const id of SB_PARAM_IDS) {{
        const el = document.getElementById(id);
        if (el) el.disabled = !enabled;
    }}
}}
setSandboxParamsEnabled(false);

function bindStanceSlider(id, valId, suffix, decimals) {{
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(valId);
    if (!slider || !valSpan) return;
    slider.addEventListener('input', () => {{
        const v = parseFloat(slider.value);
        valSpan.textContent = decimals > 0 ? v.toFixed(decimals) + suffix : Math.round(v) + suffix;
    }});
}}
function bindSliderDelta(id, valId, suffix, decimals) {{
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(valId);
    if (!slider || !valSpan) return;
    slider.addEventListener('input', () => {{
        const v = parseFloat(slider.value);
        const sign = v > 0 ? '+' : '';
        valSpan.textContent = sign + (decimals > 0 ? v.toFixed(decimals) : Math.round(v)) + suffix;
    }});
}}
function bindDisplaySlider(id, valId, suffix, decimals) {{
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(valId);
    if (!slider || !valSpan) return;
    slider.addEventListener('input', () => {{
        const v = parseFloat(slider.value);
        valSpan.textContent = decimals > 0 ? v.toFixed(decimals) + suffix : Math.round(v) + suffix;
    }});
}}
bindStanceSlider('sb-stance-width', 'sb-stance-width-val', 'x', 2);
bindStanceSlider('sb-toe-out', 'sb-toe-out-val', '°', 0);
bindSliderDelta('sb-d-dorsi', 'sb-d-dorsi-val', '°', 1);
bindSliderDelta('sb-d-knee-flex', 'sb-d-knee-flex-val', '°', 1);
bindDisplaySlider('sb-barbell-weight', 'sb-barbell-weight-val', ' kg', 0);
bindDisplaySlider('sb-body-mass', 'sb-body-mass-val', ' kg', 0);
bindDisplaySlider('sb-speed-slider', 'sb-speed-val', 'x', 1);

// When balance is locked, re-solve the (hip-shift, lean) counterbalance whenever a
// parameter changes so the COM stays over mid-foot. Holds the last balance on failure.
function attachRebalanceHook(id) {{
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {{
        if (!_balanceLocked) return;
        const rep = reps[curRep];
        if (!rep || !rep.length) return;
        const fd = rep[curFrame] || rep[0];
        if (!fd) return;
        const solved = solveBalanceLeanOffsetDeg(fd);
        const statusEl = document.getElementById('sb-balance-status');
        if (solved.success) {{
            _balanceLeanOffsetDeg = solved.leanDeg;
            _balanceHipShiftX = solved.hipShiftX || 0;
            _balanceHipShiftZ = solved.hipShiftZ || 0;
            if (statusEl) statusEl.style.color = '';
        }} else if (statusEl) {{
            statusEl.textContent = 'Holding last balance (re-balance unavailable at this setting).';
            statusEl.style.color = '#f0a040';
        }}
        updateSandbox(fd);
    }});
}}
['sb-stance-width', 'sb-toe-out', 'sb-d-dorsi', 'sb-d-knee-flex', 'sb-barbell-weight', 'sb-body-mass']
    .forEach(attachRebalanceHook);

function showSandboxBalanceUI(visible) {{
    const balanceBtn = document.getElementById('sb-balance-btn');
    const balanceHeader = document.getElementById('sb-balance-header');
    if (balanceBtn) balanceBtn.style.display = visible ? '' : 'none';
    if (balanceHeader) balanceHeader.classList.toggle('visible', visible);
}}

function getNormalizedBodyFracs() {{
    let totalFrac = 0;
    const bodyFracs = {{}};
    for (const [key, seg] of Object.entries(SEGMENT_MASS)) {{
        bodyFracs[key] = seg.frac;
        totalFrac += seg.frac;
    }}
    for (const key of Object.keys(bodyFracs)) bodyFracs[key] /= totalFrac;
    return bodyFracs;
}}

function buildSegmentCOMs(kpts, footLen) {{
    function g(i) {{ return {{ x: kpts[i*3], y: kpts[i*3+1], z: kpts[i*3+2] }}; }}
    const lS = g(5), rS = g(6), lH = g(11), rH = g(12);
    const sMid = {{ x:(lS.x+rS.x)/2, y:(lS.y+rS.y)/2, z:(lS.z+rS.z)/2 }};
    const hMid = {{ x:(lH.x+rH.x)/2, y:(lH.y+rH.y)/2, z:(lH.z+rH.z)/2 }};
    const segCOMs = {{ head: g(0), trunk: {{ x:(sMid.x+hMid.x)/2, y:(sMid.y+hMid.y)/2, z:(sMid.z+hMid.z)/2 }} }};
    for (const [key, seg] of Object.entries(SEGMENT_MASS)) {{
        if (!seg.joints || seg.joints[0] === null || seg.joints[1] === null) continue;
        if (key === 'head' || key === 'trunk' || key.startsWith('foot')) continue;
        const a = g(seg.joints[0]), b = g(seg.joints[1]);
        segCOMs[key] = {{ x:(a.x+b.x)/2, y:(a.y+b.y)/2, z:(a.z+b.z)/2 }};
    }}
    const aL = g(15), aR = g(16), kL = g(13), kR = g(14);
    function footFwd(ankle, knee) {{
        const dx = knee.x - ankle.x, dz = knee.z - ankle.z;
        const legLen = Math.sqrt(dx * dx + dz * dz) || 1;
        return {{ x: dx / legLen, z: dz / legLen }};
    }}
    const fL = footFwd(aL, kL), fR = footFwd(aR, kR);
    const footAlong = balancePointAlongFoot(footLen || FOOT_LEN_M);
    segCOMs.foot_l = {{ x: aL.x + fL.x * footAlong, y: aL.y, z: aL.z + fL.z * footAlong }};
    segCOMs.foot_r = {{ x: aR.x + fR.x * footAlong, y: aR.y, z: aR.z + fR.z * footAlong }};
    return segCOMs;
}}

function blendSegmentCOMGround(segCOMs, bodyFracs, segmentKeys, bodyMass, barWeight, barPos) {{
    const barFrac = barWeight > 0 ? barWeight / (bodyMass + barWeight) : 0;
    const bodyF = 1 - barFrac;
    let selectedFrac = 0;
    for (const key of segmentKeys) selectedFrac += bodyFracs[key] || 0;
    let cx = 0, cz = 0;
    for (const key of segmentKeys) {{
        const segmentCom = segCOMs[key];
        const frac = bodyFracs[key];
        if (!segmentCom || !frac) continue;
        cx += segmentCom.x * frac * bodyF;
        cz += segmentCom.z * frac * bodyF;
    }}
    if (selectedFrac > 1e-9) {{
        cx /= selectedFrac;
        cz /= selectedFrac;
    }}
    if (barWeight > 0 && barPos) {{
        cx = cx * bodyF + barPos.x * barFrac;
        cz = cz * bodyF + barPos.z * barFrac;
    }}
    return {{ x: cx, z: cz }};
}}

function computeLowerBodyCOMGround(kpts, footLen) {{
    const segCOMs = buildSegmentCOMs(kpts, footLen);
    const bodyFracs = getNormalizedBodyFracs();
    return blendSegmentCOMGround(segCOMs, bodyFracs, LOWER_BODY_SEG_KEYS, 1, 0, null);
}}

function solveBalanceLeanOffsetDeg(frameData) {{
    if (!frameData || !frameData.kpts) {{
        return {{ success: false, reason: 'no_frame' }};
    }}
    const capturedKpts = frameData.kpts;
    if (!capturedKpts[5] || !capturedKpts[6] || !capturedKpts[11] || !capturedKpts[12]) {{
        return {{ success: false, reason: 'missing_keypoints' }};
    }}

    const barWeight = _sv('sb-barbell-weight') || 0;
    const bodyMass = _sv('sb-body-mass') || 75;
    const deformed = deformLowerBody(capturedKpts);

    // Sagittal axis from the (deformed) hip lateral axis
    const hipLeftX = deformed[11 * 3], hipLeftZ = deformed[11 * 3 + 2];
    const hipRightX = deformed[12 * 3], hipRightZ = deformed[12 * 3 + 2];
    const hipLateralX = hipRightX - hipLeftX, hipLateralZ = hipRightZ - hipLeftZ;
    const hipLateralLen = Math.sqrt(hipLateralX * hipLateralX + hipLateralZ * hipLateralZ) || 1e-9;
    const sagittalDirX = -hipLateralZ / hipLateralLen;
    const sagittalDirZ = hipLateralX / hipLateralLen;

    // Mass fractions + head height (normalized to match computeCOM)
    const fr = getNormalizedBodyFracs();
    const armFrac = (fr.upper_arm_l || 0) + (fr.upper_arm_r || 0) +
                    (fr.forearm_l || 0) + (fr.forearm_r || 0);
    const barFrac = barWeight > 0 ? barWeight / (bodyMass + barWeight) : 0;
    const bodyF = 1 - barFrac;
    const bodyScale = AP ? AP.bodyScale : 1.0;
    const headHeight = REF.head_offset * bodyScale;

    // Balance by rotating ONLY the upper body about the fixed hip — the lower body
    // (hips included) never moves. Newton-refined against the real computeCOM.
    let lean = 0;
    let lastCom = null, lastTarget = null, lastGapPerp = 0, lastGapSagittal = 0;
    for (let iteration = 0; iteration < 16; iteration++) {{
        const posed = applyCounterbalance(deformed, 0, 0, lean);
        if (barWeight > 0) {{
            const sinL = Math.sin(posed.newLean), cosL = Math.cos(posed.newLean);
            _barbellPos = {{
                x: posed.shoulderMidX + 0.04 * sinL - 0.05 * cosL,
                y: posed.shoulderMidY + 0.04 * cosL + 0.05 * sinL,
                z: 0,
            }};
        }}
        const com = computeCOM(posed.kpts, barWeight, bodyMass, FOOT_LEN_M);
        const target = computeBalanceTargetGround(posed.kpts, FOOT_LEN_M);
        const gapX = target.x - com.groundX, gapZ = target.z - com.groundZ;
        const gapSagittal = gapX * sagittalDirX + gapZ * sagittalDirZ;
        lastCom = com; lastTarget = target;
        lastGapPerp = gapX * (-sagittalDirZ) + gapZ * sagittalDirX;
        lastGapSagittal = gapSagittal;
        if (Math.abs(gapSagittal) <= BALANCE_TARGET_EPS) break;
        const cosL = Math.cos(posed.newLean), sinL = Math.sin(posed.newLean);
        const upperMoment = (fr.trunk || 0) * 0.5 * posed.torsoLen +
                            (fr.head || 0) * (posed.torsoLen + 0.5 * headHeight) +
                            armFrac * posed.torsoLen;
        let sensLean = bodyF * upperMoment * cosL;
        if (barWeight > 0) sensLean += barFrac * (posed.torsoLen * cosL + 0.04 * cosL + 0.05 * sinL);
        // The upper body translates only in +X with lean, but the gap is measured along
        // the sagittal axis. Project the COM sensitivity onto that axis so the Newton step
        // has the correct sign for either hip orientation (otherwise lean runs to the clamp).
        sensLean *= sagittalDirX;
        if (!isFinite(sensLean) || Math.abs(sensLean) < 1e-9) break;
        // Damped, step-limited Newton (the barbell term is nonlinear in lean and can
        // overshoot); a hard bound keeps a divergent solve from running away.
        let step = gapSagittal / sensLean;
        step = Math.max(-0.3, Math.min(0.3, step));
        lean += 0.8 * step;
        const hardMax = (BALANCE_LEAN_OFFSET_MAX_DEG + 10) * Math.PI / 180;
        lean = Math.max(-hardMax, Math.min(hardMax, lean));
    }}

    if (!isFinite(lean)) {{
        return {{ success: false, reason: 'error', message: 'nonfinite_solve' }};
    }}
    const leanDeg = lean * (180 / Math.PI);
    if (Math.abs(leanDeg) > BALANCE_LEAN_OFFSET_MAX_DEG) {{
        return {{ success: false, reason: 'clamped',
                  neededDeg: parseFloat(Math.abs(leanDeg).toFixed(1)),
                  limitDeg: BALANCE_LEAN_OFFSET_MAX_DEG }};
    }}
    return {{ success: true, leanDeg, hipShiftX: 0, hipShiftZ: 0,
              com: lastCom, target: lastTarget,
              errorAlong: Math.abs(lastGapSagittal), errorPerp: Math.abs(lastGapPerp) }};
}}

function finishBalanceSearch(statusEl, balanceBtn, solveResult, bosInside) {{
    if (!solveResult.success) {{
        _balanceLocked = false;
        _balanceLeanOffsetDeg = 0;
        _balanceHipShiftX = 0;
        _balanceHipShiftZ = 0;
        let message;
        switch (solveResult.reason) {{
            case 'missing_keypoints':
                message = 'Balance needs shoulder and hip keypoints on this frame.';
                break;
            case 'clamped':
                message = `Balance needs ${{solveResult.neededDeg}}° lean, limit is ±${{solveResult.limitDeg}}°.`;
                break;
            case 'error':
                message = `Balance error: ${{solveResult.message}}`;
                break;
            default:
                message = 'No valid pose data for this frame.';
        }}
        if (statusEl) {{
            statusEl.textContent = message;
            statusEl.style.color = '#ff4444';
        }}
        if (balanceBtn) {{
            balanceBtn.disabled = false;
            balanceBtn.textContent = 'Balance';
        }}
        return;
    }}
    _balanceLeanOffsetDeg = solveResult.leanDeg;
    _balanceHipShiftX = solveResult.hipShiftX || 0;
    _balanceHipShiftZ = solveResult.hipShiftZ || 0;
    _balanceLocked = true;
    setSandboxParamsEnabled(true);
    const alongMm = (solveResult.errorAlong * 1000).toFixed(1);
    const perpMm = (solveResult.errorPerp * 1000).toFixed(1);
    const bosLabel = bosInside ? 'BOS inside' : 'BOS outside';
    if (statusEl) {{
        statusEl.innerHTML = `<span class="balance-ok">Balanced</span> (along: ${{alongMm}} mm, perp: ${{perpMm}} mm; ${{bosLabel}}). COM locked.`;
        statusEl.style.color = '';
    }}
    if (balanceBtn) {{
        balanceBtn.disabled = true;
        balanceBtn.textContent = 'Balanced';
    }}
}}

function balanceSandbox() {{
    if (_balanceLocked) return;
    const statusEl = document.getElementById('sb-balance-status');
    const balanceBtn = document.getElementById('sb-balance-btn');

    const rep = reps[curRep];
    if (!rep || !rep.length) {{
        finishBalanceSearch(statusEl, balanceBtn, {{ success: false, reason: 'no_frame' }}, false);
        return;
    }}
    const frameData = rep[curFrame] || rep[0];
    if (!frameData) {{
        finishBalanceSearch(statusEl, balanceBtn, {{ success: false, reason: 'no_frame' }}, false);
        return;
    }}

    try {{
        const solved = solveBalanceLeanOffsetDeg(frameData);
        let bosInside = false;
        if (solved.success) {{
            const verifyPose = buildSandboxKpts(frameData);
            if (verifyPose) {{
                const bos = computeBOS(verifyPose.kpts, 0, FOOT_LEN_M);
                bosInside = isBalanced(solved.com, bos).inside;
            }}
        }}
        finishBalanceSearch(statusEl, balanceBtn, solved, bosInside);
        if (solved.success) updateSandbox(frameData);
    }} catch (balanceError) {{
        console.error('Balance solve failed:', balanceError);
        finishBalanceSearch(statusEl, balanceBtn,
            {{ success: false, reason: 'error', message: balanceError.message }}, false);
    }}
}}

const balanceBtnEl = document.getElementById('sb-balance-btn');
if (balanceBtnEl) {{
    balanceBtnEl.addEventListener('click', (e) => {{
        e.preventDefault();
        e.stopPropagation();
        balanceSandbox();
    }});
}}
document.getElementById('controls').addEventListener('click', (e) => {{
    if (e.target && e.target.id === 'sb-balance-btn') balanceSandbox();
}});

// ======== MODE SWITCHING ========
function setActiveTab(tabId) {{
    ['tab-replay', 'tab-sandbox', 'tab-diagnosis'].forEach(id => {{
        const el = document.getElementById(id);
        if (el) el.classList.remove('active');
    }});
    const active = document.getElementById(tabId);
    if (active) active.classList.add('active');
    document.getElementById('replay-panel').style.display = 'none';
    document.getElementById('sandbox-panel').style.display = 'none';
    document.getElementById('diagnosis-panel').style.display = 'none';
}}
document.getElementById('tab-replay').addEventListener('click', () => {{
    viewMode = 'replay';
    setActiveTab('tab-replay');
    document.getElementById('replay-panel').style.display = '';
    showSandboxBalanceUI(false);
    hideSandboxVisuals();
    hideDiagnosisVisuals();
}});
document.getElementById('tab-sandbox').addEventListener('click', () => {{
    viewMode = 'sandbox';
    setActiveTab('tab-sandbox');
    document.getElementById('sandbox-panel').style.display = '';
    showSandboxBalanceUI(true);
    hideDiagnosisVisuals();
    const depthAngle = baselineData.peakKneeFlex;
    const depthOk = depthAngle >= 90;
    const depthEl = document.getElementById('sb-depth-check');
    if (depthEl) {{
        depthEl.innerHTML = depthOk
            ? `<span class="balance-ok">Depth: ${{depthAngle.toFixed(1)}}° — below parallel ✓</span>`
            : `<span class="balance-bad">Depth: ${{depthAngle.toFixed(1)}}° — parallel is ~90°</span>`;
    }}
}});
document.getElementById('tab-diagnosis').addEventListener('click', () => {{
    viewMode = 'diagnosis';
    setActiveTab('tab-diagnosis');
    document.getElementById('diagnosis-panel').style.display = '';
    showSandboxBalanceUI(false);
    hideSandboxVisuals();
    refreshDiagnosisView();
}});

function hideSandboxVisuals() {{
    ghostTorsoLine.visible = false;
    midfootLine.visible = false; midfootDisc.visible = false;
    barbellGroup.visible = false;
    comSphere.visible = false; comDisc.visible = false; comLine.visible = false; bosLine.visible = false;
}}

// ======== DIAGNOSIS MODE ========
const diagData = DATA.diagnosis;
let diagMorphPlaying = true;
let diagMorphFrame = 0;
let diagMorphLastT = 0;
let diagCurrentRep = 0;
let diagViewMode = 'set'; // 'set' or 'rep'

const ghostJointMat = new THREE.MeshPhongMaterial({{ color: 0x6688aa, transparent: true, opacity: 0.35 }});
const ghostBoneMat = new THREE.MeshPhongMaterial({{ color: 0x556688, transparent: true, opacity: 0.3 }});
const corrJointMat = new THREE.MeshPhongMaterial({{ color: 0x4ecdc4 }});
const corrBoneMat = new THREE.MeshPhongMaterial({{ color: 0x3aaa9a }});

const ghostJoints = [], ghostBones = [], corrJoints = [], corrBones = [];
for (let i = 0; i < 19; i++) {{
    const gj = new THREE.Mesh(new THREE.SphereGeometry(0.016, 10, 6), ghostJointMat);
    gj.visible = false; scene.add(gj); ghostJoints.push(gj);
    const cj = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 8), corrJointMat);
    cj.visible = false; scene.add(cj); corrJoints.push(cj);
}}
for (let i = 0; i < BONE_CONNECTIONS.length; i++) {{
    const gb = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 1, 5), ghostBoneMat);
    gb.visible = false; scene.add(gb); ghostBones.push(gb);
    const cb = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 1, 6), corrBoneMat);
    cb.visible = false; scene.add(cb); corrBones.push(cb);
}}

function updateDiagSkeleton(joints, bones, kpts) {{
    if (!kpts || kpts.length < 19) return;
    for (let i = 0; i < 19; i++) {{
        joints[i].position.set(kpts[i][0], kpts[i][1], kpts[i][2]);
        joints[i].visible = true;
    }}
    for (let bi = 0; bi < BONE_CONNECTIONS.length; bi++) {{
        const [si, ei] = BONE_CONNECTIONS[bi];
        const sp = joints[si].position, ep = joints[ei].position;
        const mid = new THREE.Vector3().addVectors(sp, ep).multiplyScalar(0.5);
        const dir = new THREE.Vector3().subVectors(ep, sp);
        const blen = dir.length();
        bones[bi].position.copy(mid);
        bones[bi].scale.set(1, blen, 1);
        bones[bi].quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
        bones[bi].visible = true;
    }}
}}

function hideDiagSkeleton(joints, bones) {{
    joints.forEach(j => j.visible = false);
    bones.forEach(b => b.visible = false);
}}

function showDiagnosisRepSkeleton() {{
    if (!diagData || !diagData.per_rep || !diagData.per_rep.length) return;
    const repData = diagData.per_rep[diagCurrentRep];
    if (!repData) return;
    // Main skeleton stays visible for playback; ghost+corrected overlay when corrections exist
    for (let i = 0; i < 19; i++) jointMeshes[i].visible = true;
    boneMeshes.forEach(b => b.visible = true);
    if (repData.has_correction && repData.morph_frames) {{
        updateDiagSkeleton(ghostJoints, ghostBones, repData.observed_kpts);
        updateDiagSkeleton(corrJoints, corrBones, repData.morph_frames[0]);
    }} else {{
        hideDiagSkeleton(ghostJoints, ghostBones);
        hideDiagSkeleton(corrJoints, corrBones);
    }}
}}

function showDiagnosisSetSkeleton() {{
    hideDiagSkeleton(ghostJoints, ghostBones);
    hideDiagSkeleton(corrJoints, corrBones);
    for (let i = 0; i < 19; i++) jointMeshes[i].visible = true;
    boneMeshes.forEach(b => b.visible = true);
}}

function refreshDiagnosisView() {{
    // Sync playback state with current diagnosis view
    if (diagViewMode === 'set') {{
        repFilter = -1;
        curRep = 0;
        showDiagnosisSetSkeleton();
    }} else {{
        repFilter = diagCurrentRep;
        curRep = diagCurrentRep;
        showDiagnosisRepSkeleton();
    }}
    curFrame = 0;
}}

function hideDiagnosisVisuals() {{
    hideDiagSkeleton(ghostJoints, ghostBones);
    hideDiagSkeleton(corrJoints, corrBones);
}}

function updateDiagnosisMorph(now) {{
    if (!diagData || !diagData.per_rep || !diagData.per_rep.length) return;
    if (diagViewMode === 'set') return;
    const repData = diagData.per_rep[diagCurrentRep];
    if (!repData || !repData.has_correction || !repData.morph_frames) return;
    const morphInterval = 1000 / 30;
    if (diagMorphPlaying && (now - diagMorphLastT) >= morphInterval) {{
        diagMorphFrame = (diagMorphFrame + 1) % repData.morph_frames.length;
        diagMorphLastT = now;
    }}
    updateDiagSkeleton(corrJoints, corrBones, repData.morph_frames[diagMorphFrame]);
}}

// ---- Tier card builder (shared by set and per-rep views) ----
function buildTierCards(container, tiers, expandFirst) {{
    container.innerHTML = '';
    const tierOrder = ['1', '2', '3', '0'];
    const tierLabels = {{ '1': 'Cue-correctable (fix now)', '2': 'Session-level', '3': 'Long-term', '0': 'Contextual' }};

    for (const tierKey of tierOrder) {{
        const tier = tiers[tierKey];
        if (!tier || tier.causes.length === 0) continue;

        const section = document.createElement('div');
        section.className = 'tier-section';
        section.dataset.tier = tierKey;

        const header = document.createElement('div');
        header.className = 'tier-header';
        const icon = document.createElement('span');
        icon.className = 'tier-icon' + (expandFirst && tierKey === '1' ? ' expanded' : '');
        icon.textContent = '\\u25B6';
        const label = document.createElement('span');
        label.className = 'tier-label';
        label.textContent = tierLabels[tierKey] || ('Tier ' + tierKey);
        const count = document.createElement('span');
        count.className = 'tier-count';
        count.textContent = '(' + tier.causes.length + ')';
        header.appendChild(icon);
        header.appendChild(label);
        header.appendChild(count);

        const body = document.createElement('div');
        body.className = 'tier-body' + (expandFirst && tierKey === '1' ? ' visible' : '');

        for (const cause of tier.causes) {{
            const card = document.createElement('div');
            card.className = 'cause-card';
            const ch = document.createElement('div');
            ch.className = 'cause-header';
            const cid = document.createElement('span');
            cid.className = 'cause-id';
            cid.textContent = cause.cause_id.replace(/_/g, ' ');
            const sc = document.createElement('div');
            sc.className = 'score-bar-container';
            const sf = document.createElement('div');
            sf.className = 'score-bar-fill';
            const scorePct = Math.round(cause.score * 100);
            sf.style.width = scorePct + '%';
            sf.style.background = cause.score > 0.5 ? '#4ecdc4' : cause.score > 0.3 ? '#f1c40f' : '#e67e22';
            sc.appendChild(sf);
            ch.appendChild(cid); ch.appendChild(sc);
            const expl = document.createElement('div');
            expl.className = 'cause-explanation';
            expl.textContent = cause.explanation;
            card.appendChild(ch); card.appendChild(expl);
            if (cause.implicated_by && cause.implicated_by.length > 0) {{
                const ev = document.createElement('div');
                ev.className = 'cause-evidence';
                ev.textContent = 'Evidence: ' + cause.implicated_by.join(', ').replace(/_/g, ' ');
                card.appendChild(ev);
            }}
            body.appendChild(card);
        }}

        header.addEventListener('click', () => {{
            body.classList.toggle('visible');
            icon.classList.toggle('expanded');
        }});
        section.appendChild(header);
        section.appendChild(body);
        container.appendChild(section);
    }}
}}

// ---- Symptom HTML helper ----
function renderSymptoms(element, symptoms) {{
    if (symptoms && symptoms.length > 0) {{
        element.innerHTML = 'Detected: ' + symptoms.map(
            s => '<span style="color:#a0a0ff">' + s.id.replace(/_/g, ' ') + '</span> (' + Math.round(s.severity * 100) + '%)'
        ).join(', ');
    }} else {{
        element.innerHTML = '<span style="color:#2ecc71">No form issues detected</span>';
    }}
}}

// ---- Quality score helpers ----
function scoreColor(score) {{
    if (score >= 0.7) return '#2ecc71';
    if (score >= 0.4) return '#f1c40f';
    return '#e74c3c';
}}

function renderRepScore(container, repData) {{
    if (!repData || repData.quality_score === undefined) {{
        container.innerHTML = '';
        return;
    }}
    const score = repData.quality_score;
    const subs = repData.sub_scores;
    const subDefs = [
        {{ key: 'depth', label: 'Depth', weight: '30%' }},
        {{ key: 'trunk_control', label: 'Trunk', weight: '25%' }},
        {{ key: 'knee_tracking', label: 'Knee Track', weight: '20%' }},
        {{ key: 'symmetry', label: 'Symmetry', weight: '15%' }},
        {{ key: 'ankle_utilization', label: 'Ankles', weight: '10%' }},
    ];

    let html = '<div class="quality-composite">';
    html += '<span class="quality-number" style="color:' + scoreColor(score) + '">' + Math.round(score * 100) + '</span>';
    html += '<span class="quality-label">/ 100</span>';
    html += '</div>';

    for (const sub of subDefs) {{
        const val = subs[sub.key];
        html += '<div class="quality-sub-row">';
        html += '<span class="quality-sub-label">' + sub.label + '</span>';
        html += '<div class="quality-sub-bar-bg"><div class="quality-sub-bar" style="width:' + (val * 100) + '%;background:' + scoreColor(val) + '"></div></div>';
        html += '<span class="quality-sub-val">' + Math.round(val * 100) + '</span>';
        html += '</div>';
    }}
    container.innerHTML = html;
}}

function renderSetScore(container) {{
    if (!diagData || !diagData.set_score) {{
        container.innerHTML = '';
        return;
    }}
    const ss = diagData.set_score;
    const meanPct = Math.round(ss.mean * 100);
    const trendArrow = ss.trend_slope > 0.01 ? '↑' : ss.trend_slope < -0.01 ? '↓' : '→';
    const trendColor = ss.trend_slope > 0.01 ? '#2ecc71' : ss.trend_slope < -0.01 ? '#e74c3c' : '#888';
    const trendLabel = ss.trend_slope > 0.01 ? 'Improving' : ss.trend_slope < -0.01 ? 'Degrading' : 'Steady';

    let html = '<div class="set-score-row">';
    html += '<span class="set-score-mean" style="color:' + scoreColor(ss.mean) + '">' + meanPct + '</span>';
    html += '<span class="quality-label">/ 100 avg</span>';
    html += '</div>';
    html += '<div class="set-score-detail set-score-best">Best rep: #' + ss.best_rep + '</div>';
    if (diagData.per_rep.length >= 2) {{
        html += '<div class="set-score-detail">Trend: <span class="set-score-trend" style="color:' + trendColor + '">' + trendArrow + ' ' + trendLabel + '</span></div>';
    }}
    container.innerHTML = html;
}}

// ---- Sparkline bars (set overview) ----
function buildSparklines() {{
    const container = document.getElementById('diag-sparklines');
    container.innerHTML = '';

    // Quality score sparkline (0–100 scale)
    if (diagData.per_rep.length > 0 && diagData.per_rep[0].quality_score !== undefined) {{
        const qGroup = document.createElement('div');
        qGroup.className = 'sparkline-group';
        const qLabel = document.createElement('div');
        qLabel.className = 'sparkline-label';
        qLabel.textContent = 'Quality Score';
        qGroup.appendChild(qLabel);

        for (const rep of diagData.per_rep) {{
            const score = rep.quality_score;
            const row = document.createElement('div');
            row.className = 'sparkline-row';
            const repLabel = document.createElement('span');
            repLabel.className = 'sparkline-rep';
            repLabel.textContent = 'R' + rep.rep_number;
            const barBg = document.createElement('div');
            barBg.className = 'sparkline-bar-bg';
            const bar = document.createElement('div');
            bar.className = 'sparkline-bar';
            bar.style.width = (score * 100) + '%';
            bar.style.background = scoreColor(score);
            barBg.appendChild(bar);
            const valLabel = document.createElement('span');
            valLabel.className = 'sparkline-val';
            valLabel.textContent = Math.round(score * 100);
            row.appendChild(repLabel);
            row.appendChild(barBg);
            row.appendChild(valLabel);
            qGroup.appendChild(row);
        }}
        container.appendChild(qGroup);
    }}

    const metricDefs = [
        {{ key: 'trunk_lean', label: 'Trunk Lean', unit: '°', goodMax: 25, warnMax: 40 }},
        {{ key: 'knee_valgus', label: 'Knee Valgus', unit: '°', goodMax: 8, warnMax: 15 }},
        {{ key: 'depth_angle', label: 'Squat Depth', unit: '°', goodMax: null, warnMax: null }},
        {{ key: 'dorsiflexion', label: 'Dorsiflexion', unit: '°', goodMax: null, warnMax: null }},
    ];

    for (const metric of metricDefs) {{
        const values = diagData.per_rep.map(r => r.metrics[metric.key]);
        const maxVal = Math.max(...values, 1);

        const group = document.createElement('div');
        group.className = 'sparkline-group';

        const label = document.createElement('div');
        label.className = 'sparkline-label';
        label.textContent = metric.label;
        group.appendChild(label);

        for (let repIdx = 0; repIdx < diagData.per_rep.length; repIdx++) {{
            const rep = diagData.per_rep[repIdx];
            const val = rep.metrics[metric.key];

            const row = document.createElement('div');
            row.className = 'sparkline-row';

            const repLabel = document.createElement('span');
            repLabel.className = 'sparkline-rep';
            repLabel.textContent = 'R' + rep.rep_number;

            const barBg = document.createElement('div');
            barBg.className = 'sparkline-bar-bg';
            const bar = document.createElement('div');
            bar.className = 'sparkline-bar';
            bar.style.width = (val / maxVal * 100) + '%';
            if (metric.goodMax !== null) {{
                bar.style.background = val <= metric.goodMax ? '#2ecc71'
                    : val <= metric.warnMax ? '#f1c40f' : '#e74c3c';
            }} else {{
                bar.style.background = '#4ecdc4';
            }}
            barBg.appendChild(bar);

            const valLabel = document.createElement('span');
            valLabel.className = 'sparkline-val';
            valLabel.textContent = val + metric.unit;

            row.appendChild(repLabel);
            row.appendChild(barBg);
            row.appendChild(valLabel);
            group.appendChild(row);
        }}

        container.appendChild(group);
    }}
}}

// ---- Fault map (set overview) ----
function buildFaultMap() {{
    const container = document.getElementById('diag-fault-map');
    const faultSection = document.getElementById('diag-fault-section');
    container.innerHTML = '';

    const symptoms = diagData.set_symptoms;
    if (!symptoms || symptoms.length === 0) {{
        faultSection.style.display = 'none';
        return;
    }}
    faultSection.style.display = '';

    // Header row
    const header = document.createElement('div');
    header.className = 'fault-map-row fault-map-header';
    const spacer = document.createElement('span');
    spacer.className = 'fault-map-label';
    spacer.textContent = '';
    header.appendChild(spacer);
    diagData.per_rep.forEach(rep => {{
        const cell = document.createElement('span');
        cell.className = 'fault-map-cell';
        cell.textContent = 'R' + rep.rep_number;
        header.appendChild(cell);
    }});
    container.appendChild(header);

    // Fault rows
    for (const symptom of symptoms) {{
        const row = document.createElement('div');
        row.className = 'fault-map-row';

        const label = document.createElement('span');
        label.className = 'fault-map-label';
        label.title = symptom.id.replace(/_/g, ' ');
        label.textContent = symptom.id.replace(/_/g, ' ');
        row.appendChild(label);

        diagData.per_rep.forEach(rep => {{
            const cell = document.createElement('span');
            cell.className = 'fault-map-cell';
            const isPresent = symptom.contributing_reps && symptom.contributing_reps.includes(rep.rep_number);
            const dot = document.createElement('span');
            dot.className = 'fault-dot ' + (isPresent ? 'present' : 'absent');
            cell.appendChild(dot);
            row.appendChild(cell);
        }});

        container.appendChild(row);
    }}
}}

// ---- Performance trend (set overview) ----
function diagStdDev(arr) {{
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / arr.length);
}}

function buildPerformanceTrend() {{
    const container = document.getElementById('diag-trend');
    const repList = diagData.per_rep;
    if (repList.length < 2) {{
        container.innerHTML = '<span style="color:#888">Need 2+ reps to detect trends.</span>';
        return;
    }}

    const confidences = repList.map(r => r.confidence);
    const halfIdx = Math.ceil(confidences.length / 2);
    const avgFirst = confidences.slice(0, halfIdx).reduce((a, b) => a + b, 0) / halfIdx;
    const avgSecond = confidences.slice(halfIdx).reduce((a, b) => a + b, 0) / (confidences.length - halfIdx);

    const trunkValues = repList.map(r => r.metrics.trunk_lean);
    const valgusValues = repList.map(r => r.metrics.knee_valgus);
    const trunkStd = diagStdDev(trunkValues);
    const valgusStd = diagStdDev(valgusValues);

    let html = '';

    if (avgSecond > avgFirst * 1.15) {{
        html += '<div class="trend-line trend-bad">⚠ Form degradation detected — later reps show more issues</div>';
    }} else if (avgSecond < avgFirst * 0.85) {{
        html += '<div class="trend-line trend-good">✓ Form improving over the set</div>';
    }} else {{
        html += '<div class="trend-line trend-neutral">◆ Consistent form across reps</div>';
    }}

    if (trunkStd > 5) {{
        html += '<div class="trend-line trend-warn">Trunk lean varies ±' + trunkStd.toFixed(1) + '° across reps</div>';
    }}
    if (valgusStd > 3) {{
        html += '<div class="trend-line trend-warn">Knee valgus varies ±' + valgusStd.toFixed(1) + '° across reps</div>';
    }}

    // Depth consistency
    const depthValues = repList.map(r => r.metrics.depth_angle);
    const depthStd = diagStdDev(depthValues);
    if (depthStd > 8) {{
        html += '<div class="trend-line trend-warn">Depth varies ±' + depthStd.toFixed(1) + '° across reps</div>';
    }}

    container.innerHTML = html;
}}

// ---- Switch to Set Overview ----
function switchToDiagSetView() {{
    diagViewMode = 'set';
    repFilter = -1;
    curRep = 0;
    curFrame = 0;
    document.getElementById('diag-set-view').style.display = '';
    document.getElementById('diag-rep-view').style.display = 'none';
    showDiagnosisSetSkeleton();
}}

// ---- Switch to per-rep view ----
function switchToDiagRepView(repIdx) {{
    diagViewMode = 'rep';
    diagCurrentRep = repIdx;
    diagMorphFrame = 0;
    repFilter = repIdx;
    curRep = repIdx;
    curFrame = 0;

    document.getElementById('diag-set-view').style.display = 'none';
    document.getElementById('diag-rep-view').style.display = '';

    const repData = diagData.per_rep[repIdx];
    if (!repData) return;

    // Confidence
    const confPct = Math.round(repData.confidence * 100);
    document.getElementById('diag-rep-confidence').textContent = 'Confidence: ' + confPct + '%';

    // Symptoms
    renderSymptoms(document.getElementById('diag-rep-symptoms'), repData.symptoms);

    // Quality score
    renderRepScore(document.getElementById('diag-rep-score'), repData);

    // Tiers
    buildTierCards(document.getElementById('diag-rep-tiers'), repData.tiers, true);

    // Voice cues
    const vcSection = document.getElementById('diag-voice-cues-section');
    const vcPre = document.getElementById('diag-voice-cues');
    if (repData.voice_cues && Object.keys(repData.voice_cues).length > 0) {{
        vcSection.style.display = '';
        vcPre.textContent = JSON.stringify(repData.voice_cues, null, 2);
    }} else {{
        vcSection.style.display = 'none';
    }}

    // Morph section
    const morphSection = document.getElementById('diag-morph-section');
    if (repData.has_correction && repData.morph_frames) {{
        morphSection.style.display = '';
    }} else {{
        morphSection.style.display = 'none';
    }}

    showDiagnosisRepSkeleton();
}}

// ---- Main panel builder ----
function buildDiagnosisPanel() {{
    if (!diagData) return;
    document.getElementById('tab-diagnosis').style.display = '';

    // Build rep selector
    const selector = document.getElementById('diag-rep-selector');
    selector.innerHTML = '';

    const overviewBtn = document.createElement('button');
    overviewBtn.className = 'rep-btn active';
    overviewBtn.textContent = 'Set Overview';
    overviewBtn.addEventListener('click', () => {{
        selector.querySelectorAll('.rep-btn').forEach(b => b.classList.remove('active'));
        overviewBtn.classList.add('active');
        switchToDiagSetView();
    }});
    selector.appendChild(overviewBtn);

    diagData.per_rep.forEach((repData, idx) => {{
        const btn = document.createElement('button');
        btn.className = 'rep-btn';
        btn.textContent = 'Rep ' + repData.rep_number;
        if (repData.quality_score !== undefined) {{
            btn.style.borderBottom = '3px solid ' + scoreColor(repData.quality_score);
        }}
        btn.addEventListener('click', () => {{
            selector.querySelectorAll('.rep-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            switchToDiagRepView(idx);
        }});
        selector.appendChild(btn);
    }});

    // Build Set Overview content
    const setConfPct = Math.round(diagData.set_confidence * 100);
    document.getElementById('diag-set-confidence').textContent = 'Confidence: ' + setConfPct + '%';
    renderSymptoms(document.getElementById('diag-set-symptoms'), diagData.set_symptoms);
    renderSetScore(document.getElementById('diag-set-score'));
    buildSparklines();
    buildFaultMap();
    buildPerformanceTrend();
    buildTierCards(document.getElementById('diag-set-tiers'), diagData.set_tiers, true);

    // Morph play button (correction preview)
    document.getElementById('diag-play-btn')?.addEventListener('click', () => {{
        diagMorphPlaying = !diagMorphPlaying;
        document.getElementById('diag-play-btn').innerHTML = diagMorphPlaying ? '&#9646;&#9646;' : '&#9654;';
    }});

    // Playback controls (rep/set animation)
    const diagScrub = document.getElementById('diag-frame-scrubber');
    const diagFv = document.getElementById('diag-frame-val');
    diagScrub.addEventListener('input', () => {{
        curFrame = parseInt(diagScrub.value);
        playing = false;
        document.getElementById('diag-anim-play-btn').innerHTML = '&#9654;';
    }});
    document.getElementById('diag-anim-play-btn').addEventListener('click', () => {{
        playing = !playing;
        document.getElementById('diag-anim-play-btn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
    }});
    const diagSs = document.getElementById('diag-speed-slider');
    const diagSv = document.getElementById('diag-speed-val');
    diagSs.addEventListener('input', () => {{
        speed = parseFloat(diagSs.value);
        diagSv.textContent = speed.toFixed(1) + 'x';
    }});

    // Auto-switch to diagnosis tab
    if (diagData.auto_open) {{
        document.getElementById('tab-diagnosis').click();
    }}
}}

buildDiagnosisPanel();

function balancePointAlongFoot(footLen) {{
    return BALANCE_FRAC * footLen - HEEL_OFFSET;
}}

function computeBalanceTargetGround(kpts, footLen) {{
    function g(i) {{ return {{ x: kpts[i*3], y: kpts[i*3+1], z: kpts[i*3+2] }}; }}
    const along = balancePointAlongFoot(footLen);
    const targets = [];
    for (const [ankleIdx, footIdx] of [[15, 17], [16, 18]]) {{
        const ankle = g(ankleIdx);
        const foot = footIdx < 19 && kpts[footIdx * 3] !== undefined ? g(footIdx) : null;
        let fx, fz;
        if (foot && (Math.abs(foot.x - ankle.x) > 1e-6 || Math.abs(foot.z - ankle.z) > 1e-6)) {{
            fx = foot.x - ankle.x;
            fz = foot.z - ankle.z;
        }} else {{
            const toeOutRad = (_sv('sb-toe-out') || baselineToeOut) * Math.PI / 180;
            const sideSign = ankleIdx === 15 ? -1 : 1;
            fx = Math.cos(toeOutRad);
            fz = sideSign * Math.sin(toeOutRad);
        }}
        const fl = Math.sqrt(fx * fx + fz * fz) || 1;
        fx /= fl; fz /= fl;
        targets.push({{ x: ankle.x + fx * along, z: ankle.z + fz * along }});
    }}
    if (!targets.length) return {{ x: 0, z: 0 }};
    let tx = 0, tz = 0;
    for (const t of targets) {{ tx += t.x; tz += t.z; }}
    return {{ x: tx / targets.length, z: tz / targets.length }};
}}

function updateBalanceTargetVisual(kpts, footLen) {{
    const tgt = computeBalanceTargetGround(kpts, footLen);
    const barTopY = 1.6;
    const positions = new Float32Array([tgt.x, 0.003, tgt.z, tgt.x, barTopY, tgt.z]);
    midfootLineGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    midfootLine.computeLineDistances();
    midfootLine.visible = viewMode === 'sandbox';
    midfootDisc.position.set(tgt.x, 0.001, tgt.z);
    midfootDisc.visible = viewMode === 'sandbox';
}}

// ======== FAULT CLASSIFICATION ========
function classifyLean(trunkAngleDeg) {{
    const o = 180 - trunkAngleDeg;
    return o >= LEAN_T.severe ? 'severe' : o >= LEAN_T.moderate ? 'moderate' : o >= LEAN_T.mild ? 'mild' : 'none';
}}
function classifyValgus(valgusDeg) {{
    return valgusDeg >= VALG_T.severe ? 'severe' : valgusDeg >= VALG_T.moderate ? 'moderate' : valgusDeg >= VALG_T.mild ? 'mild' : 'none';
}}
function sb(s) {{ return `<span class="severity-indicator sev-${{s}}">${{s.toUpperCase()}}</span>`; }}

// ======== REPLAY UPDATE ========
function updateReplay(fd) {{
    if (!fd) return;
    const k = fd.kpts, a = fd.angles;
    const ls = classifyLean(a.trunk_flexion);
    const vl = classifyValgus(Math.abs(a.knee_valgus_l)), vr = classifyValgus(Math.abs(a.knee_valgus_r));
    const vs = vl!=='none'||vr!=='none' ? (vl==='severe'||vr==='severe'?'severe':vl==='moderate'||vr==='moderate'?'moderate':'mild') : 'none';
    const fj = new Set();
    if (ls !== 'none') [0,1,2,3,4,5,6].forEach(j => fj.add(j));
    if (vs !== 'none') [13,14].forEach(j => fj.add(j));
    for (let i = 0; i < 19; i++) {{
        if (!k[i]) continue;
        jm[i].position.set(k[i][0], k[i][1], k[i][2]);
        jm[i].visible = true;
        const t = fj.has(i) ? 'f' : 'n';
        if (t !== js[i]) {{ jm[i].material = (t === 'f' ? matF : matN).clone(); js[i] = t; }}
    }}
    const fb = new Set();
    if (ls !== 'none') {{ fb.add('5-6'); fb.add('5-11'); fb.add('6-12'); fb.add('0-5'); fb.add('0-6'); }}
    if (vs !== 'none') {{ fb.add('11-13'); fb.add('13-15'); fb.add('12-14'); fb.add('14-16'); }}
    for (const bone of bm) {{
        const pa = jm[bone.a].position, pb = jm[bone.b].position;
        const d = new THREE.Vector3().subVectors(pb, pa), l = d.length(); d.normalize();
        bone.mesh.position.copy(pa); bone.mesh.scale.set(1, l, 1);
        bone.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), d);
        bone.mesh.material = fb.has(`${{bone.a}}-${{bone.b}}`) ? matBF : matB;
    }}
    // COM / BOS from captured keypoints
    const rkpts = new Float64Array(19 * 3);
    for (let i = 0; i < 19; i++) {{ if (k[i]) {{ rkpts[i*3]=k[i][0]; rkpts[i*3+1]=k[i][1]; rkpts[i*3+2]=k[i][2]; }} }}
    const rFootLen = AP ? (AP.foot_avg_m || REF.foot_len) : REF.foot_len;
    const rCom = computeCOM(rkpts, 0, 75, rFootLen);
    const rBos = computeBOS(rkpts, 0, rFootLen);
    const rBal = isBalanced(rCom, rBos);
    updateCOMVisuals(rCom, rBos, rBal);
    const to = (180 - a.trunk_flexion).toFixed(1);
    document.getElementById('angles-info').innerHTML = `
        <span class="lbl">Knee Flex:</span> <span class="val">${{a.knee_flex.toFixed(1)}}°</span><br>
        <span class="lbl">Trunk Angle:</span> <span class="val">${{a.trunk_flexion.toFixed(1)}}°</span> (offset: ${{to}}°)<br>
        <span class="lbl">Valgus L/R:</span> <span class="val">${{a.knee_valgus_l.toFixed(1)}}° / ${{a.knee_valgus_r.toFixed(1)}}°</span><br>
        <span class="lbl">Dorsi L/R:</span> <span class="val">${{a.dorsi_l.toFixed(1)}}° / ${{a.dorsi_r.toFixed(1)}}°</span><br>
        <span class="lbl">Hip Flex L/R:</span> <span class="val">${{a.hip_flex_l.toFixed(1)}}° / ${{a.hip_flex_r.toFixed(1)}}°</span>`;
    const fl = [];
    if (ls !== 'none') fl.push('Forward Lean ' + sb(ls));
    if (vs !== 'none') fl.push('Knee Valgus ' + sb(vs));
    document.getElementById('faults-info').innerHTML = fl.length ? fl.join('<br>') : '<span class="val-green">Clean</span>';
    document.getElementById('info-overlay').innerHTML = `
        <span class="lbl">Rep:</span> <span class="val">${{curRep+2}}</span>
        <span class="lbl" style="margin-left:12px">Frame:</span> <span class="val">${{curFrame+1}}/${{reps[curRep].length}}</span><br>
        <span class="lbl">Knee:</span> <span class="val">${{a.knee_flex.toFixed(1)}}°</span>
        <span class="lbl" style="margin-left:8px">Trunk:</span> <span class="val">${{a.trunk_flexion.toFixed(1)}}°</span>`;
}}

// ======== FK ENGINE (from fault_visualizer) ========
function computeSquatPose(phase, params, lockedShoulder) {{
    const {{ maxKneeFlex, forwardLean, kneeValgus, stanceWidth, toeOut, dorsiRatio,
             bodyScale, torsoRatio, thighRatio, shinRatio, shoulderWidthRatio,
             barbellWeight: bw }} = params;
    const deg2rad = Math.PI / 180, rad2deg = 180 / Math.PI;
    const profile = phase; // phase-matched: phase IS the normalized depth (0=standing, 1=peak)
    const kneeFlexDeg = maxKneeFlex * profile;
    const kneeFlexRad = kneeFlexDeg * deg2rad;
    const ankleDF = dorsiRatio * kneeFlexRad;
    const extraLeanRad = (forwardLean * deg2rad) * profile;
    const valgusRad = (kneeValgus * deg2rad) * profile;
    const toeOutRad = toeOut * deg2rad;

    const hw = REF.hip_width * bodyScale;
    const ankleZ = (hw / 2) * stanceWidth;
    const thl = REF.thigh_len * bodyScale * thighRatio;
    const shl = REF.shin_len * bodyScale * shinRatio;
    const tl = REF.torso_len * bodyScale * torsoRatio;
    const sw = REF.shoulder_width * bodyScale * (shoulderWidthRatio || 1.0);
    const ual = REF.upper_arm * bodyScale;
    const fal = REF.forearm * bodyScale;
    const headH = REF.head_offset * bodyScale;

    const kpts = new Float64Array(19 * 3);
    function set(idx, x, y, z) {{ kpts[idx*3]=x; kpts[idx*3+1]=y; kpts[idx*3+2]=z; }}
    function get(idx) {{ return [kpts[idx*3], kpts[idx*3+1], kpts[idx*3+2]]; }}
    const fl = FOOT_LEN_M;

    const shinTilt = ankleDF;
    const thighDirAngle = shinTilt - kneeFlexRad;
    const valgusShift = shl * Math.sin(valgusRad);

    function placeLeg(side, ax, ay, az) {{
        const sgn = side;
        const fx = Math.cos(toeOutRad), fz = sgn * Math.sin(toeOutRad);
        const lx = -fz, lz = fx;
        const medialSign = (side < 0) ? 1 : -1;
        const mx = medialSign * lx, mz = medialSign * lz;
        const kx = ax + shl * Math.sin(shinTilt) * fx + valgusShift * mx;
        const ky = ay + shl * Math.cos(shinTilt);
        const kz = az + shl * Math.sin(shinTilt) * fz + valgusShift * mz;
        const tf = thl * Math.sin(thighDirAngle), tu = thl * Math.cos(thighDirAngle);
        const hx = kx + tf * fx, hy = ky + tu, hz = kz + tf * fz;
        return {{ kx, ky, kz, hx, hy, hz }};
    }}

    set(15, 0, 0, -ankleZ); set(16, 0, 0, ankleZ);
    set(17, fl * Math.cos(toeOutRad), 0, -ankleZ - fl * Math.sin(toeOutRad));
    set(18, fl * Math.cos(toeOutRad), 0, ankleZ + fl * Math.sin(toeOutRad));
    const L = placeLeg(-1, 0, 0, -ankleZ), R = placeLeg(+1, 0, 0, ankleZ);
    set(13, L.kx, L.ky, L.kz); set(14, R.kx, R.ky, R.kz);
    const hipY = (L.hy + R.hy) / 2, hipX = (L.hx + R.hx) / 2;
    set(11, hipX, hipY, -hw/2); set(12, hipX, hipY, hw/2);

    const lHip = get(11), rHip = get(12);
    const hipMidX = (lHip[0] + rHip[0]) / 2, hipMidY = (lHip[1] + rHip[1]) / 2;

    let totalTrunkLean;
    if (lockedShoulder) {{
        const baseAngle = Math.atan2(lockedShoulder.x - hipMidX, lockedShoulder.y - hipMidY);
        const leanDelta = ((forwardLean - (lockedShoulder.refLean || forwardLean)) * deg2rad) * profile;
        totalTrunkLean = Math.max(0, baseAngle + leanDelta);
    }} else if (bw && bw > 0) {{
        const balanceX = balancePointAlongFoot(fl) * Math.cos(toeOutRad);
        const A = tl + 0.04, B = -0.05, C = balanceX - hipMidX;
        const Rv = Math.sqrt(A*A + B*B), phi = Math.atan2(B, A);
        const sinArg = Math.max(-1, Math.min(1, C / Rv));
        totalTrunkLean = Math.max(0, Math.asin(sinArg) - phi + extraLeanRad);
    }} else {{
        const balanceX = balancePointAlongFoot(fl) * Math.cos(toeOutRad);
        const sinArg = Math.max(-1, Math.min(1, (balanceX - hipMidX) / tl));
        totalTrunkLean = Math.max(0, Math.asin(sinArg) + extraLeanRad);
    }}

    const sMidX = hipMidX + tl * Math.sin(totalTrunkLean);
    const sMidY = hipMidY + tl * Math.cos(totalTrunkLean);
    set(5, sMidX, sMidY, -sw/2); set(6, sMidX, sMidY, sw/2);

    const headX = sMidX + headH * Math.sin(totalTrunkLean) * 0.5;
    const headY = sMidY + headH;
    set(0, headX, headY, 0);
    set(1, headX, headY + 0.02, -0.03 * bodyScale);
    set(2, headX, headY + 0.02, 0.03 * bodyScale);
    set(3, headX, headY - 0.01, -0.06 * bodyScale);
    set(4, headX, headY - 0.01, 0.06 * bodyScale);

    // Arms: hanging default
    const lS = get(5), rS = get(6);
    const armFwd = 0.05 * profile, armDown = 0.15 + 0.1 * profile;
    set(7, lS[0] + armFwd, lS[1] - ual * 0.7 - armDown, lS[2]);
    set(8, rS[0] + armFwd, rS[1] - ual * 0.7 - armDown, rS[2]);
    const lE = get(7), rE = get(8);
    set(9, lE[0] + 0.02, lE[1] - fal * 0.5, lE[2]);
    set(10, rE[0] + 0.02, rE[1] - fal * 0.5, rE[2]);

    const trunkAngleDeg = 180 - (totalTrunkLean * rad2deg);
    const valgusActiveDeg = kneeValgus * profile;
    const dorsiDeg = ankleDF * rad2deg;
    const totalTrunkLeanDeg = totalTrunkLean * rad2deg;

    return {{ kpts, avgKneeDeg: kneeFlexDeg, trunkAngleDeg, valgusActiveDeg, dorsiDeg,
              totalTrunkLeanDeg, hipMidX, hipMidY, shoulderMidX: sMidX, shoulderMidY: sMidY,
              toeOutRad, tl, sw, totalTrunkLean }};
}}

// ======== CAPTURED-POSE DEFORMATION (sandbox sliders) ========
// Rotate a [x,y,z] vector about an arbitrary axis by ang radians (Rodrigues).
function rotateAboutAxis(v, axis, ang) {{
    const al = Math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]) || 1e-9;
    const kx = axis[0]/al, ky = axis[1]/al, kz = axis[2]/al;
    const c = Math.cos(ang), s = Math.sin(ang);
    const dot = kx*v[0] + ky*v[1] + kz*v[2];
    const crossX = ky*v[2] - kz*v[1];
    const crossY = kz*v[0] - kx*v[2];
    const crossZ = kx*v[1] - ky*v[0];
    return [
        v[0]*c + crossX*s + kx*dot*(1-c),
        v[1]*c + crossY*s + ky*dot*(1-c),
        v[2]*c + crossZ*s + kz*dot*(1-c),
    ];
}}
// Rotate a [x,y,z] vector about the vertical (Y) axis by ang radians.
function rotateYvec(v, ang) {{
    const c = Math.cos(ang), s = Math.sin(ang);
    return [ v[0]*c + v[2]*s, v[1], -v[0]*s + v[2]*c ];
}}

// 2-link IK: place the knee so |hip-knee| == thighLen and |knee-ankle| == shinLen
// exactly (bones locked to capture). Hip H and ankle A stay fixed; the captured/prior
// knee `ref` disambiguates the bend half-plane (prevents flipping).
function solveKnee(H, A, thighLen, shinLen, ref) {{
    const HAx = A[0]-H[0], HAy = A[1]-H[1], HAz = A[2]-H[2];
    const d = Math.sqrt(HAx*HAx + HAy*HAy + HAz*HAz) || 1e-9;
    const ux = HAx/d, uy = HAy/d, uz = HAz/d;
    const dmin = Math.abs(thighLen - shinLen) + 1e-4, dmax = thighLen + shinLen - 1e-4;
    const dc = Math.max(dmin, Math.min(dmax, d));
    let cosA = (thighLen*thighLen + dc*dc - shinLen*shinLen) / (2*thighLen*dc);
    cosA = Math.max(-1, Math.min(1, cosA));
    const sinA = Math.sin(Math.acos(cosA));
    // pole = component of (ref-H) perpendicular to the hip->ankle axis
    let px = (ref[0]-H[0]), py = (ref[1]-H[1]), pz = (ref[2]-H[2]);
    const rdotu = px*ux + py*uy + pz*uz;
    px -= rdotu*ux; py -= rdotu*uy; pz -= rdotu*uz;
    let pl = Math.sqrt(px*px + py*py + pz*pz);
    if (pl < 1e-6) {{ // degenerate (straight leg): bend toward world up
        px = -ux*uy; py = 1 - uy*uy; pz = -uz*uy;
        pl = Math.sqrt(px*px + py*py + pz*pz) || 1e-9;
    }}
    px /= pl; py /= pl; pz /= pl;
    return [
        H[0] + thighLen*cosA*ux + thighLen*sinA*px,
        H[1] + thighLen*cosA*uy + thighLen*sinA*py,
        H[2] + thighLen*cosA*uz + thighLen*sinA*pz,
    ];
}}

// Rebuild the lower body (11-18) bottom-up from the grounded feet using captured
// per-side bone vectors, overriding only stance / toe-out / dorsiflexion. The hips
// fall out of the leg geometry (rigid pelvis reconciled by averaging). Returns a
// Float64Array(19*3) (upper body 0-10 copied verbatim), or null if degenerate.
function bottomUpBuild(capturedKpts, stanceWidth, toeOutDeg, dorsiDeltaDeg, kneeFlexDeltaDeg) {{
    for (const idx of [11,12,13,14,15,16]) if (!capturedKpts[idx]) return null;
    const DEG = Math.PI / 180;
    const dKnee = (kneeFlexDeltaDeg || 0) * DEG;
    const out = new Float64Array(19 * 3);
    for (let i = 0; i < 19; i++) {{
        if (!capturedKpts[i]) continue;
        out[i*3] = capturedKpts[i][0]; out[i*3+1] = capturedKpts[i][1]; out[i*3+2] = capturedKpts[i][2];
    }}
    const hipL = capturedKpts[11], hipR = capturedKpts[12];
    const latX = hipR[0]-hipL[0], latZ = hipR[2]-hipL[2];
    const latLen = Math.sqrt(latX*latX + latZ*latZ) || 1e-9;
    const latU = [latX/latLen, 0, latZ/latLen];           // hip lateral axis (ground, L->R)

    const ankL = capturedKpts[15], ankR = capturedKpts[16];
    const ankMidX = (ankL[0]+ankR[0])/2, ankMidZ = (ankL[2]+ankR[2])/2;
    const spanX = ankR[0]-ankL[0], spanZ = ankR[2]-ankL[2];
    const spanLen = Math.sqrt(spanX*spanX + spanZ*spanZ) || 1e-9;
    const spanU = [spanX/spanLen, 0, spanZ/spanLen];      // inter-ankle direction (L->R)
    const spanScale = stanceWidth / (baselineStanceWidth || 1e-9);
    const dToe = (toeOutDeg - baselineToeOut) * DEG;
    const dDorsi = dorsiDeltaDeg * DEG;

    const sides = [
        {{ aIdx:15, kIdx:13, hIdx:11, tIdx:17, sign:-1 }},
        {{ aIdx:16, kIdx:14, hIdx:12, tIdx:18, sign:+1 }},
    ];
    const hipEst = [];
    for (const S of sides) {{
        const A0 = capturedKpts[S.aIdx], K0 = capturedKpts[S.kIdx], H0 = capturedKpts[S.hIdx];
        const T0 = capturedKpts[S.tIdx] || null;
        // 1. new grounded ankle: scale lateral span about the ankle midpoint
        const halfDist = (spanLen/2) * spanScale * S.sign;
        const A = [ankMidX + spanU[0]*halfDist, A0[1], ankMidZ + spanU[2]*halfDist];
        // 2/3. shin = captured (knee-ankle), yawed by toe-out then tilted by dorsi
        let shin = [K0[0]-A0[0], K0[1]-A0[1], K0[2]-A0[2]];
        shin = rotateYvec(shin, S.sign * dToe);
        shin = rotateAboutAxis(shin, latU, dDorsi);
        const K = [A[0]+shin[0], A[1]+shin[1], A[2]+shin[2]];
        // 4. thigh = captured (hip-knee), yawed with the leg
        let thigh = [H0[0]-K0[0], H0[1]-K0[1], H0[2]-K0[2]];
        thigh = rotateYvec(thigh, S.sign * dToe);
        if (dKnee !== 0) {{ thigh = rotateAboutAxis(thigh, latU, dKnee); }}
        hipEst.push([K[0]+thigh[0], K[1]+thigh[1], K[2]+thigh[2]]);
        out[S.aIdx*3]=A[0]; out[S.aIdx*3+1]=A[1]; out[S.aIdx*3+2]=A[2];
        out[S.kIdx*3]=K[0]; out[S.kIdx*3+1]=K[1]; out[S.kIdx*3+2]=K[2];
        if (T0) {{
            let foot = [T0[0]-A0[0], T0[1]-A0[1], T0[2]-A0[2]];
            foot = rotateYvec(foot, S.sign * dToe);
            out[S.tIdx*3]=A[0]+foot[0]; out[S.tIdx*3+1]=A[1]+foot[1]; out[S.tIdx*3+2]=A[2]+foot[2];
        }}
    }}
    // 5. rigid pelvis reconciliation: hip-mid = avg estimate, keep captured half-vector
    const hipMidX = (hipEst[0][0]+hipEst[1][0])/2;
    const hipMidY = (hipEst[0][1]+hipEst[1][1])/2;
    const hipMidZ = (hipEst[0][2]+hipEst[1][2])/2;
    const halfX = (hipR[0]-hipL[0])/2, halfY = (hipR[1]-hipL[1])/2, halfZ = (hipR[2]-hipL[2])/2;
    out[11*3]=hipMidX-halfX; out[11*3+1]=hipMidY-halfY; out[11*3+2]=hipMidZ-halfZ;
    out[12*3]=hipMidX+halfX; out[12*3+1]=hipMidY+halfY; out[12*3+2]=hipMidZ+halfZ;
    // 6. re-ground guard (no-op by construction, protects against a dipped ankle)
    const minAnkleY = Math.min(out[15*3+1], out[16*3+1]);
    if (isFinite(minAnkleY) && minAnkleY < 0) {{ for (let i = 11; i <= 18; i++) out[i*3+1] -= minAnkleY; }}
    return out;
}}

// Deform the captured pose from the live sliders, preserving identity at baseline
// via delta-FK (captured + (FK_mod - FK_base)). The upper body (0-10) rides rigidly
// with the pelvis so the captured torso length/lean are preserved.
function deformLowerBody(capturedKpts) {{
    const out = new Float64Array(19 * 3);
    for (let i = 0; i < 19; i++) {{
        if (!capturedKpts[i]) continue;
        out[i*3] = capturedKpts[i][0]; out[i*3+1] = capturedKpts[i][1]; out[i*3+2] = capturedKpts[i][2];
    }}
    const sEl = document.getElementById('sb-stance-width');
    const tEl = document.getElementById('sb-toe-out');
    const dEl = document.getElementById('sb-d-dorsi');
    const kEl = document.getElementById('sb-d-knee-flex');
    const stance = sEl ? parseFloat(sEl.value) : baselineStanceWidth;
    const toeOut = tEl ? parseFloat(tEl.value) : baselineToeOut;
    const dorsi  = dEl ? parseFloat(dEl.value) : 0;
    const kneeDelta = kEl ? parseFloat(kEl.value) : 0;
    if (stance === baselineStanceWidth && toeOut === baselineToeOut && dorsi === 0 && kneeDelta === 0) return out;
    const base = bottomUpBuild(capturedKpts, baselineStanceWidth, baselineToeOut, 0, 0);
    const mod  = bottomUpBuild(capturedKpts, stance, toeOut, dorsi, kneeDelta);
    if (!base || !mod) return out;
    for (let i = 11; i <= 18; i++) {{
        if (!capturedKpts[i]) continue;
        for (let c = 0; c < 3; c++) {{
            const d = mod[i*3+c] - base[i*3+c];
            if (Number.isFinite(d)) out[i*3+c] = capturedKpts[i][c] + d;
        }}
    }}
    // Lock thigh + shin to captured lengths: re-solve each knee from the fixed hip
    // (rigid pelvis) and the fixed grounded ankle. Feet and hips stay put.
    for (const [hI, kI, aI] of [[11,13,15],[12,14,16]]) {{
        if (!capturedKpts[hI] || !capturedKpts[kI] || !capturedKpts[aI]) continue;
        const thighLen = Math.hypot(capturedKpts[hI][0]-capturedKpts[kI][0], capturedKpts[hI][1]-capturedKpts[kI][1], capturedKpts[hI][2]-capturedKpts[kI][2]);
        const shinLen = Math.hypot(capturedKpts[kI][0]-capturedKpts[aI][0], capturedKpts[kI][1]-capturedKpts[aI][1], capturedKpts[kI][2]-capturedKpts[aI][2]);
        const K = solveKnee(
            [out[hI*3], out[hI*3+1], out[hI*3+2]],
            [out[aI*3], out[aI*3+1], out[aI*3+2]],
            thighLen, shinLen,
            [out[kI*3], out[kI*3+1], out[kI*3+2]],
        );
        out[kI*3]=K[0]; out[kI*3+1]=K[1]; out[kI*3+2]=K[2];
    }}
    if (capturedKpts[11] && capturedKpts[12]) {{
        const capMidX=(capturedKpts[11][0]+capturedKpts[12][0])/2;
        const capMidY=(capturedKpts[11][1]+capturedKpts[12][1])/2;
        const capMidZ=(capturedKpts[11][2]+capturedKpts[12][2])/2;
        const defMidX=(out[11*3]+out[12*3])/2;
        const defMidY=(out[11*3+1]+out[12*3+1])/2;
        const defMidZ=(out[11*3+2]+out[12*3+2])/2;
        const dX=defMidX-capMidX, dY=defMidY-capMidY, dZ=defMidZ-capMidZ;
        for (let i = 0; i <= 10; i++) {{ if(!capturedKpts[i]) continue; out[i*3]+=dX; out[i*3+1]+=dY; out[i*3+2]+=dZ; }}
    }}
    return out;
}}

// Apply a counterbalance (hip horizontal shift + trunk lean) to a deformed pose so
// the total COM tracks the mid-foot. The pelvis + upper body (0-12) translate by the
// shift; the trunk (0-10) then leans about the shifted hip-mid. Lower legs stay put.
function applyCounterbalance(srcKpts, hipShiftX, hipShiftZ, leanDeltaRad) {{
    const k = new Float64Array(srcKpts);
    for (let i = 0; i <= 12; i++) {{ k[i*3] += hipShiftX; k[i*3+2] += hipShiftZ; }}
    const hipMidX = (k[11*3] + k[12*3]) / 2;
    const hipMidY = (k[11*3+1] + k[12*3+1]) / 2;
    const shMidX = (k[5*3] + k[6*3]) / 2;
    const shMidY = (k[5*3+1] + k[6*3+1]) / 2;
    const tDX = shMidX - hipMidX, tDY = shMidY - hipMidY;
    const torsoLen = Math.sqrt(tDX*tDX + tDY*tDY) || 1e-9;
    const currentLean = Math.atan2(tDX, tDY);
    const newLean = currentLean + leanDeltaRad;
    const newShMidX = hipMidX + torsoLen * Math.sin(newLean);
    const newShMidY = hipMidY + torsoLen * Math.cos(newLean);
    const offX = newShMidX - shMidX, offY = newShMidY - shMidY;
    for (let i = 0; i <= 10; i++) {{ k[i*3] += offX; k[i*3+1] += offY; }}
    // Only when the hip is shifted (relative to the planted feet) do the knees need
    // re-solving to keep thigh + shin locked. With no shift (the Balance case) the
    // lower body — hips included — is left completely untouched.
    if (hipShiftX !== 0 || hipShiftZ !== 0) {{
        for (const [hI, kI, aI] of [[11,13,15],[12,14,16]]) {{
            const thighLen = Math.hypot(srcKpts[hI*3]-srcKpts[kI*3], srcKpts[hI*3+1]-srcKpts[kI*3+1], srcKpts[hI*3+2]-srcKpts[kI*3+2]);
            const shinLen = Math.hypot(srcKpts[kI*3]-srcKpts[aI*3], srcKpts[kI*3+1]-srcKpts[aI*3+1], srcKpts[kI*3+2]-srcKpts[aI*3+2]);
            const K = solveKnee(
                [k[hI*3], k[hI*3+1], k[hI*3+2]],
                [k[aI*3], k[aI*3+1], k[aI*3+2]],
                thighLen, shinLen,
                [k[kI*3], k[kI*3+1], k[kI*3+2]],
            );
            k[kI*3]=K[0]; k[kI*3+1]=K[1]; k[kI*3+2]=K[2];
        }}
    }}
    return {{ kpts:k, hipMidX, hipMidY, shoulderMidX:newShMidX, shoulderMidY:newShMidY, newLean, torsoLen }};
}}

function buildSandboxKpts(fd) {{
    if (!fd || !fd.kpts) return null;

    const capturedKpts = fd.kpts;
    const capturedAngles = fd.angles;
    const kneeFlexDelta = parseFloat(document.getElementById('sb-d-knee-flex')?.value || '0');
    const deformed = deformLowerBody(capturedKpts);

    if (_balanceLocked) {{
        // Counterbalance: shift hips + lean trunk so total COM tracks the mid-foot.
        const posed = applyCounterbalance(
            deformed, _balanceHipShiftX, _balanceHipShiftZ,
            _balanceLeanOffsetDeg * (Math.PI / 180),
        );
        const kpts = posed.kpts;
        const totalTrunkLeanDeg = posed.newLean * (180 / Math.PI);
        const trunkAngleDeg = 180 - totalTrunkLeanDeg;
        return {{ kpts, trunkAngleDeg, avgKneeDeg: capturedAngles.knee_flex + kneeFlexDelta,
                  dorsiDeg: (capturedAngles.dorsi_l + capturedAngles.dorsi_r) / 2,
                  totalTrunkLeanDeg,
                  dorsiLDeg: capturedAngles.dorsi_l, dorsiRDeg: capturedAngles.dorsi_r,
                  kfLDeg: (capturedAngles.knee_flex_l || capturedAngles.knee_flex) + kneeFlexDelta,
                  kfRDeg: (capturedAngles.knee_flex_r || capturedAngles.knee_flex) + kneeFlexDelta,
                  valLDeg: capturedAngles.knee_valgus_l, valRDeg: capturedAngles.knee_valgus_r,
                  hipFlexL: capturedAngles.hip_flex_l, hipFlexR: capturedAngles.hip_flex_r,
                  hipMidX: posed.hipMidX, hipMidY: posed.hipMidY,
                  shoulderMidX: posed.shoulderMidX, shoulderMidY: posed.shoulderMidY,
                  totalTrunkLean: posed.newLean, toeOutRad: 0, torsoLen: posed.torsoLen, shoulderWidth: 0 }};
    }}

    // Non-locked path: deformed pose as-is (identity when sliders are at baseline)
    const kpts = deformed;
    const hipMidX = (kpts[11 * 3] + kpts[12 * 3]) / 2;
    const hipMidY = (kpts[11 * 3 + 1] + kpts[12 * 3 + 1]) / 2;
    const shoulderMidX = (kpts[5 * 3] + kpts[6 * 3]) / 2;
    const shoulderMidY = (kpts[5 * 3 + 1] + kpts[6 * 3 + 1]) / 2;
    const totalTrunkLeanDeg = 180 - capturedAngles.trunk_flexion;
    const trunkAngleDeg = capturedAngles.trunk_flexion;
    const totalTrunkLean = totalTrunkLeanDeg * Math.PI / 180;

    return {{ kpts, trunkAngleDeg, avgKneeDeg: capturedAngles.knee_flex + kneeFlexDelta,
              dorsiDeg: (capturedAngles.dorsi_l + capturedAngles.dorsi_r) / 2,
              totalTrunkLeanDeg,
              dorsiLDeg: capturedAngles.dorsi_l, dorsiRDeg: capturedAngles.dorsi_r,
              kfLDeg: (capturedAngles.knee_flex_l || capturedAngles.knee_flex) + kneeFlexDelta,
              kfRDeg: (capturedAngles.knee_flex_r || capturedAngles.knee_flex) + kneeFlexDelta,
              valLDeg: capturedAngles.knee_valgus_l, valRDeg: capturedAngles.knee_valgus_r,
              hipFlexL: capturedAngles.hip_flex_l, hipFlexR: capturedAngles.hip_flex_r,
              hipMidX, hipMidY, shoulderMidX, shoulderMidY,
              totalTrunkLean, toeOutRad: 0, torsoLen: 0, shoulderWidth: 0,
              anyDeltaActive: false }};
}}


// ======== COM / BOS / BALANCE ========
function computeCOM(kpts, bw, bodyMass, footLen) {{
    const segCOMs = buildSegmentCOMs(kpts, footLen);
    const bodyFracs = getNormalizedBodyFracs();
    const barFrac = bw > 0 ? bw / (bodyMass + bw) : 0;
    const bodyF = 1 - barFrac;
    let cx = 0, cy = 0, cz = 0;
    for (const [key, frac] of Object.entries(bodyFracs)) {{
        const segmentCom = segCOMs[key];
        if (!segmentCom) continue;
        cx += segmentCom.x * frac * bodyF;
        cy += segmentCom.y * frac * bodyF;
        cz += segmentCom.z * frac * bodyF;
    }}
    if (bw > 0 && _barbellPos) {{
        cx += _barbellPos.x * barFrac;
        cy += _barbellPos.y * barFrac;
        cz += _barbellPos.z * barFrac;
    }}
    return {{ x: cx, y: cy, z: cz, groundX: cx, groundZ: cz }};
}}

function computeBOS(kpts, toeOutRad, footLen) {{
    function g(i) {{ return {{ x: kpts[i*3], y: kpts[i*3+1], z: kpts[i*3+2] }}; }}
    const aL = g(15), aR = g(16), kL = g(13), kR = g(14);
    const halfW = 0.05, heelOff = HEEL_OFFSET, toeOff = footLen - heelOff;
    function footRect(ankle, knee) {{
        const dx=knee.x-ankle.x, dz=knee.z-ankle.z, l=Math.sqrt(dx*dx+dz*dz)||1;
        const fx=dx/l, fz=dz/l, lx=-fz, lz=fx;
        return [
            {{ x:ankle.x-fx*heelOff-lx*halfW, z:ankle.z-fz*heelOff-lz*halfW }},
            {{ x:ankle.x-fx*heelOff+lx*halfW, z:ankle.z-fz*heelOff+lz*halfW }},
            {{ x:ankle.x+fx*toeOff+lx*halfW, z:ankle.z+fz*toeOff+lz*halfW }},
            {{ x:ankle.x+fx*toeOff-lx*halfW, z:ankle.z+fz*toeOff-lz*halfW }},
        ];
    }}
    const pts = [...footRect(aL, kL), ...footRect(aR, kR)];
    return convexHull2D(pts);
}}
function convexHull2D(points) {{
    if (points.length <= 3) return points;
    let start = 0;
    for (let i = 1; i < points.length; i++) {{
        if (points[i].x < points[start].x || (points[i].x === points[start].x && points[i].z < points[start].z)) start = i;
    }}
    const hull = []; let cur = start;
    let guard = 0;
    do {{
        hull.push(points[cur]); let next = (cur+1) % points.length;
        for (let i = 0; i < points.length; i++) {{
            if (i === cur || i === next) continue;
            const cross = (points[next].x-points[cur].x)*(points[i].z-points[cur].z) - (points[next].z-points[cur].z)*(points[i].x-points[cur].x);
            if (cross < 0) next = i;
        }}
        cur = next;
        guard++;
    }} while (cur !== start && hull.length < points.length + 1 && guard <= points.length + 2);
    return hull.length >= 3 ? hull : points;
}}
function isBalanced(com, bos) {{
    const px=com.groundX, pz=com.groundZ; let inside=false; const n=bos.length;
    for (let i=0,j=n-1; i<n; j=i++) {{
        const xi=bos[i].x,zi=bos[i].z,xj=bos[j].x,zj=bos[j].z;
        if (((zi>pz)!==(zj>pz)) && (px<(xj-xi)*(pz-zi)/(zj-zi)+xi)) inside=!inside;
    }}
    let minDist=Infinity;
    for (let i=0,j=n-1; i<n; j=i++) {{
        const ax=bos[j].x,az=bos[j].z,bx=bos[i].x,bz=bos[i].z;
        const ex=bx-ax,ez=bz-az,t=Math.max(0,Math.min(1,((px-ax)*ex+(pz-az)*ez)/(ex*ex+ez*ez+1e-9)));
        const dx=px-(ax+t*ex),dz=pz-(az+t*ez); minDist=Math.min(minDist,Math.sqrt(dx*dx+dz*dz));
    }}
    let cx=0,cz2=0; for (const pt of bos) {{ cx+=pt.x; cz2+=pt.z; }} cx/=n; cz2/=n;
    let avgR=0; for (const pt of bos) avgR+=Math.sqrt((pt.x-cx)**2+(pt.z-cz2)**2); avgR/=n;
    return {{ inside, marginRatio: inside ? minDist/(avgR+1e-9) : -minDist/(avgR+1e-9) }};
}}

function getBarbellPosition(pose) {{
    const {{ shoulderMidX, shoulderMidY, totalTrunkLean, sw }} = pose;
    const s = Math.sin(totalTrunkLean), c = Math.cos(totalTrunkLean);
    const bx = shoulderMidX + 0.04*s + 0.05*(-c);
    const by = shoulderMidY + 0.04*c + 0.05*s;
    _barbellPos = {{ x:bx, y:by, z:0 }};
    return {{ x:bx, y:by, z:0, trunkLean: totalTrunkLean, sw }};
}}
function updateBarbellVisuals(bw, barPos, sw) {{
    if (!bw || bw <= 0) {{ barbellGroup.visible = false; return; }}
    barbellGroup.visible = true;
    const barLen = sw * 2.4; barMesh.scale.set(1, barLen, 1);
    barbellGroup.position.set(barPos.x, barPos.y, barPos.z); barbellGroup.rotation.set(0,0,0);
    const plateR = 0.05 + (bw/200)*0.15, halfBar = barLen/2;
    const spacings = [0.10, 0.16];
    for (let i = 0; i < 4; i++) {{
        const side = i < 2 ? -1 : 1, slot = i % 2;
        plateMeshes[i].scale.set(plateR/0.05, plateR/0.05, 1);
        plateMeshes[i].position.set(0, 0, side*(halfBar-0.06-spacings[slot]));
    }}
}}
function updateCOMVisuals(com, bos, bal) {{
    if (!com || !bos) {{ comSphere.visible=false; comDisc.visible=false; comLine.visible=false; bosLine.visible=false; return; }}
    comSphere.position.set(com.x,com.y,com.z); comSphere.visible=true;
    comDisc.position.set(com.groundX,0.002,com.groundZ); comDisc.visible=true;
    const lp = new Float32Array([com.groundX,0.003,com.groundZ, com.x,com.y,com.z]);
    comLineGeo.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
    comLine.computeLineDistances(); comLine.visible=true;
    const bp = []; for (const pt of bos) bp.push(pt.x, 0.002, pt.z);
    bosGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(bp), 3));
    bosLine.visible=true;
    comSphere.material.color.setHex(bal.inside ? 0x22ff22 : 0xff2222);
    comDisc.material.color.setHex(bal.inside ? 0x22ff22 : 0xff2222);
    bosLineMat.color.setHex(bal.inside ? 0x44ff88 : 0xff6644);
}}

function _sv(id) {{ const el = document.getElementById(id); return el ? parseFloat(el.value) : 0; }}

// ======== SANDBOX UPDATE ========

function updateSandbox(fd) {{
    if (!fd || !fd.kpts) return;

    const pose = buildSandboxKpts(fd);
    if (!pose) return;
    const {{ kpts, trunkAngleDeg, avgKneeDeg, dorsiDeg, totalTrunkLeanDeg,
        dorsiLDeg, dorsiRDeg, kfLDeg, kfRDeg, valLDeg, valRDeg,
        hipFlexL, hipFlexR }} = pose;

    // Fault classification from effective per-side valgus
    const faultValgus = Math.max(Math.abs(valLDeg), Math.abs(valRDeg));
    const leanSev = classifyLean(trunkAngleDeg);
    const valgusSev = classifyValgus(faultValgus);
    const fj = new Set();
    if (leanSev !== 'none') [0,1,2,3,4,5,6].forEach(j => fj.add(j));
    if (valgusSev !== 'none') [13,14].forEach(j => fj.add(j));

    for (let i = 0; i < 19; i++) {{
        jm[i].position.set(kpts[i*3], kpts[i*3+1], kpts[i*3+2]);
        jm[i].visible = true;
        const t = fj.has(i) ? 'f' : 'n';
        if (t !== js[i]) {{ jm[i].material = (t === 'f' ? matF : matN).clone(); js[i] = t; }}
    }}
    const fb = new Set();
    if (leanSev !== 'none') {{ fb.add('5-6'); fb.add('5-11'); fb.add('6-12'); fb.add('0-5'); fb.add('0-6'); }}
    if (valgusSev !== 'none') {{ fb.add('11-13'); fb.add('13-15'); fb.add('12-14'); fb.add('14-16'); }}
    for (const bone of bm) {{
        const pa = jm[bone.a].position, pb = jm[bone.b].position;
        const d = new THREE.Vector3().subVectors(pb, pa), l = d.length(); d.normalize();
        bone.mesh.position.copy(pa); bone.mesh.scale.set(1, l, 1);
        bone.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), d);
        bone.mesh.material = fb.has(`${{bone.a}}-${{bone.b}}`) ? matBF : matB;
    }}

    hideSandboxVisuals();

    const sbBodyMass = _sv('sb-body-mass') || 75;
    const sbBarWeight = _sv('sb-barbell-weight') || 0;
    const sbCom = computeCOM(kpts, sbBarWeight, sbBodyMass, FOOT_LEN_M);
    const sbBos = computeBOS(kpts, 0, FOOT_LEN_M);
    const sbBal = isBalanced(sbCom, sbBos);
    updateCOMVisuals(sbCom, sbBos, sbBal);
    updateBalanceTargetVisual(kpts, FOOT_LEN_M);

    if (_balanceLocked) {{
        const trunkLeanOffset = 180 - trunkAngleDeg;
        const leanSeverity = classifyLean(trunkAngleDeg);
        const leanClass = leanSeverity === 'none' ? 'balance-ok' : 'balance-bad';
        const trunkLeanEl = document.getElementById('sb-trunk-lean-status');
        if (trunkLeanEl) {{
            trunkLeanEl.innerHTML = `<span class="${{leanClass}}">Trunk lean: ${{trunkLeanOffset.toFixed(1)}}°</span> ${{leanSeverity !== 'none' ? sb(leanSeverity) : '✓'}}`;
        }}
    }}

    // Barbell visual
    if (sbBarWeight > 0 && _barbellPos) {{
        barbellGroup.visible = true;
        const barY = pose.shoulderMidY || 0;
        const barX = pose.shoulderMidX || 0;
        barbellGroup.position.set(barX - 0.05 * Math.cos(pose.totalTrunkLean || 0),
                                   barY + 0.04 * Math.sin(pose.totalTrunkLean || 0), 0);
        _barbellPos = {{ x: barbellGroup.position.x, y: barbellGroup.position.y, z: 0 }};
    }} else {{
        barbellGroup.visible = false;
    }}

    const leanOff = (180 - trunkAngleDeg).toFixed(1);
    const balPct = (sbBal.marginRatio * 100).toFixed(1);
    const balCls = sbBal.inside && sbBal.marginRatio >= BALANCE_MARGIN_MIN ? 'balance-ok' : 'balance-bad';
    document.getElementById('sb-angles-info').innerHTML = `
        <span class="lbl">Knee Flex L/R:</span> <span class="val">${{kfLDeg.toFixed(1)}}° / ${{kfRDeg.toFixed(1)}}°</span>
        <span class="val" style="opacity:0.5">(avg ${{avgKneeDeg.toFixed(1)}}°)</span><br>
        <span class="lbl">Trunk Angle:</span> <span class="val">${{trunkAngleDeg.toFixed(1)}}°</span> (offset: ${{leanOff}}°)
        ${{leanSev !== 'none' ? sb(leanSev) : ''}}<br>
        <span class="lbl">Dorsi L/R:</span> <span class="val">${{dorsiLDeg.toFixed(1)}}° / ${{dorsiRDeg.toFixed(1)}}°</span>
        <span class="val" style="opacity:0.5">(avg ${{dorsiDeg.toFixed(1)}}°)</span><br>
        <span class="lbl">Valgus L/R:</span> <span class="val">${{valLDeg.toFixed(1)}}° / ${{valRDeg.toFixed(1)}}°</span>
        ${{valgusSev !== 'none' ? sb(valgusSev) : ''}}<br>
        <span class="lbl">Hip Flex L/R:</span> <span class="val">${{hipFlexL.toFixed(1)}}° / ${{hipFlexR.toFixed(1)}}°</span><br>
        <span class="lbl">Phase:</span> <span class="val">${{(fd.phase || 0).toFixed(3)}}</span><br>
        <span class="lbl">Balance:</span> <span class="${{balCls}}">${{sbBal.inside ? 'OK' : 'UNSTABLE'}}</span>
        <span class="val" style="opacity:0.5"> (margin ${{balPct}}%)</span>
        ${{_balanceLocked ? '<span style="color:#f0a040; margin-left:8px">COM locked</span>' : ''}}`;

    document.getElementById('info-overlay').innerHTML = `
        <span style="color:#4ecdc4">SANDBOX</span>
        <span class="lbl" style="margin-left:8px">Rep:</span> <span class="val">${{curRep+2}}</span>
        <span class="lbl" style="margin-left:8px">Frame:</span> <span class="val">${{curFrame+1}}</span><br>
        <span class="lbl">Knee:</span> <span class="val">${{avgKneeDeg.toFixed(1)}}°</span>
        <span class="lbl" style="margin-left:8px">Trunk:</span> <span class="val">${{trunkAngleDeg.toFixed(1)}}°</span>
        <span class="lbl" style="margin-left:8px">Dorsi:</span> <span class="val">${{dorsiDeg.toFixed(1)}}°</span>`;
}}

// ======== RESIZE ========
function resize() {{ const w=container.clientWidth, h=container.clientHeight; renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix(); }}
window.addEventListener('resize', resize); resize();

// ======== MAIN ANIMATION LOOP ========
function animate(now) {{
    requestAnimationFrame(animate);
    const dt = (now - lastT) / 1000; lastT = now;
    if (!reps.length) return;
    const r = reps[curRep]; if (!r || !r.length) return;

    // Frame advancement (shared between modes)
    if (playing) {{
        frameAcc += dt * dataFps * speed;
        while (frameAcc >= 1) {{
            frameAcc -= 1; curFrame++;
            if (curFrame >= r.length) {{
                if (repFilter === -1) curRep = (curRep + 1) % reps.length;
                curFrame = 0;
            }}
        }}
    }}
    if (curFrame >= r.length) curFrame = r.length - 1;

    // Update scrubbers (whichever panel is visible)
    scrub.max = r.length - 1; scrub.value = curFrame;
    fv.textContent = `${{curFrame+1}}/${{r.length}}`;
    sbScrub.max = r.length - 1; sbScrub.value = curFrame;
    sbFv.textContent = `${{curFrame+1}}/${{r.length}}`;
    const diagScrubEl = document.getElementById('diag-frame-scrubber');
    const diagFvEl = document.getElementById('diag-frame-val');
    if (diagScrubEl) {{ diagScrubEl.max = r.length - 1; diagScrubEl.value = curFrame; }}
    if (diagFvEl) {{ diagFvEl.textContent = `${{curFrame+1}}/${{r.length}}`; }}

    const fd = r[curFrame];

    if (viewMode === 'sandbox') {{
        updateSandbox(fd);
    }} else if (viewMode === 'diagnosis') {{
        updateReplay(fd);
        if (diagViewMode === 'rep') {{
            updateDiagnosisMorph(now);
        }}
        hideSandboxVisuals();
    }} else {{
        updateReplay(fd);
        hideSandboxVisuals();
        hideDiagnosisVisuals();
    }}

    orbitCtrl.update();
    renderer.render(scene, camera);
}}
requestAnimationFrame(animate);
</script>
</body>
</html>"""


def _build_voice_cues(diagnosis, rep_summary, anthro, rom):
    """Build the exact dict the edge LLM receives to produce a voice cue.

    Keys are fault names, values are compact structured corrections with
    observed/target numbers — no prose. This is what we verify in the
    visualizer and what ships to the voice agent.
    """
    import math as _math
    from biomechanics.diagnosis.graph.parameter_deltas import dorsi_driven_targets

    cues = {}
    for cause in diagnosis.immediate_causes:
        cid = cause.cause_id
        delta = cause.parameter_delta

        if cid == "narrow_stance" and delta:
            foot_delta = delta.get("__foot_target_delta", [0] * 6)
            widen_per_side_cm = round(abs(foot_delta[2]) * 100, 1)
            current = round(rep_summary.stance_width_ratio, 2)
            dorsi_cap = rom.get("peak_dorsiflexion", 35.0)
            target_ratio, _ = dorsi_driven_targets(dorsi_cap, anthro)
            target = round(max(target_ratio, current + 0.15), 2)
            cues[cid] = {
                "fix": "widen stance",
                "current_ratio": current,
                "target_ratio": target,
                "widen_per_side_cm": widen_per_side_cm,
                "unit": "x shoulder width",
            }

        elif cid == "narrow_foot_angle" and delta:
            delta_deg = round(_math.degrees(abs(delta.get("L_ankle.ry", 0.0))), 1)
            current = round((rep_summary.foot_direction_angle_l + rep_summary.foot_direction_angle_r) / 2.0, 1)
            target = round(current + delta_deg, 1)
            cues[cid] = {
                "fix": "turn feet out",
                "current_deg": current,
                "target_deg": target,
                "increase_deg": delta_deg,
            }

        elif cid == "bracing_failure" and delta:
            correction_deg = round(_math.degrees(abs(delta.get("trunk.rx", 0.0))), 1)
            cues[cid] = {
                "fix": "brace core harder",
                "trunk_correction_deg": correction_deg,
            }

        elif cid == "knee_track_cue" and delta:
            push_out_deg = round(_math.degrees(abs(delta.get("L_hip.ry", 0.0))), 1)
            valgus = round(max(rep_summary.knee_valgus_l, rep_summary.knee_valgus_r), 1)
            cues[cid] = {
                "fix": "push knees out",
                "current_valgus_deg": valgus,
                "correction_deg": push_out_deg,
            }

        elif cid == "weight_shift_cue" and delta:
            shift_cm = round(abs(delta.get("pelvis.tx", 0.0)) * 100, 1)
            side = "left" if rep_summary.hip_y_l_at_bottom < rep_summary.hip_y_r_at_bottom else "right"
            cues[cid] = {
                "fix": "center weight",
                "shift_toward": side,
                "shift_cm": shift_cm,
            }

        elif cid == "depth_cue_unfamiliar":
            cues[cid] = {
                "fix": "squat deeper",
                "target": "parallel (hip crease at knee level)",
            }

    return cues


def _diagnosis_to_tiers(diagnosis):
    """Convert a DiagnosisResult's causes into tier dict for the viewer."""
    tier_labels = {1: "Cue-correctable", 2: "Session-level", 3: "Long-term", 0: "Contextual"}
    tiers = {}
    all_causes = (
        diagnosis.immediate_causes
        + diagnosis.session_causes
        + diagnosis.longterm_causes
        + diagnosis.contextual_notes
    )
    for cause in all_causes:
        tier_key = str(cause.tier)
        if tier_key not in tiers:
            tiers[tier_key] = {"label": tier_labels.get(cause.tier, "Other"), "causes": []}
        tiers[tier_key]["causes"].append({
            "cause_id": cause.cause_id,
            "score": cause.score,
            "explanation": cause.explanation,
            "implicated_by": cause.implicated_by,
        })
    return tiers


def _diagnosis_to_symptoms(diagnosis):
    """Convert DetectedSymptom list to viewer-friendly dicts."""
    return [
        {
            "id": symptom.symptom_id,
            "severity": symptom.severity,
            "contributing_reps": symptom.contributing_reps,
        }
        for symptom in diagnosis.detected_symptoms
    ]


def run_diagnosis(replay_reps, athlete_params, baseline):
    """Run the diagnosis pipeline on captured/loaded session data."""
    from biomechanics.diagnosis import HypothesisEngine
    from biomechanics.diagnosis.bridge import (
        build_rep_kinematic_summary,
        build_set_features,
        find_bottom_frame,
    )
    from biomechanics.diagnosis.keypoint_corrector import (
        KeypointCorrector,
        build_morph_frames,
    )
    from biomechanics.diagnosis.rep_scoring import score_set

    # Set-level diagnosis (aggregates all reps)
    set_features = build_set_features(replay_reps, athlete_params, baseline)
    engine = HypothesisEngine()
    set_diagnosis = engine.diagnose(set_features)

    # Per-rep quality scoring
    set_score_summary = score_set(
        set_features.per_rep_kinematics,
        set_features.anthropometry,
        set_features.rom,
    )

    print(f"  Confidence: {set_diagnosis.confidence:.0%}")
    print(f"  Symptoms: {[s.symptom_id for s in set_diagnosis.detected_symptoms]}")
    print(f"  Tier-1 causes: {[c.cause_id for c in set_diagnosis.immediate_causes]}")

    corrector = KeypointCorrector()
    per_rep_data = []

    for rep_idx, rep_frames in enumerate(replay_reps):
        bottom_frame = find_bottom_frame(rep_frames)
        observed_kpts = bottom_frame["kpts"]
        rep_number = rep_idx + 2

        # Per-rep diagnosis (single-rep SetFeatures)
        single_rep_features = build_set_features([rep_frames], athlete_params, baseline)
        rep_diagnosis = engine.diagnose(single_rep_features)

        corrected_kpts = corrector.correct(
            observed_kpts, rep_diagnosis,
            anthro=set_features.anthropometry,
            rom=set_features.rom,
        )
        has_correction = corrected_kpts is not None
        morph_frames = None
        if has_correction:
            morph_frames = build_morph_frames(observed_kpts, corrected_kpts, num_frames=60)

        # Voice cues — the exact dict the edge LLM would receive
        rep_summary = build_rep_kinematic_summary(bottom_frame, athlete_params, rep_number)
        voice_cues = _build_voice_cues(
            rep_diagnosis, rep_summary,
            set_features.anthropometry, set_features.rom,
        )
        if voice_cues:
            import json as _json
            print(f"  Rep {rep_number} voice cues: {_json.dumps(voice_cues, indent=2)}")

        # Kinematic metrics for sparkline comparison
        metrics = {
            "trunk_lean": round(rep_summary.trunk_pitch_at_bottom, 1),
            "knee_valgus": round(max(rep_summary.knee_valgus_l, rep_summary.knee_valgus_r), 1),
            "depth_angle": round(bottom_frame["angles"].get("knee_flex", 0.0), 1),
            "dorsiflexion": round(max(rep_summary.ankle_df_l_max, rep_summary.ankle_df_r_max), 1),
        }

        rep_score = set_score_summary.per_rep_scores[rep_idx]
        per_rep_data.append({
            "rep_number": rep_number,
            "observed_kpts": observed_kpts,
            "corrected_kpts": corrected_kpts,
            "morph_frames": morph_frames,
            "has_correction": has_correction,
            "confidence": rep_diagnosis.confidence,
            "tiers": _diagnosis_to_tiers(rep_diagnosis),
            "symptoms": _diagnosis_to_symptoms(rep_diagnosis),
            "metrics": metrics,
            "quality_score": rep_score.composite_score,
            "sub_scores": {
                "depth": rep_score.depth_score,
                "trunk_control": rep_score.trunk_control_score,
                "knee_tracking": rep_score.knee_tracking_score,
                "symmetry": rep_score.symmetry_score,
                "ankle_utilization": rep_score.ankle_utilization_score,
            },
            "voice_cues": voice_cues,
        })

    return {
        "set_confidence": set_diagnosis.confidence,
        "set_tiers": _diagnosis_to_tiers(set_diagnosis),
        "set_symptoms": _diagnosis_to_symptoms(set_diagnosis),
        "set_score": {
            "mean": set_score_summary.mean_score,
            "best_rep": set_score_summary.best_rep_number,
            "worst_rep": set_score_summary.worst_rep_number,
            "trend_slope": set_score_summary.trend_slope,
        },
        "per_rep": per_rep_data,
        "auto_open": True,
    }


def _plot_ground_plane_valgus(per_rep_data: list, output_path) -> None:
    """Generate a bird's-eye view plot of knee valgus vectors for each rep."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping valgus plot")
        return

    n_reps = len(per_rep_data)
    if n_reps == 0:
        return

    fig, axes = plt.subplots(1, n_reps, figsize=(5 * n_reps, 5), squeeze=False)

    HIP_L, HIP_R = 11, 12
    KNEE_L, KNEE_R = 13, 14
    ANKLE_L, ANKLE_R = 15, 16
    FOOT_L, FOOT_R = 17, 18

    side_cfg = [
        ("Left", HIP_L, KNEE_L, ANKLE_L, FOOT_L, "#2563eb", "#93c5fd"),
        ("Right", HIP_R, KNEE_R, ANKLE_R, FOOT_R, "#dc2626", "#fca5a5"),
    ]

    for rep_idx, rep in enumerate(per_rep_data):
        ax = axes[0][rep_idx]
        kpts = np.array(rep["observed_kpts"])

        for side_name, hip_i, knee_i, ankle_i, foot_i, dark_color, light_color in side_cfg:
            hip_vis = kpts[hip_i]
            knee_vis = kpts[knee_i]
            ankle_vis = kpts[ankle_i]
            foot_vis = kpts[foot_i]

            hip_gp = np.array([-hip_vis[2], hip_vis[0]])
            knee_gp = np.array([-knee_vis[2], knee_vis[0]])
            ankle_gp = np.array([-ankle_vis[2], ankle_vis[0]])
            foot_gp = np.array([-foot_vis[2], foot_vis[0]])

            # Foot line: ankle -> toe.  Femur: hip -> knee.  Valgus is the
            # angle between these two on the floor plane.
            ref_vec = foot_gp - ankle_gp
            knee_vec = knee_gp - hip_gp
            ref_mag = np.linalg.norm(ref_vec)
            knee_mag = np.linalg.norm(knee_vec)

            ax.scatter(*hip_gp, color=dark_color, s=60, zorder=5)
            ax.scatter(*knee_gp, color=dark_color, s=60, zorder=5, marker="D")
            ax.scatter(*ankle_gp, color=dark_color, s=60, zorder=5, marker="s")
            ax.scatter(*foot_gp, color=light_color, s=60, zorder=5, marker="^")

            if ref_mag > 1e-6:
                ax.annotate(
                    "", xy=foot_gp, xytext=ankle_gp,
                    arrowprops=dict(arrowstyle="-|>", color=dark_color, lw=2),
                )
                mid_ref = ankle_gp + ref_vec * 0.5
                ax.text(
                    mid_ref[0], mid_ref[1],
                    f"  foot {ref_mag:.3f}",
                    fontsize=7, color=dark_color, ha="left",
                )

            if knee_mag > 1e-6:
                ax.annotate(
                    "", xy=knee_gp, xytext=hip_gp,
                    arrowprops=dict(
                        arrowstyle="-|>", color=dark_color,
                        lw=2, linestyle="dashed",
                    ),
                )
                mid_knee = hip_gp + knee_vec * 0.5
                ax.text(
                    mid_knee[0], mid_knee[1],
                    f"  knee {knee_mag:.3f}",
                    fontsize=7, color=dark_color, ha="left", style="italic",
                )

        valgus_val = rep["metrics"]["knee_valgus"]
        ax.set_title(f"Rep {rep['rep_number']}  |  valgus {valgus_val:+.1f}°", fontsize=10)
        ax.set_xlabel("X (subject's left →)")
        ax.set_ylabel("Z (forward →)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    left_patch = mpatches.Patch(color="#2563eb", label="Left leg")
    right_patch = mpatches.Patch(color="#dc2626", label="Right leg")
    fig.legend(
        handles=[left_patch, right_patch],
        loc="upper center", ncol=2, fontsize=9, frameon=False,
    )
    fig.suptitle("Ground-Plane Knee Tracking (bird's-eye view)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Valgus plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Capture squats and visualize in 3D")
    parser.add_argument("--output", "-o", default=None,
                        help="Output video path (default: recordings/squat_YYYYMMDD_HHMMSS.mp4)")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open HTML in browser")
    parser.add_argument(
        "--refit",
        nargs="?",
        const="",
        default=None,
        metavar="SESSION",
        help="Regenerate HTML from last session (or a .session.json path); skips webcam capture",
    )
    parser.add_argument(
        "--shoe-size-eur",
        type=float,
        default=46,
        help="EU shoe size for foot length in sandbox balance (default: 46 ≈ 29.3 cm)",
    )
    parser.add_argument(
        "--diagnose",
        nargs="?",
        const="",
        default=None,
        metavar="SESSION",
        help="Run diagnosis on last session (or a .session.json path); shows corrected form",
    )
    args = parser.parse_args()

    recordings_dir = Path(__file__).parent.parent / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.diagnose is not None:
        if args.diagnose:
            session_path = Path(args.diagnose)
        else:
            session_path = resolve_last_session_path(recordings_dir)
            if session_path is None:
                print(
                    "ERROR: No saved session found. Run a full capture first "
                    "(creates recordings/*.session.json)."
                )
                sys.exit(1)

        payload = load_session(session_path)
        replay_reps = payload["replay_reps"]
        athlete_params = payload.get("athlete_params")
        baseline = payload["baseline"]
        fps = payload["fps"]

        if not athlete_params:
            print("ERROR: Session has no athlete params. Re-run a full capture first.")
            sys.exit(1)

        print("=" * 50)
        print("  SQUAT DIAGNOSIS")
        print("=" * 50)
        print(f"  Session → {session_path}")
        print(f"  Reps    → {len(replay_reps)} replay")

        print("\n  Running diagnosis engine...")
        diagnosis_data = run_diagnosis(replay_reps, athlete_params, baseline)

        valgus_plot_path = recordings_dir / f"squat_diagnosis_{timestamp}_valgus.png"
        _plot_ground_plane_valgus(diagnosis_data["per_rep"], valgus_plot_path)

        if args.output:
            html_path = Path(args.output)
            if html_path.suffix.lower() != ".html":
                html_path = html_path.with_suffix(".html")
        else:
            html_path = recordings_dir / f"squat_diagnosis_{timestamp}.html"

        foot_length_m = eur_size_to_foot_length_m(args.shoe_size_eur)
        html = build_html(
            baseline, replay_reps, fps, athlete_params,
            foot_length_m=foot_length_m, diagnosis_data=diagnosis_data,
        )
        html_path.write_text(html)
        print(f"\n  HTML saved: {html_path}")
        if not args.no_open:
            webbrowser.open(f"file://{html_path.resolve()}")
        return

    if args.refit is not None:
        if args.refit:
            session_path = Path(args.refit)
        else:
            session_path = resolve_last_session_path(recordings_dir)
            if session_path is None:
                print(
                    "ERROR: No saved session found. Run a full capture first "
                    "(creates recordings/*.session.json)."
                )
                sys.exit(1)

        if args.output:
            html_path = Path(args.output)
            if html_path.suffix.lower() != ".html":
                html_path = html_path.with_suffix(".html")
        else:
            html_path = recordings_dir / f"squat_refit_{timestamp}.html"

        run_refit(
            session_path,
            html_path,
            open_browser=not args.no_open,
            shoe_size_eur=args.shoe_size_eur,
        )
        return

    video_path = Path(args.output) if args.output else recordings_dir / f"squat_{timestamp}.mp4"
    html_path = video_path.with_suffix(".html")
    session_path = video_path.with_suffix(".session.json")

    print("=" * 50)
    print("  SQUAT CAPTURE & 3D REPLAY")
    print("=" * 50)
    print(f"  Video   → {video_path}")
    print(f"  HTML    → {html_path}")
    print(f"  Session → {session_path}")
    print(f"  Reps to capture: {TARGET_REPS}")
    print("=" * 50)

    frames_data, reps, rep_boundaries, fps, measured = run_capture(args.camera, video_path)

    if len(reps) < 2:
        print(f"ERROR: Need at least 2 reps, got {len(reps)}.")
        sys.exit(1)

    print(f"\nProcessing {len(reps)} reps...")
    print(f"  Using rep 1 as baseline, replaying reps 2-{len(reps)}")

    baseline, replay_reps, athlete_params = process_captured_reps(
        frames_data, rep_boundaries, measured,
    )
    print(f"  Baseline trunk offset: {baseline['peakTrunkOffset']}°")
    print(f"  Lean thresholds: {baseline['leanThresholds']}")
    print(f"  Valgus thresholds: {baseline['valgusThresholds']}")

    if athlete_params:
        print("  Athlete params:")
        print(f"    Stance width: {athlete_params['stanceWidth']}x  "
              f"Toe-out: {athlete_params['toeOut']}°")
        print(f"    Dorsi ratio: {athlete_params['dorsiRatio']}  "
              f"Body scale: {athlete_params['bodyScale']}")
        print(f"    Proportions: torso={athlete_params['torsoRatio']} "
              f"thigh={athlete_params['thighRatio']} shin={athlete_params['shinRatio']}")

    save_session(
        session_path,
        source_video=video_path,
        fps=fps,
        baseline=baseline,
        athlete_params=athlete_params,
        replay_reps=replay_reps,
        rep_count=len(reps),
    )
    print(f"  Session saved: {session_path}")

    foot_length_m = eur_size_to_foot_length_m(args.shoe_size_eur)
    print(f"  Shoe size: EU {args.shoe_size_eur} → foot length {foot_length_m * 100:.1f} cm")

    html = build_html(
        baseline, replay_reps, fps, athlete_params, foot_length_m=foot_length_m,
    )
    html_path.write_text(html)
    print(f"\nVideo saved: {video_path}")
    print(f"HTML saved:  {html_path}")
    print(f"Next refit:  python scripts/visualize_video_squats.py --refit")

    if not args.no_open:
        webbrowser.open(f"file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
